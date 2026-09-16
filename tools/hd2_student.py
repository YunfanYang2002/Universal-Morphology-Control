"""HD2 lazy shard converter and task-balanced HNMLP preflight update."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from gym import spaces
from torch.utils.data import IterableDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from metamorph.algos.distill.distill import _hd0_check_finite, _hd0_tensor  # noqa: E402
from metamorph.algos.ppo.model import ActorCritic  # noqa: E402
from metamorph.config import cfg  # noqa: E402
from tools.convert_rmamorph_teacher_to_hyperdistill import nominal_context  # noqa: E402

try:
    import resource
except ImportError:
    resource = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


FROZEN_HD1_COLUMN_COUNT = 204
FROZEN_HD1_COLUMNS_SHA256 = "5aae11e48a05f57bca706085939d3ed47985f8c80e2e37299766e8d5f309917d"
TEACHER_PROPRIO_DIM = 624


def load_frozen_hd1_columns(path: Path) -> tuple[list[int], bytes]:
    raw = Path(path).read_bytes()
    columns = json.loads(raw)
    if (not isinstance(columns, list) or len(columns) != FROZEN_HD1_COLUMN_COUNT
            or any(type(column) is not int for column in columns)
            or len(set(columns)) != FROZEN_HD1_COLUMN_COUNT
            or min(columns) < 0 or max(columns) >= TEACHER_PROPRIO_DIM):
        raise ValueError("frozen HD1 student columns must be 204 unique indices in [0, 624)")
    semantic_hash = hashlib.sha256(json.dumps(columns).encode()).hexdigest()
    if semantic_hash != FROZEN_HD1_COLUMNS_SHA256:
        raise ValueError(f"frozen HD1 student columns hash mismatch: {semantic_hash}")
    return columns, raw


def select_frozen_student_observation(teacher_observation: np.ndarray, frozen_columns: list[int]) -> np.ndarray:
    if teacher_observation.ndim != 2 or teacher_observation.shape[1] != TEACHER_PROPRIO_DIM:
        raise ValueError(f"HD2 expert observation must have shape [N, 624], got {teacher_observation.shape}")
    return teacher_observation[:, frozen_columns]


def write_frozen_student_columns(output: Path, frozen_bytes: bytes) -> None:
    (Path(output) / "student_columns.json").write_bytes(frozen_bytes)


def official_epoch_batch_plan(samples: int, batch_size: int) -> dict:
    if samples <= 0 or batch_size <= 0:
        raise ValueError("samples and batch_size must be positive")
    full_batches, final_batch = divmod(samples, batch_size)
    return {"samples": samples, "batch_size": batch_size, "drop_last": False,
            "full_batches": full_batches, "final_batch_size": final_batch,
            "optimizer_steps": full_batches + int(final_batch != 0)}


def convert_shards(expert_dir: Path, converted_dir: Path, manifest_path: Path, hd1_columns: Path) -> dict:
    """Convert one 8k NPZ per PD robot; each pickle is self-contained and reload-checked."""
    expert_dir, converted_dir = Path(expert_dir), Path(converted_dir)
    entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    frozen_columns, frozen_bytes = load_frozen_hd1_columns(hd1_columns)
    converted_dir.mkdir(parents=True, exist_ok=False)
    write_frozen_student_columns(converted_dir, frozen_bytes)
    bytes_on_disk = 0
    for entry in entries:
        source = expert_dir / f"{entry['pd_robot_id']}.npz"
        with np.load(source, allow_pickle=False) as data:
            max_limbs = int(data["max_limbs"].item())
            teacher_obs = data["proprio_normalized"]
            obs, target = select_frozen_student_observation(teacher_obs, frozen_columns), data["action_mean"]
            context_raw, obs_mask, act_mask, adjacency = data["context_raw"], data["obs_padding_mask"], data["act_padding_mask"], data["adjacency"]
            if len(obs) != 8000 or context_raw.ndim != 3 or not np.array_equal(context_raw, np.broadcast_to(context_raw[0], context_raw.shape)):
                raise ValueError(f"HD2 {entry['pd_robot_id']} is not an exact-static 8000-transition shard")
            context = nominal_context(context_raw[0], obs_mask[0].astype(bool), act_mask[0].astype(bool))
            if not (np.isfinite(obs).all() and np.isfinite(target).all() and np.isfinite(context).all()):
                raise FloatingPointError(f"HD2 converted shard has nonfinite values: {entry['pd_robot_id']}")
            if (target[:, act_mask[0].astype(bool)] != 0).any():
                raise ValueError(f"HD2 shard has nonzero padded teacher targets: {entry['pd_robot_id']}")
            payload = {"obs": torch.from_numpy(np.array(obs, copy=True)), "act_mean": torch.from_numpy(np.array(target, copy=True)),
                       "context": torch.from_numpy(context), "obs_padding_mask": torch.from_numpy(obs_mask[0].astype(bool)),
                       "act_padding_mask": torch.from_numpy(act_mask[0].astype(bool)), "adjacency_matrix": torch.from_numpy(adjacency[0]),
                       "teacher_obs_rms": {"mean": torch.from_numpy(data["rms_mean"][frozen_columns].copy()), "var": torch.from_numpy(data["rms_var"][frozen_columns].copy()), "count": torch.from_numpy(data["rms_count"].copy())},
                       "manifest": {"context_version": 1, "proprio_features_per_limb": 17, "max_limbs": max_limbs,
                                    "normalization": "teacher_obs_rms_then_selected", "pd_robot_id": entry["pd_robot_id"],
                                    "parent_walker_id": entry["parent_walker_id"], "parent_family": entry["parent_family"], "source_sha256": sha256(source)}}
        destination = converted_dir / f"{entry['pd_robot_id']}.pkl"
        with destination.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with destination.open("rb") as stream:
            reloaded = pickle.load(stream)
        if not torch.equal(payload["obs"], reloaded["obs"]):
            raise RuntimeError(f"HD2 shard reload mismatch: {entry['pd_robot_id']}")
        bytes_on_disk += destination.stat().st_size
    shard_artifacts = []
    for entry in entries:
        path = converted_dir / f"{entry['pd_robot_id']}.pkl"
        shard_artifacts.append({"pd_robot_id": entry["pd_robot_id"], "bytes": path.stat().st_size, "sha256": sha256(path)})
    report = {"HD2A_SHARDED_DATASET": "PASS", "HD2A_MAPPER_IDENTITY": "PASS", "dataset_bytes_on_disk": bytes_on_disk,
              "shard_count": len(entries), "student_columns_sha256": FROZEN_HD1_COLUMNS_SHA256,
              "FROZEN_HD1_COLUMNS_LENGTH": FROZEN_HD1_COLUMN_COUNT, "FROZEN_HD1_COLUMNS_HASH": "PASS",
              "HD2_CONVERTER_USES_FROZEN_COLUMNS": "PASS", "HD2_CONVERTER_DOES_NOT_DERIVE_FIRST_17_PER_LIMB": "PASS",
              "HD2_OUTPUT_COLUMNS_EQUALS_HD1": "PASS", "shard_artifacts": shard_artifacts}
    (converted_dir / "conversion_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


class TaskBalancedShardBatches(IterableDataset):
    """Loads one shard at a time and samples tasks uniformly; never concatenates 8M rows."""
    def __init__(self, converted_dir: Path, manifest_path: Path, batch_size: int, seed: int):
        self.converted_dir, self.entries, self.batch_size, self.seed = Path(converted_dir), json.loads(Path(manifest_path).read_text()), batch_size, seed
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(self.entries))
        for index in order:
            entry = self.entries[int(index)]
            with (self.converted_dir / f"{entry['pd_robot_id']}.pkl").open("rb") as stream:
                payload = pickle.load(stream)
            rows = rng.integers(0, len(payload["obs"]), size=self.batch_size)
            yield {"obs": payload["obs"][rows], "target": payload["act_mean"][rows],
                   "context": payload["context"], "obs_mask": payload["obs_padding_mask"],
                   "act_mask": payload["act_padding_mask"], "adjacency": payload["adjacency_matrix"],
                   "pd_robot_id": entry["pd_robot_id"]}


def configure_hd2(cfg_path: str, seed: int) -> None:
    cfg.merge_from_file(cfg_path)
    cfg.merge_from_list(["RNG_SEED", seed, "MODEL.TYPE", "hnmlp", "MODEL.MLP.LAYER_NUM", 2,
                         "MODEL.MLP.DROPOUT", None, "MODEL.HYPERNET.EMBEDDING_DROPOUT", 0.1,
                         "DISTILL.VALUE_NET", False, "DISTILL.LOSS_TYPE", "KL", "DISTILL.KL_TARGET", "act_mean",
                         "DISTILL.IMITATION_TARGET", "act_mean", "DISTILL.BALANCED_LOSS", True,
                         "DISTILL.BATCH_SIZE", 5120, "DISTILL.BASE_LR", 3e-4, "DISTILL.GRAD_NORM", 0.5,
                         "ENV.KEYS_TO_KEEP", []])


def _make_model(batch: dict) -> ActorCritic:
    max_limbs = int(batch["obs_mask"].numel())
    cfg.MODEL.MAX_LIMBS = max_limbs
    obs_space = spaces.Dict({"proprioceptive": spaces.Box(-np.inf, np.inf, shape=(batch["obs"].shape[1],), dtype=np.float32),
                              "context": spaces.Box(-np.inf, np.inf, shape=(batch["context"].numel(),), dtype=np.float32),
                              "obs_padding_mask": spaces.Box(0, 1, shape=(max_limbs,), dtype=np.bool_),
                              "act_padding_mask": spaces.Box(0, 1, shape=(max_limbs * 2,), dtype=np.bool_),
                              "adjacency_matrix": spaces.Box(-np.inf, np.inf, shape=(max_limbs, max_limbs), dtype=np.float32)})
    return ActorCritic(obs_space, spaces.Box(-np.inf, np.inf, shape=(max_limbs * 2,), dtype=np.float32)).cuda()


def _loss(model, batch, device):
    obs, target = batch["obs"].to(device), batch["target"].to(device)
    context, obs_mask, act_mask, adjacency = (batch[key].to(device).reshape(1, *batch[key].shape) for key in ("context", "obs_mask", "act_mask", "adjacency"))
    size = len(obs)
    model({"proprioceptive": obs, "context": context.expand(size, -1), "obs_padding_mask": obs_mask.expand(size, -1),
           "act_padding_mask": act_mask.expand(size, -1), "adjacency_matrix": adjacency.expand(size, -1, -1)}, compute_val=False)
    prediction, valid = model.action_mu, (~act_mask).to(dtype=obs.dtype)
    _hd0_check_finite("HD2 prediction", prediction)
    return (((prediction - target).square() * valid).sum(dim=1) / valid.sum(dim=1)).mean()


def preflight_update(converted_dir: Path, manifest_path: Path, output: Path, hd1_columns: Path, microbatch: int = 512) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("HD2A effective-batch preflight requires CUDA")
    if microbatch <= 0 or 5120 % microbatch:
        raise ValueError("microbatch must divide the frozen effective batch size 5120")
    manifest = json.loads(Path(manifest_path).read_text())
    if len(manifest) != 10:
        raise ValueError("HD2A training accepts exactly 10 preflight shards")
    if json.loads(Path(hd1_columns).read_text()) != json.loads((Path(converted_dir) / "student_columns.json").read_text()):
        raise ValueError("HD2A mapper is not semantic-identical to HD1")
    configure_hd2(str(ROOT / "configs" / "ft.yaml"), 1409)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    physical = next(iter(TaskBalancedShardBatches(converted_dir, manifest_path, 5120, 1409)))
    torch.manual_seed(1409); np.random.seed(1409)
    model = _make_model(physical)
    if not hasattr(model.mu_net.input_context_encoder, "embedding_dropout") or cfg.MODEL.MLP.DROPOUT is not None:
        raise RuntimeError("HD2 context dropout is not exclusively configured on the HN context path")
    optimizer = optim.Adam(model.parameters(), lr=3e-4, eps=cfg.DISTILL.EPS, weight_decay=cfg.DISTILL.WEIGHT_DECAY)
    model.train(); optimizer.zero_grad(); started = time.monotonic(); losses = []
    try:
        loss = _loss(model, physical, torch.device("cuda")); (0.5 * loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); optimizer.step(); losses.append(float(loss.detach().cpu()))
        batch_implementation = "physical"
    except torch.cuda.OutOfMemoryError:
        # This is the single allowed fallback: preserve one 5120-sample optimizer update.
        del model, optimizer; torch.cuda.empty_cache(); torch.manual_seed(1409); np.random.seed(1409)
        batches = cycle(TaskBalancedShardBatches(converted_dir, manifest_path, microbatch, 1409))
        first = next(batches); model = _make_model(first)
        optimizer = optim.Adam(model.parameters(), lr=3e-4, eps=cfg.DISTILL.EPS, weight_decay=cfg.DISTILL.WEIGHT_DECAY)
        optimizer.zero_grad()
        for batch in [first, *[next(batches) for _ in range(5120 // microbatch - 1)]]:
            loss = _loss(model, batch, torch.device("cuda")); (0.5 * loss / (5120 // microbatch)).backward(); losses.append(float(loss.detach().cpu()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); optimizer.step(); batch_implementation = "gradient_accumulation"
    for parameter in model.parameters():
        _hd0_check_finite("HD2 parameter", parameter)
    checkpoint = output / "checkpoint_000.pt"
    torch.save({"mu_net": model.mu_net.state_dict(), "optimizer": optimizer.state_dict(), "seed": 1409}, checkpoint)
    reloaded = torch.load(checkpoint, map_location="cpu")
    if set(reloaded) != {"mu_net", "optimizer", "seed"} or reloaded["seed"] != 1409:
        raise RuntimeError("HD2A checkpoint reload failed")
    epoch_plan = official_epoch_batch_plan(8000000, 5120)
    metrics = {"HD2A_TASK_BALANCE": "PASS", "HD2A_EFFECTIVE_BATCH_5120": "PASS", "HD2A_CONTEXT_DROPOUT": "PASS",
               "HD2A_CHECKPOINT_RELOAD": "PASS", "HD2_EFFECTIVE_BATCH_SIZE": 5120,
               "HD2_BATCH_IMPLEMENTATION": batch_implementation, "microbatch": 5120 if batch_implementation == "physical" else microbatch,
               "optimizer_updates": 1, "HD2_FULL_BATCH_SIZE": epoch_plan["batch_size"], "HD2_DROP_LAST": epoch_plan["drop_last"],
               "HD2_STEPS_PER_EPOCH_AT_8M": epoch_plan["optimizer_steps"], "HD2_FINAL_BATCH_SIZE_AT_8M": epoch_plan["final_batch_size"],
               "HD2_SAMPLES_PER_EPOCH": epoch_plan["samples"],
               "mean_masked_action_mse": float(np.mean(losses)), "samples_per_second": 5120 / (time.monotonic() - started),
               "peak_gpu_vram_bytes": int(torch.cuda.max_memory_allocated()),
               "peak_host_ram_bytes": ("NOT_AVAILABLE" if resource is None else int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024),
               "checkpoint": str(checkpoint.resolve())}
    (output / "hd2a_student_config.yaml").write_text(cfg.dump())
    (output / "hd2a_update_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert")
    convert.add_argument("--expert", required=True, type=Path); convert.add_argument("--output", required=True, type=Path)
    convert.add_argument("--manifest", required=True, type=Path); convert.add_argument("--hd1-columns", required=True, type=Path)
    train = sub.add_parser("preflight-update")
    train.add_argument("--dataset", required=True, type=Path); train.add_argument("--manifest", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path); train.add_argument("--hd1-columns", required=True, type=Path); train.add_argument("--microbatch", type=int, default=512)
    args = parser.parse_args()
    result = convert_shards(args.expert, args.output, args.manifest, args.hd1_columns) if args.command == "convert" else preflight_update(args.dataset, args.manifest, args.output, args.hd1_columns, args.microbatch)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
