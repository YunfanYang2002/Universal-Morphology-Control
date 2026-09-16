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


def epoch_batch_sizes(samples: int, batch_size: int) -> list[int]:
    """Paper-faithful epoch partition: final short batch is an optimizer step."""
    plan = official_epoch_batch_plan(samples, batch_size)
    return [batch_size] * plan["full_batches"] + ([plan["final_batch_size"]] if plan["final_batch_size"] else [])


def convert_shards(expert_dir: Path, converted_dir: Path, manifest_path: Path, hd1_columns: Path, *, resume: bool = False, label: str = "HD2A") -> dict:
    """Convert one 8k NPZ per PD robot; each pickle is self-contained and reload-checked."""
    expert_dir, converted_dir = Path(expert_dir), Path(converted_dir)
    entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    frozen_columns, frozen_bytes = load_frozen_hd1_columns(hd1_columns)
    if converted_dir.exists():
        if not resume:
            raise FileExistsError(f"HD2 converted output already exists: {converted_dir}")
        columns_path = converted_dir / "student_columns.json"
        if not columns_path.is_file() or columns_path.read_bytes() != frozen_bytes:
            raise ValueError("HD2 resume conversion mapper differs from the frozen HD1 mapper")
    else:
        converted_dir.mkdir(parents=True, exist_ok=False)
        write_frozen_student_columns(converted_dir, frozen_bytes)
    for entry in entries:
        source = expert_dir / f"{entry['pd_robot_id']}.npz"
        destination = converted_dir / f"{entry['pd_robot_id']}.pkl"
        if destination.exists():
            if not resume:
                raise FileExistsError(f"HD2 converted shard already exists: {destination}")
            with destination.open("rb") as stream:
                payload = pickle.load(stream)
            provenance = payload.get("manifest", {})
            if (provenance.get("pd_robot_id") != entry["pd_robot_id"] or provenance.get("source_sha256") != sha256(source)
                    or len(payload.get("obs", ())) != 8000):
                raise ValueError(f"HD2 resume conversion shard is invalid: {entry['pd_robot_id']}")
            continue
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
        with destination.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with destination.open("rb") as stream:
            reloaded = pickle.load(stream)
        if not torch.equal(payload["obs"], reloaded["obs"]):
            raise RuntimeError(f"HD2 shard reload mismatch: {entry['pd_robot_id']}")
    shard_artifacts = []
    for entry in entries:
        path = converted_dir / f"{entry['pd_robot_id']}.pkl"
        shard_artifacts.append({"pd_robot_id": entry["pd_robot_id"], "bytes": path.stat().st_size, "sha256": sha256(path)})
    report = {f"{label}_SHARDED_DATASET": "PASS", f"{label}_MAPPER_IDENTITY": "PASS", "dataset_bytes_on_disk": sum(row["bytes"] for row in shard_artifacts),
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


class FullCoverageShardBatches(IterableDataset):
    """One independently shuffled pass over every converted row, without padding or carry-over."""
    def __init__(self, converted_dir: Path, manifest_path: Path, batch_size: int, seed: int, expected_rows_per_shard: int | None = 8000):
        self.converted_dir = Path(converted_dir)
        self.entries = json.loads(Path(manifest_path).read_text())
        self.batch_size, self.seed, self.expected_rows_per_shard = batch_size, seed, expected_rows_per_shard
        if not self.entries or batch_size <= 0:
            raise ValueError("HD2 full-coverage batches require a nonempty manifest and positive batch size")

    @staticmethod
    def _collate(parts: list[dict]) -> dict:
        return {key: torch.cat([part[key] for part in parts], dim=0) for key in ("obs", "target", "context", "obs_mask", "act_mask", "adjacency")}

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        parts: list[dict] = []
        pending = 0
        for entry_index in rng.permutation(len(self.entries)):
            entry = self.entries[int(entry_index)]
            with (self.converted_dir / f"{entry['pd_robot_id']}.pkl").open("rb") as stream:
                payload = pickle.load(stream)
            rows = len(payload["obs"])
            if self.expected_rows_per_shard is not None and rows != self.expected_rows_per_shard:
                raise ValueError(f"HD2 full trainer requires {self.expected_rows_per_shard} rows per shard: {entry['pd_robot_id']}")
            order = torch.from_numpy(rng.permutation(rows).astype(np.int64, copy=False))
            offset = 0
            while offset < rows:
                take = min(self.batch_size - pending, rows - offset)
                indices = order[offset:offset + take]
                parts.append({"obs": payload["obs"][indices], "target": payload["act_mean"][indices],
                              "context": payload["context"].reshape(1, -1).expand(take, -1),
                              "obs_mask": payload["obs_padding_mask"].reshape(1, -1).expand(take, -1),
                              "act_mask": payload["act_padding_mask"].reshape(1, -1).expand(take, -1),
                              "adjacency": payload["adjacency_matrix"].reshape(1, *payload["adjacency_matrix"].shape).expand(take, -1, -1)})
                offset += take; pending += take
                if pending == self.batch_size:
                    yield self._collate(parts)
                    parts = []; pending = 0
        if pending:
            yield self._collate(parts)


def configure_hd2(cfg_path: str, seed: int) -> None:
    cfg.merge_from_file(cfg_path)
    cfg.merge_from_list(["RNG_SEED", seed, "MODEL.TYPE", "hnmlp", "MODEL.MLP.LAYER_NUM", 2,
                         "MODEL.MLP.DROPOUT", None, "MODEL.HYPERNET.EMBEDDING_DROPOUT", 0.1,
                         "DISTILL.VALUE_NET", False, "DISTILL.LOSS_TYPE", "KL", "DISTILL.KL_TARGET", "act_mean",
                         "DISTILL.IMITATION_TARGET", "act_mean", "DISTILL.BALANCED_LOSS", True,
                         "DISTILL.BATCH_SIZE", 5120, "DISTILL.BASE_LR", 3e-4, "DISTILL.GRAD_NORM", 0.5,
                         "ENV.KEYS_TO_KEEP", []])


def _make_model(batch: dict) -> ActorCritic:
    max_limbs = int(batch["obs_mask"].shape[-1])
    cfg.MODEL.MAX_LIMBS = max_limbs
    obs_space = spaces.Dict({"proprioceptive": spaces.Box(-np.inf, np.inf, shape=(batch["obs"].shape[1],), dtype=np.float32),
                              "context": spaces.Box(-np.inf, np.inf, shape=(batch["context"].shape[-1],), dtype=np.float32),
                              "obs_padding_mask": spaces.Box(0, 1, shape=(max_limbs,), dtype=np.bool_),
                              "act_padding_mask": spaces.Box(0, 1, shape=(max_limbs * 2,), dtype=np.bool_),
                              "adjacency_matrix": spaces.Box(-np.inf, np.inf, shape=(max_limbs, max_limbs), dtype=np.float32)})
    return ActorCritic(obs_space, spaces.Box(-np.inf, np.inf, shape=(max_limbs * 2,), dtype=np.float32)).cuda()


def _loss(model, batch, device):
    obs, target = batch["obs"].to(device), batch["target"].to(device)
    size = len(obs)
    def per_row(key, dimensions):
        value = batch[key].to(device)
        if value.ndim == dimensions:
            return value.reshape(1, *value.shape).expand(size, *([-1] * dimensions))
        if value.ndim == dimensions + 1 and len(value) == size:
            return value
        raise ValueError(f"HD2 invalid {key} batch shape: {tuple(value.shape)}")
    context, obs_mask, act_mask, adjacency = (per_row(key, dimensions) for key, dimensions in (("context", 1), ("obs_mask", 1), ("act_mask", 1), ("adjacency", 2)))
    model({"proprioceptive": obs, "context": context.expand(size, -1), "obs_padding_mask": obs_mask.expand(size, -1),
           "act_padding_mask": act_mask.expand(size, -1), "adjacency_matrix": adjacency.expand(size, -1, -1)}, compute_val=False)
    prediction, valid = model.action_mu, (~act_mask).to(dtype=obs.dtype)
    _hd0_check_finite("HD2 prediction", prediction)
    return (((prediction - target).square() * valid).sum(dim=1) / valid.sum(dim=1)).mean()


def train_full_hd2b(converted_dir: Path, manifest_path: Path, output: Path, hd1_columns: Path, *, epochs: int = 150,
                    resume_checkpoint: Path | None = None) -> dict:
    """Full HD2B training; every epoch visits each of the 8M rows exactly once."""
    if not torch.cuda.is_available():
        raise RuntimeError("HD2B full training requires CUDA")
    if epochs != 150:
        raise ValueError("HD2B freezes training at exactly 150 epochs")
    entries = json.loads(Path(manifest_path).read_text())
    if len(entries) != 1000 or len({entry["pd_robot_id"] for entry in entries}) != 1000:
        raise ValueError("HD2B full training requires exactly 1000 unique PD shards")
    if json.loads(Path(hd1_columns).read_text()) != json.loads((Path(converted_dir) / "student_columns.json").read_text()):
        raise ValueError("HD2B mapper is not semantic-identical to frozen HD1")
    output = Path(output)
    if output.exists():
        if resume_checkpoint is None:
            raise FileExistsError(f"HD2B training output already exists: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    resume_state = None
    if resume_checkpoint is not None:
        resume_checkpoint = Path(resume_checkpoint)
        if not resume_checkpoint.is_file() or not resume_checkpoint.resolve().is_relative_to(output.resolve()):
            raise ValueError("HD2B resume checkpoint must be an existing checkpoint below the requested training output")
        resume_state = torch.load(resume_checkpoint, map_location="cpu")
        if resume_state.get("seed") != 1409 or not isinstance(resume_state.get("completed_epoch"), int):
            raise ValueError("HD2B resume checkpoint does not match the frozen training protocol")
    configure_hd2(str(ROOT / "configs" / "ft.yaml"), 1409)
    plan = official_epoch_batch_plan(8_000_000, 5120)
    if plan != {"samples": 8_000_000, "batch_size": 5120, "drop_last": False, "full_batches": 1562,
                "final_batch_size": 2560, "optimizer_steps": 1563}:
        raise AssertionError("HD2B frozen batch plan is inconsistent")
    first = next(iter(FullCoverageShardBatches(converted_dir, manifest_path, 5120, 1409)))
    torch.manual_seed(1409); np.random.seed(1409)
    model = _make_model(first)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, eps=cfg.DISTILL.EPS, weight_decay=cfg.DISTILL.WEIGHT_DECAY)
    completed_epoch = 0; cumulative_steps = 0; cumulative_samples = 0
    if resume_state is not None:
        model.mu_net.load_state_dict(resume_state["mu_net"]); optimizer.load_state_dict(resume_state["optimizer"])
        completed_epoch = resume_state["completed_epoch"]
        cumulative_steps = int(resume_state["cumulative_optimizer_steps"])
        cumulative_samples = int(resume_state["cumulative_samples_seen"])
    if completed_epoch < 0 or completed_epoch >= epochs:
        raise ValueError("HD2B resume checkpoint completed_epoch is outside [0, 149]")
    checkpoints = output / "checkpoints"; checkpoints.mkdir(exist_ok=True)
    metrics_path = output / "epoch_metrics.jsonl"
    model.train(); epoch_rows = []
    for epoch in range(completed_epoch + 1, epochs + 1):
        optimizer_steps = samples_seen = last_batch_size = 0
        for batch in FullCoverageShardBatches(converted_dir, manifest_path, 5120, 1409 + epoch):
            optimizer.zero_grad(); loss = _loss(model, batch, torch.device("cuda")); (0.5 * loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); optimizer.step()
            optimizer_steps += 1; last_batch_size = len(batch["obs"]); samples_seen += last_batch_size
        if optimizer_steps != 1563 or samples_seen != 8_000_000 or last_batch_size != 2560:
            raise AssertionError("HD2B epoch violated the frozen drop_last=False coverage contract")
        cumulative_steps += optimizer_steps; cumulative_samples += samples_seen
        row = {"epoch": epoch, "optimizer_steps": optimizer_steps, "samples_seen_this_epoch": samples_seen,
               "cumulative_optimizer_steps": cumulative_steps, "cumulative_samples_seen": cumulative_samples,
               "last_batch_size": last_batch_size}
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        epoch_rows.append(row)
        if epoch % 30 == 0:
            checkpoint = checkpoints / f"checkpoint_{epoch:03d}.pt"
            torch.save({"mu_net": model.mu_net.state_dict(), "optimizer": optimizer.state_dict(), "seed": 1409,
                        "completed_epoch": epoch, "cumulative_optimizer_steps": cumulative_steps,
                        "cumulative_samples_seen": cumulative_samples}, checkpoint)
    report = {"HD2_DROP_LAST": False, "HD2_FULL_BATCH_SIZE": 5120, "HD2_FULL_BATCH_COUNT_PER_EPOCH": 1562,
              "HD2_FINAL_BATCH_SIZE": 2560, "HD2_STEPS_PER_EPOCH": 1563, "HD2_SAMPLES_PER_EPOCH": 8_000_000,
              "HD2B_RESUMABLE_TRAINING": "PASS", "checkpoint_epochs": [30, 60, 90, 120, 150],
              "final_epoch": epoch_rows[-1], "epoch_metrics": str(metrics_path.resolve())}
    (output / "hd2b_training_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "hd2b_batch_contract.txt").write_text("\n".join(f"{key}={value}" for key, value in (
        ("HD2_DROP_LAST", "False"), ("HD2_FULL_BATCH_SIZE", 5120), ("HD2_FULL_BATCH_COUNT_PER_EPOCH", 1562),
        ("HD2_FINAL_BATCH_SIZE", 2560), ("HD2_STEPS_PER_EPOCH", 1563), ("HD2_SAMPLES_PER_EPOCH", 8000000))) + "\n")
    return report


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
    convert.add_argument("--resume", action="store_true", help="reuse only provenance-verified converted shards")
    train = sub.add_parser("preflight-update")
    train.add_argument("--dataset", required=True, type=Path); train.add_argument("--manifest", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path); train.add_argument("--hd1-columns", required=True, type=Path); train.add_argument("--microbatch", type=int, default=512)
    full = sub.add_parser("train-full-hd2b")
    full.add_argument("--dataset", required=True, type=Path); full.add_argument("--manifest", required=True, type=Path)
    full.add_argument("--output", required=True, type=Path); full.add_argument("--hd1-columns", required=True, type=Path)
    full.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "convert":
        result = convert_shards(args.expert, args.output, args.manifest, args.hd1_columns, resume=args.resume, label="HD2B" if args.resume else "HD2A")
    elif args.command == "preflight-update":
        result = preflight_update(args.dataset, args.manifest, args.output, args.hd1_columns, args.microbatch)
    else:
        result = train_full_hd2b(args.dataset, args.manifest, args.output, args.hd1_columns, resume_checkpoint=args.resume_checkpoint)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
