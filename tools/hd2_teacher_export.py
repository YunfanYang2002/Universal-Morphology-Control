"""Server-only HD2A validation and exact-8000 teacher collector for static PD shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.hd0_teacher_export import (  # frozen HD0/HD1 observation, action, and context bindings
    base_env, native_adjacency, native_raw_context, proprio_labels,
    tensor_to_numpy,
)


TRANSITION_KEYS = ("proprio_normalized", "action_mean", "action_canonical", "obs_padding_mask", "act_padding_mask", "adjacency", "context_raw")
FROZEN_PROPRIO_DIM = 624


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(partial, path)


def _atomic_npz(path: Path, arrays: dict) -> None:
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(partial, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "collect"))
    parser.add_argument("--rmamorph-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--walker-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-pd-robots", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="reuse only hash-verified complete per-robot shards")
    return parser.parse_args()


def _configure(args):
    root = args.rmamorph_root.resolve()
    if not (root / "metamorph" / "config.py").is_file():
        raise FileNotFoundError(f"--rmamorph-root is not an RMAMorph checkout: {root}")
    for path, label in ((args.config, "config"), (args.checkpoint, "checkpoint"), (args.manifest, "manifest")):
        if not path.is_file():
            raise FileNotFoundError(f"HD2 {label} is missing: {path}")
    sys.path.insert(0, str(root))
    import torch
    from metamorph.algos.ppo.envs import make_vec_envs
    from metamorph.config import canonical_runtime_config_from_resolved, cfg
    from metamorph.envs.history_dynamics import canonicalize_executed_action, policy_action_spec
    from metamorph.utils import sample as su
    from tools.evaluate_v1 import apply_ob_rms, extract_checkpoint, restore_legacy_checkpoint_attributes
    canonical_runtime_config_from_resolved(args.config)
    cfg.DISTRIBUTED = False; cfg.RANK = cfg.LOCAL_RANK = 0; cfg.WORLD_SIZE = 1; cfg.PPO.NUM_ENVS = 1
    cfg.PPO.CHECKPOINT_PATH = ""; cfg.ADAPT.TEACHER_CHECKPOINT_PATH = ""; cfg.VECENV.TYPE = "DummyVecEnv"
    cfg.EXIT_ON_MJ_STEP_EXCEPTION = True; cfg.DYNAMICS.MID_EPISODE_ENABLED = False; cfg.DYNAMICS.ENABLED = True
    cfg.DYNAMICS.MOTOR_STRENGTH_RANGE = [1., 1.]; cfg.DYNAMICS.FRICTION_RANGE = [1., 1.]; cfg.DYNAMICS.MASS_RANGE = [1., 1.]
    cfg.ENV.WALKER_DIR = str(args.walker_dir.resolve())
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu"); cfg.DEVICE = str(device)
    checkpoint = torch.load(str(args.checkpoint), map_location=device)
    teacher, ob_rms = extract_checkpoint(checkpoint)
    if ob_rms is None or "proprioceptive" not in ob_rms:
        raise ValueError("Teacher checkpoint lacks proprioceptive RMS")
    restore_legacy_checkpoint_attributes(teacher); teacher = teacher.to(device).eval()
    if teacher.mu_net.__class__.__name__ != "TransformerModel" or teacher.mu_net.adapt_enabled:
        raise ValueError("HD2 requires frozen non-adaptation MetaMorph-DR teacher")
    rms = ob_rms["proprioceptive"]
    labels, max_limbs = proprio_labels(cfg), int(cfg.MODEL.MAX_LIMBS)
    return locals()


def _validate_one(runtime, entry):
    cfg, teacher, ob_rms, make_vec_envs, su = (runtime[key] for key in ("cfg", "teacher", "ob_rms", "make_vec_envs", "su"))
    cfg.ENV.WALKERS = [entry["pd_robot_id"]]; cfg.RNG_SEED = 1409; su.set_seed(cfg.RNG_SEED)
    env = make_vec_envs(training=False, norm_rew=False, num_env=1); runtime["apply_ob_rms"](env, ob_rms)
    try:
        raw = base_env(env); obs = env.reset(); mask = tensor_to_numpy(obs["obs_padding_mask"])[0].astype(bool)
        action_mask = tensor_to_numpy(obs["act_padding_mask"])[0].astype(bool)
        context = native_raw_context(raw, runtime["max_limbs"]); proprio = tensor_to_numpy(obs["proprioceptive"])[0]
        if not (np.isfinite(context).all() and np.isfinite(proprio).all() and np.isfinite(action_mask).all()):
            raise FloatingPointError(f"HD2 nonfinite reset state: {entry['pd_robot_id']}")
        if mask.shape != (runtime["max_limbs"],) or action_mask.shape != (2 * runtime["max_limbs"],) or action_mask.all():
            raise ValueError(f"HD2 invalid padding mask: {entry['pd_robot_id']}")
        return {"pd_robot_id": entry["pd_robot_id"], "xml_parse": "PASS", "metadata": "PASS", "mujoco_model": "PASS",
                "reset_finite": "PASS", "context_finite": "PASS", "proprioception_finite": "PASS", "action_mask_finite": "PASS"}
    finally:
        env.close()


def validate(runtime, entries, output):
    rows = [_validate_one(runtime, entry) for entry in entries]
    report = {"HD2_PD_XML_VALID": f"{len(rows)}/{len(rows)}", "HD2_PD_RESET_VALID": f"{len(rows)}/{len(rows)}",
              "HD2_PD_CONTEXT_VALID": f"{len(rows)}/{len(rows)}", "HD2_PD_VALIDITY": "PASS", "per_robot": rows}
    output.mkdir(parents=True, exist_ok=False); (output / "pd_validity.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def collect_one(runtime, entry, destination):
    cfg, teacher, ob_rms, make_vec_envs, su, torch = (runtime[key] for key in ("cfg", "teacher", "ob_rms", "make_vec_envs", "su", "torch"))
    canonicalize, policy_action_spec = runtime["canonicalize_executed_action"], runtime["policy_action_spec"]
    cfg.ENV.WALKERS = [entry["pd_robot_id"]]; cfg.RNG_SEED = 1409; su.set_seed(cfg.RNG_SEED)
    env = make_vec_envs(training=False, norm_rew=False, num_env=1); runtime["apply_ob_rms"](env, ob_rms)
    transitions = {key: [] for key in TRANSITION_KEYS}
    episodes, return_values, lengths, early = [], [], [], 0
    try:
        raw = base_env(env); obs = env.reset(); context = native_raw_context(raw, runtime["max_limbs"]); static_context = context.copy(); episode_start = float(raw.sim.data.qpos[0])
        while len(transitions["proprio_normalized"]) < 8000:
            if bool(raw.metadata.get("dynamics_mid_episode_perturbation")):
                raise RuntimeError("HD2 static morphology collector detected forbidden mid-episode dynamics mutation")
            proprio = tensor_to_numpy(obs["proprioceptive"])[0].astype(np.float32, copy=True)
            obs_mask = tensor_to_numpy(obs["obs_padding_mask"])[0].astype(bool, copy=True)
            act_mask = tensor_to_numpy(obs["act_padding_mask"])[0].astype(bool, copy=True)
            with torch.no_grad():
                _, distribution, _, _ = teacher(obs)
            action_mean = tensor_to_numpy(distribution.mean)[0].astype(np.float32, copy=True); action_mean[act_mask] = 0.
            valid, low, high = policy_action_spec(raw.action_space.low, raw.action_space.high, act_mask, None)
            action = canonicalize(action_mean, valid, low, high).astype(np.float32, copy=False)
            if not (np.isfinite(proprio).all() and np.isfinite(action_mean).all() and np.array_equal(action[act_mask], np.zeros(act_mask.sum(), dtype=np.float32))):
                raise FloatingPointError(f"HD2 invalid teacher transition: {entry['pd_robot_id']}")
            for key, value in (("proprio_normalized", proprio), ("action_mean", action_mean), ("action_canonical", action),
                               ("obs_padding_mask", obs_mask), ("act_padding_mask", act_mask), ("adjacency", native_adjacency(raw, runtime["max_limbs"])), ("context_raw", context)):
                transitions[key].append(value)
            obs, rewards, dones, infos = env.step(torch.as_tensor(action, device=runtime["device"]).unsqueeze(0))
            if not np.isfinite(tensor_to_numpy(rewards)).all():
                raise FloatingPointError(f"HD2 nonfinite teacher reward: {entry['pd_robot_id']}")
            if bool(dones[0]):
                episode = infos[0].get("episode")
                if episode is None:
                    raise RuntimeError("HD2 terminal transition lacks episode statistics")
                termination = "horizon" if infos[0].get("timeout") else "early_termination"; early += termination == "early_termination"
                episodes.append({"return": float(episode["r"]), "length": int(episode["l"]), "termination": termination,
                                 "forward_displacement": float(infos[0]["x_pos"]) - episode_start})
                return_values.append(float(episode["r"])); lengths.append(int(episode["l"])); obs = env.reset()
                context = native_raw_context(raw, runtime["max_limbs"])
                if not np.array_equal(context, static_context):
                    raise RuntimeError("HD2 static morphology context changed across resets")
                episode_start = float(raw.sim.data.qpos[0])
    finally:
        env.close()
    arrays = {key: np.asarray(value) for key, value in transitions.items()}
    arrays.update({"rms_mean": np.asarray(runtime["rms"].mean), "rms_var": np.asarray(runtime["rms"].var), "rms_count": np.asarray(runtime["rms"].count),
                   "per_limb_obs_labels": np.asarray(json.dumps(runtime["labels"])), "max_limbs": np.asarray(runtime["max_limbs"])})
    if len(arrays["proprio_normalized"]) != 8000:
        raise AssertionError("HD2 collector must truncate exactly at 8000")
    _atomic_npz(destination, arrays)
    completed_episode_count = len(episodes)
    final_partial_episode_steps = 8000 - sum(lengths)
    if final_partial_episode_steps < 0 or sum(lengths) + final_partial_episode_steps != 8000:
        raise AssertionError("HD2 completed and final-partial episode accounting must equal exactly 8000 transitions")
    return {"pd_robot_id": entry["pd_robot_id"], "transition_count": 8000,
            # Legacy field is retained for readers of historical HD2A artifacts only.
            "episode_count_required_to_reach_8000": completed_episode_count,
            "episode_count_required_to_reach_8000_semantics": "DEPRECATED: completed terminal episodes only; use episodes_started",
            "completed_episode_count": completed_episode_count,
            "completed_episode_length_distribution": lengths,
            "final_partial_episode_steps": final_partial_episode_steps,
            "episodes_started": completed_episode_count + int(final_partial_episode_steps > 0),
            "teacher_return_distribution": return_values,
            "episode_length_distribution": lengths,
            "early_termination_count": early, "shard_sha256": sha256(destination),
            "coverage_metrics_available": True}


def teacher_coverage_summary(metrics):
    """Cost/coverage audit only; no morphology filtering or collection-policy change."""
    per_robot = []
    missing = []
    for row in metrics:
        if not row.get("coverage_metrics_available", False):
            missing.append(row["pd_robot_id"])
            continue
        lengths = np.asarray(row["completed_episode_length_distribution"], dtype=np.float64)
        returns = np.asarray(row["teacher_return_distribution"], dtype=np.float64)
        completed = int(row["completed_episode_count"])
        partial = int(row["final_partial_episode_steps"])
        if completed <= 0 or int(lengths.sum()) + partial != int(row["transition_count"]):
            raise ValueError(f"HD2 invalid completed/final-partial accounting: {row['pd_robot_id']}")
        per_robot.append({"pd_robot_id": row["pd_robot_id"], "episodes_started": int(row["episodes_started"]),
                          "completed_episode_count": completed,
                          "early_termination_count": int(row["early_termination_count"]),
                          "early_termination_fraction": float(row["early_termination_count"] / completed),
                          "completed_episode_length_median": float(np.median(lengths)),
                          "final_partial_episode_steps": partial,
                          "teacher_return_mean": float(np.mean(returns)), "teacher_return_median": float(np.median(returns))})
    def quantiles(values):
        return {f"p{percent}": float(np.percentile(values, percent)) for percent in (50, 90, 95, 99)} | {"max": float(np.max(values))}
    def quantiles_with_minimum(values):
        return quantiles(values) | {"min": float(np.min(values))}
    unavailable = "NOT_AVAILABLE"
    return {"collection_policy": "audit_only_no_filtering", "per_pd_robot": per_robot,
            "coverage_metrics_available_count": len(per_robot), "coverage_metrics_missing_count": len(missing),
            "coverage_metrics_missing_robot_ids": missing,
            "episodes_started": quantiles([row["episodes_started"] for row in per_robot]) if per_robot else unavailable,
            "early_termination_fraction": quantiles([row["early_termination_fraction"] for row in per_robot]) if per_robot else unavailable,
            "completed_episode_length_median": quantiles_with_minimum([row["completed_episode_length_median"] for row in per_robot]) if per_robot else unavailable,
            "final_partial_episode_steps": quantiles([row["final_partial_episode_steps"] for row in per_robot]) if per_robot else unavailable,
            "teacher_return_mean": quantiles([row["teacher_return_mean"] for row in per_robot]) if per_robot else unavailable}


def _sidecar_path(output: Path, pd_robot_id: str) -> Path:
    return output / f"{pd_robot_id}.metrics.json"


def _verify_sidecar(shard: Path, entry: dict) -> dict:
    sidecar = _sidecar_path(shard.parent, entry["pd_robot_id"])
    row = json.loads(sidecar.read_text(encoding="utf-8"))
    lengths = row.get("completed_episode_length_distribution")
    required = ("completed_episode_count", "completed_episode_length_distribution", "final_partial_episode_steps", "episodes_started",
                "teacher_return_distribution", "episode_length_distribution", "early_termination_count", "shard_sha256")
    if (not all(key in row for key in required) or row.get("pd_robot_id") != entry["pd_robot_id"] or row.get("transition_count") != 8000
            or row.get("shard_sha256") != sha256(shard) or row.get("coverage_metrics_available") is not True
            or not isinstance(lengths, list) or not all(type(length) is int for length in lengths)
            or type(row.get("completed_episode_count")) is not int or type(row.get("final_partial_episode_steps")) is not int
            or type(row.get("episodes_started")) is not int or row.get("completed_episode_count") != len(lengths)
            or not isinstance(row.get("teacher_return_distribution"), list)
            or len(row["teacher_return_distribution"]) != len(lengths) or row.get("episode_length_distribution") != lengths
            or not isinstance(row.get("early_termination_count"), int) or not 0 <= row["early_termination_count"] <= len(lengths)
            or sum(lengths) + int(row["final_partial_episode_steps"]) != 8000
            or row.get("episodes_started") != len(lengths) + int(int(row["final_partial_episode_steps"]) > 0)):
        raise ValueError("invalid sidecar")
    return row | {"resume_source": "SIDECAR_VERIFIED"}


def _validate_legacy_shard(shard: Path, entry: dict) -> dict:
    """Strictly admit an old shard only when it is usable by the frozen converter."""
    try:
        with np.load(shard, allow_pickle=False) as data:
            missing = set(TRANSITION_KEYS).difference(data.files)
            if missing:
                raise ValueError(f"missing keys {sorted(missing)}")
            arrays = {key: data[key] for key in TRANSITION_KEYS}
            if any(array.ndim == 0 or array.shape[0] != 8000 for array in arrays.values()):
                raise ValueError("transition count is not 8000")
            max_limbs = int(data["max_limbs"].item()) if "max_limbs" in data.files else -1
            if (max_limbs <= 0 or arrays["proprio_normalized"].shape != (8000, FROZEN_PROPRIO_DIM)
                    or arrays["action_mean"].shape != (8000, 2 * max_limbs)
                    or arrays["action_canonical"].shape != (8000, 2 * max_limbs)
                    or arrays["obs_padding_mask"].shape != (8000, max_limbs)
                    or arrays["act_padding_mask"].shape != (8000, 2 * max_limbs)
                    or arrays["adjacency"].shape != (8000, max_limbs, max_limbs)
                    or arrays["context_raw"].shape != (8000, max_limbs, 35)):
                raise ValueError("frozen observation/action/context schema mismatch")
            for key, array in arrays.items():
                if array.dtype.kind not in "biufc" or not np.isfinite(array).all():
                    raise ValueError(f"nonfinite or nonnumeric transition array: {key}")
            if np.any(arrays["action_canonical"][arrays["act_padding_mask"].astype(bool)] != 0):
                raise ValueError("canonical action has nonzero padded slots")
            for key in ("rms_mean", "rms_var", "rms_count"):
                if key not in data.files or not np.isfinite(data[key]).all():
                    raise ValueError(f"missing or nonfinite converter normalization field: {key}")
            if data["rms_mean"].shape != (FROZEN_PROPRIO_DIM,) or data["rms_var"].shape != (FROZEN_PROPRIO_DIM,) or data["rms_count"].ndim != 0:
                raise ValueError("converter normalization schema mismatch")
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError(str(error)) from error
    return {"pd_robot_id": entry["pd_robot_id"], "transition_count": 8000, "shard_sha256": sha256(shard),
            "resume_source": "LEGACY_SHARD_VALIDATED", "coverage_metrics_available": False}


def collection_resume_plan(output: Path, entries: list[dict], legacy_validator=_validate_legacy_shard, log=print) -> tuple[dict, list[dict], dict]:
    """Use actual shard/sidecar pairs as the resume truth source, never the final aggregate."""
    existing, needed = {}, []
    stats = {"sidecar_verified": 0, "legacy_recovered": 0, "invalid": []}
    for entry in entries:
        pd_robot_id = entry["pd_robot_id"]
        shard, sidecar = output / f"{pd_robot_id}.npz", _sidecar_path(output, pd_robot_id)
        removed_partial = False
        for partial in (shard.with_name(f"{shard.name}.partial"), sidecar.with_name(f"{sidecar.name}.partial")):
            if partial.exists():
                partial.unlink(); removed_partial = True
        if removed_partial:
            log(f"STALE_PARTIAL_REMOVED={pd_robot_id}")
        if not shard.is_file():
            needed.append(entry); continue
        if sidecar.is_file():
            try:
                existing[pd_robot_id] = _verify_sidecar(shard, entry); stats["sidecar_verified"] += 1
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                log(f"SIDECAR_INVALID={pd_robot_id}"); stats["invalid"].append(pd_robot_id); needed.append(entry)
            continue
        try:
            existing[pd_robot_id] = legacy_validator(shard, entry); stats["legacy_recovered"] += 1
            log(f"RESUME_SKIP_LEGACY_VALID={pd_robot_id}")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            log(f"LEGACY_SHARD_INVALID={pd_robot_id}"); stats["invalid"].append(pd_robot_id); needed.append(entry)
    return existing, needed, stats


def _verified_existing_collection(output: Path, entries: list[dict]) -> dict:
    """Backward-compatible test helper for the per-shard source of truth."""
    return collection_resume_plan(output, entries)[0]


def main():
    args = parse_args(); runtime = _configure(args); entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.expected_pd_robots <= 0 or len(entries) != args.expected_pd_robots or len({entry["pd_robot_id"] for entry in entries}) != args.expected_pd_robots:
        raise ValueError(f"HD2 must receive exactly {args.expected_pd_robots} unique PD robots")
    if args.command == "validate":
        if args.resume:
            raise ValueError("--resume is valid only for teacher collection")
        result = validate(runtime, entries, args.output)
    else:
        if args.output.exists():
            if not args.resume:
                raise FileExistsError(f"HD2 collection output already exists: {args.output}")
        else:
            args.output.mkdir(parents=True, exist_ok=False)
        label = "HD2A" if args.expected_pd_robots == 10 else "HD2B"
        total = 8000 * args.expected_pd_robots
        if args.resume:
            existing, needed, resume_stats = collection_resume_plan(args.output, entries)
        else:
            existing, needed = {}, list(entries)
            resume_stats = {"sidecar_verified": 0, "legacy_recovered": 0, "invalid": []}
        if label == "HD2B":
            print(f"HD2B_MANIFEST_ROBOTS={len(entries)}")
            print(f"HD2B_EXISTING_VERIFIED={resume_stats['sidecar_verified']}")
            print(f"HD2B_EXISTING_LEGACY_RECOVERED={resume_stats['legacy_recovered']}")
            print(f"HD2B_NEED_COLLECTION={len(needed)}")
        newly_collected = 0
        def write_progress(last_completed_robot):
            _atomic_json(args.output / "collection_progress.json", {
                "verified_complete": len(existing), "legacy_recovered": resume_stats["legacy_recovered"],
                "newly_collected": newly_collected, "remaining": len(needed) - newly_collected,
                "last_completed_robot": last_completed_robot,
            })
        write_progress(None)
        for entry in needed:
            pd_robot_id = entry["pd_robot_id"]
            row = collect_one(runtime, entry, args.output / f"{pd_robot_id}.npz")
            _atomic_json(_sidecar_path(args.output, pd_robot_id), row)
            existing[pd_robot_id] = row; newly_collected += 1
            write_progress(pd_robot_id)
            if label == "HD2B":
                print(f"HD2B_COLLECTION_PROGRESS={len(existing)}/{len(entries)}")
        metrics = [existing[entry["pd_robot_id"]] for entry in entries]
        if sum(row["transition_count"] for row in metrics) != total:
            raise AssertionError("HD2 collection total transitions differs from the exact frozen contract")
        result = {f"{label}_EXACT_8000": "PASS", f"{label}_TOTAL_TRANSITIONS": total, "per_robot": metrics}
        if label == "HD2B":
            result["HD2B_RESUMABLE_COLLECTION"] = "PASS"
        _atomic_json(args.output / "collection_metrics.json", result)
        _atomic_json(args.output / "teacher_coverage_summary.json", teacher_coverage_summary(metrics))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
