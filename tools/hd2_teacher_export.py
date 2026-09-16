"""Server-only HD2A validation and exact-8000 teacher collector for static PD shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.hd0_teacher_export import (  # frozen HD0/HD1 observation, action, and context bindings
    base_env, native_adjacency, native_raw_context, proprio_labels,
    tensor_to_numpy,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "collect"))
    parser.add_argument("--rmamorph-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--walker-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
    transitions = {key: [] for key in ("proprio_normalized", "action_mean", "action_canonical", "obs_padding_mask", "act_padding_mask", "adjacency", "context_raw")}
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
    np.savez_compressed(destination, **arrays)
    return {"pd_robot_id": entry["pd_robot_id"], "transition_count": 8000, "episode_count_required_to_reach_8000": len(episodes),
            "teacher_return_distribution": return_values, "episode_length_distribution": lengths, "early_termination_count": early,
            "shard_sha256": sha256(destination)}


def teacher_coverage_summary(metrics):
    """Cost/coverage audit only; no morphology filtering or collection-policy change."""
    per_robot = []
    for row in metrics:
        lengths = np.asarray(row["episode_length_distribution"], dtype=np.float64)
        returns = np.asarray(row["teacher_return_distribution"], dtype=np.float64)
        episodes = int(row["episode_count_required_to_reach_8000"])
        per_robot.append({"pd_robot_id": row["pd_robot_id"], "episode_count_to_8000": episodes,
                          "early_termination_count": int(row["early_termination_count"]),
                          "early_termination_fraction": float(row["early_termination_count"] / episodes),
                          "median_episode_length": float(np.median(lengths)),
                          "teacher_return_mean": float(np.mean(returns)), "teacher_return_median": float(np.median(returns))})
    def quantiles(values):
        return {f"p{percent}": float(np.percentile(values, percent)) for percent in (50, 90, 95, 99)} | {"max": float(np.max(values))}
    return {"collection_policy": "audit_only_no_filtering", "per_pd_robot": per_robot,
            "episode_count_to_8000": quantiles([row["episode_count_to_8000"] for row in per_robot]),
            "early_termination_fraction": quantiles([row["early_termination_fraction"] for row in per_robot]),
            "median_episode_length": quantiles([row["median_episode_length"] for row in per_robot]),
            "teacher_return_mean": quantiles([row["teacher_return_mean"] for row in per_robot])}


def main():
    args = parse_args(); runtime = _configure(args); entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    if len(entries) != 10 or len({entry["pd_robot_id"] for entry in entries}) != 10:
        raise ValueError("HD2A must receive exactly ten unique PD robots")
    if args.command == "validate":
        result = validate(runtime, entries, args.output)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        metrics = [collect_one(runtime, entry, args.output / f"{entry['pd_robot_id']}.npz") for entry in entries]
        result = {"HD2A_EXACT_8000": "PASS", "HD2A_TOTAL_TRANSITIONS": sum(row["transition_count"] for row in metrics), "per_robot": metrics}
        (args.output / "collection_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.output / "teacher_coverage_summary.json").write_text(json.dumps(teacher_coverage_summary(metrics), indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
