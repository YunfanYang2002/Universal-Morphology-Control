"""Export small, no-mutation RMAMorph Teacher rollouts for HD0 conversion.

Run this script with the RMAMorph checkout as the working directory.  It adds
``--rmamorph-root`` to ``sys.path`` before importing ``metamorph`` so the
unrelated package in this checkout cannot be imported by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


CONTEXT_RAW_V1_LABELS = (
    [f"body_pos:{axis}" for axis in range(3)]
    + [f"body_ipos:{axis}" for axis in range(3)]
    + [f"body_iquat:{axis}" for axis in range(4)]
    + [f"geom_quat:{axis}" for axis in range(4)]
    + ["body_mass:0"]
    + [f"body_shape:{axis}" for axis in range(2)]
    + [f"joint{slot}/{name}:{axis}" for slot in range(2) for name, width in (("jnt_pos", 3), ("joint_range", 2), ("joint_axis", 3), ("gear", 1)) for axis in range(width)]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing or is not a regular file: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmamorph-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--walkers", nargs=3, metavar=("WALKER0", "WALKER1", "WALKER2"), help="three train walkers; defaults to the first three configured training walkers")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", required=True, type=int, help="completed episodes per walker")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1409)
    parser.add_argument("--student-policy", type=Path, help="optional frozen TorchScript policy accepting [1, selected_columns]")
    parser.add_argument("--student-columns", type=Path, help="JSON list of flattened normalized proprioceptive column indices")
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def require_tmp_output(root: Path, output: Path) -> None:
    allowed = ((root / "tmp").resolve(), (Path(__file__).resolve().parents[1] / "tmp").resolve())
    if not any(output.is_relative_to(directory) for directory in allowed):
        raise ValueError(f"--output must be below one of: {', '.join(map(str, allowed))}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")


def proprio_labels(cfg: Any) -> list[str]:
    limb_widths = {
        "body_idx": int(cfg.MODEL.MAX_LIMBS),
        "body_xpos": 3,
        "body_xquat": 4,
        "body_xvelp": 3,
        "body_xvelr": 3,
        "body_pos": 3,
        "body_ipos": 3,
        "body_iquat": 4,
        "geom_quat": 4,
        "geom_extremities": 3,
        "body_mass": 1,
        "body_shape": 2,
        "body_friction": 1,
        "qpos": 1,
        "qvel": 1,
        "jnt_pos": 3,
        "joint_range": 2,
        "joint_axis": 3,
        "gear": 1,
        "armature": 1,
        "damping": 1,
    }
    labels: list[str] = []
    for name in cfg.MODEL.PROPRIOCEPTIVE_OBS_TYPES:
        if name not in limb_widths:
            raise ValueError(f"unsupported native proprioceptive feature for explicit layout: {name}")
        labels.extend(f"{name}:{index}" for index in range(limb_widths[name]))
    # Native Agent.combine_limb_joint_obs appends two full joint slots after limb features.
    joint_labels: list[str] = []
    for name in cfg.MODEL.PROPRIOCEPTIVE_OBS_TYPES:
        if name in {"qpos", "qvel", "jnt_pos", "joint_range", "joint_axis", "gear", "armature", "damping"}:
            joint_labels.extend(f"joint{{slot}}/{name}:{index}" for index in range(limb_widths[name]))
    limb_labels = [label for label in labels if not label.startswith(("qpos:", "qvel:", "jnt_pos:", "joint_range:", "joint_axis:", "gear:", "armature:", "damping:"))]
    return limb_labels + [label.format(slot=slot) for slot in range(2) for label in joint_labels]


def base_env(vec_env: Any) -> Any:
    current = vec_env
    while hasattr(current, "venv"):
        current = current.venv
    if not hasattr(current, "envs") or len(current.envs) != 1:
        raise RuntimeError("HD0 exporter requires one native DummyVecEnv environment")
    return current.envs[0].unwrapped


def native_raw_context(env: Any, max_limbs: int) -> np.ndarray:
    """Reconstruct the requested raw_v1 static context from the live MuJoCo model."""
    agent = env.modules.get("Agent")
    if agent is None:
        raise RuntimeError("native environment has no Agent module")
    sim = env.sim
    body_idxs = np.asarray(agent.agent_body_idxs, dtype=np.int64)
    geom_idxs = np.asarray(agent.agent_geom_idxs, dtype=np.int64)
    limb_count = body_idxs.size
    if limb_count > max_limbs:
        raise ValueError(f"walker has {limb_count} limbs > MODEL.MAX_LIMBS={max_limbs}")
    context = np.zeros((max_limbs, 35), dtype=np.float32)
    body = np.concatenate((
        sim.model.body_pos[body_idxs, :],
        sim.model.body_ipos[body_idxs, :],
        sim.model.body_iquat[body_idxs, :],
        sim.model.geom_quat[geom_idxs, :],
        sim.model.body_mass[body_idxs, None],
        sim.model.geom_size[geom_idxs, :2],
    ), axis=1)
    if body.shape != (limb_count, 17):
        raise RuntimeError(f"unexpected raw body context shape: {body.shape}")
    context[:limb_count, :17] = body
    joint_mask = np.asarray(agent.joint_mask_for_node_graph, dtype=bool)
    slots = limb_count * 2
    if joint_mask.size != slots:
        raise RuntimeError(f"native joint-slot mask size {joint_mask.size} != {slots}")
    joint_rows = np.flatnonzero(joint_mask)
    joints = np.concatenate((
        sim.model.jnt_pos[1:, :], sim.model.jnt_range[1:, :],
        sim.model.jnt_axis[1:, :], sim.model.actuator_gear[:, :1],
    ), axis=1)
    if joints.shape != (joint_rows.size, 9):
        raise RuntimeError(f"unexpected raw joint context shape: {joints.shape}, slots={joint_rows.size}")
    joint_context = np.zeros((slots, 9), dtype=np.float32)
    joint_context[joint_rows] = joints
    context[:limb_count, 17:] = joint_context.reshape(limb_count, 18)
    return context


def native_adjacency(env: Any, max_limbs: int) -> np.ndarray:
    agent = env.modules.get("Agent")
    if agent is None:
        raise RuntimeError("native environment has no Agent module")
    edges = np.asarray(agent.edges, dtype=np.int64).reshape(-1, 2)
    adjacency = np.zeros((max_limbs, max_limbs), dtype=bool)
    if np.any(edges < 0) or np.any(edges >= max_limbs):
        raise ValueError(f"native edges outside MAX_LIMBS: {edges.tolist()}")
    # Native Agent._get_edges emits [child_limb, parent_limb].
    adjacency[edges[:, 0], edges[:, 1]] = True
    return adjacency


def tensor_to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def load_student_columns(path: Path, proprio_dim: int) -> np.ndarray:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not values or any(type(value) is not int for value in values):
        raise ValueError("--student-columns must contain a non-empty JSON list of integer indices")
    columns = np.asarray(values, dtype=np.int64)
    if len(set(columns.tolist())) != columns.size or np.any(columns < 0) or np.any(columns >= proprio_dim):
        raise ValueError(f"--student-columns must be unique indices in [0, {proprio_dim})")
    return columns


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("--episodes and --max-steps must be positive")
    if bool(args.student_policy) != bool(args.student_columns):
        raise ValueError("--student-policy and --student-columns must be supplied together")

    root = args.rmamorph_root.resolve()
    if not (root / "metamorph" / "config.py").is_file():
        raise FileNotFoundError(f"--rmamorph-root is not an RMAMorph checkout: {root}")
    config = require_file(resolve(root, args.config), "config")
    checkpoint = require_file(resolve(root, args.checkpoint), "checkpoint")
    output = resolve(root, args.output)
    require_tmp_output(root, output)
    sys.path.insert(0, str(root))
    import torch
    from metamorph.algos.ppo.envs import make_vec_envs
    from metamorph.config import canonical_runtime_config_from_resolved, cfg
    from metamorph.envs.history_dynamics import canonicalize_executed_action, policy_action_spec
    from metamorph.utils import sample as su
    from tools.evaluate_v1 import apply_ob_rms, extract_checkpoint, restore_legacy_checkpoint_attributes

    canonical_runtime_config_from_resolved(config)
    cfg.DISTRIBUTED = False
    cfg.RANK = cfg.LOCAL_RANK = 0
    cfg.WORLD_SIZE = 1
    cfg.PPO.NUM_ENVS = 1
    cfg.PPO.CHECKPOINT_PATH = ""
    cfg.ADAPT.TEACHER_CHECKPOINT_PATH = ""
    cfg.VECENV.TYPE = "DummyVecEnv"
    cfg.OUT_DIR = str(output)
    if bool(cfg.MIRROR_DATA_AUG):
        raise ValueError("HD0 raw_v1 export does not support mirrored native observations")
    if bool(cfg.ADAPT.ENABLED):
        raise ValueError("HD0 MetaMorph export requires ADAPT.ENABLED=false; refusing an adaptation/privileged-context checkpoint")
    if cfg.ENV.MOTION_COMMAND.ENABLED or cfg.ENV.VELOCITY_COMMAND.ENABLED:
        raise ValueError("HD0 feature mapper does not include command-conditioned observations")
    cfg.EXIT_ON_MJ_STEP_EXCEPTION = True
    if not hasattr(cfg.DYNAMICS, "MID_EPISODE_ENABLED"):
        raise ValueError("config lacks DYNAMICS.MID_EPISODE_ENABLED required for no-mutation export")
    cfg.DYNAMICS.MID_EPISODE_ENABLED = False
    cfg.DYNAMICS.ENABLED = True
    cfg.DYNAMICS.MOTOR_STRENGTH_RANGE = [1.0, 1.0]
    cfg.DYNAMICS.FRICTION_RANGE = [1.0, 1.0]
    cfg.DYNAMICS.MASS_RANGE = [1.0, 1.0]
    walkers = list(args.walkers) if args.walkers else list(cfg.ENV.WALKERS[:3])
    if len(walkers) != 3 or len(set(walkers)) != 3:
        raise ValueError("export requires exactly three distinct walkers")
    if not set(walkers).issubset(set(cfg.ENV.WALKERS)):
        raise ValueError("HD0 walkers must belong to the checkpoint config training set")
    walker_selection = "explicit --walkers" if args.walkers else "first three cfg.ENV.WALKERS"
    walker_dir = (root / "output" / "unimals_100" / "train").resolve()
    cfg.ENV.WALKER_DIR = str(walker_dir)
    walker_paths = [require_file(walker_dir / "xml" / f"{walker}.xml", f"walker XML ({walker})") for walker in walkers]
    metadata_paths = [require_file(walker_dir / "metadata" / f"{walker}.json", f"walker metadata ({walker})") for walker in walkers]

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    cfg.DEVICE = str(device)
    checkpoint_value = torch.load(str(checkpoint), map_location=device)
    teacher, ob_rms = extract_checkpoint(checkpoint_value)
    if ob_rms is None or "proprioceptive" not in ob_rms:
        raise ValueError("Teacher checkpoint lacks proprioceptive observation RMS")
    fixes = restore_legacy_checkpoint_attributes(teacher)
    teacher = teacher.to(device).eval()
    if teacher.mu_net.__class__.__name__ != "TransformerModel" or teacher.mu_net.adapt_enabled:
        raise ValueError("Checkpoint is not a non-adaptation MetaMorph universal Transformer")
    rms = ob_rms["proprioceptive"]
    rms_mean = np.array(rms.mean, dtype=np.float64, copy=True)
    rms_var = np.array(rms.var, dtype=np.float64, copy=True)
    rms_count = np.array(rms.count, dtype=np.float64, copy=True)
    per_limb_labels = proprio_labels(cfg)
    max_limbs = int(cfg.MODEL.MAX_LIMBS)
    action_dim = 2 * max_limbs
    proprio_dim = len(per_limb_labels) * max_limbs
    if rms_mean.shape != (proprio_dim,) or rms_var.shape != (proprio_dim,):
        raise ValueError(f"RMS shape incompatible with declared native layout: mean={rms_mean.shape}, expected={(proprio_dim,)}")
    if not (np.isfinite(rms_mean).all() and np.isfinite(rms_var).all() and np.isfinite(rms_count).all() and (rms_var >= 0).all()):
        raise ValueError("Invalid checkpoint RMS statistics")

    student = None
    student_columns = None
    student_sha = None
    if args.student_policy:
        policy_path = require_file(resolve(root, args.student_policy), "student policy")
        columns_path = require_file(resolve(root, args.student_columns), "student columns")
        student_columns = load_student_columns(columns_path, proprio_dim)
        student = torch.jit.load(str(policy_path), map_location=device).eval()
        student_sha = sha256_file(policy_path)

    transitions: dict[str, list[np.ndarray]] = {key: [] for key in (
        "proprio_normalized", "action_mean", "action_canonical", "episode_id", "step_id", "walker_index", "obs_padding_mask", "act_padding_mask", "adjacency"
    )}
    contexts: list[np.ndarray] = []
    returns: list[float] = []
    lengths: list[int] = []
    student_squared_errors: list[np.ndarray] = []
    episode_id = 0
    order_bindings = {}
    for walker_index, walker in enumerate(walkers):
        cfg.ENV.WALKERS = [walker]
        cfg.RNG_SEED = int(args.seed) + walker_index
        su.set_seed(cfg.RNG_SEED)
        env = make_vec_envs(training=False, norm_rew=False, num_env=1)
        apply_ob_rms(env, ob_rms)
        try:
            raw_env = base_env(env)
            obs = env.reset()
            contexts.append(native_raw_context(raw_env, max_limbs))
            agent = raw_env.modules["Agent"]
            order_bindings[walker] = {
                "body_indices": np.asarray(agent.agent_body_idxs).tolist(),
                "body_names": [raw_env.sim.model.body_id2name(int(i)) for i in agent.agent_body_idxs],
                "joint_slot_mask": np.asarray(agent.joint_mask_for_node_graph, dtype=bool).tolist(),
                "joint_names": list(raw_env.sim.model.joint_names),
                "actuator_names": list(raw_env.sim.model.actuator_names),
            }
            completed = 0
            steps = 0
            episode_step = 0
            while completed < args.episodes:
                if steps >= args.max_steps:
                    raise RuntimeError(f"walker {walker} completed only {completed}/{args.episodes} episodes within {args.max_steps} steps")
                if bool(raw_env.metadata.get("dynamics_mid_episode_perturbation")):
                    raise RuntimeError("no-mutation invariant failed: native environment has a mid-episode perturbation")
                proprio = tensor_to_numpy(obs["proprioceptive"])[0].astype(np.float32, copy=True)
                obs_mask = tensor_to_numpy(obs["obs_padding_mask"])[0].astype(bool, copy=True)
                act_mask = tensor_to_numpy(obs["act_padding_mask"])[0].astype(bool, copy=True)
                if proprio.shape != (proprio_dim,) or obs_mask.shape != (max_limbs,) or act_mask.shape != (action_dim,):
                    raise RuntimeError(f"native observation shape changed: proprio={proprio.shape}, obs_mask={obs_mask.shape}, act_mask={act_mask.shape}")
                if not np.isfinite(proprio).all():
                    raise FloatingPointError("Teacher observation is nonfinite")
                expected_mask = np.ones(action_dim, dtype=bool)
                expected_mask[:len(agent.joint_mask_for_node_graph)] = ~np.asarray(agent.joint_mask_for_node_graph, dtype=bool)
                if not np.array_equal(act_mask, expected_mask):
                    raise ValueError("Native action mask differs from physical joint-slot ordering")
                with torch.no_grad():
                    _, distribution, _, _ = teacher(obs)
                    action_mean = distribution.mean
                action_mean_np = tensor_to_numpy(action_mean)[0].astype(np.float32, copy=True)
                if action_mean_np.shape != (action_dim,) or not np.isfinite(action_mean_np).all():
                    raise RuntimeError(f"Teacher action mean is invalid: shape={action_mean_np.shape}")
                action_mean_np[act_mask] = 0.0
                valid, low, high = policy_action_spec(raw_env.action_space.low, raw_env.action_space.high, act_mask, None)
                canonical = canonicalize_executed_action(action_mean_np, valid, low, high).astype(np.float32, copy=False)
                if not np.array_equal(canonical[act_mask], np.zeros(act_mask.sum(), dtype=np.float32)):
                    raise RuntimeError("canonical action padding is not exactly zero")
                transitions["proprio_normalized"].append(proprio)
                transitions["action_mean"].append(action_mean_np)
                transitions["action_canonical"].append(canonical)
                transitions["episode_id"].append(np.asarray(episode_id, dtype=np.int64))
                transitions["step_id"].append(np.asarray(episode_step, dtype=np.int32))
                transitions["walker_index"].append(np.asarray(walker_index, dtype=np.int16))
                transitions["obs_padding_mask"].append(obs_mask)
                transitions["act_padding_mask"].append(act_mask)
                transitions["adjacency"].append(native_adjacency(raw_env, max_limbs))
                obs, rewards, dones, infos = env.step(torch.as_tensor(canonical, device=device).unsqueeze(0))
                if not np.isfinite(tensor_to_numpy(rewards)).all():
                    raise FloatingPointError("Nonfinite teacher rollout reward")
                steps += 1
                episode_step += 1
                if bool(dones[0]):
                    episode = infos[0].get("episode")
                    if episode is None:
                        raise RuntimeError("native done transition lacks RecordEpisodeStatistics episode record")
                    returns.append(float(episode["r"]))
                    lengths.append(int(episode["l"]))
                    episode_id += 1
                    completed += 1
                    episode_step = 0
                    if completed < args.episodes:
                        contexts.append(native_raw_context(raw_env, max_limbs))
        finally:
            env.close()

    student_returns: list[float] = []
    student_lengths: list[int] = []
    if student is not None:
        # A student policy is evaluated only on the first selected training walker.
        walker = walkers[0]
        cfg.ENV.WALKERS = [walker]
        cfg.RNG_SEED = int(args.seed)
        su.set_seed(cfg.RNG_SEED)
        env = make_vec_envs(training=False, norm_rew=False, num_env=1)
        apply_ob_rms(env, ob_rms)
        try:
            raw_env = base_env(env)
            obs = env.reset()
            steps = completed = 0
            while completed < args.episodes:
                if steps >= args.max_steps:
                    raise RuntimeError(f"student walker {walker} completed only {completed}/{args.episodes} episodes within {args.max_steps} steps")
                proprio = tensor_to_numpy(obs["proprioceptive"])[0].astype(np.float32, copy=False)
                if not np.isfinite(proprio).all():
                    raise FloatingPointError("Nonfinite student rollout observation")
                act_mask = tensor_to_numpy(obs["act_padding_mask"])[0].astype(bool, copy=False)
                with torch.no_grad():
                    _, teacher_distribution, _, _ = teacher(obs)
                    teacher_mean = tensor_to_numpy(teacher_distribution.mean)[0].astype(np.float32, copy=False)
                    student_input = torch.as_tensor(proprio[student_columns], dtype=torch.float32, device=device).unsqueeze(0)
                    student_action = tensor_to_numpy(student(student_input))[0].astype(np.float32, copy=False)
                if student_action.shape != (action_dim,) or not np.isfinite(student_action).all():
                    raise RuntimeError(f"TorchScript student output is invalid: shape={student_action.shape}")
                if not np.isfinite(teacher_mean).all() or np.any(student_action[act_mask] != 0):
                    raise FloatingPointError("Invalid teacher comparison or nonzero student padded action")
                valid, low, high = policy_action_spec(raw_env.action_space.low, raw_env.action_space.high, act_mask, None)
                canonical_student = canonicalize_executed_action(student_action, valid, low, high).astype(np.float32, copy=False)
                student_squared_errors.append((student_action[~act_mask] - teacher_mean[~act_mask]) ** 2)
                obs, rewards, dones, infos = env.step(torch.as_tensor(canonical_student, device=device).unsqueeze(0))
                if not np.isfinite(tensor_to_numpy(rewards)).all():
                    raise FloatingPointError("Nonfinite student rollout reward")
                steps += 1
                if bool(dones[0]):
                    episode = infos[0].get("episode")
                    if episode is None:
                        raise RuntimeError("student done transition lacks episode record")
                    student_returns.append(float(episode["r"]))
                    student_lengths.append(int(episode["l"]))
                    completed += 1
        finally:
            env.close()

    output.mkdir(parents=True)
    (output / "config.yaml").write_bytes(config.read_bytes())
    (output / "runtime_config.yaml").write_text(cfg.dump(), encoding="utf-8")
    with (output / "obs_rms.pkl").open("wb") as stream:
        pickle.dump(ob_rms, stream)
    arrays = {key: np.asarray(value) for key, value in transitions.items()}
    arrays.update({
        "context_raw": np.asarray(contexts, dtype=np.float32),
        "rms_mean": rms_mean,
        "rms_var": rms_var,
        "rms_count": rms_count,
    })
    if not all(np.isfinite(value).all() for value in arrays.values()) or not np.isfinite(returns).all():
        raise FloatingPointError("Nonfinite export arrays or episode returns")
    if not np.array_equal(rms_mean, rms.mean) or not np.array_equal(rms_var, rms.var) or not np.array_equal(rms_count, rms.count):
        raise RuntimeError("Teacher RMS changed during export")
    np.savez_compressed(output / "arrays.npz", **arrays)
    manifest = {
        "schema_version": 1,
        "context_format": "raw_v1",
        "context_raw_labels": CONTEXT_RAW_V1_LABELS,
        "max_limbs": max_limbs,
        "action_dim": action_dim,
        "per_limb_obs_labels": per_limb_labels,
        "proprio_feature_labels": [f"limb{limb}/{label}" for limb in range(max_limbs) for label in per_limb_labels],
        "walkers": walkers,
        "order_bindings": order_bindings,
        "teacher_episode_returns": returns,
        "teacher_episode_lengths": lengths,
        "walker_selection": walker_selection,
        "episode_count": episode_id,
        "transition_count": int(arrays["episode_id"].size),
        "no_mutation": {"DYNAMICS.MID_EPISODE_ENABLED": False, "DYNAMICS.ENABLED": True, "reset_multipliers": {"motor_strength": 1.0, "friction": 1.0, "mass": 1.0}, "verified_each_transition": True},
        "normalization": {"source": "checkpoint ob_rms['proprioceptive']", "arrays": ["rms_mean", "rms_var", "rms_count"], "normalized_array": "proprio_normalized", "formula": "clip((native_padded_obs-mean)/sqrt(var+1e-8),-10,10)", "update": False},
        "native_adjacency": {"array": "adjacency", "edge_order": "[child_limb,parent_limb]", "directed": True},
        "paths": {"config": str(config), "checkpoint": str(checkpoint), "walker_dir": str(walker_dir)},
        "sha256": {"config": sha256_file(config), "checkpoint": sha256_file(checkpoint), "walker_xml": {walker: sha256_file(path) for walker, path in zip(walkers, walker_paths)}, "walker_metadata": {walker: sha256_file(path) for walker, path in zip(walkers, metadata_paths)},
                   "native_sources": {name: sha256_file(root / name) for name in (
                       "metamorph/envs/modules/agent.py", "metamorph/algos/ppo/model.py",
                       "metamorph/envs/wrappers/multi_env_wrapper.py", "metamorph/envs/vec_env/vec_normalize.py")}},
        "checkpoint_compatibility_fixes": fixes,
        "teacher_policy": "native RMAMorph serialized model; deterministic distribution.mean",
        "METAMORPH_TEACHER_LOAD": "PASS",
        "TEACHER_OBS_FINITE": "PASS",
        "TEACHER_ACTION_FINITE": "PASS",
        "OBS_ORDER_BINDING": "PASS",
        "ACTION_ORDER_BINDING": "PASS",
        "ACTION_MASK_BINDING": "PASS",
        "MORPH_CONTEXT_BINDING": "PASS",
        "OBS_RMS_PROVENANCE": "PASS",
        "student_rollout": None if student is None else {"policy_sha256": student_sha, "walker": walkers[0], "columns": student_columns.tolist(), "action_mse_valid": float(np.concatenate(student_squared_errors).mean()) if student_squared_errors else None, "teacher_episode_returns": returns[:args.episodes], "teacher_episode_lengths": lengths[:args.episodes], "student_episode_returns": student_returns, "student_episode_lengths": student_lengths, "student_rollout_finite": bool(np.isfinite(student_returns).all() and np.isfinite(student_lengths).all() and np.isfinite(np.concatenate(student_squared_errors)).all())},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if student is not None:
        (output / "rollout_metrics.json").write_text(json.dumps(manifest["student_rollout"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "episodes": episode_id, "transitions": manifest["transition_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
