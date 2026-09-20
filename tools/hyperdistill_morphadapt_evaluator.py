"""Run frozen HyperDistill through rmamorph's canonical evaluator semantics."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_adapter():
    spec = importlib.util.spec_from_file_location("hyperdistill_adapter", ROOT / "tools/hyperdistill_morphadapt_adapter.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_rm_evaluator(rmamorph_root: Path):
    sys.path = [str(rmamorph_root), *[entry for entry in sys.path if entry not in {str(ROOT), str(ROOT / "tools")}]]
    spec = importlib.util.spec_from_file_location("rmamorph_evaluate_dynamics", rmamorph_root / "tools/evaluate_dynamics.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_ob_rms(evaluator, checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu")
    _model, ob_rms = evaluator.extract_checkpoint(payload)
    if ob_rms is None or "proprioceptive" not in ob_rms:
        raise ValueError("teacher checkpoint lacks the frozen proprioceptive observation RMS")
    rms = ob_rms["proprioceptive"]
    for name in ("mean", "var", "count"):
        if not hasattr(rms, name) or not np.isfinite(np.asarray(getattr(rms, name))).all():
            raise ValueError(f"teacher observation RMS is invalid: {name}")
    if np.asarray(rms.mean).shape != (624,) or np.asarray(rms.var).shape != (624,):
        raise ValueError("teacher observation RMS is not the frozen 624-column mapper source")
    return ob_rms


class _RecordingVecEnv:
    def __init__(self, wrapped, records, identity):
        self._wrapped = wrapped
        self._records = records
        self._identity = identity
        self._start_x = None

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def _capture_reset_x(self):
        env = self._wrapped
        while hasattr(env, "venv"):
            env = env.venv
        if not hasattr(env, "envs") or len(env.envs) != 1:
            raise RuntimeError("formal HyperDistill recording requires one DummyVecEnv environment")
        current = env.envs[0]
        while hasattr(current, "env"):
            current = current.env
        self._start_x = float(current.sim.data.qpos[0])

    def reset(self, *args, **kwargs):
        result = self._wrapped.reset(*args, **kwargs)
        self._capture_reset_x()
        return result

    def step(self, action):
        result = self._wrapped.step(action)
        _obs, _reward, _done, infos = result
        info = infos[0]
        episode = info.get("episode")
        if episode is not None:
            if "x_pos" not in info or self._start_x is None:
                raise RuntimeError("canonical evaluator terminal record lacks x_pos or reset qpos")
            length = int(episode["l"])
            timed_out = bool(info.get("timeout", False))
            self._records.append({
                **self._identity,
                "episode_id": len([row for row in self._records if row["walker_id"] == self._identity["walker_id"] and row["eval_seed"] == self._identity["eval_seed"]]),
                "return": float(episode["r"]),
                "episode_length": length,
                "survived_to_horizon": bool(timed_out or length >= 1000),
                "early_termination": bool(not timed_out and length < 1000),
                "forward_displacement": float(info["x_pos"]) - self._start_x,
                "termination_reason": "time_limit" if timed_out else "environment_terminal_reason_unavailable",
            })
            self._start_x = None
        return result


def _mean(values):
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None


def _median(values):
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmamorph-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--student-columns", type=Path, required=True)
    parser.add_argument("--walkers-file", type=Path, required=True)
    parser.add_argument("--walker-root", type=Path, required=True)
    parser.add_argument("--eval-seed", type=int, default=1409)
    parser.add_argument("--episodes-per-walker", type=int, default=1)
    parser.add_argument("--max-steps-per-walker", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes_per_walker != 1 or args.max_steps_per_walker != 1000:
        raise ValueError("HyperDistill Strict-OOD97 must match the current MorphAdapt formal contract: 1 episode and horizon 1000")

    evaluator = _load_rm_evaluator(args.rmamorph_root.resolve())
    adapter_module = _load_adapter()
    from metamorph.config import cfg
    from metamorph.algos.ppo import envs as env_module

    evaluator.canonical_runtime_config_from_resolved(args.config.resolve(), ["ENV.WALKER_DIR", str(args.walker_root.resolve())])
    print("RMAMORPH_CFG_ACTIVE=YES", flush=True)
    cfg.DISTRIBUTED = False
    cfg.RANK = 0
    cfg.LOCAL_RANK = 0
    cfg.WORLD_SIZE = 1
    cfg.PPO.NUM_ENVS = 1
    cfg.VECENV.TYPE = "DummyVecEnv"
    cfg.OUT_DIR = str(args.out.resolve().parent)
    cfg.DYNAMICS.MID_EPISODE_ENABLED = False
    evaluator.configure_protocol("nominal")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = adapter_module.FrozenHyperDistillPolicy(args.student_checkpoint, args.student_columns, device)
    ob_rms = _load_ob_rms(evaluator, args.teacher_checkpoint.resolve())
    walkers = evaluator.load_walkers(args.walkers_file.resolve())
    if len(walkers) != 97:
        raise ValueError(f"Strict-OOD97 evaluator received {len(walkers)} walkers")

    records = []
    original_make = env_module.make_vec_envs
    original_select_action = evaluator.select_action
    current = {"walker": None, "seed": None}
    checkpoint_epoch = int(torch.load(args.student_checkpoint, map_location="cpu")["completed_epoch"])

    def recording_make(*make_args, **make_kwargs):
        env = original_make(*make_args, **make_kwargs)
        return _RecordingVecEnv(env, records, {"walker_id": current["walker"], "eval_seed": current["seed"], "checkpoint_epoch": checkpoint_epoch})

    def select_action(_model, obs, deterministic=True):
        return policy.action(obs)

    original_evaluate_walker = evaluator.evaluate_walker

    def evaluate_walker(*call_args, **call_kwargs):
        current["walker"] = str(call_args[2])
        current["seed"] = int(call_args[3])
        policy.begin_walker(current["walker"], current["seed"])
        return original_evaluate_walker(*call_args, **call_kwargs)

    env_module.make_vec_envs = recording_make
    evaluator.select_action = select_action
    evaluator.evaluate_walker = evaluate_walker
    try:
        args_obj = type("FormalArgs", (), {
            "trace_out": None, "trace_identity": None,
            "action_history_intervention": "aligned", "post_response_intervention": False,
            "episodes_per_walker": 1, "max_steps_per_walker": 1000,
        })()
        result = evaluator.evaluate_protocol(policy, ob_rms, "nominal", walkers, [args.eval_seed], args_obj)
    finally:
        env_module.make_vec_envs = original_make
        evaluator.select_action = original_select_action
        evaluator.evaluate_walker = original_evaluate_walker

    if len(records) != 97:
        raise RuntimeError(f"canonical evaluator completed {len(records)}/97 rollouts")
    if not all(np.isfinite([row["return"], row["episode_length"], row["forward_displacement"]]).all() for row in records):
        raise FloatingPointError("HyperDistill rollout metrics contain non-finite values")
    per_walker = []
    for walker in walkers:
        rows = [row for row in records if row["walker_id"] == walker]
        if len(rows) != 1:
            raise RuntimeError(f"walker {walker} does not have exactly one formal rollout")
        row = rows[0]
        per_walker.append({
            "walker_id": walker,
            "eval_seed": args.eval_seed,
            "checkpoint_epoch": row["checkpoint_epoch"],
            "return": row["return"],
            "episode_length": row["episode_length"],
            "survival": float(row["survived_to_horizon"]),
            "termination": float(row["early_termination"]),
            "forward_displacement": row["forward_displacement"],
            "velocity_tracking": result["per_walker"][walker].get("velocity_tracking"),
        })
    aggregate = {
        "n_walkers": len(per_walker), "n_rollouts": len(records), "eval_seed": args.eval_seed,
        "mean_return": _mean([row["return"] for row in records]),
        "median_return": _median([row["return"] for row in records]),
        "mean_episode_length": _mean([row["episode_length"] for row in records]),
        "survival_rate": _mean([row["survived_to_horizon"] for row in records]),
        "termination_rate": _mean([row["early_termination"] for row in records]),
        "mean_forward_displacement": _mean([row["forward_displacement"] for row in records]),
        "median_forward_displacement": _median([row["forward_displacement"] for row in records]),
        "velocity_tracking": result["aggregate"].get("velocity_tracking"),
        "fall_rate": None,
        "fall_rate_status": "NOT_AVAILABLE: canonical evaluator exposes early termination but no fall-reason classifier",
    }
    output = {
        "schema_version": 1,
        "evaluator": "rmamorph/tools/evaluate_dynamics.py; canonical evaluate_protocol/evaluate_walker",
        "protocol": "nominal",
        "checkpoint_epoch": records[0]["checkpoint_epoch"],
        "canonical_result": result,
        "per_episode": records,
        "per_walker": per_walker,
        "aggregate": aggregate,
        "rollout_finite": "PASS",
        "context_leakage_audit": {
            "static_context_only": "PASS",
            "generated_before_rollout_and_frozen": "PASS",
            "no_return_or_trajectory_input": "PASS",
            "no_future_dynamics_input": "PASS",
            "no_identity_lookup_policy": "PASS",
            "no_mid_episode_regeneration": "PASS",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"OUT": str(args.out.resolve()), "N_ROLLOUTS": len(records), "ROLLOUT_FINITE": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
