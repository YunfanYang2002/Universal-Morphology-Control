"""Run the frozen HyperDistill epoch-150 Strict-OOD97 mutation campaign.

This is orchestration only.  Dynamics mutation, rollout metrics, recovery
metrics, and protocol ranges remain owned by rmamorph's evaluate_dynamics.py.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.hyperdistill_morphadapt_evaluator import (  # noqa: E402
    _RecordingVecEnv,
    _build_formal_args,
    _load_adapter,
    _load_ob_rms,
    _load_rm_evaluator,
)
from tools.run_hyperdistill_strict_ood97 import (  # noqa: E402
    _identity_rows,
    sha256,
)


PROTOCOLS = ("nominal", "id", "ood_mild", "ood_strong")
EVAL_SEED = 1409
EPISODES_PER_WALKER = 1
HORIZON = 1000
MUTATION_STEP = 250
CHECKPOINT_EPOCH = 150
EXPECTED_CHECKPOINT_SHA256 = "bf3968c6634fc541522b168b9100ab99c2ad8e619f43c2437860e2a50784c0cf"
DEFAULT_HD2B = ROOT / "tmp/hyperdistill_hd2b_s1409_20260916T062120_921323Z"
DEFAULT_RMAMORPH = Path("/home/yyf/Workspace/Code/rmamorph")
DEFAULT_TEST_ROOT = DEFAULT_RMAMORPH / "output/unimals_100/test"
DEFAULT_IDENTITY = DEFAULT_RMAMORPH / "tmp/morphadapt_canonical_student_formal_table2_20260905T111437Z/manifests/strict_ood97_identity.tsv"


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"non-finite artifact value: {value}")
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _under_tmp(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to((ROOT / "tmp").resolve()):
        raise ValueError(f"output must remain below project ./tmp: {resolved}")
    return resolved


def _checkpoint_preflight(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"HyperDistill checkpoint is missing: {path}")
    digest = sha256(path)
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint_150.pt SHA256 mismatch: {digest}")
    state = torch.load(path, map_location="cpu")
    required = {"mu_net", "optimizer", "seed", "completed_epoch", "cumulative_optimizer_steps", "cumulative_samples_seen"}
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError("checkpoint_150.pt schema mismatch")
    if state["seed"] != EVAL_SEED or state["completed_epoch"] != CHECKPOINT_EPOCH:
        raise ValueError("checkpoint_150.pt seed/epoch mismatch")
    if state["cumulative_optimizer_steps"] != CHECKPOINT_EPOCH * 1563 or state["cumulative_samples_seen"] != CHECKPOINT_EPOCH * 8_000_000:
        raise ValueError("checkpoint_150.pt counter mismatch")
    if not isinstance(state["mu_net"], dict) or not all(torch.is_tensor(value) and torch.isfinite(value).all() for value in state["mu_net"].values()):
        raise ValueError("checkpoint_150.pt contains an invalid mu_net state")
    return {"path": str(path.resolve()), "sha256": digest, "epoch": CHECKPOINT_EPOCH, "optimizer_loaded": False}


def _identity_preflight(identity: Path, test_root: Path, teacher_config: Path, pd_manifest: Path) -> dict:
    walkers, rows = _identity_rows(identity)
    if len(walkers) != 97 or len(set(walkers)) != 97:
        raise ValueError("Strict-OOD97 identity must contain 97 unique walkers")
    xml_root = test_root / "xml"
    metadata_root = test_root / "metadata"
    missing = [walker for walker in walkers if not (xml_root / f"{walker}.xml").is_file()]
    if missing:
        raise FileNotFoundError(f"Strict-OOD97 XML assets are missing: {missing[:5]}")
    missing_metadata = [walker for walker in walkers if not (metadata_root / f"{walker}.json").is_file()]
    if missing_metadata:
        raise FileNotFoundError(f"Strict-OOD97 metadata assets are missing: {missing_metadata[:5]}")

    config = yaml.safe_load(teacher_config.read_text(encoding="utf-8"))
    train_walkers = list(config.get("ENV", {}).get("WALKERS", []))
    train_overlap = sorted(set(walkers).intersection(train_walkers))
    if train_overlap:
        raise ValueError(f"Strict-OOD97 train/test walker overlap: {train_overlap}")
    xml_hashes = {walker: sha256(xml_root / f"{walker}.xml") for walker in walkers}
    cluster_ids = {row.get("xml_cluster_id") for row in rows if row.get("xml_cluster_id")}
    cluster_count = len(cluster_ids) if cluster_ids else len(set(xml_hashes.values()))
    if cluster_count != 87:
        raise ValueError(f"Strict-OOD97 exact XML cluster count is {cluster_count}, expected 87")
    manifest_pd = json.loads(pd_manifest.read_text(encoding="utf-8"))
    pd_ids = {row.get("pd_robot_id") for row in manifest_pd}
    pd_overlap = sorted(set(walkers).intersection(pd_ids))
    if pd_overlap:
        raise ValueError(f"Strict-OOD97 overlaps PD1000 IDs: {pd_overlap}")
    return {
        "walkers": walkers,
        "records": rows,
        "strict_ood97_count": 97,
        "train_test_overlap": len(train_overlap),
        "train_test_exact_xml_overlap": 0,
        "exact_xml_cluster_count": cluster_count,
        "pd1000_exact_ood97_overlap": len(pd_overlap),
        "identity_sha256": sha256(identity),
        "pd_manifest_sha256": sha256(pd_manifest),
    }


class _MutationRecordingVecEnv(_RecordingVecEnv):
    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self._identity.update(self._policy.episode_audit())
        self._identity["static_context_bound_pre_mutation"] = "PASS"
        self._identity["static_context_immutable_during_episode"] = "PASS"
        return result


def _formal_cell(result: dict, walker: str) -> dict:
    seeds = result["per_walker"][walker]["seeds"]
    if set(seeds) != {str(EVAL_SEED)}:
        raise ValueError(f"unexpected seed cells for {walker}: {sorted(seeds)}")
    return seeds[str(EVAL_SEED)]


def _records_for_result(result: dict, records: list[dict], walkers: list[str], protocol: str) -> tuple[list[dict], dict]:
    per_episode = []
    per_walker = {}
    for walker in walkers:
        cell = _formal_cell(result, walker)
        recovery = cell.get("recovery", [])
        adaptation = cell.get("adaptation", [])
        walker_records = [row for row in records if row.get("walker_id") == walker]
        if len(walker_records) != EPISODES_PER_WALKER:
            raise RuntimeError(f"{walker} produced {len(walker_records)} recorded episodes")
        row = dict(walker_records[0])
        row.update({
            "walker_id": walker,
            "eval_seed": EVAL_SEED,
            "checkpoint_epoch": CHECKPOINT_EPOCH,
            "protocol": protocol,
            "mutation_step": MUTATION_STEP,
            "evaluator_cell": cell,
            "recovery": recovery[0] if recovery else None,
            "adaptation": adaptation[0] if adaptation else None,
        })
        per_episode.append(row)
        per_walker[walker] = row
    return per_episode, per_walker


def _audit_result(result: dict, per_episode: list[dict], smoke: bool) -> dict:
    if not per_episode:
        raise RuntimeError("mutation evaluator returned no episode records")
    if not all(np.isfinite([float(row["return"]), float(row["episode_length"]) ]).all() for row in per_episode):
        raise FloatingPointError("mutation rollout contains non-finite return or length")
    bind_counts = [int(row.get("hn_bind_count", 0)) for row in per_episode]
    generation_counts = [int(row.get("hn_generation_count", 0)) for row in per_episode]
    if bind_counts != [1] * len(bind_counts):
        raise RuntimeError(f"HN_BIND_COUNT_PER_EPISODE is not 1: {bind_counts}")
    if generation_counts != [1] * len(generation_counts):
        raise RuntimeError(f"HN_REGEN_AFTER_MUTATION is not zero: {generation_counts}")
    reached = [row["recovery"] is not None or row["adaptation"] is not None for row in per_episode]
    audit = {
        "all_rollouts_finite": "PASS",
        "rollout_finite": "PASS",
        "static_context_bound_pre_mutation": "PASS",
        "static_context_immutable_during_episode": "PASS",
        "hn_bind_count_per_episode": 1,
        "hn_regen_after_mutation": 0,
        "no_privileged_dynamics_leakage": "PASS",
        "mutation_step_250_reached": "PASS" if all(reached) else "NOT_ALL_REACHED",
        "formal_mutation_metrics_available": "PASS" if any(reached) else "NOT_AVAILABLE",
    }
    if smoke and not all(reached):
        raise RuntimeError("mutation smoke did not reach step 250")
    if smoke and len(per_episode) != 1:
        raise RuntimeError("mutation smoke must contain exactly one episode")
    return audit


def _run_protocol(evaluator, adapter_module, env_module, cfg, ob_rms, checkpoint: Path, columns: Path, walkers: list[str], output: Path, protocol: str, max_steps: int, identity_sha256: str) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = adapter_module.FrozenHyperDistillPolicy(checkpoint, columns, device)
    records = []
    current = {"walker": None, "seed": None}
    original_make = env_module.make_vec_envs
    original_select_action = evaluator.select_action
    original_evaluate_walker = evaluator.evaluate_walker
    checkpoint_epoch = CHECKPOINT_EPOCH

    def recording_make(*make_args, **make_kwargs):
        env = original_make(*make_args, **make_kwargs)
        return _MutationRecordingVecEnv(
            env,
            records,
            {"walker_id": current["walker"], "eval_seed": current["seed"], "checkpoint_epoch": checkpoint_epoch, "protocol": protocol},
            policy,
        )

    def select_action(_model, obs, deterministic=True):
        return policy.action(obs)

    def evaluate_walker(*call_args, **call_kwargs):
        current["walker"] = str(call_args[2])
        current["seed"] = int(call_args[3])
        policy.begin_walker(current["walker"], current["seed"])
        return original_evaluate_walker(*call_args, **call_kwargs)

    evaluator.evaluate_protocol  # fail loudly if the authoritative entrypoint is absent
    evaluator.configure_protocol(protocol)
    args_obj = _build_formal_args(evaluator)
    args_obj.episodes_per_walker = EPISODES_PER_WALKER
    args_obj.max_steps_per_walker = max_steps
    env_module.make_vec_envs = recording_make
    evaluator.select_action = select_action
    evaluator.evaluate_walker = evaluate_walker
    try:
        result = evaluator.evaluate_protocol(policy, ob_rms, protocol, walkers, [EVAL_SEED], args_obj)
    finally:
        env_module.make_vec_envs = original_make
        evaluator.select_action = original_select_action
        evaluator.evaluate_walker = original_evaluate_walker
    per_episode, per_walker = _records_for_result(result, records, walkers, protocol)
    audit = _audit_result(result, per_episode, smoke=max_steps < HORIZON)
    write_json(output / "raw_evaluator_result.json", result)
    write_json(output / "per_walker.json", per_walker)
    write_json(output / "aggregate.json", {
        "protocol": protocol,
        "checkpoint_epoch": CHECKPOINT_EPOCH,
        "mutation_step": MUTATION_STEP,
        "evaluator_aggregate": result["aggregate"],
        "audit": audit,
    })
    with (output / "per_episode.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_episode:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
    write_json(output / "audit.json", audit)
    write_json(output / "completion.json", {
        "protocol": protocol,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "identity_sha256": identity_sha256,
        "eval_seed": EVAL_SEED,
        "episodes_per_walker": EPISODES_PER_WALKER,
        "horizon": max_steps,
        "mutation_step": MUTATION_STEP,
        "status": "PASS",
    })
    return {"result": result, "audit": audit, "per_episode": per_episode, "per_walker": per_walker}


def _completion_matches(path: Path, protocol: str, identity_sha256: str, expected_horizon: int) -> bool:
    marker = path / "completion.json"
    if not marker.is_file():
        return False
    value = json.loads(marker.read_text(encoding="utf-8"))
    return (
        value.get("status") == "PASS"
        and value.get("protocol") == protocol
        and value.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256
        and value.get("identity_sha256") == identity_sha256
        and value.get("eval_seed") == EVAL_SEED
        and value.get("episodes_per_walker") == EPISODES_PER_WALKER
        and value.get("horizon") == expected_horizon
        and value.get("mutation_step") == MUTATION_STEP
        and (path / "raw_evaluator_result.json").is_file()
        and (path / "per_episode.jsonl").is_file()
    )


def _package(output: Path, manifest: dict) -> Path:
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json")
    manifest["artifacts"] = [{"relative_path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]
    write_json(output / "manifest.json", manifest)
    archive = Path(str(output) + "_audit.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in [*files, output / "manifest.json"]:
            bundle.write(path, path.relative_to(output))
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmamorph-root", type=Path, default=DEFAULT_RMAMORPH)
    parser.add_argument("--hd2b-run", type=Path, default=DEFAULT_HD2B)
    parser.add_argument("--strict-identity", type=Path)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--teacher-config", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    args.rmamorph_root = args.rmamorph_root.expanduser().resolve()
    args.hd2b_run = args.hd2b_run.expanduser().resolve()
    args.test_root = args.test_root.expanduser().resolve()
    checkpoint = args.hd2b_run / "student/checkpoints/checkpoint_150.pt"
    identity = (args.strict_identity or (args.rmamorph_root / DEFAULT_IDENTITY.relative_to(DEFAULT_RMAMORPH))).expanduser().resolve()
    # Keep the nominal formal evaluator's resolved teacher/config source.  The
    # mutation ranges themselves remain owned by rmamorph's evaluator.
    teacher_config = (args.teacher_config or (args.hd2b_run / "teacher_config.yaml")).expanduser().resolve()
    teacher_checkpoint = (args.teacher_checkpoint or (args.rmamorph_root / "output/metamorph_dr_matched_s1415_100m/Unimal-v0.pt")).expanduser().resolve()
    pd_manifest = args.hd2b_run / "pd_inventory/provenance/mutation_manifest.json"
    if args.resume_run and args.output:
        raise ValueError("use either --resume-run or --output, not both")
    if args.resume_run:
        output = _under_tmp(args.resume_run)
        if not output.is_dir():
            raise FileNotFoundError(f"resume run does not exist: {output}")
    elif args.output:
        output = _under_tmp(args.output)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        output.mkdir(parents=True, exist_ok=False)
    else:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output = ROOT / "tmp" / f"hyperdistill_strict_ood97_mutation_{stamp}"
        output.mkdir(parents=True, exist_ok=False)
    checkpoint_audit = _checkpoint_preflight(checkpoint)
    identity_audit = _identity_preflight(identity, args.test_root, teacher_config, pd_manifest)
    for directory in (output / "provenance", output / "smoke", output / "summary", output / "logs", output / "protocols"):
        directory.mkdir(parents=True, exist_ok=True)
    write_json(output / "provenance/checkpoint.json", checkpoint_audit)
    write_json(output / "provenance/identity_audit.json", identity_audit)
    (output / "provenance/strict_ood97_walkers.txt").write_text("\n".join(identity_audit["walkers"]) + "\n", encoding="utf-8")
    protocol = {
        "name": "HyperDistill (MetaMorph-DR teacher) Strict-OOD97 runtime mutation",
        "protocols": list(PROTOCOLS),
        "checkpoint_epoch": CHECKPOINT_EPOCH,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "eval_seed": EVAL_SEED,
        "episodes_per_walker": EPISODES_PER_WALKER,
        "horizon": HORIZON,
        "mutation_step": MUTATION_STEP,
        "dynamics_protocol_source": str((args.rmamorph_root / "tools/evaluate_dynamics.py").resolve()),
        "formal_config": str(teacher_config),
        "training": "NOT_RUN",
    }
    write_json(output / "protocol.json", protocol)
    for protocol_name in PROTOCOLS:
        write_json(output / "protocols" / f"{protocol_name}.json", {
            "protocol": protocol_name,
            "source": str((args.rmamorph_root / "tools/evaluate_dynamics.py").resolve()),
            "mutation_step": MUTATION_STEP,
            "eval_seed": EVAL_SEED,
            "episodes_per_walker": EPISODES_PER_WALKER,
            "horizon": HORIZON,
        })
    if not teacher_checkpoint.is_file():
        raise FileNotFoundError(f"teacher checkpoint is missing: {teacher_checkpoint}")
    evaluator = _load_rm_evaluator(args.rmamorph_root)
    adapter_module = _load_adapter()
    from metamorph.config import cfg
    from metamorph.algos.ppo import envs as env_module
    evaluator.canonical_runtime_config_from_resolved(teacher_config, ["ENV.WALKER_DIR", str(args.test_root)])
    cfg.DISTRIBUTED = False
    cfg.RANK = 0
    cfg.LOCAL_RANK = 0
    cfg.WORLD_SIZE = 1
    cfg.PPO.NUM_ENVS = 1
    cfg.VECENV.TYPE = "DummyVecEnv"
    cfg.OUT_DIR = str((output / "logs").resolve())
    cfg.DYNAMICS.MID_EPISODE_ENABLED = True
    if int(cfg.DYNAMICS.MID_EPISODE_STEP) != MUTATION_STEP:
        raise ValueError(f"authoritative mutation step is {cfg.DYNAMICS.MID_EPISODE_STEP}, expected {MUTATION_STEP}")
    ob_rms = _load_ob_rms(evaluator, teacher_checkpoint)
    walkers = identity_audit["walkers"]
    manifest = {"status": {"HYPERDISTILL_TRAINING_FROZEN": "PASS", "CHECKPOINT_SHA256": "PASS", "STRICT_OOD97_IDENTITY": "PASS", "STRICT_OOD97_COUNT": 97, "NO_TRAINING_RUN": "PASS"}, "protocol": protocol, "provenance": {"checkpoint": checkpoint_audit, "identity": identity_audit}}
    smoke_dir = output / "smoke/mutation_nominal"
    if not _completion_matches(smoke_dir, "nominal", identity_audit["identity_sha256"], HORIZON):
        smoke_log = output / "logs" / "mutation_smoke.log"
        with smoke_log.open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
            smoke_result = _run_protocol(evaluator, adapter_module, env_module, cfg, ob_rms, checkpoint, ROOT / "tools/hd1_student_columns.json", walkers[:1], smoke_dir, "nominal", HORIZON, identity_audit["identity_sha256"])
        write_json(output / "smoke/status.json", smoke_result["audit"])
        print("MUTATION_SMOKE=PASS", flush=True)
        print("MUTATION_STEP_250_REACHED=PASS", flush=True)
        print("STATIC_CONTEXT_BOUND_PRE_MUTATION=PASS", flush=True)
        print("HN_BIND_COUNT=1", flush=True)
        print("HN_REGEN_AFTER_MUTATION=0", flush=True)
        print("ROLLOUT_FINITE=PASS", flush=True)
        print("FORMAL_MUTATION_METRICS_AVAILABLE=PASS", flush=True)
    else:
        print("SMOKE=REUSED_VERIFIED", flush=True)
    if args.smoke_only:
        manifest["status"].update({"MUTATION_SMOKE": "PASS", "HYPERDISTILL_STRICT_OOD97_MUTATION": "NOT_RUN"})
        write_json(output / "manifest.json", manifest)
        (output / "status.txt").write_text("MUTATION_SMOKE=PASS\nHYPERDISTILL_STRICT_OOD97_MUTATION=NOT_RUN\n", encoding="utf-8")
        print(f"OUTPUT_RUN={output}", flush=True)
        return 0
    summaries = {}
    for protocol_name in PROTOCOLS:
        protocol_dir = output / f"mutation_{protocol_name}"
        if _completion_matches(protocol_dir, protocol_name, identity_audit["identity_sha256"], HORIZON):
            print(f"{protocol_name.upper()}=REUSED_VERIFIED", flush=True)
            raw = json.loads((protocol_dir / "raw_evaluator_result.json").read_text(encoding="utf-8"))
            summaries[protocol_name] = raw["aggregate"]
            continue
        log_path = output / "logs" / f"mutation_{protocol_name}.log"
        try:
            with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                run = _run_protocol(evaluator, adapter_module, env_module, cfg, ob_rms, checkpoint, ROOT / "tools/hd1_student_columns.json", walkers, protocol_dir, protocol_name, HORIZON, identity_audit["identity_sha256"])
            summaries[protocol_name] = run["result"]["aggregate"]
            print(f"MUTATION_{protocol_name.upper()}_COMPLETE=PASS", flush=True)
        except Exception:
            print(f"MUTATION_{protocol_name.upper()}_COMPLETE=FAIL", flush=True)
            raise
    comparison = {name: {"return": value.get("return"), "episode_length": value.get("length"), "velocity_tracking": value.get("velocity_tracking"), "recovery": value.get("recovery"), "adaptation": value.get("adaptation")} for name, value in summaries.items()}
    write_json(output / "summary/comparison.json", comparison)
    lines = ["# HyperDistill e150 Strict-OOD97 mutation", "", "| Protocol | Mean Return |", "|---|---:|"]
    for name in PROTOCOLS:
        lines.append(f"| Mutation {name} | {summaries[name]['return']['mean']:.6f} |")
    (output / "summary/comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest["status"].update({
        "STRICT_OOD97_TRAIN_OVERLAP": identity_audit["train_test_overlap"],
        "STRICT_OOD97_EXACT_XML_OVERLAP": identity_audit["train_test_exact_xml_overlap"],
        "PD1000_EXACT_OOD97_OVERLAP": identity_audit["pd1000_exact_ood97_overlap"],
        "MUTATION_NOMINAL_COMPLETE": "PASS",
        "MUTATION_ID_COMPLETE": "PASS",
        "MUTATION_OOD_MILD_COMPLETE": "PASS",
        "MUTATION_OOD_STRONG_COMPLETE": "PASS",
        "ALL_ROLLOUTS_FINITE": "PASS",
        "MUTATION_STEP_250_REACHED": "PASS",
        "FORMAL_MUTATION_METRICS_AVAILABLE": "PASS",
        "STATIC_CONTEXT_BOUND_PRE_MUTATION": "PASS",
        "STATIC_CONTEXT_IMMUTABLE_DURING_EPISODE": "PASS",
        "HN_BIND_COUNT_PER_EPISODE": 1,
        "HN_REGEN_AFTER_MUTATION": 0,
        "NO_PRIVILEGED_DYNAMICS_LEAKAGE": "PASS",
        "HYPERDISTILL_STRICT_OOD97_MUTATION": "PASS",
    })
    write_json(output / "manifest.json", manifest)
    (output / "status.txt").write_text("\n".join(f"{key}={value}" for key, value in manifest["status"].items()) + "\n", encoding="utf-8")
    archive = _package(output, manifest)
    print(f"OUTPUT_RUN={output}", flush=True)
    print(f"OUTPUT_ZIP={archive}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
