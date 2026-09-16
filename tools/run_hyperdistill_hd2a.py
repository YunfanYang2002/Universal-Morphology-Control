"""Run HD2A only: ten official PD robots, exact 80k teacher transitions, and one 5120-sample update."""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hd1_protocol import sha256  # noqa: E402
from tools.hd2_protocol import PROTOCOL, prepare_pd_inventory  # noqa: E402

TEACHER_SHA = "ed5bfbcaa57a04aafd58411b8735faa392377d1cd9e16c784a8ddc0ce8fd3d84"
CONFIG_SHA = "90764721ea3474e1e9b2f177b479d74e1fdc5b61d4a0e292cf06f32c2fb08f51"
STATUS_KEYS = """HD2A_PD_GENERATION HD2A_PD_VALIDITY HD2A_EXACT_8000 HD2A_TOTAL_TRANSITIONS
HD2A_SHARDED_DATASET HD2A_MAPPER_IDENTITY HD2A_TASK_BALANCE HD2A_EFFECTIVE_BATCH_5120
HD2A_CONTEXT_DROPOUT HD2A_CHECKPOINT_RELOAD HD2A_FINAL""".split()
AUDIT_SUFFIXES = {".json", ".yaml", ".yml", ".txt", ".log", ".csv", ".md"}


def prepare_inventory_stage(config: Path, ood_pool: Path, pd_source: Path, run_root: Path) -> tuple[Path, dict]:
    """Give the protocol-owned child directory to the fail-closed inventory creator."""
    inventory_dir = run_root / "pd_inventory"
    inventory = prepare_pd_inventory(config, ood_pool, pd_source, inventory_dir, preflight_count=10)
    return inventory_dir, inventory


def serialize_status(status: dict) -> dict:
    return {key: ("NOT_MEASURED" if key == "HD2A_FINAL" else "NOT_RUN") if value is None else value for key, value in status.items()}


def verify_resume_run(run_root: Path, config: Path, checkpoint: Path) -> tuple[Path, dict]:
    """Accept only a complete, hash-bound HD2A collection and return its inventory."""
    run_root = Path(run_root).resolve()
    if not run_root.is_dir() or not run_root.is_relative_to((ROOT / "tmp").resolve()):
        raise ValueError("RESUME_REJECTED: --resume-run must be an existing run below project ./tmp")
    protocol_path, teacher_config_path, manifest_path = (run_root / name for name in ("frozen_protocol.json", "teacher_config.yaml", "manifest.json"))
    if not all(path.is_file() for path in (protocol_path, teacher_config_path, manifest_path)):
        raise ValueError("RESUME_REJECTED: missing frozen run provenance")
    if json.loads(protocol_path.read_text()) != PROTOCOL or sha256(teacher_config_path) != sha256(config):
        raise ValueError("RESUME_REJECTED: frozen protocol or teacher config differs")
    old_manifest = json.loads(manifest_path.read_text())
    teacher = old_manifest.get("teacher", {})
    if teacher.get("checkpoint_sha256") != sha256(checkpoint) or teacher.get("config_sha256") != sha256(config):
        raise ValueError("RESUME_REJECTED: teacher checkpoint/config hash differs")
    inventory_dir = run_root / "pd_inventory"
    selection_path = inventory_dir / "provenance" / "mutation_manifest.json"
    parents_path = inventory_dir / "provenance" / "parents.json"
    validity_path = run_root / "pd_validity" / "pd_validity.json"
    collection_path = run_root / "expert_shards" / "collection_metrics.json"
    if not all(path.is_file() for path in (selection_path, parents_path, validity_path, collection_path)):
        raise ValueError("RESUME_REJECTED: PD inventory, validity, or collection provenance is missing")
    selected, parents = json.loads(selection_path.read_text()), json.loads(parents_path.read_text())
    if len(parents) != 100 or len(selected) != 10 or len({row.get("pd_robot_id") for row in selected}) != 10:
        raise ValueError("RESUME_REJECTED: PD inventory counts differ from frozen HD2A")
    for row in selected:
        for path_key, hash_key in (("materialized_xml_path", "pd_xml_sha256"), ("materialized_metadata_path", "pd_metadata_sha256")):
            path = Path(row.get(path_key, ""))
            if not path.is_file() or sha256(path) != row.get(hash_key):
                raise ValueError(f"RESUME_REJECTED: PD inventory hash mismatch for {row.get('pd_robot_id')}")
    validity = json.loads(validity_path.read_text())
    if validity.get("HD2_PD_VALIDITY") != "PASS" or {row.get("pd_robot_id") for row in validity.get("per_robot", [])} != {row["pd_robot_id"] for row in selected}:
        raise ValueError("RESUME_REJECTED: PD validity is not a complete PASS")
    collection = json.loads(collection_path.read_text())
    rows = collection.get("per_robot", [])
    if collection.get("HD2A_EXACT_8000") != "PASS" or collection.get("HD2A_TOTAL_TRANSITIONS") != 80000 or len(rows) != 10:
        raise ValueError("RESUME_REJECTED: collection totals are not exact HD2A 80k")
    for row in rows:
        shard = run_root / "expert_shards" / f"{row.get('pd_robot_id')}.npz"
        if row.get("transition_count") != 8000 or not shard.is_file() or sha256(shard) != row.get("shard_sha256"):
            raise ValueError(f"RESUME_REJECTED: expert shard is absent or hash-mismatched for {row.get('pd_robot_id')}")
    return inventory_dir, collection


def stage_state(run_root: Path, name: str, completion_file: str) -> str:
    stage = run_root / name
    if not stage.exists():
        return "RUN"
    return "COMPLETED" if (stage / completion_file).is_file() else "INCOMPLETE"


def resolve_resume_stage(run_root: Path, name: str, completion_file: str, validator) -> tuple[str, dict | str | None]:
    state = stage_state(run_root, name, completion_file)
    if state == "INCOMPLETE":
        shutil.rmtree(run_root / name)
        return "RUN", "CLEARED_INCOMPLETE"
    if state == "COMPLETED":
        return "SKIP", validator()
    return "RUN", None


def validate_converted_stage(run_root: Path, selection_path: Path) -> dict:
    stage = run_root / "converted_shards"
    audit = json.loads((stage / "conversion_audit.json").read_text())
    required = {"HD2A_SHARDED_DATASET": "PASS", "HD2A_MAPPER_IDENTITY": "PASS", "FROZEN_HD1_COLUMNS_LENGTH": 204,
                "FROZEN_HD1_COLUMNS_HASH": "PASS", "HD2_CONVERTER_USES_FROZEN_COLUMNS": "PASS",
                "HD2_CONVERTER_DOES_NOT_DERIVE_FIRST_17_PER_LIMB": "PASS", "HD2_OUTPUT_COLUMNS_EQUALS_HD1": "PASS",
                "student_columns_sha256": "5aae11e48a05f57bca706085939d3ed47985f8c80e2e37299766e8d5f309917d", "shard_count": 10}
    if any(audit.get(key) != value for key, value in required.items()):
        raise ValueError("RESUME_REJECTED: converted-shard completion audit differs from frozen contract")
    columns = stage / "student_columns.json"
    if not columns.is_file() or sha256(columns) != required["student_columns_sha256"]:
        raise ValueError("RESUME_REJECTED: converted student_columns.json differs from frozen HD1 mapper")
    entries = json.loads(selection_path.read_text())
    expected = {entry["pd_robot_id"] for entry in entries}
    actual = {path.stem for path in stage.glob("*.pkl")}
    if actual != expected:
        raise ValueError("RESUME_REJECTED: converted shard inventory differs from selected PD robots")
    verified = []
    for pd_robot_id in sorted(expected):
        path = stage / f"{pd_robot_id}.pkl"
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        source = run_root / "expert_shards" / f"{pd_robot_id}.npz"
        if payload.get("manifest", {}).get("pd_robot_id") != pd_robot_id or payload.get("manifest", {}).get("source_sha256") != sha256(source):
            raise ValueError(f"RESUME_REJECTED: converted provenance mismatch for {pd_robot_id}")
        verified.append({"pd_robot_id": pd_robot_id, "sha256": sha256(path)})
    return {"HD2A_SHARDED_DATASET": "PASS", "HD2A_MAPPER_IDENTITY": "PASS", "CONVERT_STAGE": "REUSED_VERIFIED", "verified_shards": verified}


def validate_student_stage(run_root: Path) -> dict:
    stage = run_root / "student"
    metrics_path, checkpoint = stage / "hd2a_update_metrics.json", stage / "checkpoint_000.pt"
    metrics = json.loads(metrics_path.read_text())
    required = {"HD2A_TASK_BALANCE": "PASS", "HD2A_EFFECTIVE_BATCH_5120": "PASS", "HD2A_CONTEXT_DROPOUT": "PASS",
                "HD2A_CHECKPOINT_RELOAD": "PASS", "HD2_EFFECTIVE_BATCH_SIZE": 5120, "HD2_BATCH_IMPLEMENTATION": "physical",
                "microbatch": 5120, "optimizer_updates": 1}
    if any(metrics.get(key) != value for key, value in required.items()) or not checkpoint.is_file():
        raise ValueError("RESUME_REJECTED: student preflight completion audit differs from frozen contract")
    import torch
    state = torch.load(checkpoint, map_location="cpu")
    if set(state) != {"mu_net", "optimizer", "seed"} or state["seed"] != 1409:
        raise ValueError("RESUME_REJECTED: student checkpoint is unreadable or incompatible")
    return {key: metrics[key] for key in ("HD2A_TASK_BALANCE", "HD2A_EFFECTIVE_BATCH_5120", "HD2A_CONTEXT_DROPOUT", "HD2A_CHECKPOINT_RELOAD")} | {"STUDENT_PREFLIGHT_STAGE": "REUSED_VERIFIED", "checkpoint_sha256": sha256(checkpoint)}


def archive_stale_failure(run_root: Path) -> None:
    failure = run_root / "failure.txt"
    if not failure.is_file():
        return
    history_path = run_root / "resume_history.json"
    history = json.loads(history_path.read_text()) if history_path.is_file() else []
    attempt = len(history) + 2
    archive = run_root / "attempt_history" / f"attempt_{attempt:03d}_failure.txt"
    archive.parent.mkdir(exist_ok=True)
    reason = failure.read_text()
    shutil.move(str(failure), archive)
    history.append({"attempt": attempt, "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "failure_reason": reason, "resume_verification": "PASS"})
    history_path.write_text(json.dumps(history, indent=2) + "\n")


def package(output: Path) -> Path:
    manifest_path = output / "manifest.json"; manifest = json.loads(manifest_path.read_text())
    files = [path for path in output.rglob("*") if path.is_file() and path != manifest_path and "temp" not in path.relative_to(output).parts]
    manifest["artifacts"] = [{"relative_path": str(path.relative_to(output)), "path": str(path.resolve()), "bytes": path.stat().st_size,
                              "sha256": sha256(path), "included_in_audit_zip": path.suffix.lower() in AUDIT_SUFFIXES} for path in files]
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    archive = output.parent / f"{output.name}_audit.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for path in [*filter(lambda value: value.suffix.lower() in AUDIT_SUFFIXES, files), manifest_path]:
            zipped.write(path, path.relative_to(output))
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmamorph-root", type=Path, default=Path.home() / "Workspace/Code/rmamorph")
    parser.add_argument("--config", type=Path, default=Path("output/metamorph_dr_matched_s1415_100m/config.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("output/metamorph_dr_matched_s1415_100m/Unimal-v0.pt"))
    parser.add_argument("--hd1-student-columns", type=Path, default=ROOT / "tools" / "hd1_student_columns.json", help="frozen HD1 student_columns.json; byte identity is mandatory")
    parser.add_argument("--pd-source", type=Path, default=ROOT / "data/train_mutate_1000")
    parser.add_argument("--ood-pool", type=Path, default=Path("configs/morphadapt_metamorph_dr_formal_ood98.txt"))
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--resume-run", type=Path, help="verified existing HD2A run under ./tmp; resumes at conversion")
    args = parser.parse_args(); os.chdir(ROOT)
    fresh_run = args.resume_run is None
    if fresh_run:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output = ROOT / "tmp" / f"hyperdistill_hd2a_preflight_{stamp}"; output.mkdir(parents=True, exist_ok=False)
    else:
        output = args.resume_run.resolve()
        if not output.is_dir() or not output.is_relative_to((ROOT / "tmp").resolve()):
            raise ValueError("--resume-run must be an existing directory below project ./tmp")
    (output / "temp").mkdir(parents=True, exist_ok=True)
    for key in ("TMP", "TEMP", "TMPDIR"): os.environ[key] = str(output / "temp")
    status = {key: None for key in STATUS_KEYS}; unmeasured = set(STATUS_KEYS)
    manifest = {"protocol": PROTOCOL, "stages": [], "unmeasured": [], "commands": [], "python": sys.executable,
                "python_version": sys.version, "host": platform.node(), "platform": platform.platform(),
                "arguments": {key: str(value) for key, value in vars(args).items()}}
    def update(values):
        for key, value in values.items():
            if key in status:
                status[key] = value; unmeasured.discard(key)
    def stage(name, script, arguments, cwd):
        command = [sys.executable, str(ROOT / "tools" / script), *map(str, arguments)]
        environment = os.environ.copy(); environment["PYTHONPATH"] = str(cwd); environment["PYTHONUNBUFFERED"] = "1"
        manifest["commands"].append({"stage": name, "cwd": str(cwd), "argv": command}); print(f"STAGE={name}", flush=True)
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout: log.write(line); print(line, end="", flush=True)
                if process.wait(): raise subprocess.CalledProcessError(process.returncode, command)
        manifest["stages"].append(name)
    code = 1
    try:
        teacher_root = args.rmamorph_root.expanduser().resolve(); config = args.config if args.config.is_absolute() else teacher_root / args.config
        checkpoint = args.checkpoint if args.checkpoint.is_absolute() else teacher_root / args.checkpoint
        ood_pool = args.ood_pool if args.ood_pool.is_absolute() else teacher_root / args.ood_pool
        if sha256(checkpoint) != TEACHER_SHA or sha256(config) != CONFIG_SHA:
            raise ValueError("HD2 teacher/config differs from frozen HD0/HD1 pair")
        if not args.hd1_student_columns.is_file(): raise FileNotFoundError("actual frozen HD1 student_columns.json is required")
        if fresh_run:
            (output / "teacher_config.yaml").write_bytes(config.read_bytes()); (output / "frozen_protocol.json").write_text(json.dumps(PROTOCOL, indent=2))
        manifest["teacher"] = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "config": str(config), "config_sha256": sha256(config)}
        manifest["git"] = {label: {"sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                                   "tracked_changes": subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=repo, text=True)} for label, repo in (("hyperdistill", ROOT), ("rmamorph", teacher_root))}
        manifest["environment_versions"] = {}
        for name in ("torch", "numpy", "gym", "mujoco-py", "PyYAML"):
            try: manifest["environment_versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: manifest["environment_versions"][name] = "NOT_AVAILABLE"
        if fresh_run:
            inventory_dir, inventory = prepare_inventory_stage(config, ood_pool, args.pd_source, output)
            update({"HD2A_PD_GENERATION": "PASS"}); selection = inventory_dir / "provenance" / "mutation_manifest.json"; walker_dir = inventory_dir / "pd1000"
            common = ["--rmamorph-root", teacher_root, "--config", config, "--checkpoint", checkpoint, "--manifest", selection, "--walker-dir", walker_dir]
            stage("pd_validity", "hd2_teacher_export.py", ["validate", *common, "--output", output / "pd_validity"], teacher_root)
            validity = json.loads((output / "pd_validity" / "pd_validity.json").read_text()); update({"HD2A_PD_VALIDITY": "PASS" if validity["HD2_PD_VALIDITY"] == "PASS" else "FAIL"})
            stage("expert_collection", "hd2_teacher_export.py", ["collect", *common, "--output", output / "expert_shards"], teacher_root)
            collection = json.loads((output / "expert_shards" / "collection_metrics.json").read_text()); update({"HD2A_EXACT_8000": collection["HD2A_EXACT_8000"], "HD2A_TOTAL_TRANSITIONS": collection["HD2A_TOTAL_TRANSITIONS"]})
            if collection["HD2A_TOTAL_TRANSITIONS"] != 80000: raise AssertionError("HD2A requires exactly 80000 transitions")
        else:
            inventory_dir, collection = verify_resume_run(output, config, checkpoint)
            selection = inventory_dir / "provenance" / "mutation_manifest.json"
            update({"HD2A_PD_GENERATION": "PASS", "HD2A_PD_VALIDITY": "PASS", "HD2A_EXACT_8000": "PASS", "HD2A_TOTAL_TRANSITIONS": 80000})
            manifest["resume"] = {"run_root": str(output), "collection_metrics_sha256": sha256(output / "expert_shards" / "collection_metrics.json"), "verified": "PASS"}
            archive_stale_failure(output)
        convert_state = "RUN" if fresh_run else stage_state(output, "converted_shards", "conversion_audit.json")
        student_state = "RUN" if fresh_run else stage_state(output, "student", "hd2a_update_metrics.json")
        if not fresh_run and student_state == "COMPLETED" and convert_state != "COMPLETED":
            raise ValueError("RESUME_REJECTED: completed student preflight lacks a completed converted-shard stage")
        if not fresh_run:
            convert_state, conversion = resolve_resume_stage(output, "converted_shards", "conversion_audit.json", lambda: validate_converted_stage(output, selection))
            if convert_state == "RUN" and conversion: manifest.setdefault("resume_stages", {})["convert"] = conversion
        if convert_state == "SKIP":
            update(conversion)
            manifest.setdefault("resume_stages", {})["convert"] = conversion["CONVERT_STAGE"]
        else:
            stage("convert", "hd2_student.py", ["convert", "--expert", output / "expert_shards", "--output", output / "converted_shards", "--manifest", selection, "--hd1-columns", args.hd1_student_columns], ROOT)
            conversion = json.loads((output / "converted_shards" / "conversion_audit.json").read_text()); update(conversion)
        if not fresh_run:
            student_state, student = resolve_resume_stage(output, "student", "hd2a_update_metrics.json", lambda: validate_student_stage(output))
            if student_state == "RUN" and student: manifest.setdefault("resume_stages", {})["student"] = student
        if student_state == "SKIP":
            update(student)
            manifest.setdefault("resume_stages", {})["student"] = student["STUDENT_PREFLIGHT_STAGE"]
        else:
            stage("one_effective_batch", "hd2_student.py", ["preflight-update", "--dataset", output / "converted_shards", "--manifest", selection, "--output", output / "student", "--hd1-columns", args.hd1_student_columns, "--microbatch", args.microbatch], ROOT)
            update(json.loads((output / "student" / "hd2a_update_metrics.json").read_text()))
        required = ["HD2A_PD_GENERATION", "HD2A_PD_VALIDITY", "HD2A_EXACT_8000", "HD2A_SHARDED_DATASET", "HD2A_MAPPER_IDENTITY", "HD2A_TASK_BALANCE", "HD2A_EFFECTIVE_BATCH_5120", "HD2A_CONTEXT_DROPOUT", "HD2A_CHECKPOINT_RELOAD"]
        status["HD2A_FINAL"] = "PASS" if all(status[key] == "PASS" for key in required) and status["HD2A_TOTAL_TRANSITIONS"] == 80000 else "FAIL"; unmeasured.discard("HD2A_FINAL")
        code = 0 if status["HD2A_FINAL"] == "PASS" else 1
    except Exception:
        manifest["failure"] = traceback.format_exc(); (output / "failure.txt").write_text(manifest["failure"]); print(manifest["failure"], file=sys.stderr)
    finally:
        serialized_status = serialize_status(status)
        manifest["status"] = serialized_status; manifest["unmeasured"] = sorted(unmeasured); manifest["HD2B"] = "NOT_AUTHORIZED; this launcher never starts full HD2B"
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)); (output / "status.json").write_text(json.dumps(serialized_status, indent=2))
        (output / "status.txt").write_text("\n".join(f"{key}={value}" for key, value in serialized_status.items()) + "\n")
        print((output / "status.txt").read_text(), end=""); print(f"OUTPUT_ZIP={package(output)}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
