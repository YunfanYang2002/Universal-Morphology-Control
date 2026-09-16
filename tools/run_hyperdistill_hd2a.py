"""Run HD2A only: ten official PD robots, exact 80k teacher transitions, and one 5120-sample update."""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
import platform
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
    args = parser.parse_args(); os.chdir(ROOT)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / "tmp" / f"hyperdistill_hd2a_preflight_{stamp}"; (output / "temp").mkdir(parents=True)
    for key in ("TMP", "TEMP", "TMPDIR"): os.environ[key] = str(output / "temp")
    status = {key: "FAIL" for key in STATUS_KEYS}; unmeasured = set(STATUS_KEYS)
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
        (output / "teacher_config.yaml").write_bytes(config.read_bytes()); (output / "frozen_protocol.json").write_text(json.dumps(PROTOCOL, indent=2))
        manifest["teacher"] = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "config": str(config), "config_sha256": sha256(config)}
        manifest["git"] = {label: {"sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                                   "tracked_changes": subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=repo, text=True)} for label, repo in (("hyperdistill", ROOT), ("rmamorph", teacher_root))}
        manifest["environment_versions"] = {}
        for name in ("torch", "numpy", "gym", "mujoco-py", "PyYAML"):
            try: manifest["environment_versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: manifest["environment_versions"][name] = "NOT_AVAILABLE"
        inventory = prepare_pd_inventory(config, ood_pool, args.pd_source, output, preflight_count=10)
        update({"HD2A_PD_GENERATION": "PASS"}); selection = output / "provenance" / "mutation_manifest.json"; walker_dir = output / "pd1000"
        common = ["--rmamorph-root", teacher_root, "--config", config, "--checkpoint", checkpoint, "--manifest", selection, "--walker-dir", walker_dir]
        stage("pd_validity", "hd2_teacher_export.py", ["validate", *common, "--output", output / "pd_validity"], teacher_root)
        validity = json.loads((output / "pd_validity" / "pd_validity.json").read_text()); update({"HD2A_PD_VALIDITY": "PASS" if validity["HD2_PD_VALIDITY"] == "PASS" else "FAIL"})
        stage("expert_collection", "hd2_teacher_export.py", ["collect", *common, "--output", output / "expert_shards"], teacher_root)
        collection = json.loads((output / "expert_shards" / "collection_metrics.json").read_text()); update({"HD2A_EXACT_8000": collection["HD2A_EXACT_8000"], "HD2A_TOTAL_TRANSITIONS": collection["HD2A_TOTAL_TRANSITIONS"]})
        if collection["HD2A_TOTAL_TRANSITIONS"] != 80000: raise AssertionError("HD2A requires exactly 80000 transitions")
        stage("convert", "hd2_student.py", ["convert", "--expert", output / "expert_shards", "--output", output / "converted_shards", "--manifest", selection, "--hd1-columns", args.hd1_student_columns], ROOT)
        conversion = json.loads((output / "converted_shards" / "conversion_audit.json").read_text()); update(conversion)
        stage("one_effective_batch", "hd2_student.py", ["preflight-update", "--dataset", output / "converted_shards", "--manifest", selection, "--output", output / "student", "--hd1-columns", args.hd1_student_columns, "--microbatch", args.microbatch], ROOT)
        update(json.loads((output / "student" / "hd2a_update_metrics.json").read_text()))
        required = ["HD2A_PD_GENERATION", "HD2A_PD_VALIDITY", "HD2A_EXACT_8000", "HD2A_SHARDED_DATASET", "HD2A_MAPPER_IDENTITY", "HD2A_TASK_BALANCE", "HD2A_EFFECTIVE_BATCH_5120", "HD2A_CONTEXT_DROPOUT", "HD2A_CHECKPOINT_RELOAD"]
        status["HD2A_FINAL"] = "PASS" if all(status[key] == "PASS" for key in required) and status["HD2A_TOTAL_TRANSITIONS"] == 80000 else "FAIL"; unmeasured.discard("HD2A_FINAL")
        code = 0 if status["HD2A_FINAL"] == "PASS" else 1
    except Exception:
        manifest["failure"] = traceback.format_exc(); (output / "failure.txt").write_text(manifest["failure"]); print(manifest["failure"], file=sys.stderr)
    finally:
        manifest["status"] = status; manifest["unmeasured"] = sorted(unmeasured); manifest["HD2B"] = "NOT_AUTHORIZED; this launcher never starts full HD2B"
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)); (output / "status.json").write_text(json.dumps(status, indent=2))
        (output / "status.txt").write_text("\n".join(f"{key}={value}" for key, value in status.items()) + "\n")
        print((output / "status.txt").read_text(), end=""); print(f"OUTPUT_ZIP={package(output)}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
