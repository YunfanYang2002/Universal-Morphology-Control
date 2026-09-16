"""Launch the frozen full HD2B protocol; all artifacts remain under project ./tmp."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hd1_protocol import sha256  # noqa: E402
from tools.hd2_protocol import PROTOCOL, prepare_pd_inventory  # noqa: E402
from tools.run_hyperdistill_hd2a import CONFIG_SHA, TEACHER_SHA, package  # noqa: E402


HD2B_PROTOCOL = {**PROTOCOL, "stage": "HD2B full paper-faithful training", "drop_last": False,
                 "full_batch_count_per_epoch": 1562, "final_batch_size": 2560, "optimizer_steps_per_epoch": 1563}


def _under_tmp(path: Path) -> Path:
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT / "tmp").resolve()):
        raise ValueError("HD2B output and resume paths must be below project ./tmp")
    return path


def _verify_inventory(run: Path) -> Path:
    inventory = run / "pd_inventory"; manifest = inventory / "provenance" / "mutation_manifest.json"
    if not manifest.is_file():
        raise ValueError("HD2B resume inventory manifest is missing")
    entries = json.loads(manifest.read_text())
    if len(entries) != 1000 or len({row.get("pd_robot_id") for row in entries}) != 1000:
        raise ValueError("HD2B resume inventory is not the frozen 1000-PD corpus")
    for row in entries:
        for path_key, hash_key in (("materialized_xml_path", "pd_xml_sha256"), ("materialized_metadata_path", "pd_metadata_sha256")):
            path = Path(row.get(path_key, ""))
            if not path.is_file() or sha256(path) != row.get(hash_key):
                raise ValueError(f"HD2B resume inventory hash mismatch: {row.get('pd_robot_id')}")
    return inventory


def _latest_checkpoint(student: Path) -> Path | None:
    checkpoints = sorted((student / "checkpoints").glob("checkpoint_*.pt")) if student.is_dir() else []
    if (student / "hd2b_training_summary.json").is_file():
        return None
    return checkpoints[-1] if checkpoints else None


def _verify_completed_training(student: Path) -> None:
    summary_path = student / "hd2b_training_summary.json"
    summary = json.loads(summary_path.read_text())
    required = {"HD2_DROP_LAST": False, "HD2_FULL_BATCH_SIZE": 5120, "HD2_FULL_BATCH_COUNT_PER_EPOCH": 1562,
                "HD2_FINAL_BATCH_SIZE": 2560, "HD2_STEPS_PER_EPOCH": 1563, "HD2_SAMPLES_PER_EPOCH": 8000000,
                "HD2B_RESUMABLE_TRAINING": "PASS"}
    if any(summary.get(key) != value for key, value in required.items()) or not (student / "checkpoints" / "checkpoint_150.pt").is_file():
        raise ValueError("HD2B completed training summary or epoch-150 checkpoint is invalid")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmamorph-root", type=Path, default=Path.home() / "Workspace/Code/rmamorph")
    parser.add_argument("--config", type=Path, default=Path("output/metamorph_dr_matched_s1415_100m/config.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("output/metamorph_dr_matched_s1415_100m/Unimal-v0.pt"))
    parser.add_argument("--hd1-student-columns", type=Path, default=ROOT / "tools" / "hd1_student_columns.json")
    parser.add_argument("--pd-source", type=Path, default=ROOT / "data/train_mutate_1000")
    parser.add_argument("--ood-pool", type=Path, default=Path("configs/morphadapt_metamorph_dr_formal_ood98.txt"))
    parser.add_argument("--resume-run", type=Path, help="existing HD2B run below project ./tmp")
    args = parser.parse_args(); os.chdir(ROOT)
    fresh = args.resume_run is None
    if fresh:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output = ROOT / "tmp" / f"hyperdistill_hd2b_s1409_{stamp}"; output.mkdir(parents=True, exist_ok=False)
    else:
        output = _under_tmp(args.resume_run)
        if not output.is_dir():
            raise ValueError("--resume-run must name an existing HD2B run")
    (output / "temp").mkdir(exist_ok=True)
    for key in ("TMP", "TEMP", "TMPDIR"):
        os.environ[key] = str(output / "temp")
    manifest = {"protocol": HD2B_PROTOCOL, "commands": [], "stages": [], "arguments": {key: str(value) for key, value in vars(args).items()}}
    def stage(name: str, script: str, arguments: list[object], cwd: Path) -> None:
        command = [sys.executable, str(ROOT / "tools" / script), *map(str, arguments)]
        manifest["commands"].append({"stage": name, "cwd": str(cwd), "argv": command}); print(f"STAGE={name}", flush=True)
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=cwd, env={**os.environ, "PYTHONPATH": str(cwd), "PYTHONUNBUFFERED": "1"}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    log.write(line); print(line, end="", flush=True)
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)
        manifest["stages"].append(name)
    try:
        teacher_root = args.rmamorph_root.expanduser().resolve()
        config = args.config if args.config.is_absolute() else teacher_root / args.config
        checkpoint = args.checkpoint if args.checkpoint.is_absolute() else teacher_root / args.checkpoint
        ood_pool = args.ood_pool if args.ood_pool.is_absolute() else teacher_root / args.ood_pool
        if sha256(checkpoint) != TEACHER_SHA or sha256(config) != CONFIG_SHA:
            raise ValueError("HD2B teacher/config differs from frozen HD0/HD1 pair")
        if fresh:
            (output / "frozen_protocol.json").write_text(json.dumps(HD2B_PROTOCOL, indent=2) + "\n")
            (output / "teacher_config.yaml").write_bytes(config.read_bytes())
            prepare_pd_inventory(config, ood_pool, args.pd_source, output / "pd_inventory", preflight_count=None)
        elif json.loads((output / "frozen_protocol.json").read_text()) != HD2B_PROTOCOL or sha256(output / "teacher_config.yaml") != sha256(config):
            raise ValueError("HD2B resume frozen protocol or teacher config differs")
        inventory = _verify_inventory(output); selection = inventory / "provenance" / "mutation_manifest.json"
        common = ["--rmamorph-root", teacher_root, "--config", config, "--checkpoint", checkpoint, "--manifest", selection,
                  "--walker-dir", inventory / "pd1000", "--expected-pd-robots", 1000]
        if not (output / "pd_validity" / "pd_validity.json").is_file():
            stage("pd_validity", "hd2_teacher_export.py", ["validate", *common, "--output", output / "pd_validity"], teacher_root)
        stage("expert_collection", "hd2_teacher_export.py", ["collect", *common, "--output", output / "expert_shards", "--resume"], teacher_root)
        stage("convert", "hd2_student.py", ["convert", "--expert", output / "expert_shards", "--output", output / "converted_shards",
                                               "--manifest", selection, "--hd1-columns", args.hd1_student_columns, "--resume"], ROOT)
        student = output / "student"; latest = _latest_checkpoint(student)
        if (student / "hd2b_training_summary.json").is_file():
            _verify_completed_training(student)
            manifest["training"] = "REUSED_VERIFIED_COMPLETE"
        else:
            arguments = ["train-full-hd2b", "--dataset", output / "converted_shards", "--manifest", selection,
                         "--output", student, "--hd1-columns", args.hd1_student_columns]
            if latest is not None:
                arguments.extend(["--resume-checkpoint", latest])
            elif student.exists():
                raise ValueError("HD2B interrupted training has no epoch-30 checkpoint; refusing unsafe restart")
            stage("student_training", "hd2_student.py", arguments, ROOT)
        manifest["HD2B_READY"] = "EXECUTED"
        code = 0
    except Exception:
        manifest["failure"] = traceback.format_exc(); (output / "failure.txt").write_text(manifest["failure"]); print(manifest["failure"], file=sys.stderr)
        code = 1
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        print(f"OUTPUT_ZIP={package(output)}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
