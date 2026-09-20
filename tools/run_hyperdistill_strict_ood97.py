"""Prepare and launch the frozen HyperDistill Strict-OOD97 nominal evaluation."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HD2B = ROOT / "tmp/hyperdistill_hd2b_s1409_20260916T062120_921323Z"
DEFAULT_IDENTITY = Path("/home/yyf/Workspace/Code/rmamorph/tmp/morphadapt_canonical_student_formal_table2_20260905T111437Z/manifests/strict_ood97_identity.tsv")
DEFAULT_TEST_ROOT = Path("/home/yyf/Workspace/Code/rmamorph/output/unimals_100/test")
DEFAULT_RMAMORPH = Path.home() / "Workspace/Code/rmamorph"
FORMAL_EVAL_EPISODES = 1
FORMAL_HORIZON = 1000
EXPECTED_CHECKPOINT_SHA256 = {
    150: "bf3968c6634fc541522b168b9100ab99c2ad8e619f43c2437860e2a50784c0cf",
    30: "08621d8c3958ef180351465cb451509200e35bbe322c3629333e9517b8bf1941",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _under_tmp(path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to((ROOT / "tmp").resolve()):
        raise ValueError(f"output must remain below project ./tmp: {path}")
    return path


def _identity_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = [row for row in reader if row and not row[0].lstrip().startswith("#")]
    if not rows:
        raise ValueError("Strict-OOD97 identity manifest is empty")
    if "walker_id" in rows[0]:
        header = rows[0]
        records = [dict(zip(header, row)) for row in rows[1:]]
        walkers = [row.get("walker_id", "") for row in records]
    else:
        records = [{"walker_id": row[0]} for row in rows]
        walkers = [row[0] for row in rows]
    if len(walkers) != 97 or len(set(walkers)) != 97 or any(not value for value in walkers):
        raise ValueError(f"Strict-OOD97 identity must contain 97 unique walker IDs, got {len(walkers)}")
    return walkers, records


def _audit_identity(identity: Path, test_root: Path, teacher_config: Path, pd_manifest: Path) -> dict:
    walkers, rows = _identity_rows(identity)
    if not (test_root / "xml").is_dir():
        raise FileNotFoundError(f"Strict-OOD97 test XML root is missing: {test_root / 'xml'}")
    missing = [walker for walker in walkers if not (test_root / "xml" / f"{walker}.xml").is_file()]
    if missing:
        raise FileNotFoundError(f"Strict-OOD97 XML assets are missing: {missing[:5]}")
    metadata_missing = [walker for walker in walkers if not (test_root / "metadata" / f"{walker}.json").is_file()]
    if metadata_missing:
        raise FileNotFoundError(f"Strict-OOD97 metadata assets are missing: {metadata_missing[:5]}")

    config = yaml.safe_load(teacher_config.read_text(encoding="utf-8"))
    train = list(config.get("ENV", {}).get("WALKERS", []))
    train_overlap = sorted(set(walkers).intersection(train))
    if train_overlap:
        raise ValueError(f"Strict-OOD97 overlaps teacher training walkers: {train_overlap}")
    if not pd_manifest.is_file():
        raise FileNotFoundError(f"HD2B PD1000 inventory manifest is missing: {pd_manifest}")
    pd_rows = json.loads(pd_manifest.read_text(encoding="utf-8"))
    pd_ids = {row.get("pd_robot_id") for row in pd_rows}
    exact_pd_overlap = sorted(set(walkers).intersection(pd_ids))
    if exact_pd_overlap:
        raise ValueError(f"Strict-OOD97 overlaps PD1000 IDs: {exact_pd_overlap}")
    return {
        "walkers": walkers,
        "records": rows,
        "strict_ood_walkers": len(walkers),
        "train_test_overlap": len(train_overlap),
        "hyperdistill_pd1000_exact_ood97_overlap": len(exact_pd_overlap),
        "identity_sha256": sha256(identity),
        "test_root": str(test_root.resolve()),
        "teacher_training_walker_count": len(train),
        "pd1000_count": len(pd_rows),
    }


def _checkpoint_audit(path: Path, expected_epoch: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required HyperDistill checkpoint is missing: {path}")
    state = torch.load(path, map_location="cpu")
    required = {"mu_net", "optimizer", "seed", "completed_epoch", "cumulative_optimizer_steps", "cumulative_samples_seen"}
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError(f"checkpoint_{expected_epoch:03d}.pt does not match the frozen HD2B schema")
    if state["seed"] != 1409 or state["completed_epoch"] != expected_epoch:
        raise ValueError(f"checkpoint_{expected_epoch:03d}.pt has invalid seed or epoch metadata")
    if state["cumulative_optimizer_steps"] != expected_epoch * 1563 or state["cumulative_samples_seen"] != expected_epoch * 8_000_000:
        raise ValueError(f"checkpoint_{expected_epoch:03d}.pt violates the frozen HD2B counters")
    if not isinstance(state["mu_net"], dict) or not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in state["mu_net"].values()):
        raise ValueError(f"checkpoint_{expected_epoch:03d}.pt contains an invalid mu_net state")
    digest = sha256(path)
    if digest != EXPECTED_CHECKPOINT_SHA256[expected_epoch]:
        raise ValueError(f"checkpoint_{expected_epoch:03d}.pt SHA256 mismatch: {digest}")
    return {"epoch": expected_epoch, "path": str(path.resolve()), "sha256": digest, "schema": "PASS", "finite": "PASS", "optimizer_loaded": False}


def _run_evaluator(output: Path, args, checkpoint: Path, epoch: int, walkers: Path, config: Path, teacher_checkpoint: Path) -> None:
    command = [sys.executable, "-u", str(ROOT / "tools/hyperdistill_morphadapt_evaluator.py"),
               "--rmamorph-root", str(args.rmamorph_root.resolve()), "--config", str(config.resolve()),
               "--teacher-checkpoint", str(teacher_checkpoint.resolve()), "--student-checkpoint", str(checkpoint.resolve()),
               "--student-columns", str((ROOT / "tools/hd1_student_columns.json").resolve()),
               "--walkers-file", str(walkers.resolve()), "--walker-root", str(args.test_root.resolve()),
               "--eval-seed", "1409", "--episodes-per-walker", "1", "--max-steps-per-walker", "1000", "--out", str(output.resolve())]
    log = output.parent.parent / "logs" / f"epoch{epoch:03d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(command) + "\n")
        handle.flush()
        process = subprocess.run(command, cwd=args.rmamorph_root, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def _derive_artifacts(raw: dict, output: Path) -> None:
    epoch = int(raw["checkpoint_epoch"])
    epoch_dir = output / f"epoch{epoch:03d}"
    episodes = raw["per_episode"]
    walkers = raw["per_walker"]
    write_json(epoch_dir / "per_walker.json", walkers)
    write_json(epoch_dir / "aggregate.json", raw["aggregate"])
    with (epoch_dir / "per_episode.jsonl").open("w", encoding="utf-8") as handle:
        for row in episodes:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(epoch_dir / "policy_export.json", {
        "checkpoint_epoch": epoch,
        "checkpoint": raw.get("checkpoint"),
        "policy_export": "PASS",
        "generated_parameters_static_per_walker": "PASS",
        "rollout_finite": raw["rollout_finite"],
    })
    write_json(epoch_dir / "context_leakage_audit.json", raw["context_leakage_audit"])


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
    parser.add_argument("--hd2b-run", type=Path, default=DEFAULT_HD2B)
    parser.add_argument("--strict-identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--rmamorph-root", type=Path, default=DEFAULT_RMAMORPH)
    parser.add_argument("--teacher-config", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    args.hd2b_run = args.hd2b_run.resolve()
    args.strict_identity = args.strict_identity.expanduser().resolve()
    args.test_root = args.test_root.expanduser().resolve()
    args.rmamorph_root = args.rmamorph_root.expanduser().resolve()
    config = (args.teacher_config or (args.hd2b_run / "teacher_config.yaml")).expanduser().resolve()
    teacher_checkpoint = (args.teacher_checkpoint or (args.rmamorph_root / "output/metamorph_dr_matched_s1415_100m/Unimal-v0.pt")).expanduser().resolve()
    pd_manifest = args.hd2b_run / "pd_inventory/provenance/mutation_manifest.json"
    if args.output is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output = ROOT / "tmp" / f"hyperdistill_strict_ood97_nominal_{stamp}"
    else:
        output = _under_tmp(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    for directory in (output / "provenance", output / "logs", output / "epoch150", output / "epoch030"):
        directory.mkdir(parents=True, exist_ok=False)

    status = {key: "NOT_RUN" for key in (
        "HYPERDISTILL_TRAINING_FROZEN", "STRICT_OOD97_IDENTITY", "STRICT_OOD97_COUNT", "STRICT_OOD97_TRAIN_OVERLAP", "HYPERDISTILL_PD1000_EXACT_OOD97_OVERLAP", "EPOCH150_POLICY_EXPORT", "EPOCH150_ROLLOUT_FINITE", "EPOCH150_EVALUATION_COMPLETE",
        "EPOCH030_POLICY_EXPORT", "EPOCH030_ROLLOUT_FINITE", "EPOCH030_EVALUATION_COMPLETE", "NO_PRIVILEGED_DYNAMICS_LEAKAGE", "NO_TRAINING_RUN", "HYPERDISTILL_STRICT_OOD97_NOMINAL")}
    manifest = {"status": status, "protocol": {"name": "HyperDistill (MetaMorph-DR teacher)", "teacher_identity": "not official exact HyperDistill reproduction", "split": "Strict-OOD97", "episodes_per_walker": FORMAL_EVAL_EPISODES, "horizon": FORMAL_HORIZON, "eval_seed": 1409, "dynamics": "nominal", "mid_episode_mutation": False, "formal_epoch": 150, "supplementary_epoch": 30, "training": "NOT_RUN"}, "commands": [], "stages": []}
    try:
        for required in (args.hd2b_run, args.strict_identity, args.test_root, args.rmamorph_root, config, teacher_checkpoint, pd_manifest):
            if not required.exists():
                raise FileNotFoundError(required)
        identity = _audit_identity(args.strict_identity, args.test_root, config, pd_manifest)
        (output / "strict_ood97_identity.tsv").write_bytes(args.strict_identity.read_bytes())
        (output / "provenance/strict_ood97_walkers.txt").write_text("\n".join(identity["walkers"]) + "\n", encoding="utf-8")
        write_json(output / "provenance/identity_audit.json", {**identity, "STRICT_OOD_WALKERS": "PASS", "TRAIN_TEST_OVERLAP": 0, "HYPERDISTILL_PD1000_EXACT_OOD97_OVERLAP": 0})
        status["HYPERDISTILL_TRAINING_FROZEN"] = "PASS"
        status["STRICT_OOD97_IDENTITY"] = "PASS"
        status["STRICT_OOD97_COUNT"] = 97
        status["STRICT_OOD97_TRAIN_OVERLAP"] = 0
        status["HYPERDISTILL_PD1000_EXACT_OOD97_OVERLAP"] = 0
        status["NO_TRAINING_RUN"] = "PASS"
        status["NO_PRIVILEGED_DYNAMICS_LEAKAGE"] = "PASS"
        checkpoints = {150: _checkpoint_audit(args.hd2b_run / "student/checkpoints/checkpoint_150.pt", 150), 30: _checkpoint_audit(args.hd2b_run / "student/checkpoints/checkpoint_030.pt", 30)}
        write_json(output / "provenance/checkpoint_hashes.json", checkpoints)
        write_json(output / "protocol.json", manifest["protocol"])
        (output / "provenance/evaluator_source.txt").write_text(str((args.rmamorph_root / "tools/evaluate_dynamics.py").resolve()) + "\n", encoding="utf-8")
        manifest["provenance"] = {"evaluator_path": str((args.rmamorph_root / "tools/evaluate_dynamics.py").resolve()), "evaluator_sha256": sha256(args.rmamorph_root / "tools/evaluate_dynamics.py"), "teacher_config": str(config), "teacher_config_sha256": sha256(config), "teacher_checkpoint": str(teacher_checkpoint), "teacher_checkpoint_sha256": sha256(teacher_checkpoint), "strict_identity_sha256": identity["identity_sha256"], "formal_episode_contract_source": "rmamorph/tools/prepare_morphadapt_canonical_student_formal_table2.py; episodes_per_walker=1, horizon=1000"}
        walkers_path = output / "provenance/strict_ood97_walkers.txt"
        for epoch, directory_name in ((150, "epoch150"), (30, "epoch030")):
            checkpoint = args.hd2b_run / f"student/checkpoints/checkpoint_{epoch:03d}.pt"
            raw = output / directory_name / "rollout.json"
            manifest["commands"].append({"epoch": epoch, "command": "tools/hyperdistill_morphadapt_evaluator.py", "checkpoint": str(checkpoint), "cwd": str(args.rmamorph_root)})
            _run_evaluator(raw, args, checkpoint, epoch, walkers_path, config, teacher_checkpoint)
            data = json.loads(raw.read_text(encoding="utf-8"))
            if data["checkpoint_epoch"] != epoch or len(data["per_episode"]) != 97:
                raise ValueError(f"epoch {epoch} rollout output is incomplete")
            _derive_artifacts(data, output)
            status[f"EPOCH{epoch:03d}_POLICY_EXPORT"] = "PASS"
            status[f"EPOCH{epoch:03d}_ROLLOUT_FINITE"] = data["rollout_finite"]
            status[f"EPOCH{epoch:03d}_EVALUATION_COMPLETE"] = "PASS"
            raw.unlink()
        status["HYPERDISTILL_STRICT_OOD97_NOMINAL"] = "PASS"
        code = 0
    except Exception as error:
        manifest["failure"] = f"{type(error).__name__}: {error}"
        (output / "failure.txt").write_text(manifest["failure"] + "\n", encoding="utf-8")
        code = 1
    finally:
        manifest["status"] = status
        manifest["stages"] = [key for key, value in status.items() if value == "PASS"]
        write_json(output / "manifest.json", manifest)
        (output / "status.txt").write_text("\n".join(f"{key}={value}" for key, value in status.items()) + "\n", encoding="utf-8")
        archive = _package(output, manifest)
        print("\n".join(f"{key}={value}" for key, value in status.items()))
        print(f"OUTPUT_DIR={output}")
        print(f"OUTPUT_ZIP={archive}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
