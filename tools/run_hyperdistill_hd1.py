"""Run only the frozen reduced HD1 protocol and package small audit evidence."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.hd1_protocol import PROTOCOL, evaluate_gates, prepare_selection, sha256


STATUS_KEYS = """HD1_TRAIN_WALKER_COUNT HD1_TRAIN_FAMILY_BALANCE HD1_TRAIN_TEST_OVERLAP
HD1_EXPERT_EXPORT HD1_DATASET_RELOAD HD1_TRAIN_TRANSITIONS HD1_VALID_TRANSITIONS
HD1_STUDENT_TRAIN HD1_VALIDATION_FINITE HD1_TRAIN_SMOKE HD1_OOD_SMOKE
HD1_RUNTIME_FINITE HD1_CROSS_MORPH_CONTROL HD1_NORMALIZED_RETURN_GATE HD1_FINAL
OOD_VALID_WALKERS OOD_MEDIAN_STUDENT_TEACHER_RATIO OOD_MEAN_STUDENT_RETURN
OOD_MEAN_TEACHER_RETURN OOD_NUMERICAL_FAILURES""".split()
NUMERIC = {"HD1_TRAIN_WALKER_COUNT", "HD1_TRAIN_TEST_OVERLAP", "HD1_TRAIN_TRANSITIONS", "HD1_VALID_TRANSITIONS", *[k for k in STATUS_KEYS if k.startswith("OOD_")]}
TEACHER_SHA = "ed5bfbcaa57a04aafd58411b8735faa392377d1cd9e16c784a8ddc0ce8fd3d84"
CONFIG_SHA = "90764721ea3474e1e9b2f177b479d74e1fdc5b61d4a0e292cf06f32c2fb08f51"
AUDIT_SUFFIXES = {".json", ".yaml", ".yml", ".txt", ".log", ".csv", ".md"}


def package(output):
    """Raw trajectories and model binaries stay on disk with hashes in manifest."""
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = [p for p in output.rglob("*") if p.is_file() and p != manifest_path and "temp" not in p.relative_to(output).parts]
    manifest["artifacts"] = [{"path": str(p.resolve()), "relative_path": str(p.relative_to(output)),
                              "bytes": p.stat().st_size, "sha256": sha256(p),
                              "included_in_package": p.suffix.lower() in AUDIT_SUFFIXES} for p in files]
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    included = [p for p in files if p.suffix.lower() in AUDIT_SUFFIXES] + [manifest_path]
    archive = Path(str(output) + "_package.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in included:
            z.write(path, path.relative_to(output))
    if archive.stat().st_size > 25 * 1024 * 1024:
        slim = Path(str(output) + "_audit_slim.zip")
        with zipfile.ZipFile(slim, "w", zipfile.ZIP_DEFLATED) as z:
            for path in included:
                if path.suffix == ".log" and path.stat().st_size > 65536:
                    with path.open("rb") as stream:
                        stream.seek(-65536, 2)
                        z.writestr(str(path.relative_to(output)) + ".tail.txt", stream.read())
                else:
                    z.write(path, path.relative_to(output))
        print(f"AUDIT_SLIM_ZIP={slim}")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmamorph-root", type=Path, default=Path.home() / "Workspace/Code/rmamorph")
    base = Path("output/metamorph_dr_matched_s1415_100m")
    parser.add_argument("--config", type=Path, default=base / "config.yaml")
    parser.add_argument("--checkpoint", type=Path, default=base / "Unimal-v0.pt")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "tmp" / f"hyperdistill_hd1_{stamp}"
    (output / "temp").mkdir(parents=True)
    for key in ("TMP", "TEMP", "TMPDIR"):
        os.environ[key] = str(output / "temp")
    status = {key: None if key in NUMERIC else "FAIL" for key in STATUS_KEYS}
    manifest = {"protocol": PROTOCOL, "stages": [], "unmeasured": list(STATUS_KEYS),
                "commands": [], "python": sys.executable, "python_version": sys.version,
                "host": platform.node(), "platform": platform.platform(), "arguments": vars(args).copy()}
    manifest["arguments"] = {key: str(value) for key, value in manifest["arguments"].items()}
    (output / "frozen_protocol.json").write_text(json.dumps(PROTOCOL, indent=2))

    def update(values):
        for key, value in values.items():
            if key in status:
                status[key] = value
                if key in manifest["unmeasured"]:
                    manifest["unmeasured"].remove(key)

    def stage(name, script, arguments, cwd):
        command = [sys.executable, str(root / "tools" / script), *map(str, arguments)]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(cwd)
        env["PYTHONUNBUFFERED"] = "1"
        manifest["commands"].append({"stage": name, "cwd": str(cwd), "argv": command})
        manifest["active_stage"] = name
        print(f"STAGE={name}", flush=True)
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            log.write(json.dumps(command) + "\n")
            log.flush()
            with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                returncode = process.wait()
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
        manifest["stages"].append(name)
        manifest["active_stage"] = None

    code = 1
    try:
        teacher_root = args.rmamorph_root.expanduser().resolve()
        config = args.config.expanduser()
        checkpoint = args.checkpoint.expanduser()
        config = config if config.is_absolute() else teacher_root / config
        checkpoint = checkpoint if checkpoint.is_absolute() else teacher_root / checkpoint
        assert sha256(checkpoint) == TEACHER_SHA, "HD1 teacher must remain the HD0 MetaMorph-DR checkpoint"
        assert sha256(config) == CONFIG_SHA, "HD1 teacher config differs from the frozen HD0 pair"
        (output / "teacher_config.yaml").write_bytes(config.read_bytes())
        manifest["teacher"] = {"checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "config_path": str(config), "config_sha256": sha256(config)}
        manifest["git"] = {}
        for label, repo in (("hyperdistill", root), ("rmamorph", teacher_root)):
            manifest["git"][label] = {
                "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "tracked_changes": subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=repo, text=True),
            }
        manifest["environment_versions"] = {}
        for name in ("torch", "numpy", "gym", "mujoco-py", "PyYAML"):
            try:
                manifest["environment_versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                manifest["environment_versions"][name] = "NOT_AVAILABLE"
        selection = prepare_selection(config, teacher_root / "configs/morphadapt_metamorph_dr_formal_ood98.txt",
                                      teacher_root / "output/unimals_100/train", teacher_root / "output/unimals_100/test", output / "provenance")
        update(selection)
        print("HD1_TRAIN_WALKER_COUNT=18\nHD1_TRAIN_FAMILY_BALANCE=PASS\nHD1_TRAIN_FAMILY_COUNTS=6/6/6\nHD1_TRAIN_TEST_OVERLAP=0", flush=True)
        selection_path = output / "provenance/selection.json"
        common = ["--rmamorph-root", teacher_root, "--config", config, "--checkpoint", checkpoint,
                  "--hd1-selection", selection_path, "--episodes", 3, "--max-steps", 3000]

        def export_dataset(name):
            exported = output / f"{name}_expert"
            dataset = output / f"{name}_dataset"
            stage(f"{name}_expert", "hd0_teacher_export.py", [*common, "--hd1-set", name, "--output", exported], teacher_root)
            stage(f"{name}_convert", "convert_rmamorph_teacher_to_hyperdistill.py",
                  ["--source", exported, "--output", dataset, "--expected-walkers", json.dumps([row["walker_id"] for row in selection[name]])], root)
            return exported, dataset

        train_expert, train_dataset = export_dataset("train")
        train_manifest = json.loads((train_expert / "manifest.json").read_text())
        assert train_manifest["episode_count"] == 54 and train_manifest["transition_count"] <= 54000
        update({"HD1_EXPERT_EXPORT": "PASS", "HD1_DATASET_RELOAD": "PASS"})
        stage("student_train", "hd1_student.py", ["train", "--dataset", train_dataset, "--selection", selection_path, "--output", output / "student"], root)
        metrics = json.loads((output / "student/hd1_student_metrics.json").read_text())
        update(metrics)
        update({"HD1_TRAIN_TRANSITIONS": metrics["train_transitions"], "HD1_VALID_TRANSITIONS": metrics["valid_transitions"]})
        assert 0 < status["HD1_TRAIN_TRANSITIONS"] <= 36000 and 0 < status["HD1_VALID_TRANSITIONS"] <= 18000
        assert status["HD1_STUDENT_TRAIN"] == status["HD1_VALIDATION_FINITE"] == "PASS"
        smoke = {name: export_dataset(name) for name in ("train_smoke", "ood_smoke")}
        columns = train_dataset / "student_columns.json"
        assert all(columns.read_bytes() == (dataset / "student_columns.json").read_bytes() for _, dataset in smoke.values()), "Smoke feature columns differ from frozen training mapper"
        stage("student_freeze", "hd1_student.py", ["export", "--checkpoint", output / "student/hd1_student_checkpoint.pt",
              "--contexts", *[dataset for _, dataset in smoke.values()], "--selection", selection_path, "--output", output / "policies"], root)
        evaluations = {}
        for name, (reference, _) in smoke.items():
            destination = output / f"{name}_evaluation"
            stage(f"{name}_evaluation", "hd0_teacher_export.py", [*common, "--hd1-set", name,
                  "--teacher-reference", reference, "--student-policy-dir", output / "policies",
                  "--student-columns", columns, "--output", destination], teacher_root)
            result = json.loads((destination / "rollout_metrics.json").read_text())
            assert result["student_rollout_finite"] is True and len(result["per_walker"]) == 6
            assert {row["walker_id"] for row in result["per_walker"]} == {row["walker_id"] for row in selection[name]}
            evaluations[name] = result
            update({"HD1_TRAIN_SMOKE" if name == "train_smoke" else "HD1_OOD_SMOKE": "PASS"})
        gates = evaluate_gates(evaluations["ood_smoke"]["per_walker"])
        seen_metrics = evaluate_gates(evaluations["train_smoke"]["per_walker"])
        (output / "hd1_train_smoke_metrics.json").write_text(json.dumps({
            "per_walker": seen_metrics["per_walker"], "ratio_definition": PROTOCOL["ratio_definition"]
        }, indent=2, allow_nan=False))
        (output / "hd1_scientific_metrics.json").write_text(json.dumps(gates, indent=2, allow_nan=False))
        update(gates)
        if gates["HD1_FINAL"] == "FAIL":
            (output / "hd1_failure_diagnosis.json").write_text(json.dumps({
                "conclusion": "Reduced gate failed; implementation audit precedes any claim of insufficient generalization. No automatic retuning.",
                "ordered_review": [
                    {"check": "train18/family/split", "family_counts": selection["family_counts"], "overlap": selection["HD1_TRAIN_TEST_OVERLAP"]},
                    {"check": "per-walker context", "evidence": "policies/hd1_export_stats.json and teacher-reference reset comparisons"},
                    {"check": "generated parameters at reset", "evidence": "one named frozen policy per morphology; raw reset context compared on every episode"},
                    {"check": "batch morphology indexing", "evidence": "source test_hd1_student.py; review training inputs before excluding an implementation issue"},
                    {"check": "RMS binding", "evidence": "exact RMS comparison across train18 and smoke versus student checkpoint"},
                    {"check": "action mask", "evidence": "converter validation and strict padded-output check during rollout"},
                    {"check": "fit versus generalization", "training_metrics": metrics},
                ],
            }, indent=2, allow_nan=False))
        code = 0 if status["HD1_FINAL"] == "PASS" else 1
    except Exception:
        manifest["failure"] = traceback.format_exc()
        (output / "failure.txt").write_text(manifest["failure"])
        print(manifest["failure"], file=sys.stderr)
    finally:
        manifest["status"] = status
        manifest["missing_evidence_semantics"] = "FAIL + unmeasured means no result; do not diagnose scientific generalization from an infrastructure failure"
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
        (output / "status.json").write_text(json.dumps(status, indent=2, allow_nan=False))
        lines = "\n".join(
            f"{key}={('NOT_RUN' if key in manifest['unmeasured'] else 'UNDEFINED') if value is None else value}"
            for key, value in status.items()
        )
        (output / "status.txt").write_text(lines + "\n")
        print(lines)
        print(f"OUTPUT_ZIP={package(output)}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
