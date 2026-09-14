"""Run the bounded HD0 stages in isolated Python processes and package evidence."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback


GATES = """METAMORPH_TEACHER_LOAD TEACHER_OBS_FINITE TEACHER_ACTION_FINITE
OBS_ORDER_BINDING ACTION_ORDER_BINDING ACTION_MASK_BINDING MORPH_CONTEXT_BINDING
OBS_RMS_PROVENANCE HYPERDISTILL_DATASET_EXPORT HYPERDISTILL_DATASET_RELOAD
STATIC_CONTEXT_FREEZE NO_MUTATION_LEAKAGE SINGLE_WALKER_OVERFIT HD0_FINAL""".split()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmamorph-root", type=Path, default=Path.home() / "Workspace/Code/rmamorph")
    teacher_dir = Path("output/metamorph_dr_matched_s1415_100m")
    parser.add_argument("--config", type=Path, default=teacher_dir / "config.yaml")
    parser.add_argument("--checkpoint", type=Path, default=teacher_dir / "Unimal-v0.pt")
    parser.add_argument("--walkers", nargs=3, help="Defaults to the first three walkers in the teacher training config")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--python", default=sys.executable, help="Interpreter of the verified runtime environment")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "tmp" / f"hyperdistill_hd0_{stamp}"
    output.mkdir(parents=True)
    statuses = {key: "FAIL" for key in GATES}
    evidence = {"status": statuses, "completed_stages": [], "not_run": list(GATES),
                "baseline": "HyperDistill (MetaMorph teacher)",
                "host": platform.node(), "interpreter": args.python,
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}

    def stage(name, script, arguments, cwd):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(cwd)
        env["TMPDIR"] = env["TMP"] = env["TEMP"] = str(output / "temp")
        env["PYTHONUNBUFFERED"] = "1"
        command = [args.python, str(root / "tools" / script), *map(str, arguments)]
        print(f"STAGE={name}", flush=True)
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            log.write(json.dumps(command) + "\n")
            log.flush()
            subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        evidence["completed_stages"].append(name)

    def record_gates(path):
        measured = json.loads(path.read_text())
        for key in GATES[:-2]:
            if key in measured:
                assert measured[key] in ("PASS", "FAIL"), f"Invalid gate value: {key}"
                statuses[key] = measured[key]
                if key in evidence["not_run"]:
                    evidence["not_run"].remove(key)
        return measured

    result = 1
    try:
        assert args.episodes >= 2, "At least two independent episodes required"
        assert 1 <= args.epochs <= 100, "HD0 is limited to 100 epochs"
        if args.walkers:
            assert len(set(args.walkers)) == 3
        teacher_root = args.rmamorph_root.expanduser().resolve()
        config = args.config.expanduser()
        checkpoint = args.checkpoint.expanduser()
        if not config.is_absolute():
            config = teacher_root / config
        if not checkpoint.is_absolute():
            checkpoint = teacher_root / checkpoint
        for path in (teacher_root, config, checkpoint):
            assert path.exists(), f"Required path unavailable: {path}"
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == "ed5bfbcaa57a04aafd58411b8735faa392377d1cd9e16c784a8ddc0ce8fd3d84", "Checkpoint differs from the selected HD0 MetaMorph teacher"
        assert hashlib.sha256(config.read_bytes()).hexdigest() == "90764721ea3474e1e9b2f177b479d74e1fdc5b61d4a0e292cf06f32c2fb08f51", "Config differs from the selected teacher's paired training config"
        evidence["checkpoint_sha256"] = digest.hexdigest()
        shutil.copyfile(config, output / "selected_teacher_config.yaml")
        (output / "teacher_selection.json").write_text(json.dumps({
            "checkpoint": str(checkpoint), "checkpoint_sha256": digest.hexdigest(),
            "config": str(config), "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "runtime_validation": "NOT_RUN until teacher_export completes",
        }, indent=2))
        (output / "temp").mkdir()
        common = ["--rmamorph-root", teacher_root, "--config", config.resolve(),
                  "--checkpoint", checkpoint.resolve(),
                  "--episodes", args.episodes]
        if args.walkers:
            common += ["--walkers", *args.walkers]
        stage("teacher_export", "hd0_teacher_export.py", [*common, "--output", output / "teacher"], teacher_root)
        teacher = record_gates(output / "teacher" / "manifest.json")
        walkers = teacher["walkers"]
        stage("dataset", "convert_rmamorph_teacher_to_hyperdistill.py",
              ["--source", output / "teacher", "--output", output / "dataset"], root)
        mapping = record_gates(output / "dataset" / "mapping.json")
        stage("student", "hd0_student.py", ["--dataset", output / "dataset" / f"{walkers[0]}.pkl",
              "--walker", walkers[0], "--output", output / "student", "--epochs", args.epochs], root)
        student = record_gates(output / "student" / "hd0_student_metrics.json")
        stage("student_rollout", "hd0_teacher_export.py", [*common, "--output", output / "rollout",
              "--student-policy", output / "student" / "hd0_student.ts",
              "--student-columns", output / "dataset" / "student_columns.json"], teacher_root)
        # Only measured evidence may pass the aggregate gate. No stage exit code
        # alone establishes the numerical or morphology contracts.
        rollout = record_gates(output / "rollout" / "rollout_metrics.json")
        rollout_export = json.loads((output / "rollout" / "manifest.json").read_text())
        assert teacher["sha256"] == rollout_export["sha256"], "Teacher or morphology provenance changed between stages"
        assert teacher["order_bindings"] == rollout_export["order_bindings"], "Morphology ordering changed between stages"
        import numpy as np
        with np.load(output / "teacher" / "arrays.npz") as before, np.load(output / "rollout" / "arrays.npz") as after:
            assert np.array_equal(before["context_raw"][0], after["context_raw"][0]), "Nominal deployment context differs from the distilled context"
        for key in GATES[:-2]:
            values = [source.get(key) for source in (teacher, mapping, student, rollout)]
            assert "PASS" in values and "FAIL" not in values, f"Missing or failed gate evidence: {key}: {values}"
            statuses[key] = "PASS"
        assert student["final_train_mse"] < student["initial_train_mse"], "Train imitation did not improve"
        assert student["final_validation_mse"] < student["initial_validation_mse"], "Held-out imitation did not improve"
        assert student["imitation_converged"], "Imitation convergence criterion not met"
        assert rollout["student_rollout_finite"] is True, "Student rollout nonfinite"
        statuses["SINGLE_WALKER_OVERFIT"] = statuses["HD0_FINAL"] = "PASS"
        evidence["not_run"] = []
        result = 0
    except Exception:
        # The workflow boundary preserves the original traceback, then packages
        # failure evidence and returns failure; no downstream stage is run.
        evidence["failure"] = traceback.format_exc()
        (output / "failure.txt").write_text(evidence["failure"], encoding="utf-8")
        print(evidence["failure"], file=sys.stderr)
    finally:
        evidence["note"] = "FAIL with not_run means the required evidence was not obtained; it is not a measured scientific negative result."
        (output / "gate.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        report = "\n".join(f"{key}={value}" for key, value in statuses.items())
        (output / "gate.txt").write_text(report + "\n", encoding="utf-8")
        print(report)
        archive = shutil.make_archive(str(output) + "_package", "zip", root_dir=output)
        print(f"OUTPUT_ZIP={archive}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
