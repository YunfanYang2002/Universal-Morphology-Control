import json
import sys
import tempfile
import unittest
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hd1_protocol import evaluate_gates, prepare_selection  # noqa: E402
from tools.run_hyperdistill_hd1 import package  # noqa: E402


class Hd1ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "tmp").mkdir(exist_ok=True)

    def rows(self, locomotion=6, teacher=10.0, student=5.0):
        rows = []
        for i, fam in enumerate(["floor", "floor", "mvt", "mvt", "vt", "vt"]):
            good = i < locomotion
            rows.append({
                "walker_id": f"{fam}-{i}",
                "teacher_episode_returns": [teacher] * 3,
                "student_episode_returns": [student] * 3,
                "teacher_episode_lengths": [500] * 3,
                "student_episode_lengths": [500 if good else 499] * 3,
                "student_forward_displacements": [1.0 if good else 0.99] * 3,
                "action_mse_valid": 0.1,
                "numerical_failures": 0,
            })
        return rows

    def test_gate_boundaries_and_failures(self):
        result = evaluate_gates(self.rows(locomotion=4))
        self.assertEqual(result["HD1_RUNTIME_FINITE"], "PASS")
        self.assertEqual(result["HD1_CROSS_MORPH_CONTROL"], "PASS")
        self.assertEqual(result["HD1_NORMALIZED_RETURN_GATE"], "PASS")
        self.assertEqual(result["HD1_FINAL"], "PASS")
        self.assertEqual(result["OOD_MEDIAN_STUDENT_TEACHER_RATIO"], 0.5)

        result = evaluate_gates(self.rows(locomotion=3))
        self.assertEqual(result["HD1_CROSS_MORPH_CONTROL"], "FAIL")
        self.assertEqual(result["HD1_FINAL"], "FAIL")

    def test_undefined_teacher_ratio_never_divides(self):
        for teacher in (0.0, -1.0, 1e-7):
            result = evaluate_gates(self.rows(teacher=teacher, student=5.0))
            self.assertEqual(result["HD1_NORMALIZED_RETURN_GATE"], "FAIL")
            self.assertTrue(all(not row["normalized_return_defined"] for row in result["per_walker"]))
            self.assertIsNone(result["OOD_MEDIAN_STUDENT_TEACHER_RATIO"])
            self.assertEqual(result["HD1_FINAL"], "FAIL")

    def test_nonfinite_and_numerical_failures_are_runtime_failures(self):
        rows = self.rows()
        rows[0]["student_episode_returns"][1] = float("nan")
        result = evaluate_gates(rows)
        self.assertEqual(result["HD1_RUNTIME_FINITE"], "FAIL")
        self.assertEqual(result["HD1_FINAL"], "FAIL")
        self.assertIsNone(result["per_walker"][0]["normalized_return"])

        rows = self.rows()
        rows[1]["numerical_failures"] = 2
        result = evaluate_gates(rows)
        self.assertEqual(result["HD1_RUNTIME_FINITE"], "FAIL")
        self.assertEqual(result["OOD_NUMERICAL_FAILURES"], 2)
        self.assertEqual(result["OOD_LOCOMOTION_WALKERS"], 5)

    def make_assets(self, root, train, ood):
        for name in ("train", "test"):
            for sub in ("xml", "metadata"):
                (root / name / sub).mkdir(parents=True)
        for walker in train:
            (root / "train" / "xml" / f"{walker}.xml").write_text("<agent/>")
            (root / "train" / "metadata" / f"{walker}.json").write_text(json.dumps({"id": walker}))
        for walker in ood:
            (root / "test" / "xml" / f"{walker}.xml").write_text("<agent/>")
            (root / "test" / "metadata" / f"{walker}.json").write_text(json.dumps({"id": walker}))

    def test_prepare_selection_is_balanced_ordered_and_asset_bound(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as td:
            root = Path(td)
            train = [f"{fam}-{i}" for i in range(7) for fam in ("vt", "floor", "mvt")]
            ood = [f"{fam}-ood{i}" for fam in ("mvt", "vt", "floor") for i in range(2)]
            self.make_assets(root, train, ood)
            (root / "config.yaml").write_text("ENV:\n  WALKERS:\n" + "".join(f"    - {w}\n" for w in train))
            (root / "ood.txt").write_text("\n".join(ood) + "\n")
            selection = prepare_selection(root / "config.yaml", root / "ood.txt", root / "train", root / "test", root / "out")
            self.assertEqual([r["walker_id"] for r in selection["train"]], [f"{fam}-{i}" for fam in ("floor", "mvt", "vt") for i in range(6)])
            self.assertEqual([r["walker_id"] for r in selection["train_smoke"]], [f"{fam}-{i}" for fam in ("floor", "mvt", "vt") for i in range(2)])
            self.assertEqual([r["walker_id"] for r in selection["ood_smoke"]], [f"{fam}-ood{i}" for fam in ("floor", "mvt", "vt") for i in range(2)])
            self.assertEqual(selection["HD1_TRAIN_WALKER_COUNT"], 18)

    def test_prepare_selection_rejects_overlap_and_missing_assets(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as td:
            root = Path(td)
            train = [f"{fam}-{i}" for fam in ("floor", "mvt", "vt") for i in range(6)]
            ood = [f"{fam}-ood{i}" for fam in ("floor", "mvt", "vt") for i in range(2)]
            self.make_assets(root, train, ood)
            config = root / "config.yaml"
            config.write_text("ENV:\n  WALKERS:\n" + "".join(f"    - {w}\n" for w in train))
            pool = root / "ood.txt"
            pool.write_text("\n".join(ood) + "\n")
            pool.write_text(train[0] + "\n" + "\n".join(ood[1:]) + "\n")
            with self.assertRaises(AssertionError):
                prepare_selection(config, pool, root / "train", root / "test", root / "out_overlap")
            pool.write_text("\n".join(ood) + "\n")
            (root / "test" / "xml" / f"{ood[0]}.xml").unlink()
            with self.assertRaises(FileNotFoundError):
                prepare_selection(config, pool, root / "train", root / "test", root / "out_missing")

    def test_package_keeps_audit_manifest_and_excludes_raw_and_model_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as td:
            output = Path(td) / "run"
            output.mkdir()
            (output / "manifest.json").write_text(json.dumps({"stages": ["synthetic"]}))
            (output / "metrics.json").write_text(json.dumps({"status": "PASS"}))
            npz = output / "raw_arrays.npz"
            npz.write_bytes(b"synthetic raw arrays")
            (output / "checkpoint.pt").write_bytes(b"synthetic model")
            (output / "walker.pkl").write_bytes(b"synthetic pickle")
            archive = package(output)
            manifest = json.loads((output / "manifest.json").read_text())
            artifacts = {item["relative_path"]: item for item in manifest["artifacts"]}
            self.assertEqual(set(artifacts), {"metrics.json", "raw_arrays.npz", "checkpoint.pt", "walker.pkl"})
            for path, item in artifacts.items():
                self.assertEqual(item["bytes"], (output / path).stat().st_size)
                self.assertTrue(item["sha256"])
                self.assertEqual(item["path"], str((output / path).resolve()))
            with zipfile.ZipFile(archive) as zipped:
                names = set(zipped.namelist())
            self.assertEqual(names, {"manifest.json", "metrics.json"})


if __name__ == "__main__":
    unittest.main()
