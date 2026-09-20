import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hd2c_protocol import EPOCHS, OOD_WALKERS, prepare_selection, summarize  # noqa: E402


class Hd2cProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "tmp").mkdir(exist_ok=True)

    def assets(self, root):
        train = [f"floor-1409-{i}" for i in range(34)] + [f"mvt-5506-{i}" for i in range(33)] + [f"vt-1409-{i}" for i in range(33)]
        source = root / "test"
        for sub in ("xml", "metadata"):
            (source / sub).mkdir(parents=True)
        for walker in OOD_WALKERS:
            (source / "xml" / f"{walker}.xml").write_text("<agent />")
            (source / "metadata" / f"{walker}.json").write_text(json.dumps({"walker": walker}))
        entries = []
        for index in range(1000):
            parent = train[index // 10]
            entries.append({"pd_robot_id": f"{parent}-mutate-{index % 10}", "parent_walker_id": parent})
        return train, source, entries

    def test_selection_requires_frozen_inventory_and_materializes_ood_only(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as td:
            root = Path(td); train, source, entries = self.assets(root)
            config = root / "config.yaml"; config.write_text(yaml.safe_dump({"ENV": {"WALKERS": train}}))
            ood = root / "ood.txt"; ood.write_text("\n".join(OOD_WALKERS + tuple(f"floor-extra-{i}" for i in range(92))) + "\n")
            manifest = root / "pd.json"; manifest.write_text(json.dumps(entries))
            result = prepare_selection(config, ood, source, manifest, root / "selection")
            self.assertEqual(result["train"], [])
            self.assertEqual(result["train_smoke"], [])
            self.assertEqual([r["walker_id"] for r in result["ood_smoke"]], list(OOD_WALKERS))
            self.assertEqual(result["sources"]["pd_robot_count"], 1000)
            self.assertTrue(all(row["xml_sha256"] and row["metadata_sha256"] for row in result["ood_smoke"]))
            self.assertTrue((root / "selection" / "selection.json").is_file())
            self.assertEqual((root / "selection/hd2c_ood_smoke_walkers.txt").read_text().splitlines(), list(OOD_WALKERS))

    def test_selection_rejects_wrong_pd_count_or_missing_ood_asset(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as td:
            root = Path(td); train, source, entries = self.assets(root)
            config = root / "config.yaml"; config.write_text(yaml.safe_dump({"ENV": {"WALKERS": train}}))
            ood = root / "ood.txt"; ood.write_text("\n".join(OOD_WALKERS) + "\n")
            manifest = root / "pd.json"; manifest.write_text(json.dumps(entries[:-1]))
            with self.assertRaises(AssertionError):
                prepare_selection(config, ood, source, manifest, root / "bad_count")
            manifest.write_text(json.dumps(entries))
            (source / "xml" / f"{OOD_WALKERS[0]}.xml").unlink()
            with self.assertRaises(FileNotFoundError):
                prepare_selection(config, ood, source, manifest, root / "missing_asset")

            # A PD filename equal to an OOD ID is forbidden even when its parent is valid.
            (source / "xml" / f"{OOD_WALKERS[0]}.xml").write_text("<agent />")
            entries[0]["pd_robot_id"] = OOD_WALKERS[0]
            manifest.write_text(json.dumps(entries))
            with self.assertRaises(AssertionError):
                prepare_selection(config, ood, source, manifest, root / "pd_ood_overlap")

    def row(self, walker, good=True):
        return {"walker_id": walker, "teacher_episode_returns": [10.0] * 3,
                "student_episode_returns": [5.0 if good else 1.0] * 3,
                "teacher_episode_lengths": [500] * 3, "student_episode_lengths": [500] * 3,
                "student_forward_displacements": [1.0] * 3, "action_mse_valid": 0.1,
                "numerical_failures": 0}

    def test_summary_uses_exact_id_order_and_formal_epoch_only(self):
        evaluations = {}
        for epoch in EPOCHS:
            evaluations[epoch] = {"per_walker": [self.row(walker, good=epoch != 150) for walker in OOD_WALKERS]}
        curve, formal, comparison = summarize(evaluations)
        self.assertEqual([point["epoch"] for point in curve], list(EPOCHS))
        self.assertEqual(curve[0]["median_closed_loop_action_mse"], 0.1)
        self.assertEqual(curve[3]["median_student_teacher_ratio"], 0.5)
        self.assertEqual(formal["HD2_FINAL"], "FAIL")
        self.assertEqual(formal["HD2_CROSS_MORPH_CONTROL"], "PASS")
        self.assertEqual(comparison["HD1"]["approx_median_ratio"], 0.105)
        self.assertEqual(comparison["HD2_epoch150"]["median_ratio"], 0.1)
        self.assertEqual(formal['per_walker'][0]['median_student_episode_length'], 500)
        self.assertIn("no causal", comparison["QUALIFIED_COMPARISON"])

    def test_summary_rejects_wrong_order(self):
        evaluations = {epoch: {"per_walker": [self.row(walker) for walker in reversed(OOD_WALKERS)]} for epoch in EPOCHS}
        with self.assertRaises(AssertionError):
            summarize(evaluations)


if __name__ == "__main__":
    unittest.main()
