"""Static and filesystem-only contracts for the opt-in HD1 interface."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Hd1ExportTests(unittest.TestCase):
    def test_selection_validates_assets_hashes_and_split_membership(self):
        namespace = {}
        source = (ROOT / "tools" / "hd0_teacher_export.py").read_text(encoding="utf-8")
        prefix = source[:source.index("def main()")]
        exec(prefix, namespace)
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as td:
            root = Path(td)
            config = root / "config.yaml"
            ood_pool = root / "ood.txt"
            config.write_text("ENV:\n  WALKERS: [train_a]\n", encoding="utf-8")
            ood_pool.write_text("ood_b\n", encoding="utf-8")
            rows = []
            for walker in ("train_a", "ood_b"):
                xml = root / (walker + ".xml")
                metadata = root / (walker + ".json")
                xml.write_text("xml", encoding="utf-8")
                metadata.write_text("{}", encoding="utf-8")
                rows.append({"walker_id": walker, "xml_path": str(xml), "metadata_path": str(metadata),
                             "xml_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
                             "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                             "in_teacher_training": walker == "train_a", "in_frozen_ood": walker == "ood_b"})
            selection = {"sources": {"teacher_config": str(config), "teacher_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                                     "ood_pool": str(ood_pool), "ood_pool_sha256": hashlib.sha256(ood_pool.read_bytes()).hexdigest()},
                         "train_smoke": [rows[0]], "ood_smoke": [rows[1]]}
            path = root / "selection.json"
            path.write_text(json.dumps(selection), encoding="utf-8")
            self.assertEqual(namespace["load_hd1_selection"](path, "train_smoke", ["train_a"])[0]["walker_id"], "train_a")
            self.assertEqual(namespace["load_hd1_selection"](path, "ood_smoke", ["train_a"])[0]["walker_id"], "ood_b")
            with self.assertRaises(ValueError):
                namespace["load_hd1_selection"](path, "ood_smoke", ["ood_b"])

    def test_hd0_count_gate_remains_opt_in(self):
        source = (ROOT / "tools" / "convert_rmamorph_teacher_to_hyperdistill.py").read_text(encoding="utf-8")
        self.assertIn("def convert(source, output, expected_walkers=None)", source)
        self.assertIn("HD0 export requires exactly three training walkers", source)
        self.assertIn("--expected-walkers", source)


if __name__ == "__main__":
    unittest.main()
