import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hd2_protocol import prepare_pd_inventory  # noqa: E402


class Hd2ProtocolTests(unittest.TestCase):
    def make_corpus(self, root):
        train = [f"{family}-1409-{index}" for family, index in (("floor", 0), ("mvt", 1), ("vt", 2))]
        # The production protocol rejects anything other than train100; this fixture patches the YAML after construction.
        source = root / "source"; (source / "xml").mkdir(parents=True); (source / "metadata").mkdir()
        for parent in train:
            for variant in range(10):
                name = parent if variant == 0 else f"{parent}-mutate-{variant - 1}-dof"
                (source / "xml" / f"{name}.xml").write_text("<agent />")
                (source / "metadata" / f"{name}.json").write_text(json.dumps({"dof": 1}))
        return train, source

    def test_real_protocol_rejects_non_train100_before_generation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            root = Path(temporary); train, source = self.make_corpus(root)
            config = root / "config.yaml"; config.write_text(yaml.safe_dump({"ENV": {"WALKERS": train}})); ood = root / "ood.txt"; ood.write_text("floor-ood\n")
            with self.assertRaisesRegex(ValueError, "train100"):
                prepare_pd_inventory(config, ood, source, root / "out", preflight_count=10)

    def test_repository_static_pd1000_has_documented_shape(self):
        source = ROOT / "data" / "train_mutate_1000"
        parents = sorted(path.stem for path in (source / "metadata").glob("*.json") if "-mutate-" not in path.stem)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            root = Path(temporary)
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({"ENV": {"WALKERS": parents}}))
            ood = root / "ood.txt"; ood.write_text("floor-heldout\n")
            result = prepare_pd_inventory(config, ood, source, root / "out", preflight_count=10)
            self.assertEqual(result["HD2_BASE_PARENT_COUNT"], 100)
            self.assertEqual(result["HD2_PD_ROBOT_COUNT"], 1000)
            self.assertEqual(len(result["selected_pd_robots"]), 10)
            self.assertEqual(len({row["parent_walker_id"] for row in result["selected_pd_robots"]}), 10)


if __name__ == "__main__":
    unittest.main()
