import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hd2_protocol import prepare_pd_inventory  # noqa: E402
from tools.run_hyperdistill_hd2a import PROTOCOL, prepare_inventory_stage, serialize_status, STATUS_KEYS, sha256, verify_resume_run  # noqa: E402


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

    def test_runner_owned_root_and_protocol_owned_inventory_fail_closed(self):
        source = ROOT / "data" / "train_mutate_1000"
        parents = sorted(path.stem for path in (source / "metadata").glob("*.json") if "-mutate-" not in path.stem)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            run_root = Path(temporary) / "hyperdistill_hd2a_preflight_test"
            run_root.mkdir()
            (run_root / "frozen_protocol.json").write_text("{}")
            (run_root / "teacher_config.yaml").write_text("frozen: true\n")
            config = run_root / "teacher_train100.yaml"
            config.write_text(yaml.safe_dump({"ENV": {"WALKERS": parents}}))
            ood = run_root / "ood.txt"; ood.write_text("floor-heldout\n")
            inventory_dir = run_root / "pd_inventory"
            self.assertTrue(run_root.exists())
            self.assertFalse(inventory_dir.exists())
            created, result = prepare_inventory_stage(config, ood, source, run_root)
            self.assertEqual(created, inventory_dir)
            self.assertTrue(created.exists())
            self.assertEqual(result["HD2_PD_PROVENANCE"], "PASS")
            with self.assertRaises(FileExistsError):
                prepare_inventory_stage(config, ood, source, run_root)

    def test_unexecuted_gates_are_not_serialized_as_failures(self):
        status = {key: None for key in STATUS_KEYS}
        serialized = serialize_status(status)
        self.assertTrue(all(serialized[key] == "NOT_RUN" for key in STATUS_KEYS if key != "HD2A_FINAL"))
        self.assertEqual(serialized["HD2A_FINAL"], "NOT_MEASURED")

    def test_resume_requires_complete_hash_bound_collection(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            run = Path(temporary)
            config, checkpoint = run / "config.yaml", run / "teacher.pt"
            config.write_text("teacher: frozen\n"); checkpoint.write_bytes(b"teacher")
            (run / "frozen_protocol.json").write_text(json.dumps(PROTOCOL))
            (run / "teacher_config.yaml").write_bytes(config.read_bytes())
            selected, per_robot = [], []
            for index in range(10):
                pd_id = f"floor-pd-{index}"
                xml = run / "pd_inventory" / "pd1000" / "xml" / f"{pd_id}.xml"
                metadata = run / "pd_inventory" / "pd1000" / "metadata" / f"{pd_id}.json"
                xml.parent.mkdir(parents=True, exist_ok=True); metadata.parent.mkdir(parents=True, exist_ok=True)
                xml.write_text("<agent />"); metadata.write_text("{}")
                selected.append({"pd_robot_id": pd_id, "materialized_xml_path": str(xml), "materialized_metadata_path": str(metadata),
                                 "pd_xml_sha256": sha256(xml), "pd_metadata_sha256": sha256(metadata)})
                shard = run / "expert_shards" / f"{pd_id}.npz"; shard.parent.mkdir(exist_ok=True); shard.write_bytes(pd_id.encode())
                per_robot.append({"pd_robot_id": pd_id, "transition_count": 8000, "shard_sha256": sha256(shard)})
            provenance = run / "pd_inventory" / "provenance"; provenance.mkdir(parents=True)
            (provenance / "mutation_manifest.json").write_text(json.dumps(selected))
            (provenance / "parents.json").write_text(json.dumps([{"walker_id": index} for index in range(100)]))
            validity = {"HD2_PD_VALIDITY": "PASS", "per_robot": [{"pd_robot_id": row["pd_robot_id"]} for row in selected]}
            (run / "pd_validity").mkdir(); (run / "pd_validity" / "pd_validity.json").write_text(json.dumps(validity))
            collection = {"HD2A_EXACT_8000": "PASS", "HD2A_TOTAL_TRANSITIONS": 80000, "per_robot": per_robot}
            (run / "expert_shards" / "collection_metrics.json").write_text(json.dumps(collection))
            manifest = {"teacher": {"checkpoint_sha256": sha256(checkpoint), "config_sha256": sha256(config)}}
            (run / "manifest.json").write_text(json.dumps(manifest))
            inventory, resumed = verify_resume_run(run, config, checkpoint)
            self.assertEqual(inventory, run / "pd_inventory")
            self.assertEqual(resumed["HD2A_TOTAL_TRANSITIONS"], 80000)
            (run / "expert_shards" / "floor-pd-0.npz").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "RESUME_REJECTED"):
                verify_resume_run(run, config, checkpoint)


if __name__ == "__main__":
    unittest.main()
