import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hd2_protocol import prepare_pd_inventory  # noqa: E402
from tools.run_hyperdistill_hd2a import (  # noqa: E402
    PROTOCOL, archive_stale_failure, prepare_inventory_stage, resolve_resume_stage,
    serialize_status, STATUS_KEYS, sha256, validate_converted_stage, validate_student_stage,
    verify_resume_run,
)


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

    def test_completed_downstream_stages_promote_without_rerun(self):
        import torch
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            run = Path(temporary); selection = []
            expert = run / "expert_shards"; expert.mkdir()
            converted = run / "converted_shards"; converted.mkdir()
            columns = (ROOT / "tools" / "hd1_student_columns.json").read_bytes()
            (converted / "student_columns.json").write_bytes(columns)
            for index in range(10):
                pd_id = f"floor-pd-{index}"; selection.append({"pd_robot_id": pd_id})
                source = expert / f"{pd_id}.npz"; source.write_bytes(pd_id.encode())
                with (converted / f"{pd_id}.pkl").open("wb") as stream:
                    pickle.dump({"manifest": {"pd_robot_id": pd_id, "source_sha256": sha256(source)}}, stream)
            selection_path = run / "selection.json"; selection_path.write_text(json.dumps(selection))
            audit = {"HD2A_SHARDED_DATASET": "PASS", "HD2A_MAPPER_IDENTITY": "PASS", "FROZEN_HD1_COLUMNS_LENGTH": 204,
                     "FROZEN_HD1_COLUMNS_HASH": "PASS", "HD2_CONVERTER_USES_FROZEN_COLUMNS": "PASS",
                     "HD2_CONVERTER_DOES_NOT_DERIVE_FIRST_17_PER_LIMB": "PASS", "HD2_OUTPUT_COLUMNS_EQUALS_HD1": "PASS",
                     "student_columns_sha256": sha256(converted / "student_columns.json"), "shard_count": 10}
            (converted / "conversion_audit.json").write_text(json.dumps(audit))
            state, result = resolve_resume_stage(run, "converted_shards", "conversion_audit.json", lambda: validate_converted_stage(run, selection_path))
            self.assertEqual((state, result["CONVERT_STAGE"]), ("SKIP", "REUSED_VERIFIED"))
            student = run / "student"; student.mkdir()
            metrics = {"HD2A_TASK_BALANCE": "PASS", "HD2A_EFFECTIVE_BATCH_5120": "PASS", "HD2A_CONTEXT_DROPOUT": "PASS",
                       "HD2A_CHECKPOINT_RELOAD": "PASS", "HD2_EFFECTIVE_BATCH_SIZE": 5120, "HD2_BATCH_IMPLEMENTATION": "physical",
                       "microbatch": 5120, "optimizer_updates": 1}
            (student / "hd2a_update_metrics.json").write_text(json.dumps(metrics))
            torch.save({"mu_net": {}, "optimizer": {}, "seed": 1409}, student / "checkpoint_000.pt")
            state, result = resolve_resume_stage(run, "student", "hd2a_update_metrics.json", lambda: validate_student_stage(run))
            self.assertEqual((state, result["STUDENT_PREFLIGHT_STAGE"]), ("SKIP", "REUSED_VERIFIED"))
            audit["student_columns_sha256"] = "corrupt"; (converted / "conversion_audit.json").write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "RESUME_REJECTED"):
                resolve_resume_stage(run, "converted_shards", "conversion_audit.json", lambda: validate_converted_stage(run, selection_path))

    def test_absent_incomplete_and_stale_failure_resume_states(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            run = Path(temporary)
            self.assertEqual(resolve_resume_stage(run, "converted_shards", "conversion_audit.json", lambda: {}), ("RUN", None))
            (run / "converted_shards").mkdir()
            self.assertEqual(resolve_resume_stage(run, "converted_shards", "conversion_audit.json", lambda: {}), ("RUN", "CLEARED_INCOMPLETE"))
            self.assertFalse((run / "converted_shards").exists())
            (run / "failure.txt").write_text("old failure")
            archive_stale_failure(run)
            self.assertFalse((run / "failure.txt").exists())
            self.assertTrue((run / "attempt_history" / "attempt_002_failure.txt").is_file())


if __name__ == "__main__":
    unittest.main()
