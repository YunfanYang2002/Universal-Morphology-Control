import json
import hashlib
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hd2_student import (  # noqa: E402
    FROZEN_HD1_COLUMNS_SHA256,
    FullCoverageShardBatches,
    TaskBalancedShardBatches,
    convert_shards,
    epoch_batch_sizes,
    load_frozen_hd1_columns,
    official_epoch_batch_plan,
    select_frozen_student_observation,
    write_frozen_student_columns,
)


class Hd2ShardTests(unittest.TestCase):
    def test_official_epoch_retains_half_final_batch(self):
        plan = official_epoch_batch_plan(8000000, 5120)
        self.assertEqual(plan["optimizer_steps"], 1563)
        self.assertEqual(plan["final_batch_size"], 2560)
        self.assertFalse(plan["drop_last"])
        self.assertEqual(plan["full_batches"] * 5120 + plan["final_batch_size"], 8000000)
        batch_sizes = epoch_batch_sizes(8000000, 5120)
        self.assertEqual(len(batch_sizes), 1563)
        self.assertEqual(sum(batch_sizes), 8000000)
        self.assertEqual(batch_sizes[-1], 2560)

    def test_full_coverage_batching_has_no_padding_or_carry_between_epochs(self):
        self.assertEqual(epoch_batch_sizes(13, 5), [5, 5, 3])
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            root = Path(temporary); entries = []
            for pd_id, values in (("floor-pd-0", range(7)), ("floor-pd-1", range(7, 13))):
                entries.append({"pd_robot_id": pd_id})
                observations = torch.tensor([[value] for value in values], dtype=torch.float32)
                payload = {"obs": observations, "act_mean": torch.zeros(len(observations), 2), "context": torch.zeros(1),
                           "obs_padding_mask": torch.tensor([False]), "act_padding_mask": torch.tensor([False, False]),
                           "adjacency_matrix": torch.zeros(1, 1)}
                with (root / f"{pd_id}.pkl").open("wb") as stream:
                    pickle.dump(payload, stream)
            manifest = root / "manifest.json"; manifest.write_text(json.dumps(entries))
            batches = list(FullCoverageShardBatches(root, manifest, batch_size=5, seed=1409, expected_rows_per_shard=None))
            self.assertEqual([len(batch["obs"]) for batch in batches], [5, 5, 3])
            self.assertEqual(sorted(int(value) for batch in batches for value in batch["obs"][:, 0]), list(range(13)))

    def test_converter_uses_authoritative_hd1_columns_not_first_17_per_limb(self):
        columns_path = ROOT / "tools" / "hd1_student_columns.json"
        columns, raw = load_frozen_hd1_columns(columns_path)
        self.assertEqual(len(columns), 204)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), FROZEN_HD1_COLUMNS_SHA256)
        teacher_obs = torch.arange(624, dtype=torch.float32).reshape(1, 624).numpy()
        converted = select_frozen_student_observation(teacher_obs, columns)
        self.assertTrue((converted[0] == teacher_obs[0, columns]).all())
        self.assertNotEqual(columns, [base + offset for base in range(0, 624, 52) for offset in range(17)])
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            write_frozen_student_columns(Path(temporary), raw)
            self.assertEqual((Path(temporary) / "student_columns.json").read_bytes(), raw)

    def test_lazy_task_batches_open_one_named_shard_per_yield(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            root = Path(temporary); entries = []
            for index in range(10):
                pd_id = f"floor-pd-{index}"
                entries.append({"pd_robot_id": pd_id})
                payload = {"obs": torch.full((8, 3), index, dtype=torch.float32), "act_mean": torch.zeros(8, 2),
                           "context": torch.zeros(3), "obs_padding_mask": torch.tensor([False]),
                           "act_padding_mask": torch.tensor([False, False]), "adjacency_matrix": torch.zeros(1, 1)}
                with (root / f"{pd_id}.pkl").open("wb") as stream:
                    pickle.dump(payload, stream)
            manifest = root / "manifest.json"; manifest.write_text(json.dumps(entries))
            batches = list(TaskBalancedShardBatches(root, manifest, batch_size=4, seed=1409))
            self.assertEqual(len(batches), 10)
            self.assertEqual({batch["pd_robot_id"] for batch in batches}, {entry["pd_robot_id"] for entry in entries})
            self.assertTrue(all(batch["obs"].shape == (4, 3) for batch in batches))

    def test_resume_conversion_reuses_only_source_hash_bound_shard(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            root = Path(temporary); expert = root / "expert"; expert.mkdir(); converted = root / "converted"; converted.mkdir()
            pd_id = "floor-pd-0"; source = expert / f"{pd_id}.npz"; source.write_bytes(b"already collected")
            columns = (ROOT / "tools" / "hd1_student_columns.json").read_bytes(); write_frozen_student_columns(converted, columns)
            payload = {"obs": torch.zeros(8000, 3), "manifest": {"pd_robot_id": pd_id,
                       "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}
            with (converted / f"{pd_id}.pkl").open("wb") as stream:
                pickle.dump(payload, stream)
            manifest = root / "manifest.json"; manifest.write_text(json.dumps([{"pd_robot_id": pd_id}]))
            report = convert_shards(expert, converted, manifest, ROOT / "tools" / "hd1_student_columns.json", resume=True, label="HD2B")
            self.assertEqual(report["HD2B_SHARDED_DATASET"], "PASS")
            source.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "invalid"):
                convert_shards(expert, converted, manifest, ROOT / "tools" / "hd1_student_columns.json", resume=True, label="HD2B")


if __name__ == "__main__":
    unittest.main()
