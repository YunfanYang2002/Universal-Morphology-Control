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
    TaskBalancedShardBatches,
    load_frozen_hd1_columns,
    select_frozen_student_observation,
    write_frozen_student_columns,
)


class Hd2ShardTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
