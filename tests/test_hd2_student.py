import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hd2_student import TaskBalancedShardBatches  # noqa: E402


class Hd2ShardTests(unittest.TestCase):
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
