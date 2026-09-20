"""Synthetic-only frozen HyperDistill adapter checks; no simulator or training."""
import tempfile
import unittest
from pathlib import Path

import torch

from tools.hyperdistill_morphadapt_adapter import FrozenHyperDistillPolicy, _HNMLP
from metamorph.config import cfg


class HyperDistillMorphAdaptAdapterTests(unittest.TestCase):
    def test_loads_frozen_state_generates_once_and_masks_padding(self):
        with tempfile.TemporaryDirectory(dir="tmp") as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint_030.pt"
            cfg.MODEL.MLP.LAYER_NUM = 2
            cfg.MODEL.MLP.DROPOUT = None
            cfg.MODEL.HYPERNET.EMBEDDING_DROPOUT = 0.1
            model = _HNMLP()
            torch.save({
                "mu_net": model.state_dict(), "optimizer": {}, "seed": 1409,
                "completed_epoch": 30, "cumulative_optimizer_steps": 30 * 1563,
                "cumulative_samples_seen": 30 * 8_000_000,
            }, checkpoint)
            policy = FrozenHyperDistillPolicy(
                checkpoint, Path("tools/hd1_student_columns.json"), torch.device("cpu")
            )
            obs_mask = torch.tensor([[False] * 4 + [True] * 8])
            act_mask = torch.tensor([[False] * 8 + [True] * 16])
            observation = {
                "proprioceptive": torch.zeros(1, 624), "context": torch.zeros(1, 420),
                "obs_padding_mask": obs_mask, "act_padding_mask": act_mask,
            }
            policy.begin_walker("synthetic", 1409)
            action = policy.action(observation)
            self.assertEqual(tuple(action.shape), (1, 24))
            self.assertTrue(torch.isfinite(action).all())
            self.assertTrue(torch.equal(action[act_mask], torch.zeros(16)))
            observation["context"][0, 0] = 1.0
            with self.assertRaisesRegex(RuntimeError, "context changed"):
                policy.action(observation)


if __name__ == "__main__":
    unittest.main()
