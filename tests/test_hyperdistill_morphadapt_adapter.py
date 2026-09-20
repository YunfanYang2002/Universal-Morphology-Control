"""Synthetic-only frozen HyperDistill adapter checks; no simulator or training."""
import tempfile
import unittest
from pathlib import Path

import torch

from tools.hd2_student_constructor import build_hd2_hnmlp_model
from tools.hd2c_protocol import OOD_WALKERS
from tools.hyperdistill_morphadapt_adapter import FrozenHyperDistillPolicy


class HyperDistillMorphAdaptAdapterTests(unittest.TestCase):
    def test_loads_frozen_state_generates_once_and_masks_padding(self):
        with tempfile.TemporaryDirectory(dir="tmp") as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint_030.pt"
            model = build_hd2_hnmlp_model()
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
            raw_context = torch.linspace(-0.1, 0.1, 12 * 35).reshape(12, 35)
            observation = {
                "proprioceptive": torch.zeros(1, 624),
                "obs_padding_mask": obs_mask, "act_padding_mask": act_mask,
            }
            policy.begin_walker(OOD_WALKERS[0], 1409)
            policy.bind_morphology(raw_context=raw_context, obs_mask=obs_mask[0], act_mask=act_mask[0])
            action = policy.action(observation)
            self.assertEqual(tuple(action.shape), (1, 24))
            self.assertTrue(torch.isfinite(action).all())
            self.assertTrue(torch.equal(action[act_mask], torch.zeros(16)))
            with self.assertRaisesRegex(RuntimeError, "already bound"):
                policy.bind_morphology(raw_context=raw_context, obs_mask=obs_mask[0], act_mask=act_mask[0])

    def test_static_context_binding_matches_context_input_path(self):
        with tempfile.TemporaryDirectory(dir="tmp") as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint_030.pt"
            reference = build_hd2_hnmlp_model()
            reference.eval()
            torch.save({
                "mu_net": reference.state_dict(), "optimizer": {}, "seed": 1409,
                "completed_epoch": 30, "cumulative_optimizer_steps": 30 * 1563,
                "cumulative_samples_seen": 30 * 8_000_000,
            }, checkpoint)
            policy = FrozenHyperDistillPolicy(
                checkpoint, Path("tools/hd1_student_columns.json"), torch.device("cpu")
            )
            obs_mask = torch.tensor([False] * 4 + [True] * 8)
            act_mask = torch.tensor([False] * 8 + [True] * 16)
            raw_context = torch.linspace(-0.1, 0.1, 12 * 35).reshape(12, 35)
            context = policy._context(raw_context, obs_mask, act_mask)
            reference.generate_params(context, obs_mask.reshape(1, -1))
            proprio = torch.linspace(-1.0, 1.0, 624).reshape(1, 624)
            selected = proprio.index_select(1, policy.columns)
            expected, _ = reference(selected, obs_mask.reshape(1, -1))
            expected = expected.clamp(-1.0, 1.0).masked_fill(act_mask.reshape(1, -1), 0.0)
            policy.begin_walker(OOD_WALKERS[0], 1409)
            policy.bind_morphology(raw_context=raw_context, obs_mask=obs_mask, act_mask=act_mask)
            actual = policy.action({
                "proprioceptive": proprio,
                "obs_padding_mask": obs_mask,
                "act_padding_mask": act_mask,
            })
            self.assertLessEqual(float((expected - actual).abs().max()), 1e-6)
            print("STATIC_CONTEXT_BINDING_EQUIVALENCE=PASS")


if __name__ == "__main__":
    unittest.main()
