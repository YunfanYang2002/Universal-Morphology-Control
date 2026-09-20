"""Evaluator API compatibility regression; no training or real simulator run."""
import tempfile
import unittest
from pathlib import Path

import torch

from tools.hd2_student_constructor import build_hd2_hnmlp_model
from tools.hyperdistill_morphadapt_adapter import FrozenHyperDistillPolicy
from tools.hyperdistill_morphadapt_evaluator import (
    _build_formal_args,
    _evaluator_args_contract,
    _load_rm_evaluator,
)


ROOT = Path(__file__).resolve().parents[1]
RMAMORPH_ROOT = Path(r"D:\CODES\master\rmamorph")


class HyperDistillEvaluatorContractTests(unittest.TestCase):
    @unittest.skipUnless(
        (RMAMORPH_ROOT / "tools" / "evaluate_dynamics.py").is_file(),
        "local rmamorph checkout is unavailable",
    )
    def test_full_args_namespace_reaches_one_walker_smoke(self):
        evaluator = _load_rm_evaluator(RMAMORPH_ROOT)
        evaluator.canonical_runtime_config_from_resolved(
            RMAMORPH_ROOT / "configs" / "dynamics_teacher_smoke.yaml", []
        )
        args = _build_formal_args(evaluator)
        expected = {
            "trace_out", "trace_identity", "replay_out",
            "action_history_intervention", "post_response_intervention",
            "episodes_per_walker", "max_steps_per_walker",
        }
        self.assertTrue(expected.issubset(set(_evaluator_args_contract(evaluator))))
        self.assertEqual(vars(args)["replay_out"], None)
        self.assertEqual(vars(args)["action_history_intervention"], "aligned")
        self.assertFalse(vars(args)["post_response_intervention"])
        self.assertEqual(vars(args)["episodes_per_walker"], 1)
        self.assertEqual(vars(args)["max_steps_per_walker"], 1000)

        model = build_hd2_hnmlp_model()
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            checkpoint = Path(directory) / "checkpoint_030.pt"
            torch.save({
                "mu_net": model.state_dict(), "optimizer": {}, "seed": 1409,
                "completed_epoch": 30, "cumulative_optimizer_steps": 30 * 1563,
                "cumulative_samples_seen": 30 * 8_000_000,
            }, checkpoint)
            policy = FrozenHyperDistillPolicy(
                checkpoint, ROOT / "tools" / "hd1_student_columns.json", torch.device("cpu")
            )

        observation = {
            "proprioceptive": torch.zeros(1, 624),
            "context": torch.zeros(1, 420),
            "obs_padding_mask": torch.tensor([[False] * 4 + [True] * 8]),
            "act_padding_mask": torch.tensor([[False] * 8 + [True] * 16]),
        }
        original_evaluate_walker = evaluator.evaluate_walker

        def one_step_walker(model, ob_rms, walker, seed, episodes, max_steps, *call_args):
            self.assertEqual((walker, seed, episodes, max_steps), ("synthetic", 1409, 1, 1000))
            action = policy.action(observation)
            self.assertTrue(torch.isfinite(action).all())
            self.assertTrue(torch.equal(action[observation["act_padding_mask"]], torch.zeros(16)))
            return [0.0], [1], [{"motor_strength": 1.0, "friction": 1.0, "mass": 1.0}], [], [], []

        evaluator.evaluate_walker = one_step_walker
        try:
            result = evaluator.evaluate_protocol(policy, None, "nominal", ["synthetic"], [1409], args)
        finally:
            evaluator.evaluate_walker = original_evaluate_walker

        self.assertEqual(result["aggregate"]["length"]["mean"], 1.0)
        print("FORMAL_ARGS_CONTRACT=PASS")
        print("MISSING_EVALUATOR_ARGS=0")
        print("ONE_WALKER_ROLLOUT=PASS")
        print("POLICY_OUTPUT_FINITE=PASS")
        print("PADDED_ACTION_ZERO=PASS")


if __name__ == "__main__":
    unittest.main()
