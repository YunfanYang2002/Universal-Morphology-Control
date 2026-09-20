"""Synthetic contracts for the server-only HyperDistill mutation launcher."""
import tempfile
import unittest
from pathlib import Path

from tools.run_hyperdistill_strict_ood97_mutation import (
    HORIZON,
    MUTATION_STEP,
    PROTOCOLS,
    ROOT,
    _checkpoint_preflight,
    _under_tmp,
)


class _CountingFrozenPolicy:
    def __init__(self):
        self.bind_count = 0
        self.generation_count = 0

    def bind(self):
        if self.bind_count:
            raise RuntimeError("second bind")
        self.bind_count += 1
        self.generation_count += 1

    def action(self, _step):
        return "frozen"


class HyperDistillMutationRunnerTests(unittest.TestCase):
    def test_frozen_protocol_and_output_contract(self):
        self.assertEqual(PROTOCOLS, ("nominal", "id", "ood_mild", "ood_strong"))
        self.assertEqual(MUTATION_STEP, 250)
        self.assertEqual(HORIZON, 1000)
        self.assertTrue(_under_tmp(ROOT / "tmp" / "mutation_contract"))
        with self.assertRaises(ValueError):
            _under_tmp(ROOT.parent / "mutation_contract")

    def test_server_checkpoint_preflight_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            with self.assertRaises(FileNotFoundError):
                _checkpoint_preflight(Path(directory) / "checkpoint_150.pt")

    def test_bind_once_across_runtime_mutation(self):
        policy = _CountingFrozenPolicy()
        policy.bind()
        for step in range(HORIZON):
            if step == MUTATION_STEP:
                self.assertEqual(policy.bind_count, 1)
            self.assertEqual(policy.action(step), "frozen")
        self.assertEqual(policy.bind_count, 1)
        self.assertEqual(policy.generation_count, 1)
        print("HN_BIND_COUNT=1")
        print("HN_REGEN_AFTER_MUTATION=0")


if __name__ == "__main__":
    unittest.main()
