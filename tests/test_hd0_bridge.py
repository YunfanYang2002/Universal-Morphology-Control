import json
import ast
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.convert_rmamorph_teacher_to_hyperdistill import (  # noqa: E402
    CONTEXT_RANGES,
    JOINT_RANGES,
    STUDENT_LABELS,
    convert,
    nominal_context,
)


class Hd0BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "tmp").mkdir(exist_ok=True)

    def make_source(self, root, *, rows_per_walker=2, invalid_action=False,
                    mutate_context=False, nonfinite_rms=False):
        labels = [f"unused:{i}" for i in range(18)] + STUDENT_LABELS
        limbs = 3
        episodes = 6
        rows = episodes * rows_per_walker
        obs = np.arange(rows * limbs * len(labels), dtype=np.float64).reshape(
            rows, limbs * len(labels)
        )
        action_mask = np.array([False, False, False, False, True, True])
        obs_mask = np.array([False, False, True])
        action = np.zeros((rows, limbs * 2), dtype=np.float64)
        action_mean = np.zeros_like(action)
        action[:, ~action_mask] = 0.25
        if invalid_action:
            action[0, 4] = 0.1
        context_ranges = CONTEXT_RANGES + JOINT_RANGES * 2
        context = np.concatenate(
            [((np.asarray(lo) + np.asarray(hi)) / 2) for lo, hi in context_ranges]
        )
        context_raw = np.repeat(context[None, None, :], episodes * limbs, axis=0).reshape(episodes, limbs, 35)
        if mutate_context:
            context_raw[1, 0, 0] += 0.01
        walker_index = np.repeat(np.arange(3), rows_per_walker * 2)
        episode_id = np.repeat(np.arange(episodes), rows_per_walker)
        np.savez(
            root / "arrays.npz",
            proprio_normalized=obs,
            action_mean=action_mean,
            action_canonical=action,
            rms_mean=np.arange(obs.shape[1], dtype=np.float64),
            rms_var=np.ones(obs.shape[1], dtype=np.float64),
            rms_count=np.array(123.0),
            walker_index=walker_index,
            episode_id=episode_id,
            obs_padding_mask=np.repeat(obs_mask[None, :], rows, axis=0),
            act_padding_mask=np.repeat(action_mask[None, :], rows, axis=0),
            adjacency=np.repeat(np.eye(limbs)[None, :, :], rows, axis=0),
            context_raw=context_raw,
        )
        manifest = {
            "context_format": "raw_v1",
            "per_limb_obs_labels": labels,
            "max_limbs": limbs,
            "walkers": ["w0", "w1", "w2"],
        }
        (root / "manifest.json").write_text(json.dumps(manifest))

    def test_range_tables_match_official_agent_source(self):
        source = ROOT / "metamorph" / "envs" / "modules" / "agent.py"
        tree = ast.parse(source.read_text())
        init = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
            and any(isinstance(parent, ast.ClassDef) and parent.name == "Agent"
                    for parent in ast.walk(tree) if node in getattr(parent, "body", []))
        )
        tables = {}
        for node in init.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Attribute) or target.attr not in {
                "limb_context_range", "joint_context_range"
            }:
                continue
            tables[target.attr] = {
                ast.literal_eval(key): [
                    ast.literal_eval(value.args[0])[0], ast.literal_eval(value.args[0])[1]
                ]
                for key, value in zip(node.value.keys, node.value.values)
            }
        self.assertEqual(
            [tuple(map(list, pair)) for pair in tables["limb_context_range"].values()],
            CONTEXT_RANGES,
        )
        self.assertEqual(
            [tuple(map(list, pair)) for pair in tables["joint_context_range"].values()],
            JOINT_RANGES,
        )

    def test_export_selects_columns_rms_and_reloads(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as td:
            source, output = Path(td) / "source", Path(td) / "output"
            source.mkdir()
            self.make_source(source)
            convert(source, output)
            columns = json.loads((output / "student_columns.json").read_text())
            self.assertEqual(columns, [limb * 35 + 18 + i for limb in range(3) for i in range(17)])
            with np.load(output / "selected_obs_rms.npz") as rms:
                self.assertTrue(np.array_equal(rms["mean"], np.asarray(columns, dtype=float)))
            with (output / "w0.pkl").open("rb") as stream:
                payload = pickle.load(stream)
            self.assertEqual(tuple(payload["obs"].shape), (4, 51))
            with np.load(source / "arrays.npz") as data:
                self.assertTrue(np.array_equal(payload["obs"].numpy(), data["proprio_normalized"][:4, columns]))
            self.assertEqual(payload["manifest"]["normalization"], "teacher_obs_rms_then_selected")

    def test_context_scaling_matches_official_ranges_and_zeros_padded_slots(self):
        ranges = CONTEXT_RANGES + JOINT_RANGES * 2
        raw = np.repeat(np.concatenate([np.asarray(lo) for lo, _ in ranges])[None, :], 3, axis=0)
        result = nominal_context(raw, np.array([False, False, True]), np.array([False, False, False, False, True, True]))
        expected = np.concatenate([
            -(np.asarray(lo) != np.asarray(hi)).astype(float) for lo, hi in ranges
        ])
        expected = expected.astype(np.float32)
        self.assertTrue(np.array_equal(result.reshape(3, 35)[0, :17], expected[:17]))
        self.assertTrue(np.array_equal(result.reshape(3, 35)[2, 17:], np.zeros(18, dtype=np.float32)))

        # A live limb's valid joint slot must survive beside its invalid slot.
        raw[1] = np.concatenate([
            np.asarray(lo) + 0.25 * (np.asarray(hi) - np.asarray(lo))
            for lo, hi in ranges
        ])
        result = nominal_context(raw, np.array([False, False, True]), np.array([False, False, False, True, True, True]))
        scaled = 2 * (raw[1] - np.concatenate([np.asarray(lo) for lo, _ in ranges])) / (
            np.concatenate([np.asarray(hi) for _, hi in ranges])
            - np.concatenate([np.asarray(lo) for lo, _ in ranges]) + 1e-8
        ) - (np.concatenate([np.asarray(lo) for lo, _ in ranges]) != np.concatenate([np.asarray(hi) for _, hi in ranges])).astype(float)
        row = result.reshape(3, 35)[1]
        self.assertTrue(np.allclose(row[17:26], scaled[17:26].astype(np.float32)))
        self.assertTrue(np.array_equal(row[26:35], np.zeros(9, dtype=np.float32)))

    def test_rejects_invalid_action_and_context_mutation(self):
        for invalid_action, mutate_context in ((True, False), (False, True)):
            with self.subTest(invalid_action=invalid_action, mutate_context=mutate_context):
                with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as td:
                    source, output = Path(td) / "source", Path(td) / "output"
                    source.mkdir()
                    self.make_source(source, invalid_action=invalid_action, mutate_context=mutate_context)
                    with self.assertRaises(AssertionError):
                        convert(source, output)


if __name__ == "__main__":
    unittest.main()
