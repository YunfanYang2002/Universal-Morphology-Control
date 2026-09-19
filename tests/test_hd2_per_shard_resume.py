import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hd2_teacher_export import (  # noqa: E402
    _atomic_json,
    _atomic_npz,
    _validate_legacy_shard,
    _verify_sidecar,
    collection_resume_plan,
    teacher_coverage_summary,
)


def sidecar_row(pd_robot_id: str, shard: Path, *, shard_hash: str | None = None) -> dict:
    return {"pd_robot_id": pd_robot_id, "transition_count": 8000,
            "completed_episode_count": 1, "completed_episode_length_distribution": [7998],
            "final_partial_episode_steps": 2, "episodes_started": 2,
            "teacher_return_distribution": [0.0], "episode_length_distribution": [7998],
            "early_termination_count": 0, "shard_sha256": shard_hash or hashlib.sha256(shard.read_bytes()).hexdigest(),
            "coverage_metrics_available": True}


def write_legacy_npz(path: Path, transitions: int = 8000) -> None:
    max_limbs = 1
    np.savez_compressed(path,
        proprio_normalized=np.zeros((transitions, 624), dtype=np.float32),
        action_mean=np.zeros((transitions, 2), dtype=np.float32),
        action_canonical=np.zeros((transitions, 2), dtype=np.float32),
        obs_padding_mask=np.zeros((transitions, max_limbs), dtype=bool),
        act_padding_mask=np.zeros((transitions, 2 * max_limbs), dtype=bool),
        adjacency=np.zeros((transitions, max_limbs, max_limbs), dtype=bool),
        context_raw=np.zeros((transitions, max_limbs, 35), dtype=np.float32),
        rms_mean=np.zeros(624, dtype=np.float32), rms_var=np.ones(624, dtype=np.float32),
        rms_count=np.asarray(1.0, dtype=np.float64), max_limbs=np.asarray(max_limbs))


class Hd2PerShardResumeTests(unittest.TestCase):
    def test_legacy_595_shards_without_aggregate_need_only_405(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            output = Path(temporary); entries = [{"pd_robot_id": f"pd-{index:04d}"} for index in range(1000)]
            for entry in entries[:595]:
                np.savez_compressed(output / f"{entry['pd_robot_id']}.npz", fixture=np.asarray([1], dtype=np.int8))
            def valid_legacy(path, entry):
                return {"pd_robot_id": entry["pd_robot_id"], "transition_count": 8000,
                        "shard_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "resume_source": "LEGACY_SHARD_VALIDATED", "coverage_metrics_available": False}
            existing, needed, stats = collection_resume_plan(output, entries, legacy_validator=valid_legacy, log=lambda _: None)
            self.assertEqual(len(existing), 595)
            self.assertEqual(stats["legacy_recovered"], 595)
            self.assertEqual(len(needed), 405)
            recollected_existing = len(set(existing).intersection(entry["pd_robot_id"] for entry in needed))
            self.assertEqual(recollected_existing, 0)

    def test_legacy_schema_validation_and_bad_transition_count(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            output = Path(temporary); entry = {"pd_robot_id": "pd-0"}; shard = output / "pd-0.npz"
            write_legacy_npz(shard)
            row = _validate_legacy_shard(shard, entry)
            self.assertEqual(row["resume_source"], "LEGACY_SHARD_VALIDATED")
            self.assertFalse(row["coverage_metrics_available"])
            write_legacy_npz(shard, transitions=7999)
            existing, needed, _ = collection_resume_plan(output, [entry], log=lambda _: None)
            self.assertEqual(existing, {})
            self.assertEqual(needed, [entry])

    def test_sidecar_corruption_hash_mismatch_and_stale_partial_recollect_only_one(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            output = Path(temporary)
            valid, corrupt, mismatch, stale = ({"pd_robot_id": value} for value in ("valid", "corrupt", "mismatch", "stale"))
            valid_shard = output / "valid.npz"; write_legacy_npz(valid_shard)
            _atomic_json(output / "valid.metrics.json", sidecar_row("valid", valid_shard))
            corrupt_shard = output / "corrupt.npz"; corrupt_shard.write_bytes(b"not a zip")
            mismatch_shard = output / "mismatch.npz"; mismatch_shard.write_bytes(b"mismatch")
            _atomic_json(output / "mismatch.metrics.json", sidecar_row("mismatch", mismatch_shard, shard_hash="wrong"))
            (output / "stale.npz.partial").write_bytes(b"interrupted")
            existing, needed, _ = collection_resume_plan(output, [valid, corrupt, mismatch, stale], log=lambda _: None)
            self.assertEqual(set(existing), {"valid"})
            self.assertEqual({entry["pd_robot_id"] for entry in needed}, {"corrupt", "mismatch", "stale"})
            self.assertFalse((output / "stale.npz.partial").exists())

    def test_restart_skips_completed_sidecars_and_atomic_writes_leave_no_partial(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            output = Path(temporary); entries = [{"pd_robot_id": f"pd-{index}"} for index in range(5)]
            for entry in entries[:3]:
                shard = output / f"{entry['pd_robot_id']}.npz"; _atomic_npz(shard, {"value": np.asarray([1, 2, 3])})
                _atomic_json(output / f"{entry['pd_robot_id']}.metrics.json", sidecar_row(entry["pd_robot_id"], shard))
                self.assertFalse((output / f"{entry['pd_robot_id']}.npz.partial").exists())
                self.assertFalse((output / f"{entry['pd_robot_id']}.metrics.json.partial").exists())
                with np.load(shard, allow_pickle=False) as data:
                    self.assertTrue(np.array_equal(data["value"], [1, 2, 3]))
                self.assertEqual(_verify_sidecar(shard, entry)["resume_source"], "SIDECAR_VERIFIED")
            existing, needed, stats = collection_resume_plan(output, entries, log=lambda _: None)
            self.assertEqual(len(existing), 3)
            self.assertEqual(stats["sidecar_verified"], 3)
            self.assertEqual([entry["pd_robot_id"] for entry in needed], ["pd-3", "pd-4"])

    def test_coverage_summary_excludes_legacy_rows_without_fabrication(self):
        known = sidecar_row("known", Path("known.npz"), shard_hash="known")
        legacy = {"pd_robot_id": "legacy", "transition_count": 8000, "shard_sha256": "legacy",
                  "resume_source": "LEGACY_SHARD_VALIDATED", "coverage_metrics_available": False}
        summary = teacher_coverage_summary([known, legacy])
        self.assertEqual(summary["coverage_metrics_available_count"], 1)
        self.assertEqual(summary["coverage_metrics_missing_count"], 1)
        self.assertEqual(summary["coverage_metrics_missing_robot_ids"], ["legacy"])
        self.assertEqual(summary["episodes_started"]["p50"], 2.0)


if __name__ == "__main__":
    unittest.main()
