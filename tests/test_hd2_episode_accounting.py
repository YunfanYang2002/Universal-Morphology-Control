import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hd2_collection_accounting_audit import corrected_row, write_corrected_audit  # noqa: E402
from tools.hd2_teacher_export import _verified_existing_collection, teacher_coverage_summary  # noqa: E402


class Hd2EpisodeAccountingTests(unittest.TestCase):
    def test_completed_plus_final_partial_is_exact_8000(self):
        examples = (([7125], 875), ([7872], 128), ([7998], 2))
        for lengths, expected_partial in examples:
            row = corrected_row({"pd_robot_id": "vt-pd", "transition_count": 8000,
                                 "episode_count_required_to_reach_8000": len(lengths),
                                 "episode_length_distribution": lengths})
            self.assertEqual(row["final_partial_episode_steps"], expected_partial)
            self.assertEqual(row["episodes_started"], len(lengths) + 1)
            self.assertEqual(sum(row["completed_episode_length_distribution"]) + row["final_partial_episode_steps"], 8000)
            self.assertIn("DEPRECATED", row["episode_count_required_to_reach_8000_semantics"])

    def test_historical_audit_writes_corrected_summary_without_mutating_shard(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            run = Path(temporary); expert = run / "expert_shards"; expert.mkdir()
            shard = expert / "vt-pd.npz"; shard.write_bytes(b"immutable expert transitions")
            source_hash = hashlib.sha256(shard.read_bytes()).hexdigest()
            metrics = {"per_robot": [{"pd_robot_id": "vt-pd", "transition_count": 8000, "shard_sha256": source_hash,
                                        "episode_count_required_to_reach_8000": 1, "episode_length_distribution": [7125],
                                        "teacher_return_distribution": [1.0], "early_termination_count": 0}]}
            source = expert / "collection_metrics.json"; source.write_text(json.dumps(metrics))
            before = shard.read_bytes(); result = write_corrected_audit(run)
            self.assertEqual(shard.read_bytes(), before)
            self.assertEqual(result["HD2_EPISODE_ACCOUNTING_REPAIR"], "PASS")
            corrected = json.loads((expert / "collection_accounting_corrected.json").read_text())
            self.assertEqual(corrected["per_robot"][0]["final_partial_episode_steps"], 875)
            coverage = teacher_coverage_summary(corrected["per_robot"])
            self.assertEqual(coverage["episodes_started"]["p50"], 2.0)

    def test_resume_collection_accepts_only_complete_hash_bound_robots(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            output = Path(temporary); pd_id = "floor-pd-0"; shard = output / f"{pd_id}.npz"; shard.write_bytes(b"complete")
            digest = hashlib.sha256(shard.read_bytes()).hexdigest()
            row = {"pd_robot_id": pd_id, "transition_count": 8000, "shard_sha256": digest,
                   "episode_length_distribution": [7998], "teacher_return_distribution": [0.0], "early_termination_count": 0}
            (output / "collection_metrics.json").write_text(json.dumps({"per_robot": [row]}))
            verified = _verified_existing_collection(output, [{"pd_robot_id": pd_id}])
            self.assertEqual(verified[pd_id]["final_partial_episode_steps"], 2)
            shard.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash-mismatched"):
                _verified_existing_collection(output, [{"pd_robot_id": pd_id}])


if __name__ == "__main__":
    unittest.main()
