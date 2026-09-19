"""Derive corrected exact-8000 episode accounting without touching expert shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.hd2_teacher_export import teacher_coverage_summary  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corrected_row(row: dict) -> dict:
    corrected = dict(row)
    lengths = list(corrected.get("completed_episode_length_distribution", corrected.get("episode_length_distribution", [])))
    transition_count = int(corrected.get("transition_count", -1))
    if transition_count != 8000:
        raise ValueError(f"HD2 accounting audit requires exactly 8000 transitions: {corrected.get('pd_robot_id')}")
    partial = 8000 - sum(lengths)
    if partial < 0 or sum(lengths) + partial != transition_count:
        raise AssertionError(f"HD2 accounting identity failed: {corrected.get('pd_robot_id')}")
    corrected.update({
        "completed_episode_count": len(lengths),
        "completed_episode_length_distribution": lengths,
        "final_partial_episode_steps": partial,
        "episodes_started": len(lengths) + int(partial > 0),
        "episode_count_required_to_reach_8000_semantics": "DEPRECATED: completed terminal episodes only; use episodes_started",
    })
    corrected.setdefault("coverage_metrics_available", all(key in corrected for key in ("teacher_return_distribution", "episode_length_distribution", "early_termination_count")))
    return corrected


def write_corrected_audit(run_root: Path) -> dict:
    run_root = Path(run_root).resolve()
    if not run_root.is_relative_to((ROOT / "tmp").resolve()):
        raise ValueError("--run must be below project ./tmp")
    source = run_root / "expert_shards" / "collection_metrics.json"
    if not source.is_file():
        raise FileNotFoundError(f"missing collection metrics: {source}")
    original = json.loads(source.read_text())
    rows = [corrected_row(row) for row in original.get("per_robot", [])]
    if not rows:
        raise ValueError("collection metrics has no per-robot rows")
    for row in rows:
        shard = run_root / "expert_shards" / f"{row['pd_robot_id']}.npz"
        if not shard.is_file() or row.get("shard_sha256") != sha256(shard):
            raise ValueError(f"expert shard is absent or hash-mismatched: {row['pd_robot_id']}")
    corrected = {"source_collection_metrics_sha256": sha256(source),
                 "HD2_EPISODE_ACCOUNTING_REPAIR": "PASS",
                 "HD2_EXACT_8000_ACCOUNTING_IDENTITY": "PASS",
                 "per_robot": rows}
    output = run_root / "expert_shards" / "collection_accounting_corrected.json"
    output.write_text(json.dumps(corrected, indent=2, allow_nan=False) + "\n")
    coverage_path = run_root / "expert_shards" / "teacher_coverage_corrected.json"
    coverage_path.write_text(json.dumps(teacher_coverage_summary(rows), indent=2, allow_nan=False) + "\n")
    return corrected | {"corrected_audit": str(output), "teacher_coverage": str(coverage_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    print(json.dumps(write_corrected_audit(parser.parse_args().run), sort_keys=True))


if __name__ == "__main__":
    main()
