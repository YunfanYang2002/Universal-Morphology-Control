"""Frozen HD2C evaluation-only protocol helpers; no training or simulator imports."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median
import xml.etree.ElementTree as ET

import yaml

from tools.hd1_protocol import PROTOCOL as HD1_PROTOCOL, evaluate_gates, family, sha256


EPOCHS = (30, 60, 90, 120, 150)
OOD_WALKERS = (
    "floor-1409-0-14-01-12-42-39", "floor-1409-1-13-01-12-18-30",
    "mvt-5506-0-13-17-12-26-41", "mvt-5506-1-15-17-07-32-47",
    "vt-1409-1-6-02-22-04-53", "vt-1409-10-12-02-22-35-53",
)
OOD6 = OOD_WALKERS
PROTOCOL = {
    "name": "HyperDistill (MetaMorph-DR teacher)", "stage": "HD2C evaluation-only",
    "seed": 1409, "epochs": list(EPOCHS), "episodes": 3, "horizon": 1000,
    "mapper": "HD1 frozen mapper", "mapper_sha256": "5aae11e48a05f57bca706085939d3ed47985f8c80e2e37299766e8d5f309917d",
    "dynamics": "nominal", "reset_multipliers": {"motor_strength": 1.0, "friction": 1.0, "mass": 1.0},
    "MID_EPISODE_ENABLED": False, "training": "NOT_RUN", "OOD_TRAINING_SAMPLES": 0,
    "gate_thresholds": {key: HD1_PROTOCOL[key] for key in (
        "locomotion_min_median_length", "locomotion_min_median_forward_displacement_m",
        "locomotion_min_walkers", "ratio_teacher_epsilon", "min_median_valid_ratio")},
    "ratio_definition": HD1_PROTOCOL["ratio_definition"],
    "ood_walkers": list(OOD_WALKERS), "diagnostic_epoch_checkpoints": list(EPOCHS[:-1]),
    "formal_epoch": 150,
}


def _read_ids(path: Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _asset_row(walker: str, test_dir: Path, ood: set[str]) -> dict:
    xml = test_dir / "xml" / f"{walker}.xml"
    metadata = test_dir / "metadata" / f"{walker}.json"
    if not xml.is_file() or not metadata.is_file():
        raise FileNotFoundError(f"HD2C OOD asset missing: {xml} or {metadata}")
    ET.parse(xml)
    json.loads(metadata.read_text())
    return {"walker_id": walker, "family": family(walker),
            "xml_path": str(xml.resolve()), "metadata_path": str(metadata.resolve()),
            "xml_sha256": sha256(xml), "metadata_sha256": sha256(metadata),
            "in_frozen_ood": walker in ood, "in_teacher_training": False,
            "provenance": "frozen HD2C OOD evaluation-only pool"}


def prepare_selection(config, ood_pool, test_dir, pd_manifest, output):
    """Validate the frozen PD1000 inventory and materialize only six OOD rows."""
    config, ood_pool, test_dir, pd_manifest, output = map(Path, (config, ood_pool, test_dir, pd_manifest, output))
    train = yaml.safe_load(config.read_text())["ENV"]["WALKERS"]
    ood = _read_ids(ood_pool)
    assert len(train) == 100 and len(set(train)) == 100, "HD2C requires exactly 100 unique teacher parents"
    assert len(set(ood)) == len(ood), "Duplicate OOD IDs"
    selected_ood = tuple(walker for group in ("floor", "mvt", "vt")
                         for walker in [w for w in ood if family(w) == group][:2])
    assert selected_ood == OOD_WALKERS, "HD2C first-two-per-family OOD selection differs from frozen six"
    assert not set(train).intersection(ood), "HD2C teacher parents overlap OOD"
    entries = json.loads(pd_manifest.read_text())
    assert isinstance(entries, list), "PD manifest must be a list"
    assert len(entries) == 1000 and len({r["pd_robot_id"] for r in entries}) == 1000
    parents = {r["parent_walker_id"] for r in entries}
    assert len(parents) == 100 and parents == set(train), "PD parent inventory differs from teacher training"
    assert not parents.intersection(ood), "HD2C OOD overlaps PD parents"
    assert not {r["pd_robot_id"] for r in entries}.intersection(ood), "HD2C PD IDs overlap OOD IDs"
    output.mkdir(parents=True, exist_ok=True)
    rows = [_asset_row(walker, test_dir, set(OOD_WALKERS)) for walker in OOD_WALKERS]
    selection = {
        "protocol": PROTOCOL,
        "sources": {"teacher_config": str(config.resolve()), "teacher_config_sha256": sha256(config),
                    "ood_pool": str(ood_pool.resolve()), "ood_pool_sha256": sha256(ood_pool),
                    "pd_manifest": str(pd_manifest.resolve()), "pd_manifest_sha256": sha256(pd_manifest),
                    "teacher_training_parent_count": 100, "pd_robot_count": 1000},
        "train": [], "train_smoke": [], "ood_smoke": rows,
        "HD2C_PD_INVENTORY": "PASS", "HD2C_OOD_SELECTION": "PASS",
        "HD2C_TEACHER_PARENT_MATCH": "PASS", "HD2C_TRAINING": "NOT_RUN",
        "OOD6_COUNT": 6, "OOD6_TRAIN_PARENT_OVERLAP": 0, "OOD6_PD1000_EXACT_ID_OVERLAP": 0,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2))
    (output / "hd2c_ood_smoke_walkers.txt").write_text("\n".join(OOD_WALKERS) + "\n")
    return selection


def summarize(evaluations: dict[int, dict]):
    """Summarize supplied rollout metrics; this function never trains or reruns evaluation."""
    assert set(evaluations) == set(EPOCHS), "HD2C requires all five checkpoints"
    curve = []
    formal = None
    expected = list(OOD_WALKERS)
    for epoch in EPOCHS:
        rollout = evaluations[epoch]
        rows = rollout["per_walker"]
        assert [row["walker_id"] for row in rows] == expected, "HD2C OOD order differs from frozen tuple"
        gates = evaluate_gates(rows)
        enriched = []
        for row in gates["per_walker"]:
            enriched.append({**row, "median_student_episode_length": median(row["student_episode_lengths"]),
                             "median_student_forward_displacement": median(row["student_forward_displacements"])})
        curve.append({"epoch": epoch, "mean_teacher_return": gates["OOD_MEAN_TEACHER_RETURN"],
                      "mean_student_return": gates["OOD_MEAN_STUDENT_RETURN"],
                      "median_student_teacher_ratio": gates["OOD_MEDIAN_STUDENT_TEACHER_RATIO"],
                      "median_closed_loop_action_mse": median([row["action_mse_valid"] for row in rows]),
                      "numerical_failures": gates["OOD_NUMERICAL_FAILURES"],
                      "locomotion_walkers": gates["OOD_LOCOMOTION_WALKERS"],
                      "runtime_finite": gates["HD1_RUNTIME_FINITE"], "per_walker": enriched})
        if epoch == PROTOCOL["formal_epoch"]:
            formal = {key.replace("HD1_", "HD2_", 1) if key.startswith("HD1_") else key: value
                      for key, value in gates.items()}
            formal["per_walker"] = enriched
    formal["formal_epoch"] = PROTOCOL["formal_epoch"]
    hd2_ratio = formal["OOD_MEDIAN_STUDENT_TEACHER_RATIO"]
    comparison = {
        "HD1": {"morphologies": 18, "approx_transitions": 52000, "epochs": 50,
                "approx_median_ratio": 0.105, "HD1_FINAL": "FAIL", "source": "user-supplied prior result"},
        "HD2_epoch150": {"pd_robots": 1000, "transitions": 8000000, "epochs": 150,
                         "median_ratio": hd2_ratio, "HD2_FINAL": formal["HD2_FINAL"]},
        "median_ratio_difference_from_approx_HD1": hd2_ratio - 0.105 if hd2_ratio is not None else None,
        "simultaneously_changed": ["morphology count", "sample count", "epoch count", "batch/loss fidelity"],
        "QUALIFIED_COMPARISON": "Descriptive scale comparison only; no causal or statistical superiority claim.",
        "STATISTICAL_SIGNIFICANCE": "NOT_MEASURED",
        "TEACHER_WEAKNESS_CAVEAT": "User-reported HD2B audit of 405 PD robots with complete coverage found a nontrivial fraction with weak teacher control; no numerical fraction is inferred.",
        "conclusion": ("HyperDistill (MetaMorph-DR teacher), under our paper-scale distillation setup, "
                       + ("passed" if formal["HD2_FINAL"] == "PASS" else "did not pass")
                       + " the frozen unseen-morphology gate."),
    }
    return curve, formal, comparison
