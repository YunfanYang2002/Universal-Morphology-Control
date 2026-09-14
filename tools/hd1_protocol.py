"""Frozen reduced HD1 selection and scientific gates (no simulator imports)."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import yaml


FAMILIES = ("floor", "mvt", "vt")
PROTOCOL = {
    "name": "HyperDistill (MetaMorph-DR teacher)", "stage": "HD1",
    "seed": 1409, "epochs": 50, "batch_size": 64, "episodes": 3, "horizon": 1000,
    "train_episodes": [0, 1], "validation_episodes": [2],
    "dynamics": "nominal", "mutation": False,
    "locomotion_min_median_length": 500,
    "locomotion_min_median_forward_displacement_m": 1.0,
    "locomotion_min_walkers": 4,
    "ratio_teacher_epsilon": 1e-6, "min_median_valid_ratio": 0.50,
    "ratio_definition": "mean(student episode returns) / mean(teacher episode returns) only if teacher mean > 1e-6; otherwise undefined",
    "displacement_definition": "native terminal info[x_pos] minus reset sim.data.qpos[0], before vector auto-reset, in metres",
    "selection_rule": "first six per family in frozen teacher train config; first two per family in frozen OOD list; first two selected train walkers per family for train-smoke",
    "hd0_status": "PASS (user-supplied real-server result; not rerun)",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family(walker):
    value = walker.split("-", 1)[0]
    if value not in FAMILIES:
        raise ValueError(f"Unrecognized frozen walker family: {walker}")
    return value


def prepare_selection(config, ood_pool, train_dir, test_dir, output):
    config, ood_pool, output = Path(config), Path(ood_pool), Path(output)
    train = yaml.safe_load(config.read_text())["ENV"]["WALKERS"]
    ood = [line.strip() for line in ood_pool.read_text().splitlines() if line.strip() and not line.startswith("#")]
    assert len(set(train)) == len(train) and len(set(ood)) == len(ood), "Duplicate frozen pool IDs"
    assert not set(train).intersection(ood), "Frozen OOD pool overlaps teacher training"
    for walker in train + ood:
        family(walker)
    chosen = [walker for group in FAMILIES for walker in [w for w in train if family(w) == group][:6]]
    seen = [walker for group in FAMILIES for walker in [w for w in chosen if family(w) == group][:2]]
    unseen = [walker for group in FAMILIES for walker in [w for w in ood if family(w) == group][:2]]
    assert len(chosen) == 18 and len(seen) == len(unseen) == 6
    output.mkdir(parents=True, exist_ok=True)
    selection = {"protocol": PROTOCOL, "sources": {
        "teacher_config": str(config.resolve()), "teacher_config_sha256": sha256(config),
        "ood_pool": str(ood_pool.resolve()), "ood_pool_sha256": sha256(ood_pool),
        "teacher_training_count": len(train), "ood_pool_count": len(ood),
    }}
    for name, walkers, directory, provenance in (
        ("train", chosen, Path(train_dir), "frozen teacher training config"),
        ("train_smoke", seen, Path(train_dir), "subset of HD1 train18"),
        ("ood_smoke", unseen, Path(test_dir), "frozen walker-disjoint OOD pool; nominal smoke only"),
    ):
        rows = []
        for walker in walkers:
            xml = directory / "xml" / f"{walker}.xml"
            metadata = directory / "metadata" / f"{walker}.json"
            if not xml.is_file() or not metadata.is_file():
                raise FileNotFoundError(f"Frozen HD1 selection asset missing: {xml} or {metadata}; no walker substitution")
            # Check parseability without constructing a new environment.
            import xml.etree.ElementTree as ET
            ET.parse(xml)
            json.loads(metadata.read_text())
            rows.append({"walker_id": walker, "family": family(walker),
                         "xml_path": str(xml.resolve()), "metadata_path": str(metadata.resolve()),
                         "xml_sha256": sha256(xml), "metadata_sha256": sha256(metadata),
                         "provenance": provenance, "in_teacher_training": walker in train,
                         "in_frozen_ood": walker in ood})
        selection[name] = rows
        (output / f"hd1_{name}_walkers.txt").write_text("\n".join(walkers) + "\n")
        (output / f"hd1_{name}_provenance.json").write_text(json.dumps(rows, indent=2))
    selection.update({"HD1_TRAIN_WALKER_COUNT": 18, "HD1_TRAIN_FAMILY_BALANCE": "PASS",
                      "HD1_TRAIN_TEST_OVERLAP": 0, "family_counts": {f: 6 for f in FAMILIES}})
    (output / "selection.json").write_text(json.dumps(selection, indent=2))
    (output / "frozen_protocol.json").write_text(json.dumps(PROTOCOL, indent=2))
    return selection


def evaluate_gates(rows):
    """Consume exactly six OOD walker records; undefined ratios are explicit."""
    assert len(rows) == 6 and len({r["walker_id"] for r in rows}) == 6, "OOD smoke must contain six unique walkers"
    assert {f: sum(family(r["walker_id"]) == f for r in rows) for f in FAMILIES} == {f: 2 for f in FAMILIES}
    results, ratios, teacher_means, student_means = [], [], [], []
    runtime_ok = True
    failures = 0
    for row in rows:
        tr, sr = row["teacher_episode_returns"], row["student_episode_returns"]
        tl, sl = row["teacher_episode_lengths"], row["student_episode_lengths"]
        displacement = row["student_forward_displacements"]
        assert all(len(values) == 3 for values in (tr, sr, tl, sl, displacement)), "Exactly three complete episodes required"
        finite = all(math.isfinite(float(v)) for values in (tr, sr, tl, sl, displacement) for v in values)
        finite = finite and math.isfinite(float(row["action_mse_valid"]))
        numerical_failures = row["numerical_failures"]
        assert isinstance(numerical_failures, int) and numerical_failures >= 0
        failures += numerical_failures
        runtime_ok &= finite and numerical_failures == 0
        teacher_return = mean(tr) if finite else None
        student_return = mean(sr) if finite else None
        ratio = student_return / teacher_return if finite and teacher_return > PROTOCOL["ratio_teacher_epsilon"] else None
        locomotion = (finite and numerical_failures == 0
                      and median(sl) >= PROTOCOL["locomotion_min_median_length"]
                      and median(displacement) >= PROTOCOL["locomotion_min_median_forward_displacement_m"])
        results.append({**row, "teacher_mean_return": teacher_return, "student_mean_return": student_return,
                        "normalized_return": ratio, "normalized_return_defined": ratio is not None,
                        "substantive_locomotion": bool(locomotion)})
        if ratio is not None:
            ratios.append(ratio)
        if finite:
            teacher_means.append(teacher_return)
            student_means.append(student_return)
    median_ratio = median(ratios) if ratios else None
    control = sum(row["substantive_locomotion"] for row in results) >= PROTOCOL["locomotion_min_walkers"]
    normalized = median_ratio is not None and median_ratio >= PROTOCOL["min_median_valid_ratio"]
    return {"per_walker": results,
            "HD1_RUNTIME_FINITE": "PASS" if runtime_ok else "FAIL",
            "HD1_CROSS_MORPH_CONTROL": "PASS" if control else "FAIL",
            "HD1_NORMALIZED_RETURN_GATE": "PASS" if normalized else "FAIL",
            "HD1_FINAL": "PASS" if runtime_ok and control and normalized else "FAIL",
            "OOD_VALID_WALKERS": len(ratios), "OOD_MEDIAN_STUDENT_TEACHER_RATIO": median_ratio,
            "OOD_MEAN_STUDENT_RETURN": mean(student_means) if len(student_means) == 6 else None,
            "OOD_MEAN_TEACHER_RETURN": mean(teacher_means) if len(teacher_means) == 6 else None,
            "OOD_NUMERICAL_FAILURES": failures,
            "OOD_LOCOMOTION_WALKERS": sum(row["substantive_locomotion"] for row in results)}
