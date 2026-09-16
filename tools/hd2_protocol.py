"""Frozen HD2 fidelity inventory: reuse the official static train_mutate_1000 corpus."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import yaml


FAMILIES = ("floor", "mvt", "vt")
PD_PER_PARENT = 10
MUTATION_RE = re.compile(r"^(?P<parent>.+)-mutate-(?P<index>[0-8])-(?P<operators>.+)$")
PROTOCOL = {
    "name": "HyperDistill (MetaMorph-DR teacher)", "stage": "HD2A paper-scale infrastructure preflight",
    "teacher": "metamorph_dr_matched_s1415_100m/Unimal-v0.pt", "seed": 1409,
    "base_parent_robots": 100, "pd_robots": 1000, "transitions_per_pd_robot": 8000,
    "total_expert_transitions": 8000000, "epochs": 150, "effective_batch_size": 5120,
    "optimizer": "Adam", "lr": 3e-4, "grad_norm": 0.5, "balanced_loss": True,
    "context_embedding_dropout": 0.1, "base_mlp_dropout": None,
    "mutation_source": "official repository data/train_mutate_1000 static morphology corpus",
    "mutation_semantics": "one original parent plus mutate-0 through mutate-8 variants; no dynamics mutation",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family(walker_id: str) -> str:
    result = walker_id.split("-", 1)[0]
    if result not in FAMILIES:
        raise ValueError(f"unrecognized PD walker family: {walker_id}")
    return result


def _mutation_fields(stem: str) -> dict:
    match = MUTATION_RE.match(stem)
    if match is None:
        return {"parent_walker_id": stem, "variant_index": 0, "is_original_parent": True,
                "mutation_count": 0, "mutation_sequence": [], "mutation_operators": [],
                "mutation_parameters": "NOT_APPLICABLE_ORIGINAL_PARENT"}
    operators = match.group("operators").split("-")
    # The official asset stores mutation operator names in its filename, not their sampled values.
    return {"parent_walker_id": match.group("parent"), "variant_index": int(match.group("index")) + 1,
            "is_original_parent": False, "mutation_count": len(operators), "mutation_sequence": operators,
            "mutation_operators": operators, "mutation_parameters": "NOT_AVAILABLE_IN_OFFICIAL_STATIC_ASSET"}


def _seed_from_parent(parent: str) -> int:
    fields = parent.split("-")
    if len(fields) < 2 or not fields[1].isdigit():
        raise ValueError(f"official parent id does not expose its seed token: {parent}")
    return int(fields[1])


def prepare_pd_inventory(teacher_config: Path, ood_pool: Path, source: Path, output: Path,
                         preflight_count: int | None = None) -> dict:
    """Audit static PD1000 assets and materialize a 10-robot representative HD2A subset.

    This intentionally fails if the repository corpus no longer has the documented 1+9
    structure or if it differs from the frozen teacher train100.  It never regenerates
    morphology or substitutes a walker.
    """
    teacher_config, ood_pool, source, output = map(Path, (teacher_config, ood_pool, source, output))
    train = yaml.safe_load(teacher_config.read_text(encoding="utf-8"))["ENV"]["WALKERS"]
    ood = [line.strip() for line in ood_pool.read_text(encoding="utf-8").splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    if len(train) != 100 or len(set(train)) != 100:
        raise ValueError(f"HD2 requires frozen train100, found {len(train)}")
    if set(train).intersection(ood):
        raise ValueError("frozen teacher train100 overlaps OOD pool")
    xml_dir, metadata_dir = source / "xml", source / "metadata"
    if not xml_dir.is_dir() or not metadata_dir.is_dir():
        raise FileNotFoundError(f"official train_mutate_1000 XML/metadata directories are required below {source}")
    xml_names = {path.stem for path in xml_dir.glob("*.xml")}
    metadata_names = {path.stem for path in metadata_dir.glob("*.json")}
    if xml_names != metadata_names:
        raise ValueError("MUTATION_PROTOCOL_MISMATCH: XML and metadata PD inventories differ")
    groups: dict[str, list[str]] = defaultdict(list)
    for pd_id in sorted(xml_names):
        groups[_mutation_fields(pd_id)["parent_walker_id"]].append(pd_id)
    if len(groups) != 100 or set(groups) != set(train):
        raise ValueError("MUTATION_PROTOCOL_MISMATCH: official PD parents are not exactly frozen train100")
    for parent, rows in groups.items():
        fields = {_mutation_fields(row)["variant_index"] for row in rows}
        if len(rows) != PD_PER_PARENT or fields != set(range(PD_PER_PARENT)):
            raise ValueError(f"MUTATION_PROTOCOL_MISMATCH: {parent} is not 1 original + 9 variants")
    if set(groups).intersection(ood):
        raise ValueError("MUTATION_PROTOCOL_MISMATCH: OOD walker is a PD parent")
    entries = []
    for parent in train:
        for pd_id in sorted(groups[parent], key=lambda value: _mutation_fields(value)["variant_index"]):
            mutation = _mutation_fields(pd_id)
            xml, metadata = xml_dir / f"{pd_id}.xml", metadata_dir / f"{pd_id}.json"
            ET.parse(xml)
            json.loads(metadata.read_text(encoding="utf-8"))
            entries.append({"pd_robot_id": pd_id, "parent_walker_id": parent, "parent_family": family(parent),
                            "parent_split": "train", **mutation, "mutation_seed": _seed_from_parent(parent),
                            "mutation_seed_provenance": "official parent identifier seed token",
                            "xml_path": str(xml.resolve()), "metadata_path": str(metadata.resolve()),
                            "parent_xml_sha256": sha256(xml_dir / f"{parent}.xml"),
                            "parent_metadata_sha256": sha256(metadata_dir / f"{parent}.json"),
                            "pd_xml_sha256": sha256(xml), "pd_metadata_sha256": sha256(metadata)})
    if len(entries) != 1000:
        raise AssertionError("HD2 PD inventory must contain exactly 1000 robots")
    selected = entries if preflight_count is None else _select_preflight(entries, preflight_count)
    output.mkdir(parents=True, exist_ok=False)
    (output / "pd1000" / "xml").mkdir(parents=True)
    (output / "pd1000" / "metadata").mkdir(parents=True)
    # Copy only the selected assets for HD2A; HD2B calls this without a subset.
    for entry in selected:
        shutil.copy2(entry["xml_path"], output / "pd1000" / "xml" / f"{entry['pd_robot_id']}.xml")
        shutil.copy2(entry["metadata_path"], output / "pd1000" / "metadata" / f"{entry['pd_robot_id']}.json")
        entry["materialized_xml_path"] = str((output / "pd1000" / "xml" / f"{entry['pd_robot_id']}.xml").resolve())
        entry["materialized_metadata_path"] = str((output / "pd1000" / "metadata" / f"{entry['pd_robot_id']}.json").resolve())
    provenance = output / "provenance"
    provenance.mkdir()
    parents = [{"walker_id": walker, "family": family(walker), "split": "train"} for walker in train]
    split_audit = {"HD2_BASE_PARENT_COUNT": len(parents), "HD2_PD_ROBOT_COUNT": len(entries),
                   "HD2_PARENT_TEST_COUNT": 0, "HD2_STRICT_OOD97_OVERLAP": 0,
                   "HD2_OOD6_PARENT_OVERLAP": 0, "HD2_PD_PROVENANCE": "PASS",
                   "MUTATION_PROTOCOL": "PASS", "MUTATION_PROTOCOL_DETAIL": "1 original + 9 filename-traced mutated variants per frozen train parent",
                   "MUTATION_PARAMETER_EVIDENCE": "NOT_AVAILABLE_IN_OFFICIAL_STATIC_ASSET"}
    (provenance / "parents.json").write_text(json.dumps(parents, indent=2) + "\n", encoding="utf-8")
    (provenance / "pd1000_manifest.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    (provenance / "mutation_manifest.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    (provenance / "split_audit.json").write_text(json.dumps(split_audit, indent=2) + "\n", encoding="utf-8")
    return {**split_audit, "selected_pd_robots": selected, "all_pd_robots": entries,
            "selected_parent_families": dict(Counter(row["parent_family"] for row in selected))}


def _select_preflight(entries: list[dict], count: int) -> list[dict]:
    if count != 10:
        raise ValueError("HD2A requires exactly 10 PD robots")
    # Deterministic stratification traverses the frozen config order while covering every family.
    selected, used_parents = [], set()
    by_family = {name: [row for row in entries if row["parent_family"] == name and row["variant_index"] == 0] for name in FAMILIES}
    for name in FAMILIES:
        row = by_family[name][0]
        selected.append(row)
        used_parents.add(row["parent_walker_id"])
    for row in entries:
        if len(selected) == count:
            break
        if row["variant_index"] and row["parent_walker_id"] not in used_parents:
            selected.append(row)
            used_parents.add(row["parent_walker_id"])
    if len(selected) != count or len(used_parents) != count:
        raise AssertionError("HD2A preflight selection must cover ten distinct parents")
    return selected
