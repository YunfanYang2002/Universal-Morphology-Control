"""Convert the audited HD0 export; never fit or invert observation RMS."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch


STUDENT_LABELS = [
    *[f"{name}:{i}" for name, size in (
        ("body_xpos", 3), ("body_xvelp", 3), ("body_xvelr", 3),
        ("body_xquat", 4)) for i in range(size)],
    *[f"joint{j}/{name}:0" for j in range(2) for name in ("qpos", "qvel")],
]
# Official Agent.get_context, context_version=1. These are fixed hardware
# scaling ranges, not empirical observation normalization statistics.
CONTEXT_RANGES = [
    ([-.5, -.45, -.49], [.5, .45, 2.04]),
    ([-.225, -.225, -.225], [.225, .225, 0.]),
    ([.70710678, -.70710678, -.70710678, 0.], [1., .70710678, .70710678, 0.]),
    ([.70710678, -.70710678, -.70710678, 0.], [1., .70710678, .70710678, 0.]),
    ([1.17809725], [4.1887902]), ([.05, 0.], [.1, .22627417]),
]
JOINT_RANGES = [
    ([-.05, -.05, 0.], [.05, .05, .05]),
    ([-1.57079633, 0.], [0., 1.57079633]),
    ([-.5000024, -.5000024, -1.], [1., 1., 1.]), ([0.], [300.]),
]


def nominal_context(raw, obs_mask, act_mask):
    """Scale reset-only raw context, then restore nonexistent slots to zero."""
    assert raw.shape == (len(obs_mask), 35), raw.shape
    assert act_mask.shape == (len(obs_mask) * 2,)
    ranges = CONTEXT_RANGES + JOINT_RANGES * 2
    lo = np.concatenate([x[0] for x in ranges])
    hi = np.concatenate([x[1] for x in ranges])
    out = -(lo != hi).astype(float) + 2 * (raw - lo) / (hi - lo + 1e-8)
    joints = out[:, 17:].copy().reshape(-1, 9)
    joints[act_mask] = 0
    out[:, 17:] = joints.reshape(len(obs_mask), 18)
    out[obs_mask] = 0
    assert np.isfinite(out).all(), "Nonfinite nominal context"
    return out.astype(np.float32).ravel()


def convert(source, output, expected_walkers=None):
    source, output = Path(source), Path(output)
    manifest = json.loads((source / "manifest.json").read_text())
    assert manifest["context_format"] == "raw_v1"
    labels = manifest["per_limb_obs_labels"]
    assert len(set(labels)) == len(labels), "Ambiguous teacher feature labels"
    selected = [labels.index(label) for label in STUDENT_LABELS]
    limbs = manifest["max_limbs"]
    columns = [limb * len(labels) + column for limb in range(limbs) for column in selected]
    output.mkdir(parents=True, exist_ok=False)
    (output / "student_columns.json").write_text(json.dumps(columns))
    with np.load(source / "arrays.npz", allow_pickle=False) as data:
        obs = data["proprio_normalized"]
        assert obs.ndim == 2 and obs.shape[1] == limbs * len(labels)
        assert data["action_mean"].shape == data["action_canonical"].shape == (len(obs), limbs * 2)
        assert data["rms_mean"].shape == data["rms_var"].shape == (obs.shape[1],)
        assert np.isfinite(data["rms_mean"]).all() and np.isfinite(data["rms_var"]).all()
        assert (data["rms_var"] >= 0).all()
        rms = {"mean": data["rms_mean"][columns], "var": data["rms_var"][columns],
               "count": data["rms_count"].copy()}
        np.savez(output / "selected_obs_rms.npz", **rms)
        if expected_walkers is None:
            assert len(manifest["walkers"]) == 3, "HD0 export requires exactly three training walkers"
        else:
            assert list(manifest["walkers"]) == list(expected_walkers), "HD1 walker order differs from selected inventory"
        for walker_index, walker in enumerate(manifest["walkers"]):
            assert Path(walker).name == walker and walker not in (".", "..")
            rows = np.flatnonzero(data["walker_index"] == walker_index)
            assert len(rows), f"Missing walker: {walker}"
            episodes = np.unique(data["episode_id"][rows])
            assert len(episodes) >= 2, "Need disjoint train/validation trajectories"
            first = rows[0]
            om = data["obs_padding_mask"][first].astype(bool)
            am = data["act_padding_mask"][first].astype(bool)
            assert om.shape == (limbs,) and am.shape == (limbs * 2,) and (~am).any()
            assert np.array_equal(data["obs_padding_mask"][rows], np.broadcast_to(om, (len(rows), limbs)))
            assert np.array_equal(data["act_padding_mask"][rows], np.broadcast_to(am, (len(rows), limbs * 2)))
            assert am.reshape(limbs, 2)[om].all(), "Padded limbs expose actions"
            context = nominal_context(data["context_raw"][int(episodes[0])], om, am)
            for episode in episodes:
                assert np.array_equal(context, nominal_context(data["context_raw"][int(episode)], om, am)), "Nominal context changed across resets"
            adjacency = data["adjacency"][first]
            assert adjacency.shape == (limbs, limbs)
            assert np.array_equal(data["adjacency"][rows], np.broadcast_to(adjacency, (len(rows), limbs, limbs)))
            tensors = {
                "obs": obs[rows][:, columns], "act": data["action_canonical"][rows],
                "act_mean": data["action_mean"][rows], "episode_id": data["episode_id"][rows],
                "context": context, "obs_padding_mask": om, "act_padding_mask": am,
                "adjacency_matrix": adjacency,
            }
            for name, value in tensors.items():
                assert np.isfinite(value).all(), f"Nonfinite {walker}/{name}"
            for name in ("act", "act_mean"):
                assert (tensors[name][:, am] == 0).all(), f"Nonzero padded {name}"
            assert (np.abs(tensors["act"]) <= 1).all()
            tensors = {key: torch.from_numpy(np.array(value, copy=True)) for key, value in tensors.items()}
            payload = dict(tensors)
            payload["teacher_obs_rms"] = {key: torch.from_numpy(np.array(value, copy=True)) for key, value in rms.items()}
            payload["manifest"] = {"context_version": 1, "proprio_features_per_limb": 17,
                                   "max_limbs": limbs, "normalization": "teacher_obs_rms_then_selected",
                                   "walker_id": walker, "teacher_provenance": manifest}
            path = output / f"{walker}.pkl"
            with path.open("wb") as stream:
                pickle.dump(payload, stream)
            with path.open("rb") as stream:
                reloaded = pickle.load(stream)
            assert all(torch.equal(value, reloaded[key]) for key, value in tensors.items())
    mapping = {
        "baseline": "HyperDistill (MetaMorph teacher)", "teacher_export": manifest,
        "student_labels_per_limb": STUDENT_LABELS, "teacher_columns": columns,
        "normalization": "Select columns from native teacher-normalized observations; no second normalization or RMS fitting",
        "context": "Official context_version=1 affine ranges; reset-only raw nominal values; padded limb and joint slots zero",
        "action": "Identity: two canonical joint slots per limb; True mask means invalid; invalid targets exactly zero",
        "source_sha256": hashlib.sha256((source / "arrays.npz").read_bytes()).hexdigest(),
        "HYPERDISTILL_DATASET_EXPORT": "PASS", "HYPERDISTILL_DATASET_RELOAD": "PASS",
        "OBS_ORDER_BINDING": "PASS", "ACTION_MASK_BINDING": "PASS",
    }
    (output / "mapping.json").write_text(json.dumps(mapping, indent=2))
    print("HYPERDISTILL_DATASET_EXPORT=PASS\nHYPERDISTILL_DATASET_RELOAD=PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-walkers", help="optional JSON list for HD1 selected walker order")
    args = parser.parse_args()
    expected = None if args.expected_walkers is None else json.loads(args.expected_walkers)
    if expected is not None and (not isinstance(expected, list) or not all(isinstance(value, str) for value in expected)):
        raise ValueError("--expected-walkers must be a JSON list of walker IDs")
    convert(args.source, args.output, expected_walkers=expected)
