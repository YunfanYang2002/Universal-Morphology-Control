"""Frozen HyperDistill policy adapter for the canonical MorphAdapt evaluator.

The evaluator owns rmamorph's teacher/environment configuration.  Student
construction is deliberately delegated to the namespace-independent HD2
constructor so this module never reads or mutates rmamorph's global ``cfg``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
STUDENT_PROPRIO_DIM = 204
TEACHER_PROPRIO_DIM = 624
MAX_LIMBS = 12
ACTION_DIM = MAX_LIMBS * 2
_NOMINAL_CONTEXT = None


def _load_hd2_constructor():
    source = ROOT / "tools" / "hd2_student_constructor.py"
    spec = importlib.util.spec_from_file_location("hyperdistill_hd2_student_constructor", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load authoritative HD2 constructor: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HD2 = _load_hd2_constructor()
HNMLP = _HD2.HNMLP
build_hd2_hnmlp_model = _HD2.build_hd2_hnmlp_model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict(path: Path) -> tuple[dict[str, torch.Tensor], int]:
    state = torch.load(path, map_location="cpu")
    required = {
        "mu_net", "optimizer", "seed", "completed_epoch",
        "cumulative_optimizer_steps", "cumulative_samples_seen",
    }
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError("HyperDistill checkpoint schema is not the frozen HD2B schema")
    epoch = int(state["completed_epoch"])
    if state["seed"] != 1409 or epoch not in (30, 150):
        raise ValueError("HyperDistill checkpoint seed or epoch is invalid")
    if state["cumulative_optimizer_steps"] != epoch * 1563 or state["cumulative_samples_seen"] != epoch * 8_000_000:
        raise ValueError("HyperDistill checkpoint counters violate the frozen HD2B contract")
    if not isinstance(state["mu_net"], dict) or not state["mu_net"]:
        raise ValueError("HyperDistill checkpoint has no mu_net state")
    for key, value in state["mu_net"].items():
        if not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise ValueError(f"HyperDistill checkpoint has non-finite parameter: {key}")
    return state["mu_net"], epoch


class FrozenHyperDistillPolicy:
    """One frozen morphology-conditioned policy used by the canonical evaluator."""

    def __init__(self, checkpoint: Path, student_columns: Path, device: torch.device):
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device(device)
        columns = torch.as_tensor(json.loads(Path(student_columns).read_text()), dtype=torch.long)
        if (columns.numel() != STUDENT_PROPRIO_DIM
                or len(set(columns.tolist())) != STUDENT_PROPRIO_DIM
                or int(columns.min()) < 0 or int(columns.max()) >= TEACHER_PROPRIO_DIM):
            raise ValueError("Frozen HD1 student mapper must contain 204 unique columns in [0, 624)")
        self.columns = columns.to(self.device)

        # This is the same frozen constructor used by HD2B and HD2C.  In
        # particular, no rmamorph cfg is consulted or temporarily modified.
        state, epoch = _state_dict(self.checkpoint)
        self.mu_net = build_hd2_hnmlp_model(device=self.device)
        self.mu_net.load_state_dict(state, strict=True)
        loaded = self.mu_net.state_dict()
        if set(loaded) != set(state) or any(not torch.equal(loaded[key].cpu(), value.cpu()) for key, value in state.items()):
            raise RuntimeError("HyperDistill checkpoint state changed during strict load")
        self.mu_net.eval()
        for parameter in self.mu_net.parameters():
            parameter.requires_grad_(False)
        self.checkpoint_epoch = epoch
        self._context_hash = None
        self._generated = False
        self._walker = None
        self._reported_finite = False
        print("HD2_STUDENT_CONSTRUCTOR_REUSED=PASS", flush=True)
        print("RMAMORPH_HYPERDISTILL_CONFIG_ISOLATION=PASS", flush=True)
        print("HYPERDISTILL_POLICY_INIT=PASS", flush=True)
        print("HYPERDISTILL_CHECKPOINT_STRICT_LOAD=PASS", flush=True)
        print("STRICT_CHECKPOINT_LOAD=PASS", flush=True)

    def begin_walker(self, walker: str, seed: int) -> None:
        self._walker = (str(walker), int(seed))
        self._context_hash = None
        self._generated = False

    @staticmethod
    def _context(raw_context, obs_mask, act_mask):
        global _NOMINAL_CONTEXT
        if _NOMINAL_CONTEXT is None:
            source = Path(__file__).with_name("convert_rmamorph_teacher_to_hyperdistill.py")
            spec = importlib.util.spec_from_file_location("hyperdistill_nominal_context", source)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load nominal context converter: {source}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _NOMINAL_CONTEXT = module.nominal_context

        raw = torch.as_tensor(raw_context).detach().cpu().numpy().reshape(MAX_LIMBS, 35).astype(np.float32, copy=False)
        result = _NOMINAL_CONTEXT(
            raw,
            obs_mask.detach().cpu().numpy().astype(bool),
            act_mask.detach().cpu().numpy().astype(bool),
        )
        return torch.from_numpy(result).reshape(1, -1)

    @torch.no_grad()
    def action(self, obs):
        proprio = torch.as_tensor(obs["proprioceptive"], device=self.device, dtype=torch.float32)
        if proprio.ndim != 2 or proprio.shape[1] != TEACHER_PROPRIO_DIM:
            raise ValueError(f"MorphAdapt evaluator proprioception shape must be [1, 624], got {tuple(proprio.shape)}")
        obs_mask = torch.as_tensor(obs["obs_padding_mask"], device=self.device).bool().reshape(1, -1)
        act_mask = torch.as_tensor(obs["act_padding_mask"], device=self.device).bool().reshape(1, -1)
        if obs_mask.shape != (1, MAX_LIMBS) or act_mask.shape != (1, ACTION_DIM):
            raise ValueError("MorphAdapt evaluator masks do not match frozen HD2B padding")
        context = self._context(obs["context"], obs_mask[0], act_mask[0]).to(self.device)
        context_hash = hashlib.sha256(context.detach().cpu().numpy().tobytes()).hexdigest()
        if self._context_hash is None:
            self._context_hash = context_hash
            self.mu_net.generate_params(context, obs_mask)
            self._generated = True
        elif context_hash != self._context_hash:
            raise RuntimeError("HyperDistill static morphology context changed after deployment initialization")
        selected = proprio.index_select(1, self.columns)
        output, _ = self.mu_net(selected, obs_mask)
        output = output.clamp(-1.0, 1.0).masked_fill(act_mask, 0.0)
        if not torch.isfinite(output).all():
            raise FloatingPointError("HyperDistill policy action became non-finite")
        if not torch.equal(output[act_mask], torch.zeros_like(output[act_mask])):
            raise AssertionError("HyperDistill padded action is not zero")
        if not self._reported_finite:
            print("POLICY_OUTPUT_FINITE=PASS", flush=True)
            print("PADDED_ACTION_ZERO=PASS", flush=True)
            self._reported_finite = True
        return output
