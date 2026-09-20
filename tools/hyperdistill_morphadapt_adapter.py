"""Frozen HyperDistill HN-MLP adapter for the MorphAdapt evaluator.

This module intentionally contains inference-only network definitions.  The
MorphAdapt repository does not ship the HyperDistill HN-MLP implementation, so
the small exact network surface needed to load the frozen HD2B state dict is
kept here.  No optimizer, checkpoint resume, or training path is exposed.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metamorph.config import cfg


STUDENT_PROPRIO_DIM = 204
TEACHER_PROPRIO_DIM = 624
MAX_LIMBS = 12
CONTEXT_DIM = MAX_LIMBS * 35
ACTION_DIM = MAX_LIMBS * 2
_NOMINAL_CONTEXT = None


class _TransformerEncoderLayerResidual(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        return src + self.dropout2(src2)


class _TransformerEncoder(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer)])
        self.num_layers = 1
        self.norm = None

    def forward(self, src, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_key_padding_mask=src_key_padding_mask)
        return src


class _ContextEncoder(nn.Module):
    def __init__(self, context_size, model_args):
        super().__init__()
        self.model_args = copy.deepcopy(model_args)
        if self.model_args.CONTEXT_ENCODER_TYPE != "transformer":
            raise ValueError("Frozen HD2B adapter requires the transformer context encoder")
        self.context_embed = nn.Linear(context_size, self.model_args.CONTEXT_EMBED_SIZE)
        layer = _TransformerEncoderLayerResidual(
            self.model_args.CONTEXT_EMBED_SIZE,
            self.model_args.CONTEXT_TF_ENCODER_NHEAD,
            self.model_args.CONTEXT_TF_ENCODER_FF_DIM,
            0.0,
        )
        self.context_encoder = _TransformerEncoder(layer)

    def forward(self, context, obs_mask):
        value = self.context_embed(context)
        value = self.context_encoder(value, src_key_padding_mask=obs_mask)
        if self.model_args.EMBEDDING_DROPOUT is not None:
            value = F.dropout(value, p=self.model_args.EMBEDDING_DROPOUT, training=self.training)
        return value


class _HypernetLayer(nn.Module):
    def __init__(self, base_input_dim, base_output_dim, init_dim, model_args):
        super().__init__()
        self.model_args = copy.deepcopy(model_args)
        self.HN_weight = nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, base_input_dim * base_output_dim)
        self.HN_bias = nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, base_output_dim)
        # Initialization is irrelevant after strict checkpoint loading, but
        # retaining the official construction keeps missing-key diagnostics local.
        nn.init.zeros_(self.HN_weight.weight)
        nn.init.normal_(self.HN_weight.bias, std=np.sqrt(1.0 / init_dim))
        nn.init.zeros_(self.HN_bias.weight)
        nn.init.zeros_(self.HN_bias.bias)

    def forward(self, value):
        return self.HN_weight(value), self.HN_bias(value)


class _HNMLP(nn.Module):
    def __init__(self, obs_dim=STUDENT_PROPRIO_DIM, context_dim=CONTEXT_DIM, action_dim=ACTION_DIM):
        super().__init__()
        model_args = copy.deepcopy(cfg.MODEL)
        hyper_args = copy.deepcopy(cfg.MODEL.HYPERNET)
        mlp_args = copy.deepcopy(cfg.MODEL.MLP)
        self.seq_len = MAX_LIMBS
        self.limb_obs_size = obs_dim // self.seq_len
        self.limb_out_dim = action_dim // self.seq_len
        self.HN_args = hyper_args
        self.base_args = mlp_args

        context_size = context_dim // self.seq_len
        self.input_context_encoder = _ContextEncoder(context_size, hyper_args)
        self.input_HN_layer = _HypernetLayer(self.limb_obs_size, mlp_args.HIDDEN_DIM, obs_dim, hyper_args)
        if not hyper_args.SHARE_CONTEXT_ENCODER:
            self.output_context_encoder = _ContextEncoder(context_size, hyper_args)
        self.output_HN_layer = _HypernetLayer(mlp_args.HIDDEN_DIM, self.limb_out_dim, mlp_args.HIDDEN_DIM, hyper_args)
        if mlp_args.LAYER_NUM > 1:
            if not hyper_args.SHARE_CONTEXT_ENCODER:
                self.hidden_context_encoder = _ContextEncoder(context_size, hyper_args)
            self.hidden_dims = [mlp_args.HIDDEN_DIM for _ in range(mlp_args.LAYER_NUM)]
            self.hidden_HN_layers = nn.ModuleList(
                _HypernetLayer(self.hidden_dims[i], self.hidden_dims[i + 1], self.hidden_dims[i], hyper_args)
                for i in range(mlp_args.LAYER_NUM - 1)
            )

    @torch.no_grad()
    def generate_params(self, context, obs_mask):
        batch_size = context.shape[0]
        context = context.view(batch_size, self.seq_len, -1)
        input_embedding = self.input_context_encoder(context, obs_mask)
        self.input_weight, self.input_bias = self.input_HN_layer(input_embedding)
        self.input_weight = self.input_weight.view(batch_size, self.seq_len, self.limb_obs_size, self.base_args.HIDDEN_DIM)
        if self.base_args.LAYER_NUM > 1:
            hidden_embedding = input_embedding if self.HN_args.SHARE_CONTEXT_ENCODER else self.hidden_context_encoder(context, obs_mask)
            valid = (~obs_mask).to(dtype=hidden_embedding.dtype).unsqueeze(-1)
            hidden_embedding = (hidden_embedding * valid).sum(dim=1) / valid.squeeze(-1).sum(dim=1, keepdim=True)
            self.hidden_weights, self.hidden_bias = [], []
            for layer in self.hidden_HN_layers:
                weight, bias = layer(hidden_embedding)
                self.hidden_weights.append(weight.view(batch_size, self.hidden_dims[0], self.hidden_dims[1]))
                self.hidden_bias.append(bias)
        output_embedding = input_embedding if self.HN_args.SHARE_CONTEXT_ENCODER else self.output_context_encoder(context, obs_mask)
        self.output_weight, self.output_bias = self.output_HN_layer(output_embedding)
        self.output_weight = self.output_weight.view(batch_size, self.seq_len, self.base_args.HIDDEN_DIM, self.limb_out_dim)

    def forward(self, obs, obs_mask):
        batch_size = obs.shape[0]
        obs = obs.view(batch_size, self.seq_len, -1)
        embedding = (obs[:, :, :, None] * self.input_weight).sum(dim=-2) + self.input_bias
        embedding = embedding * (~obs_mask).to(dtype=embedding.dtype)[:, :, None]
        if self.HN_args.INPUT_AGGREGATION == "limb_num":
            embedding = embedding.sum(dim=1) / (~obs_mask).sum(dim=1, keepdim=True)
        elif self.HN_args.INPUT_AGGREGATION == "sqrt_limb_num":
            embedding = embedding.sum(dim=1) / torch.sqrt((~obs_mask).sum(dim=1, keepdim=True).to(dtype=embedding.dtype))
        elif self.HN_args.INPUT_AGGREGATION == "max_limb_num":
            embedding = embedding.mean(dim=1)
        else:
            embedding = embedding.sum(dim=1)
        embedding = F.relu(embedding)
        for weight, bias in zip(self.hidden_weights, self.hidden_bias):
            embedding = (embedding[:, :, None] * weight).sum(dim=1) + bias
            embedding = F.relu(embedding)
        return ((embedding[:, None, :, None] * self.output_weight).sum(dim=-2) + self.output_bias).reshape(batch_size, -1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or set(state) != {
        "mu_net", "optimizer", "seed", "completed_epoch", "cumulative_optimizer_steps", "cumulative_samples_seen",
    }:
        raise ValueError("HyperDistill checkpoint schema is not the frozen HD2B schema")
    if state["seed"] != 1409 or state["completed_epoch"] not in (30, 150):
        raise ValueError("HyperDistill checkpoint seed or epoch is invalid")
    expected_steps = state["completed_epoch"] * 1563
    expected_samples = state["completed_epoch"] * 8_000_000
    if state["cumulative_optimizer_steps"] != expected_steps or state["cumulative_samples_seen"] != expected_samples:
        raise ValueError("HyperDistill checkpoint counters violate the frozen HD2B contract")
    if not isinstance(state["mu_net"], dict) or not state["mu_net"]:
        raise ValueError("HyperDistill checkpoint has no mu_net state")
    for key, value in state["mu_net"].items():
        if not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise ValueError(f"HyperDistill checkpoint has non-finite parameter: {key}")
    return state["mu_net"]


class FrozenHyperDistillPolicy:
    """One frozen morphology-conditioned policy used by the canonical evaluator."""

    def __init__(self, checkpoint: Path, student_columns: Path, device: torch.device):
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device(device)
        columns = torch.as_tensor(json.loads(Path(student_columns).read_text()), dtype=torch.long)
        if columns.numel() != STUDENT_PROPRIO_DIM or len(set(columns.tolist())) != STUDENT_PROPRIO_DIM or int(columns.min()) < 0 or int(columns.max()) >= TEACHER_PROPRIO_DIM:
            raise ValueError("Frozen HD1 student mapper must contain 204 unique columns in [0, 624)")
        self.columns = columns.to(self.device)

        snapshot = cfg.clone()
        try:
            cfg.MODEL.MAX_LIMBS = MAX_LIMBS
            cfg.MODEL.MLP.LAYER_NUM = 2
            cfg.MODEL.MLP.DROPOUT = None
            cfg.MODEL.HYPERNET.EMBEDDING_DROPOUT = 0.1
            self.mu_net = _HNMLP().to(self.device)
        finally:
            cfg.clear()
            cfg.update(snapshot)
        self.mu_net.load_state_dict(_state_dict(self.checkpoint), strict=True)
        self.mu_net.eval()
        for parameter in self.mu_net.parameters():
            parameter.requires_grad_(False)
        self._context_hash = None
        self._generated = False
        self._walker = None

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
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            _NOMINAL_CONTEXT = module.nominal_context

        raw = torch.as_tensor(raw_context).detach().cpu().numpy().reshape(MAX_LIMBS, 35).astype(np.float32, copy=False)
        result = _NOMINAL_CONTEXT(raw, obs_mask.detach().cpu().numpy().astype(bool), act_mask.detach().cpu().numpy().astype(bool))
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
        output = self.mu_net(selected, obs_mask)
        output = output.clamp(-1.0, 1.0).masked_fill(act_mask, 0.0)
        if not torch.isfinite(output).all():
            raise FloatingPointError("HyperDistill policy action became non-finite")
        return output
