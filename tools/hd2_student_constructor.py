"""Authoritative, namespace-independent HD2 HN-MLP constructor.

The formal MorphAdapt evaluator runs with rmamorph's ``metamorph.cfg``.  This
module deliberately has no dependency on that global configuration namespace.
The constants below are the frozen HD2B constructor settings already used by
HD2 training and HD2C checkpoint export.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


STUDENT_PROPRIO_DIM = 204
TEACHER_PROPRIO_DIM = 624
MAX_LIMBS = 12
CONTEXT_DIM = MAX_LIMBS * 35
ACTION_DIM = MAX_LIMBS * 2

FROZEN_HYPERNET = SimpleNamespace(
    CONTEXT_ENCODER_TYPE="transformer", CONTEXT_EMBED_SIZE=128,
    CONTEXT_MASK=True, CONTEXT_TF_ENCODER_NHEAD=2,
    CONTEXT_TF_ENCODER_FF_DIM=256, ENCODER_LAYER_NUM=3,
    EMBEDDING_DROPOUT=0.1, SHARE_CONTEXT_ENCODER=False,
    INPUT_AGGREGATION="sum",
)
FROZEN_MLP = SimpleNamespace(HIDDEN_DIM=256, LAYER_NUM=2, DROPOUT=None)


class _TransformerEncoderLayerResidual(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
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
    def __init__(self, context_size):
        super().__init__()
        self.model_args = copy.deepcopy(FROZEN_HYPERNET)
        self.context_embed = nn.Linear(context_size, self.model_args.CONTEXT_EMBED_SIZE)
        self.context_encoder = _TransformerEncoder(_TransformerEncoderLayerResidual(
            self.model_args.CONTEXT_EMBED_SIZE,
            self.model_args.CONTEXT_TF_ENCODER_NHEAD,
            self.model_args.CONTEXT_TF_ENCODER_FF_DIM,
        ))
        self.embedding_dropout = nn.Dropout(p=self.model_args.EMBEDDING_DROPOUT)

    def forward(self, context, obs_mask):
        value = self.context_embed(context)
        value = self.context_encoder(value, src_key_padding_mask=obs_mask)
        return self.embedding_dropout(value)


class _HypernetLayer(nn.Module):
    def __init__(self, base_input_dim, base_output_dim, init_dim):
        super().__init__()
        self.HN_weight = nn.Linear(FROZEN_HYPERNET.CONTEXT_EMBED_SIZE, base_input_dim * base_output_dim)
        self.HN_bias = nn.Linear(FROZEN_HYPERNET.CONTEXT_EMBED_SIZE, base_output_dim)
        nn.init.zeros_(self.HN_weight.weight)
        nn.init.normal_(self.HN_weight.bias, std=np.sqrt(1.0 / init_dim))
        nn.init.zeros_(self.HN_bias.weight)
        nn.init.zeros_(self.HN_bias.bias)

    def forward(self, value):
        return self.HN_weight(value), self.HN_bias(value)


class HNMLP(nn.Module):
    """Frozen-HD2-compatible HNMLP with the official state-dict layout."""

    def __init__(self, obs_dim=STUDENT_PROPRIO_DIM, context_dim=CONTEXT_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.seq_len = MAX_LIMBS
        self.limb_obs_size = obs_dim // self.seq_len
        self.limb_out_dim = action_dim // self.seq_len
        self.HN_args = copy.deepcopy(FROZEN_HYPERNET)
        self.base_args = copy.deepcopy(FROZEN_MLP)
        context_size = context_dim // self.seq_len
        self.input_context_encoder = _ContextEncoder(context_size)
        self.input_HN_layer = _HypernetLayer(self.limb_obs_size, self.base_args.HIDDEN_DIM, obs_dim)
        if not self.HN_args.SHARE_CONTEXT_ENCODER:
            self.output_context_encoder = _ContextEncoder(context_size)
        self.output_HN_layer = _HypernetLayer(self.base_args.HIDDEN_DIM, self.limb_out_dim, self.base_args.HIDDEN_DIM)
        if self.base_args.LAYER_NUM > 1:
            if not self.HN_args.SHARE_CONTEXT_ENCODER:
                self.hidden_context_encoder = _ContextEncoder(context_size)
            self.hidden_dims = [self.base_args.HIDDEN_DIM for _ in range(self.base_args.LAYER_NUM)]
            self.hidden_HN_layers = nn.ModuleList(
                _HypernetLayer(self.hidden_dims[i], self.hidden_dims[i + 1], self.hidden_dims[i])
                for i in range(self.base_args.LAYER_NUM - 1)
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
            denominator = valid.squeeze(-1).sum(dim=1, keepdim=True)
            hidden_embedding = (hidden_embedding * valid).sum(dim=1) / denominator
            self.hidden_weights, self.hidden_bias = [], []
            for layer in self.hidden_HN_layers:
                weight, bias = layer(hidden_embedding)
                self.hidden_weights.append(weight.view(batch_size, self.hidden_dims[0], self.hidden_dims[1]))
                self.hidden_bias.append(bias)
        output_embedding = input_embedding if self.HN_args.SHARE_CONTEXT_ENCODER else self.output_context_encoder(context, obs_mask)
        self.output_weight, self.output_bias = self.output_HN_layer(output_embedding)
        self.output_weight = self.output_weight.view(batch_size, self.seq_len, self.base_args.HIDDEN_DIM, self.limb_out_dim)

    def forward(self, obs, obs_mask, obs_env=None, obs_cm_mask=None, obs_context=None, morphology_info=None, **_kwargs):
        batch_size = obs.shape[0]
        obs = obs.view(batch_size, self.seq_len, -1)
        embedding = (obs[:, :, :, None] * self.input_weight).sum(dim=-2) + self.input_bias
        embedding = embedding * (~obs_mask).to(dtype=embedding.dtype)[:, :, None]
        embedding = embedding.sum(dim=1)
        embedding = F.relu(embedding)
        for weight, bias in zip(self.hidden_weights, self.hidden_bias):
            embedding = (embedding[:, :, None] * weight).sum(dim=1) + bias
            embedding = F.relu(embedding)
        output = (embedding[:, None, :, None] * self.output_weight).sum(dim=-2) + self.output_bias
        return output.reshape(batch_size, -1), None


class HD2StudentActor(nn.Module):
    """Training/export shell exposing the small ActorCritic surface HD2 uses."""

    def __init__(self, device=None):
        super().__init__()
        self.mu_net = HNMLP()
        if device is not None:
            self.to(device)
        self.action_mu = None

    def forward(self, obs, compute_val=False, **_kwargs):
        if self.training:
            self.mu_net.generate_params(obs["context"], obs["obs_padding_mask"].bool())
        self.action_mu, _ = self.mu_net(
            obs["proprioceptive"], obs["obs_padding_mask"].bool(),
            obs_context=obs.get("context"),
        )
        return 0.0, None, None, None


def build_hd2_hnmlp_model(device=None) -> HNMLP:
    """Build the frozen HD2 HNMLP without reading any runtime global cfg."""
    model = HNMLP()
    return model if device is None else model.to(device)


def build_hd2_student_actor(device=None) -> HD2StudentActor:
    return HD2StudentActor(device=device)
