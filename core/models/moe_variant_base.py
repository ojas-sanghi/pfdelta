"""Reusable building blocks for Graph MoE variants in PFDelta.

This module contains a configurable MoE backbone used by many variant files.
Each variant registers its own model name while changing design knobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv, SAGEConv, TAGConv


@dataclass
class MoEVariantStats:
    probs: Tensor
    weights: Tensor
    topk_idx: Tensor
    entropy_mean: Tensor
    importance: Tensor
    load: Tensor
    aux_loss: Tensor


class _ExpertBlock(nn.Module):
    def __init__(self, expert_kind: str, hidden_dim: int, out_dim: int, n_layers: int, tag_k: int = 2):
        super().__init__()
        if n_layers < 2:
            raise ValueError("n_layers must be >=2")
        self.kind = expert_kind
        conv_ctor = {
            "graphconv": lambda i, o: GraphConv(i, o),
            "sage": lambda i, o: SAGEConv(i, o),
            "tag": lambda i, o: TAGConv(i, o, K=tag_k),
        }[expert_kind]
        self.start = conv_ctor(hidden_dim, hidden_dim)
        self.mid = nn.ModuleList([conv_ctor(hidden_dim, hidden_dim) for _ in range(n_layers - 2)])
        self.end = conv_ctor(hidden_dim, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.start(x, edge_index))
        for layer in self.mid:
            x = F.relu(layer(x, edge_index))
        return self.end(x, edge_index)


class ConfigurableGraphMoE(nn.Module):
    """A configurable graph MoE that supports multiple routing/expert choices."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        n_experts: int = 4,
        top_k: int = 2,
        x_start: int = 4,
        x_end: int = 10,
        expert_kinds: Optional[List[str]] = None,
        router_mode: str = "combined",  # combined|learned|size_only|dual
        gating_mode: str = "topk",  # topk|dense|gumbel
        size_prior_strength: float = 0.3,
        noisy_std: float = 0.0,
        temperature: float = 1.0,
        hierarchical: bool = False,
        weak_strong: bool = False,
        confidence_gate: bool = False,
        tag_k: int = 2,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.router_mode = router_mode
        self.gating_mode = gating_mode
        self.size_prior_strength = size_prior_strength
        self.noisy_std = noisy_std
        self.temperature = temperature
        self.hierarchical = hierarchical
        self.weak_strong = weak_strong
        self.confidence_gate = confidence_gate
        self.x_start = x_start
        self.x_end = x_end

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.weak_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

        if expert_kinds is None:
            expert_kinds = ["graphconv"] * n_experts
        if len(expert_kinds) < n_experts:
            expert_kinds = expert_kinds + [expert_kinds[-1]] * (n_experts - len(expert_kinds))

        self.experts = nn.ModuleList(
            [_ExpertBlock(expert_kinds[i], hidden_dim, out_dim, n_layers, tag_k=tag_k) for i in range(n_experts)]
        )

        self.router = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts),
        )
        self.router2 = nn.Sequential(nn.Linear(hidden_dim + 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_experts))

        self.cluster_router = nn.Sequential(nn.Linear(hidden_dim + 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))

        self.confidence_net = nn.Sequential(nn.Linear(2, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())
        self.size_centers = nn.Parameter(torch.linspace(0.0, 1.0, n_experts))

        self.variant_stats: Optional[MoEVariantStats] = None

    def _batch(self, data: HeteroData, n_nodes: int) -> Tensor:
        if hasattr(data["bus"], "batch") and data["bus"].batch is not None:
            return data["bus"].batch
        return torch.zeros(n_nodes, dtype=torch.long, device=data["bus"].x.device)

    def _size_feats(self, batch: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        g = int(batch.max().item()) + 1
        n = torch.bincount(batch, minlength=g).float().clamp_min(1)
        e = torch.bincount(batch[edge_index[0]], minlength=g).float().clamp_min(0)
        density = e / (n * (n - 1)).clamp_min(1)
        logn = torch.log(n)
        logn_norm = (logn - logn.min()) / (logn.max() - logn.min() + 1e-6)
        feats = torch.stack([n, e, density], dim=-1)
        feats = (feats - feats.mean(0, keepdim=True)) / (feats.std(0, keepdim=True, unbiased=False) + 1e-6)
        return feats[batch], logn_norm[batch]

    def _router_logits(self, h: Tensor, size_feats: Tensor, logn_norm: Tensor) -> Tensor:
        inp = torch.cat([h, size_feats], dim=-1)
        learned = self.router(inp)
        if self.router_mode == "learned":
            logits = learned
        elif self.router_mode == "size_only":
            logits = -((logn_norm.unsqueeze(-1) - self.size_centers.unsqueeze(0)) ** 2)
        else:
            prior = -((logn_norm.unsqueeze(-1) - self.size_centers.unsqueeze(0)) ** 2)
            logits = (1 - self.size_prior_strength) * learned + self.size_prior_strength * prior
            if self.router_mode == "dual":
                logits = 0.5 * logits + 0.5 * self.router2(inp)

        if self.hierarchical:
            coarse = F.softmax(self.cluster_router(inp), dim=-1)
            cluster_mask = torch.zeros_like(logits)
            half = self.n_experts // 2
            cluster_mask[:, :half] = coarse[:, 0:1]
            cluster_mask[:, half:] = coarse[:, 1:2]
            logits = logits + torch.log(cluster_mask + 1e-8)

        if self.training and self.noisy_std > 0:
            logits = logits + torch.randn_like(logits) * self.noisy_std
        return logits / max(self.temperature, 1e-6)

    def _gate(self, logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.gating_mode == "gumbel" and self.training:
            probs = F.gumbel_softmax(logits, tau=max(self.temperature, 1e-6), hard=False, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)

        if self.gating_mode == "dense":
            weights = probs
            topk_idx = torch.topk(probs, k=min(self.top_k, probs.size(-1)), dim=-1).indices
            return probs, weights, topk_idx

        top_vals, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
        top_vals = top_vals / (top_vals.sum(-1, keepdim=True) + 1e-8)
        weights = torch.zeros_like(probs)
        weights.scatter_(1, top_idx, top_vals)
        return probs, weights, top_idx

    def _aux(self, probs: Tensor, weights: Tensor) -> Tensor:
        importance = probs.sum(0)
        importance = importance / (importance.sum() + 1e-8)
        load = (weights > 0).float().sum(0)
        load = load / (load.sum() + 1e-8)
        entropy_mean = (-(probs * torch.log(probs + 1e-8)).sum(-1)).mean()
        imp_cv2 = torch.var(importance) / (torch.mean(importance) ** 2 + 1e-8)
        load_cv2 = torch.var(load) / (torch.mean(load) ** 2 + 1e-8)
        aux = imp_cv2 + load_cv2 + 0.001 * torch.mean(torch.logsumexp(probs, dim=-1) ** 2)
        self.variant_stats = MoEVariantStats(
            probs=probs.detach(),
            weights=weights.detach(),
            topk_idx=torch.topk(weights, k=min(self.top_k, weights.size(-1)), dim=-1).indices.detach(),
            entropy_mean=entropy_mean.detach(),
            importance=importance.detach(),
            load=load.detach(),
            aux_loss=aux.detach(),
        )
        return aux

    def forward(self, data: HeteroData) -> Tensor:
        x = data["bus"].x[:, self.x_start : self.x_end]
        edge_index = data["bus", "branch", "bus"].edge_index
        h = self.encoder(x)

        batch = self._batch(data, h.size(0))
        size_feats, logn_norm = self._size_feats(batch, edge_index)
        logits = self._router_logits(h, size_feats, logn_norm)
        probs, weights, _ = self._gate(logits)

        expert_outs = [expert(h, edge_index) for expert in self.experts]
        y_moe = (torch.stack(expert_outs, dim=1) * weights.unsqueeze(-1)).sum(1)

        if self.weak_strong or self.confidence_gate:
            y_weak = self.weak_head(h)
            dispersion = torch.var(F.softmax(y_weak, dim=-1), dim=-1, keepdim=True)
            neg_entropy = -torch.sum(
                F.softmax(y_weak, dim=-1) * torch.log_softmax(y_weak, dim=-1), dim=-1, keepdim=True
            )
            conf_inp = torch.cat([dispersion, neg_entropy], dim=-1)
            if self.confidence_gate:
                conf = self.confidence_net(conf_inp)
            else:
                conf = dispersion.clamp(0, 1)
            y = conf * y_weak + (1 - conf) * y_moe
        else:
            y = y_moe

        self._aux(probs, weights)
        return y
