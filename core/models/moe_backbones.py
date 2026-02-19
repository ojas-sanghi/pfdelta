"""Reusable graph Mixture-of-Experts backbones for PFDelta.

This module centralizes MoE building blocks so many concrete variants can be
registered in separate files with clear, isolated behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv, NNConv, SAGEConv, TAGConv


@dataclass
class MoERoutingInfo:
    probs: Tensor
    sparse_weights: Tensor
    topk_indices: Tensor
    entropy: Tensor
    expert_importance: Tensor
    expert_load: Tensor
    load_balance_loss: Tensor
    z_loss: Tensor


class ExpertBlock(nn.Module):
    """Configurable expert block used across MoE variants."""

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        expert_type: str = "graphconv",
        edge_dim: int = 5,
        tag_k: int = 2,
    ):
        super().__init__()
        self.expert_type = expert_type
        self.edge_dim = edge_dim

        if expert_type == "mlp":
            layers: List[nn.Module] = []
            in_dim = hidden_dim
            for _ in range(max(1, n_layers - 1)):
                layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
                in_dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.net = nn.Sequential(*layers)
            return

        if n_layers < 2:
            raise ValueError("n_layers must be >= 2 for graph experts")

        self.in_proj = self._make_conv(expert_type, hidden_dim, hidden_dim, edge_dim, tag_k)
        self.hid = nn.ModuleList(
            [self._make_conv(expert_type, hidden_dim, hidden_dim, edge_dim, tag_k) for _ in range(n_layers - 2)]
        )
        self.out_proj = self._make_conv(expert_type, hidden_dim, out_dim, edge_dim, tag_k)

    def _make_conv(self, t: str, in_ch: int, out_ch: int, edge_dim: int, tag_k: int):
        if t == "graphconv":
            return GraphConv(in_ch, out_ch)
        if t == "sage":
            return SAGEConv(in_ch, out_ch)
        if t == "tag":
            return TAGConv(in_ch, out_ch, K=tag_k)
        if t == "nnconv":
            edge_net = nn.Sequential(
                nn.Linear(edge_dim, in_ch),
                nn.ReLU(),
                nn.Linear(in_ch, in_ch * out_ch),
            )
            return NNConv(in_ch, out_ch, nn=edge_net)
        raise ValueError(f"Unknown expert_type: {t}")

    def _apply_conv(self, conv, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor]) -> Tensor:
        if isinstance(conv, NNConv):
            if edge_attr is None:
                edge_attr = torch.zeros(edge_index.shape[1], self.edge_dim, device=x.device)
            return conv(x, edge_index, edge_attr)
        return conv(x, edge_index)

    def forward(self, h: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor]) -> Tensor:
        if self.expert_type == "mlp":
            return self.net(h)

        x = F.relu(self._apply_conv(self.in_proj, h, edge_index, edge_attr))
        for conv in self.hid:
            x = F.relu(self._apply_conv(conv, x, edge_index, edge_attr))
        return self._apply_conv(self.out_proj, x, edge_index, edge_attr)


class GraphMoEBackbone(nn.Module):
    """Flexible MoE backbone supporting many routing and expert strategies."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        n_experts: int = 4,
        top_k: int = 2,
        router_hidden_dim: int = 128,
        expert_type: str = "graphconv",
        secondary_expert_type: Optional[str] = None,
        edge_dim: int = 5,
        tag_k: int = 2,
        router_mode: str = "topk",
        size_feature_mode: str = "log_nodes_edges",
        size_prior_strength: float = 0.3,
        size_temperature: float = 1.0,
        noisy_gating_std: float = 0.0,
        expert_dropout: float = 0.0,
        graph_context_strength: float = 0.0,
        adaptive_topk: bool = False,
        confidence_gate: bool = False,
        load_balance_coef: float = 0.01,
        z_loss_coef: float = 0.001,
        x_start: int = 4,
        x_end: int = 10,
        **unused,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(max(1, top_k), n_experts)
        self.router_mode = router_mode
        self.size_feature_mode = size_feature_mode
        self.size_prior_strength = size_prior_strength
        self.size_temperature = size_temperature
        self.noisy_gating_std = noisy_gating_std
        self.expert_dropout = expert_dropout
        self.graph_context_strength = graph_context_strength
        self.adaptive_topk = adaptive_topk
        self.confidence_gate = confidence_gate
        self.load_balance_coef = load_balance_coef
        self.z_loss_coef = z_loss_coef
        self.x_start = x_start
        self.x_end = x_end
        self.edge_dim = edge_dim
        self.routing_info: Optional[MoERoutingInfo] = None

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        size_feat_dim = {"nodes_only": 1, "log_nodes_edges": 2, "nodes_edges_density": 3}.get(size_feature_mode, 2)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim + size_feat_dim, router_hidden_dim),
            nn.LayerNorm(router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, n_experts),
        )

        if adaptive_topk:
            self.k_predictor = nn.Sequential(
                nn.Linear(hidden_dim, router_hidden_dim),
                nn.ReLU(),
                nn.Linear(router_hidden_dim, n_experts),
            )

        self.experts = nn.ModuleList()
        for idx in range(n_experts):
            chosen = expert_type
            if secondary_expert_type is not None and idx % 2 == 1:
                chosen = secondary_expert_type
            self.experts.append(
                ExpertBlock(
                    hidden_dim=hidden_dim,
                    out_dim=out_dim,
                    n_layers=n_layers,
                    expert_type=chosen,
                    edge_dim=edge_dim,
                    tag_k=tag_k,
                )
            )

        self.size_centers = nn.Parameter(torch.linspace(0.0, 1.0, steps=n_experts))

        if confidence_gate:
            # Mowst-inspired gate derived from weak expert confidence.
            self.weak_expert = ExpertBlock(hidden_dim, out_dim, max(2, n_layers - 1), expert_type="mlp")
            self.confidence_head = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, 1),
                nn.Sigmoid(),
            )

    def _batch(self, data: HeteroData, n: int) -> Tensor:
        if hasattr(data["bus"], "batch") and data["bus"].batch is not None:
            return data["bus"].batch
        return torch.zeros(n, dtype=torch.long, device=data["bus"].x.device)

    def _size_features(self, batch: Tensor, edge_index: Tensor, num_graphs: int) -> Tuple[Tensor, Tensor]:
        nodes = torch.bincount(batch, minlength=num_graphs).float().clamp_min(1.0)
        edges = torch.bincount(batch[edge_index[0]], minlength=num_graphs).float().clamp_min(0.0)
        log_n = torch.log(nodes)
        log_e = torch.log1p(edges)
        density = edges / (nodes * (nodes - 1.0)).clamp_min(1.0)
        if self.size_feature_mode == "nodes_only":
            f = nodes[:, None]
        elif self.size_feature_mode == "nodes_edges_density":
            f = torch.stack([nodes, edges, density], dim=-1)
        else:
            f = torch.stack([log_n, log_e], dim=-1)
        f = (f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True, unbiased=False) + 1e-6)
        log_n_norm = (log_n - log_n.min()) / (log_n.max() - log_n.min() + 1e-6)
        return f[batch], log_n_norm[batch]

    def _route(self, h: Tensor, size_f: Tensor, size_norm: Tensor, training: bool) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        logits = self.router(torch.cat([h, size_f], dim=-1))
        prior = -((size_norm[:, None] - self.size_centers[None, :]) ** 2) / max(self.size_temperature, 1e-6)
        logits = (1 - self.size_prior_strength) * logits + self.size_prior_strength * prior

        if self.graph_context_strength > 0:
            context = logits.mean(dim=0, keepdim=True)
            logits = logits + self.graph_context_strength * context

        if training and self.noisy_gating_std > 0:
            logits = logits + torch.randn_like(logits) * self.noisy_gating_std

        probs = torch.softmax(logits, dim=-1)

        if self.router_mode == "dense":
            weights = probs
            idx = torch.topk(weights, k=min(self.top_k, self.n_experts), dim=-1).indices
            return logits, probs, weights, idx

        if self.adaptive_topk:
            k_logits = self.k_predictor(h)
            # convert to [1..n_experts], take argmax class index +1
            pred_k = torch.argmax(k_logits, dim=-1) + 1
        else:
            pred_k = torch.full((h.shape[0],), self.top_k, device=h.device, dtype=torch.long)

        weights = torch.zeros_like(probs)
        all_idx = []
        for i in range(h.shape[0]):
            k_i = int(min(max(1, pred_k[i].item()), self.n_experts))
            v, ix = torch.topk(probs[i], k=k_i)
            v = v / (v.sum() + 1e-8)
            weights[i, ix] = v
            if k_i < self.top_k:
                pad = ix.new_full((self.top_k - k_i,), ix[0].item())
                ix = torch.cat([ix, pad], dim=0)
            all_idx.append(ix[: self.top_k])
        idx = torch.stack(all_idx, dim=0)

        if self.expert_dropout > 0 and training:
            keep = (torch.rand_like(weights) > self.expert_dropout).float()
            weights = weights * keep
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        return logits, probs, weights, idx

    def _routing_metrics(self, logits: Tensor, probs: Tensor, weights: Tensor) -> MoERoutingInfo:
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        importance = probs.sum(dim=0)
        importance = importance / (importance.sum() + 1e-8)
        load = (weights > 0).float().sum(dim=0)
        load = load / (load.sum() + 1e-8)
        imp_cv2 = torch.var(importance) / (torch.mean(importance) ** 2 + 1e-8)
        load_cv2 = torch.var(load) / (torch.mean(load) ** 2 + 1e-8)
        lb = self.load_balance_coef * (imp_cv2 + load_cv2)
        z = self.z_loss_coef * torch.mean(torch.logsumexp(logits, dim=-1) ** 2)
        topk = torch.topk(weights, k=min(self.top_k, self.n_experts), dim=-1).indices
        return MoERoutingInfo(
            probs=probs.detach(),
            sparse_weights=weights.detach(),
            topk_indices=topk.detach(),
            entropy=entropy.detach(),
            expert_importance=importance.detach(),
            expert_load=load.detach(),
            load_balance_loss=lb.detach(),
            z_loss=z.detach(),
        )

    def forward(self, data: HeteroData) -> Tensor:
        x = data["bus"].x[:, self.x_start : self.x_end]
        edge_index = data["bus", "branch", "bus"].edge_index
        edge_attr = getattr(data["bus", "branch", "bus"], "edge_attr", None)

        h = self.encoder(x)
        batch = self._batch(data, h.size(0))
        num_graphs = int(batch.max().item()) + 1
        size_f, size_norm = self._size_features(batch, edge_index, num_graphs)

        logits, probs, weights, idx = self._route(h, size_f, size_norm, self.training)

        outs = [e(h, edge_index, edge_attr) for e in self.experts]
        stacked = torch.stack(outs, dim=1)
        y = torch.sum(stacked * weights.unsqueeze(-1), dim=1)

        if self.confidence_gate:
            weak = self.weak_expert(h, edge_index, edge_attr)
            conf = self.confidence_head(torch.abs(weak)).clamp(0.0, 1.0)
            # high confidence => trust weak expert more, low => trust MoE GNN experts
            y = conf * weak + (1.0 - conf) * y

        self.routing_info = self._routing_metrics(logits, probs, weights)
        # Keep compatibility with previous implementation naming.
        self.router_stats = self.routing_info
        return y
