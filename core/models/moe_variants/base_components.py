from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GATConv, GraphConv, SAGEConv, TAGConv


@dataclass
class MoEDiagnostics:
    probs: Tensor
    weights: Tensor
    topk: Tensor
    entropy: Tensor
    importance: Tensor
    load: Tensor
    aux_loss: Tensor


CONV_MAP = {
    "graphconv": GraphConv,
    "sage": SAGEConv,
    "gcn": GCNConv,
    "tag": TAGConv,
    "gat": GATConv,
}


class ExpertTower(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, n_layers: int, conv_type: str = "graphconv"):
        super().__init__()
        conv = CONV_MAP[conv_type]
        self.conv_type = conv_type
        if conv_type == "tag":
            self.start = conv(hidden_dim, hidden_dim, K=2)
            self.mid = nn.ModuleList([conv(hidden_dim, hidden_dim, K=2) for _ in range(max(0, n_layers - 2))])
            self.end = conv(hidden_dim, out_dim, K=2)
        elif conv_type == "gat":
            self.start = conv(hidden_dim, hidden_dim // 2, heads=2, concat=True)
            self.mid = nn.ModuleList([conv(hidden_dim, hidden_dim // 2, heads=2, concat=True) for _ in range(max(0, n_layers - 2))])
            self.end = conv(hidden_dim, out_dim, heads=1, concat=True)
        else:
            self.start = conv(hidden_dim, hidden_dim)
            self.mid = nn.ModuleList([conv(hidden_dim, hidden_dim) for _ in range(max(0, n_layers - 2))])
            self.end = conv(hidden_dim, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.start(x, edge_index))
        for layer in self.mid:
            x = F.relu(layer(x, edge_index))
        return self.end(x, edge_index)


class GenericGraphMoE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        n_experts: int = 4,
        top_k: int = 2,
        expert_conv_types: Optional[List[str]] = None,
        gating_mode: str = "sparse",
        noisy_gating_std: float = 0.0,
        use_size_features: bool = True,
        size_prior_strength: float = 0.35,
        x_start: int = 4,
        x_end: int = 10,
        router_hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gating_mode = gating_mode
        self.noisy_gating_std = noisy_gating_std
        self.use_size_features = use_size_features
        self.size_prior_strength = size_prior_strength
        self.x_start = x_start
        self.x_end = x_end

        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        router_in = hidden_dim + (3 if use_size_features else 0)
        self.router = nn.Sequential(nn.Linear(router_in, router_hidden_dim), nn.ReLU(), nn.Linear(router_hidden_dim, n_experts))

        if expert_conv_types is None:
            expert_conv_types = ["graphconv"] * n_experts
        assert len(expert_conv_types) == n_experts
        self.experts = nn.ModuleList(
            [ExpertTower(hidden_dim, out_dim, n_layers, ctype) for ctype in expert_conv_types]
        )
        self.size_centers = nn.Parameter(torch.linspace(0.0, 1.0, steps=n_experts))
        self.diagnostics: Optional[MoEDiagnostics] = None

    def _batch(self, data: HeteroData, n_nodes: int) -> Tensor:
        if hasattr(data["bus"], "batch") and data["bus"].batch is not None:
            return data["bus"].batch
        return torch.zeros(n_nodes, dtype=torch.long, device=data["bus"].x.device)

    def _size_feats(self, batch: Tensor, edge_index: Tensor, num_graphs: int) -> Tuple[Tensor, Tensor]:
        n = torch.bincount(batch, minlength=num_graphs).float().clamp_min(1.0)
        e = torch.bincount(batch[edge_index[0]], minlength=num_graphs).float()
        density = e / (n * (n - 1.0)).clamp_min(1.0)
        log_n = torch.log(n)
        log_n_norm = (log_n - log_n.min()) / (log_n.max() - log_n.min() + 1e-6)
        graph_feats = torch.stack([n, e, density], dim=-1)
        graph_feats = (graph_feats - graph_feats.mean(0, keepdim=True)) / (graph_feats.std(0, keepdim=True, unbiased=False) + 1e-6)
        return graph_feats[batch], log_n_norm[batch]

    def _gate(self, logits: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.training and self.noisy_gating_std > 0:
            logits = logits + torch.randn_like(logits) * self.noisy_gating_std
        probs = F.softmax(logits, dim=-1)
        if self.gating_mode == "dense":
            return probs, probs, torch.arange(self.n_experts, device=probs.device).unsqueeze(0).repeat(probs.size(0),1)
        if self.gating_mode == "hard":
            idx = probs.argmax(dim=-1, keepdim=True)
            weights = torch.zeros_like(probs).scatter_(1, idx, 1.0)
            return probs, weights, idx
        vals, idx = torch.topk(probs, k=self.top_k, dim=-1)
        vals = vals / (vals.sum(dim=-1, keepdim=True) + 1e-8)
        weights = torch.zeros_like(probs).scatter_(1, idx, vals)
        return probs, weights, idx

    def forward(self, data: HeteroData) -> Tensor:
        x = data["bus"].x[:, self.x_start:self.x_end]
        edge_index = data["bus", "branch", "bus"].edge_index
        h = self.encoder(x)
        batch = self._batch(data, h.size(0))
        num_graphs = int(batch.max().item()) + 1

        if self.use_size_features:
            size_feats, log_n_norm = self._size_feats(batch, edge_index, num_graphs)
            logits = self.router(torch.cat([h, size_feats], dim=-1))
            prior = -((log_n_norm.unsqueeze(-1) - self.size_centers.unsqueeze(0)) ** 2)
            logits = (1 - self.size_prior_strength) * logits + self.size_prior_strength * prior
        else:
            logits = self.router(h)

        probs, weights, topk = self._gate(logits)
        outs = torch.stack([ex(h, edge_index) for ex in self.experts], dim=1)
        pred = torch.sum(outs * weights.unsqueeze(-1), dim=1)

        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        importance = probs.mean(dim=0)
        load = (weights > 0).float().mean(dim=0)
        aux = torch.var(importance) + torch.var(load) + 1e-3 * torch.mean(torch.logsumexp(logits, dim=-1) ** 2)
        self.diagnostics = MoEDiagnostics(
            probs=probs.detach(),
            weights=weights.detach(),
            topk=topk.detach(),
            entropy=entropy.detach(),
            importance=importance.detach(),
            load=load.detach(),
            aux_loss=aux.detach(),
        )
        return pred


class MowstStyleMoE(nn.Module):
    """Weak/strong MoE inspired by Mowst: MLP weak expert + GNN strong expert."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, learned_confidence: bool = False, x_start: int = 4, x_end: int = 10):
        super().__init__()
        self.x_start = x_start
        self.x_end = x_end
        self.weak = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
        self.strong_encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.strong = ExpertTower(hidden_dim, out_dim, n_layers, "graphconv")
        self.learned_confidence = learned_confidence
        if learned_confidence:
            self.conf_net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        self.diagnostics = None

    def _dispersion(self, logits: Tensor) -> Tensor:
        probs = F.softmax(logits, dim=-1)
        var = torch.var(probs, dim=-1, unbiased=False)
        ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        ent = ent / torch.log(torch.tensor(float(probs.size(-1)), device=probs.device))
        return var, 1.0 - ent

    def forward(self, data: HeteroData) -> Tensor:
        x = data["bus"].x[:, self.x_start:self.x_end]
        edge_index = data["bus", "branch", "bus"].edge_index
        weak_out = self.weak(x)
        strong_out = self.strong(self.strong_encoder(x), edge_index)

        var, conf_proxy = self._dispersion(weak_out)
        if self.learned_confidence:
            conf = self.conf_net(torch.stack([var, conf_proxy], dim=-1)).squeeze(-1)
        else:
            conf = torch.clamp(0.5 * (var + conf_proxy), 0.0, 1.0)

        out = conf.unsqueeze(-1) * weak_out + (1 - conf.unsqueeze(-1)) * strong_out
        self.diagnostics = {
            "confidence": conf.detach(),
            "weak_weight": conf.mean().detach(),
            "strong_weight": (1 - conf).mean().detach(),
        }
        return out
