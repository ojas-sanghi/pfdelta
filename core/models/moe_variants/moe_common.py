from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv, TAGConv


@dataclass
class MoERoutingSummary:
    probs: Tensor
    sparse_weights: Tensor
    topk: Tensor
    entropy: Tensor
    importance: Tensor
    load: Tensor


class SharedBusEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GraphConvExpert(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, n_layers: int):
        super().__init__()
        self.start = GraphConv(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(GraphConv(hidden_dim, hidden_dim) for _ in range(max(0, n_layers - 2)))
        self.end = GraphConv(hidden_dim, out_dim)

    def forward(self, h: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.start(h, edge_index))
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        return self.end(x, edge_index)


class TAGExpert(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, hop_k: int):
        super().__init__()
        self.tag1 = TAGConv(hidden_dim, hidden_dim, K=hop_k)
        self.tag2 = TAGConv(hidden_dim, out_dim, K=hop_k)

    def forward(self, h: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.tag1(h, edge_index))
        return self.tag2(x, edge_index)


class BaseRouter(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_experts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def extract_bus_inputs(data: HeteroData, x_start: int, x_end: int) -> Tuple[Tensor, Tensor, Tensor]:
    x = data["bus"].x[:, x_start:x_end]
    edge_index = data["bus", "branch", "bus"].edge_index
    if hasattr(data["bus"], "batch") and data["bus"].batch is not None:
        batch = data["bus"].batch
    else:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    return x, edge_index, batch


def graph_size_features(batch: Tensor, edge_index: Tensor, mode: str = "log_nodes_edges") -> Tuple[Tensor, Tensor]:
    num_graphs = int(batch.max().item()) + 1
    n = torch.bincount(batch, minlength=num_graphs).float().clamp_min(1)
    e = torch.bincount(batch[edge_index[0]], minlength=num_graphs).float()
    log_n = torch.log(n)
    log_e = torch.log1p(e)
    density = e / (n * (n - 1).clamp_min(1))

    if mode == "nodes_only":
        g = n.unsqueeze(-1)
    elif mode == "nodes_edges_density":
        g = torch.stack([n, e, density], dim=-1)
    else:
        g = torch.stack([log_n, log_e], dim=-1)

    g = (g - g.mean(0, keepdim=True)) / (g.std(0, keepdim=True, unbiased=False) + 1e-6)
    node_features = g[batch]
    log_n_norm = (log_n - log_n.min()) / (log_n.max() - log_n.min() + 1e-6)
    return node_features, log_n_norm[batch]


def sparse_routing(logits: Tensor, top_k: int, noisy_std: float = 0.0, training: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    if training and noisy_std > 0:
        logits = logits + torch.randn_like(logits) * noisy_std
    probs = F.softmax(logits, dim=-1)
    vals, idx = torch.topk(probs, k=top_k, dim=-1)
    vals = vals / (vals.sum(-1, keepdim=True) + 1e-8)
    sparse = torch.zeros_like(probs)
    sparse.scatter_(1, idx, vals)
    return probs, sparse, idx


def summarize_routing(probs: Tensor, sparse: Tensor, topk: Tensor) -> MoERoutingSummary:
    entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean().detach()
    importance = probs.sum(0)
    importance = (importance / (importance.sum() + 1e-8)).detach()
    load = (sparse > 0).float().sum(0)
    load = (load / (load.sum() + 1e-8)).detach()
    return MoERoutingSummary(
        probs=probs.detach(),
        sparse_weights=sparse.detach(),
        topk=topk.detach(),
        entropy=entropy,
        importance=importance,
        load=load,
    )
