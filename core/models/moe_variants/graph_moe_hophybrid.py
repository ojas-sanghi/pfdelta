from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from core.models.moe_variants.moe_common import (
    BaseRouter,
    SharedBusEncoder,
    TAGExpert,
    extract_bus_inputs,
    graph_size_features,
    sparse_routing,
    summarize_routing,
)
from core.utils.registry import registry


@registry.register_model("graph_moe_hophybrid")
class GraphMoEHopHybrid(nn.Module):
    """Graph MoE where experts differ by TAGConv hop radius (K)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_experts: int = 4, top_k: int = 2, hop_ks: list[int] | None = None, x_start: int = 4, x_end: int = 10):
        super().__init__()
        self.x_start, self.x_end = x_start, x_end
        self.encoder = SharedBusEncoder(in_dim, hidden_dim)
        self.router = BaseRouter(hidden_dim + 3, hidden_dim, n_experts)
        hop_ks = hop_ks or [1, 1, 2, 3]
        self.experts = nn.ModuleList(TAGExpert(hidden_dim, out_dim, hop_ks[i % len(hop_ks)]) for i in range(n_experts))
        self.top_k = top_k
        self.routing = None

    def forward(self, data: HeteroData) -> Tensor:
        x, edge_index, batch = extract_bus_inputs(data, self.x_start, self.x_end)
        h = self.encoder(x)
        size_feat, _ = graph_size_features(batch, edge_index, mode="nodes_edges_density")
        logits = self.router(torch.cat([h, size_feat], dim=-1))
        probs, sparse, topk = sparse_routing(logits, self.top_k, noisy_std=0.05, training=self.training)
        ys = torch.stack([expert(h, edge_index) for expert in self.experts], dim=1)
        out = (ys * sparse.unsqueeze(-1)).sum(1)
        self.routing = summarize_routing(probs, sparse, topk)
        return out
