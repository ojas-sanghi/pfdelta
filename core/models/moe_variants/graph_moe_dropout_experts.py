from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from core.models.moe_variants.moe_common import BaseRouter, GraphConvExpert, SharedBusEncoder, extract_bus_inputs, graph_size_features, sparse_routing
from core.utils.registry import registry


@registry.register_model("graph_moe_dropout_experts")
class GraphMoEDropoutExperts(nn.Module):
    """Sparse MoE with expert dropout for robustness and anti-collapse."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, n_experts: int = 6, top_k: int = 2, expert_dropout: float = 0.1, x_start: int = 4, x_end: int = 10):
        super().__init__()
        self.x_start, self.x_end = x_start, x_end
        self.encoder = SharedBusEncoder(in_dim, hidden_dim)
        self.router = BaseRouter(hidden_dim + 2, hidden_dim, n_experts)
        self.experts = nn.ModuleList(GraphConvExpert(hidden_dim, out_dim, n_layers) for _ in range(n_experts))
        self.top_k = top_k
        self.expert_dropout = expert_dropout

    def forward(self, data: HeteroData) -> Tensor:
        x, edge_index, batch = extract_bus_inputs(data, self.x_start, self.x_end)
        h = self.encoder(x)
        size_feat, _ = graph_size_features(batch, edge_index)
        logits = self.router(torch.cat([h, size_feat], dim=-1))
        if self.training and self.expert_dropout > 0:
            mask = torch.bernoulli(torch.full_like(logits, 1 - self.expert_dropout))
            logits = logits + torch.log(mask + 1e-8)
        _, sparse, _ = sparse_routing(logits, self.top_k, training=self.training)
        y = torch.stack([e(h, edge_index) for e in self.experts], dim=1)
        return (y * sparse.unsqueeze(-1)).sum(1)
