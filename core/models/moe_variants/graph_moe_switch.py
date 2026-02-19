from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from core.models.moe_variants.moe_common import BaseRouter, GraphConvExpert, SharedBusEncoder, extract_bus_inputs, graph_size_features
from core.utils.registry import registry


@registry.register_model("graph_moe_switch")
class GraphMoESwitch(nn.Module):
    """Switch-style MoE using top-1 expert selection for each node."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, n_experts: int = 4, x_start: int = 4, x_end: int = 10):
        super().__init__()
        self.x_start, self.x_end = x_start, x_end
        self.encoder = SharedBusEncoder(in_dim, hidden_dim)
        self.router = BaseRouter(hidden_dim + 2, hidden_dim, n_experts)
        self.experts = nn.ModuleList(GraphConvExpert(hidden_dim, out_dim, n_layers) for _ in range(n_experts))
        self.expert_histogram = None

    def forward(self, data: HeteroData) -> Tensor:
        x, edge_index, batch = extract_bus_inputs(data, self.x_start, self.x_end)
        h = self.encoder(x)
        size_feat, _ = graph_size_features(batch, edge_index)
        logits = self.router(torch.cat([h, size_feat], dim=-1))
        idx = torch.argmax(logits, dim=-1)
        y_all = torch.stack([e(h, edge_index) for e in self.experts], dim=1)
        out = y_all[torch.arange(y_all.size(0), device=y_all.device), idx]
        self.expert_histogram = torch.bincount(idx, minlength=len(self.experts)).detach()
        return out
