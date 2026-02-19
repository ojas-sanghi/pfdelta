from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from core.models.moe_variants.moe_common import GraphConvExpert, SharedBusEncoder, extract_bus_inputs, graph_size_features
from core.utils.registry import registry


@registry.register_model("graph_moe_prioronly")
class GraphMoEPriorOnly(nn.Module):
    """MoE with deterministic case-size routing (no learned router)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, n_experts: int = 4, x_start: int = 4, x_end: int = 10):
        super().__init__()
        self.x_start, self.x_end = x_start, x_end
        self.encoder = SharedBusEncoder(in_dim, hidden_dim)
        self.experts = nn.ModuleList(GraphConvExpert(hidden_dim, out_dim, n_layers) for _ in range(n_experts))
        self.centers = nn.Parameter(torch.linspace(0, 1, n_experts))

    def forward(self, data: HeteroData) -> Tensor:
        x, edge_index, batch = extract_bus_inputs(data, self.x_start, self.x_end)
        h = self.encoder(x)
        _, logn = graph_size_features(batch, edge_index)
        probs = torch.softmax(-((logn.unsqueeze(-1) - self.centers.unsqueeze(0)) ** 2), dim=-1)
        y = torch.stack([e(h, edge_index) for e in self.experts], dim=1)
        return (y * probs.unsqueeze(-1)).sum(1)
