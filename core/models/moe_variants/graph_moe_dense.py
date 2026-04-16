from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from core.models.moe_variants.moe_common import BaseRouter, GraphConvExpert, SharedBusEncoder, extract_bus_inputs, graph_size_features
from core.utils.registry import registry


@registry.register_model("graph_moe_dense")
class GraphMoEDense(nn.Module):
    """Dense MoE where all experts contribute via softmax routing weights."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, n_experts: int = 6, x_start: int = 4, x_end: int = 10):
        super().__init__()
        self.x_start, self.x_end = x_start, x_end
        self.encoder = SharedBusEncoder(in_dim, hidden_dim)
        self.router = BaseRouter(hidden_dim + 1, hidden_dim, n_experts)
        self.experts = nn.ModuleList(GraphConvExpert(hidden_dim, out_dim, n_layers) for _ in range(n_experts))
        self.mean_entropy = None

    def forward(self, data: HeteroData) -> Tensor:
        x, edge_index, batch = extract_bus_inputs(data, self.x_start, self.x_end)
        h = self.encoder(x)
        size_feat, _ = graph_size_features(batch, edge_index, mode="nodes_only")
        probs = torch.softmax(self.router(torch.cat([h, size_feat], dim=-1)), dim=-1)
        y = torch.stack([e(h, edge_index) for e in self.experts], dim=1)
        out = (y * probs.unsqueeze(-1)).sum(1)
        self.mean_entropy = (-(probs * torch.log(probs + 1e-8)).sum(-1).mean()).detach()
        return out
