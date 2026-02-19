from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from core.models.moe_variants.moe_common import BaseRouter, GraphConvExpert, SharedBusEncoder, TAGExpert, extract_bus_inputs, graph_size_features, sparse_routing, summarize_routing
from core.utils.registry import registry


@registry.register_model("graph_moe_multiscale_combo")
class GraphMoEMultiscaleCombo(nn.Module):
    """Combination pathway: mixed expert families + case-size prior + noisy sparse routing."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, top_k: int = 2, x_start: int = 4, x_end: int = 10):
        super().__init__()
        self.x_start, self.x_end = x_start, x_end
        self.encoder = SharedBusEncoder(in_dim, hidden_dim)
        self.router = BaseRouter(hidden_dim + 3, hidden_dim, 6)
        self.experts = nn.ModuleList([
            GraphConvExpert(hidden_dim, out_dim, n_layers),
            GraphConvExpert(hidden_dim, out_dim, n_layers),
            TAGExpert(hidden_dim, out_dim, hop_k=1),
            TAGExpert(hidden_dim, out_dim, hop_k=2),
            TAGExpert(hidden_dim, out_dim, hop_k=3),
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)),
        ])
        self.size_centers = nn.Parameter(torch.linspace(0, 1, 6))
        self.top_k = top_k
        self.routing = None

    def forward(self, data: HeteroData) -> Tensor:
        x, edge_index, batch = extract_bus_inputs(data, self.x_start, self.x_end)
        h = self.encoder(x)
        size_feat, logn = graph_size_features(batch, edge_index, mode="nodes_edges_density")
        logits = self.router(torch.cat([h, size_feat], dim=-1))
        prior = -((logn.unsqueeze(-1) - self.size_centers.unsqueeze(0)) ** 2)
        logits = 0.75 * logits + 0.25 * prior
        probs, sparse, topk = sparse_routing(logits, self.top_k, noisy_std=0.1, training=self.training)
        ys = []
        for expert in self.experts:
            if isinstance(expert, nn.Sequential):
                ys.append(expert(h))
            else:
                ys.append(expert(h, edge_index))
        y = torch.stack(ys, dim=1)
        self.routing = summarize_routing(probs, sparse, topk)
        return (y * sparse.unsqueeze(-1)).sum(1)
