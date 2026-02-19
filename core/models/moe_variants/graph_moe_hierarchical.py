from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from core.models.moe_variants.moe_common import BaseRouter, GraphConvExpert, SharedBusEncoder, extract_bus_inputs, graph_size_features, sparse_routing
from core.utils.registry import registry


@registry.register_model("graph_moe_hierarchical")
class GraphMoEHierarchical(nn.Module):
    """Hierarchical MoE: graph-level coarse routing then node-level fine routing."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, n_groups: int = 2, experts_per_group: int = 2, x_start: int = 4, x_end: int = 10):
        super().__init__()
        self.x_start, self.x_end = x_start, x_end
        self.encoder = SharedBusEncoder(in_dim, hidden_dim)
        self.group_router = BaseRouter(2, hidden_dim, n_groups)
        self.node_router = BaseRouter(hidden_dim + 2, hidden_dim, experts_per_group)
        self.experts = nn.ModuleList(
            nn.ModuleList(GraphConvExpert(hidden_dim, out_dim, n_layers) for _ in range(experts_per_group))
            for _ in range(n_groups)
        )
        self.last_group_probs = None

    def forward(self, data: HeteroData) -> Tensor:
        x, edge_index, batch = extract_bus_inputs(data, self.x_start, self.x_end)
        h = self.encoder(x)
        size_feat, _ = graph_size_features(batch, edge_index)

        num_graphs = int(batch.max().item()) + 1
        graph_stats = []
        for gid in range(num_graphs):
            mask = batch == gid
            graph_stats.append(torch.stack([x[mask].mean(), x[mask].std(unbiased=False)]))
        graph_stats = torch.stack(graph_stats)
        group_probs = torch.softmax(self.group_router(graph_stats), dim=-1)
        node_group_probs = group_probs[batch]

        node_logits = self.node_router(torch.cat([h, size_feat], dim=-1))
        _, sparse, _ = sparse_routing(node_logits, top_k=1, training=self.training)

        outputs = []
        for g, experts in enumerate(self.experts):
            y = torch.stack([e(h, edge_index) for e in experts], dim=1)
            y = (y * sparse.unsqueeze(-1)).sum(1)
            outputs.append(y * node_group_probs[:, g:g+1])
        self.last_group_probs = group_probs.detach()
        return torch.stack(outputs, dim=0).sum(0)
