from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from core.models.moe_variants.moe_common import GraphConvExpert, SharedBusEncoder, extract_bus_inputs, graph_size_features, sparse_routing, summarize_routing
from core.utils.registry import registry


@registry.register_model("graph_moe_attention_router")
class GraphMoEAttentionRouter(nn.Module):
    """MoE with self-attentive router over node hidden + size features."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, n_experts: int = 4, top_k: int = 2, x_start: int = 4, x_end: int = 10):
        super().__init__()
        self.x_start, self.x_end = x_start, x_end
        self.encoder = SharedBusEncoder(in_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim + 2, hidden_dim)
        self.k = nn.Linear(hidden_dim + 2, hidden_dim)
        self.v = nn.Linear(hidden_dim + 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_experts)
        self.experts = nn.ModuleList(GraphConvExpert(hidden_dim, out_dim, n_layers) for _ in range(n_experts))
        self.top_k = top_k
        self.routing = None

    def forward(self, data: HeteroData) -> Tensor:
        x, edge_index, batch = extract_bus_inputs(data, self.x_start, self.x_end)
        h = self.encoder(x)
        size_feat, _ = graph_size_features(batch, edge_index)
        z = torch.cat([h, size_feat], dim=-1)
        attn = torch.softmax((self.q(z) @ self.k(z).T) / (z.size(-1) ** 0.5), dim=-1)
        fused = attn @ self.v(z)
        logits = self.out(fused)
        probs, sparse, topk = sparse_routing(logits, self.top_k, training=self.training)
        y = torch.stack([e(h, edge_index) for e in self.experts], dim=1)
        self.routing = summarize_routing(probs, sparse, topk)
        return (y * sparse.unsqueeze(-1)).sum(1)
