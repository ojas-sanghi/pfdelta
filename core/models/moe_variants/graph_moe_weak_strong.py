from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from core.models.moe_variants.moe_common import GraphConvExpert, extract_bus_inputs
from core.utils.registry import registry


@registry.register_model("graph_moe_weak_strong")
class GraphMoEWeakStrong(nn.Module):
    """Mowst-inspired weak/strong MoE: MLP weak expert + GNN strong expert."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, x_start: int = 4, x_end: int = 10):
        super().__init__()
        self.x_start, self.x_end = x_start, x_end
        self.weak = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
        self.strong = GraphConvExpert(hidden_dim, out_dim, n_layers)
        self.pre = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.confidence_mlp = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
        self.routing_confidence = None

    def forward(self, data: HeteroData) -> Tensor:
        x, edge_index, _ = extract_bus_inputs(data, self.x_start, self.x_end)
        weak_out = self.weak(x)
        h = self.pre(x)
        strong_out = self.strong(h, edge_index)
        probs = torch.softmax(weak_out, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1, keepdim=True)
        dispersion = probs.var(-1, keepdim=True)
        conf = torch.sigmoid(self.confidence_mlp(torch.cat([entropy, dispersion], dim=-1)))
        self.routing_confidence = conf.detach().mean()
        return conf * weak_out + (1.0 - conf) * strong_out
