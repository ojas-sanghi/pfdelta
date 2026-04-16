import torch.nn as nn

from core.models.moe_variant_base import ConfigurableGraphMoE
from core.utils.registry import registry


@registry.register_model("graph_moe_v18_many_experts")
class GraphMoeV18ManyExperts(nn.Module):
    """graph_moe_v18_many_experts: auto-generated MoE variant for systematic experimentation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int):
        super().__init__()
        self.model = ConfigurableGraphMoE(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            router_mode='combined',
        gating_mode='topk',
        n_experts=6,
        top_k=2,
        expert_kinds=['graphconv', 'sage', 'tag', 'graphconv', 'tag', 'sage'],
        )

    def forward(self, data):
        return self.model(data)
