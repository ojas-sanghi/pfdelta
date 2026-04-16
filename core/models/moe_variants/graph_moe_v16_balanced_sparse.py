import torch.nn as nn

from core.utils.registry import registry
from core.models.moe_variants.base_components import GenericGraphMoE


@registry.register_model("graph_moe_v16_balanced_sparse")
class GraphMoeV16BalancedSparse(nn.Module):
    """Variant `graph_moe_v16_balanced_sparse` of Graph MoE."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, n_experts: int = 4, top_k: int = 2, **kwargs):
        super().__init__()
        self.model = GenericGraphMoE(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, n_layers=n_layers, n_experts=n_experts, top_k=min(3,n_experts), noisy_gating_std=0.05)

    def forward(self, data):
        return self.model(data)
