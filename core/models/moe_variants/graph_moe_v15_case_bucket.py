import torch.nn as nn

from core.utils.registry import registry
from core.models.moe_variants.base_components import GenericGraphMoE


@registry.register_model("graph_moe_v15_case_bucket")
class GraphMoeV15CaseBucket(nn.Module):
    """Variant `graph_moe_v15_case_bucket` of Graph MoE."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, n_experts: int = 4, top_k: int = 2, **kwargs):
        super().__init__()
        self.model = GenericGraphMoE(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, n_layers=n_layers, n_experts=6, top_k=1, gating_mode="hard", size_prior_strength=0.9)

    def forward(self, data):
        return self.model(data)
