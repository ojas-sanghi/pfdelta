import torch.nn as nn

from core.utils.registry import registry
from core.models.moe_variants.base_components import GenericGraphMoE


@registry.register_model("graph_moe_v23_combo_noisy_hopmix")
class GraphMoeV23ComboNoisyHopmix(nn.Module):
    """Variant `graph_moe_v23_combo_noisy_hopmix` of Graph MoE."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, n_experts: int = 4, top_k: int = 2, **kwargs):
        super().__init__()
        self.model = GenericGraphMoE(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, n_layers=max(4,n_layers), n_experts=6, top_k=2, noisy_gating_std=0.2, expert_conv_types=["graphconv","sage","gcn","tag","graphconv","tag"], size_prior_strength=0.5)

    def forward(self, data):
        return self.model(data)
