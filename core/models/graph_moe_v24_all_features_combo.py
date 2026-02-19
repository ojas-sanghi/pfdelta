import torch.nn as nn

from core.models.moe_variant_base import ConfigurableGraphMoE
from core.utils.registry import registry


@registry.register_model("graph_moe_v24_all_features_combo")
class GraphMoeV24AllFeaturesCombo(nn.Module):
    """graph_moe_v24_all_features_combo: auto-generated MoE variant for systematic experimentation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int):
        super().__init__()
        self.model = ConfigurableGraphMoE(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            router_mode='dual',
        gating_mode='gumbel',
        hierarchical=True,
        weak_strong=True,
        confidence_gate=True,
        noisy_std=0.1,
        temperature=0.85,
        expert_kinds=['graphconv', 'tag', 'sage', 'tag'],
        )

    def forward(self, data):
        return self.model(data)
