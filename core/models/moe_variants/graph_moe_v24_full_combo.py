"""Auto-registered MoE variant: graph_moe_v24_full_combo."""

from core.utils.registry import registry
from core.models.moe_backbones import GraphMoEBackbone


@registry.register_model("graph_moe_v24_full_combo")
class GraphMoeV24FullCombo(GraphMoEBackbone):
    """graph_moe_v24_full_combo variant with explicit fixed defaults."""

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, **kwargs):
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            expert_type="graphconv",
            secondary_expert_type='tag',
            router_mode="topk",
            adaptive_topk=True,
            confidence_gate=True,
            size_feature_mode="nodes_edges_density",
            size_prior_strength=0.35,
            noisy_gating_std=0.1,
            expert_dropout=0.15,
            graph_context_strength=0.0,
            **kwargs,
        )
