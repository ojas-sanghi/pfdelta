"""Auto-registered MoE variant: graph_moe_v17_confidence_noisy."""

from core.utils.registry import registry
from core.models.moe_backbones import GraphMoEBackbone


@registry.register_model("graph_moe_v17_confidence_noisy")
class GraphMoeV17ConfidenceNoisy(GraphMoEBackbone):
    """graph_moe_v17_confidence_noisy variant with explicit fixed defaults."""

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, **kwargs):
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            expert_type="graphconv",
            secondary_expert_type=None,
            router_mode="topk",
            adaptive_topk=False,
            confidence_gate=True,
            size_feature_mode="log_nodes_edges",
            size_prior_strength=0.25,
            noisy_gating_std=0.1,
            expert_dropout=0.0,
            graph_context_strength=0.0,
            **kwargs,
        )
