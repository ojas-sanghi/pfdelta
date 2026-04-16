"""Auto-registered MoE variant: graph_moe_v14_adaptive_topk."""

from core.utils.registry import registry
from core.models.moe_backbones import GraphMoEBackbone


@registry.register_model("graph_moe_v14_adaptive_topk")
class GraphMoeV14AdaptiveTopk(GraphMoEBackbone):
    """graph_moe_v14_adaptive_topk variant with explicit fixed defaults."""

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, **kwargs):
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            expert_type="graphconv",
            secondary_expert_type=None,
            router_mode="topk",
            adaptive_topk=True,
            confidence_gate=False,
            size_feature_mode="log_nodes_edges",
            size_prior_strength=0.35,
            noisy_gating_std=0.0,
            expert_dropout=0.0,
            graph_context_strength=0.0,
            **kwargs,
        )
