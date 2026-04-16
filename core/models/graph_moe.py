"""Default graph MoE model (upgraded) for PFDelta."""

from core.utils.registry import registry
from core.models.moe_backbones import GraphMoEBackbone


@registry.register_model("graph_moe")
class GraphMoE(GraphMoEBackbone):
    """Default case-size-aware sparse graph MoE used as primary baseline."""

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, **kwargs):
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            expert_type="graphconv",
            router_mode="topk",
            size_feature_mode="log_nodes_edges",
            size_prior_strength=0.35,
            noisy_gating_std=0.05,
            adaptive_topk=False,
            confidence_gate=False,
            **kwargs,
        )
