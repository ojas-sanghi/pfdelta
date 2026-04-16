"""Auto-registered MoE variant: graph_moe_v19_dense_confidence."""

from core.utils.registry import registry
from core.models.moe_backbones import GraphMoEBackbone


@registry.register_model("graph_moe_v19_dense_confidence")
class GraphMoeV19DenseConfidence(GraphMoEBackbone):
    """graph_moe_v19_dense_confidence variant with explicit fixed defaults."""

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, **kwargs):
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            expert_type="graphconv",
            secondary_expert_type=None,
            router_mode="dense",
            adaptive_topk=False,
            confidence_gate=True,
            size_feature_mode="log_nodes_edges",
            size_prior_strength=0.25,
            noisy_gating_std=0.0,
            expert_dropout=0.0,
            graph_context_strength=0.0,
            **kwargs,
        )
