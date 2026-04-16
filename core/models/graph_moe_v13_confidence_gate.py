import torch.nn as nn

from core.models.moe_variant_base import ConfigurableGraphMoE
from core.utils.registry import registry


@registry.register_model("graph_moe_v13_confidence_gate")
class GraphMoeV13ConfidenceGate(nn.Module):
    """graph_moe_v13_confidence_gate: auto-generated MoE variant for systematic experimentation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int):
        super().__init__()
        self.model = ConfigurableGraphMoE(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            router_mode='combined',
        gating_mode='topk',
        weak_strong=True,
        confidence_gate=True,
        )

    def forward(self, data):
        return self.model(data)
