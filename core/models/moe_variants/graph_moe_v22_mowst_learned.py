import torch.nn as nn

from core.utils.registry import registry
from core.models.moe_variants.base_components import MowstStyleMoE


@registry.register_model("graph_moe_v22_mowst_learned")
class GraphMoeV22MowstLearned(nn.Module):
    """Mowst-style variant `graph_moe_v22_mowst_learned`."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, **kwargs):
        super().__init__()
        self.model = MowstStyleMoE(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, n_layers=n_layers, learned_confidence=True)

    def forward(self, data):
        return self.model(data)
