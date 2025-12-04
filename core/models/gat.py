import torch.nn as nn
from torch_geometric.nn import GAT
from torch import Tensor
from torch_geometric.data import HeteroData

from core.utils.registry import registry


@registry.register_model("gat")
class GATModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, n_layers: int, out_channels: int):
        super().__init__()
        self.gat: GAT = GAT(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=n_layers, v2=True)

    def forward(self, data: HeteroData) -> Tensor:
        x = data["bus"].x[:, 4:10]
        edge_index = data["bus", "branch", "bus"].edge_index

        x = self.gat(x=x, edge_index=edge_index)
        return x
