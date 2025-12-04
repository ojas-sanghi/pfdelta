import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch import Tensor
from torch_geometric.data import HeteroData

from core.utils.registry import registry


@registry.register_model("sageconv")
class SAGEConvModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, n_layers: int, out_channels: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(n_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
        self.layers.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, data: HeteroData) -> Tensor:
        x = data["bus"].x[:, 4:10]
        edge_index = data["bus", "branch", "bus"].edge_index
        
        for conv in self.layers:
            x = conv(x, edge_index)
            x = nn.functional.relu(x)
        return x