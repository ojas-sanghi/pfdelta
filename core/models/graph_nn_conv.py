import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv, NNConv

from core.utils.registry import registry


@registry.register_model("graph_nn_conv")
class GraphNNConvModel(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, out_dim, n_layers):
        super().__init__()

        def make_edge_net(in_channels, out_channels):
            return nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_channels * out_channels),
            )

        self.start_conv = NNConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            nn=make_edge_net(in_dim, hidden_dim),
        )

        self.layers = nn.ModuleList()
        for _ in range(n_layers - 2):
            self.layers.append(
                NNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    nn=make_edge_net(hidden_dim, hidden_dim),
                )
            )

        self.end_conv = NNConv(
            in_channels=hidden_dim,
            out_channels=out_dim,
            nn=make_edge_net(hidden_dim, out_dim),
        )

    def forward(self, data: HeteroData):
        x = data["bus"].x[:, 4:10]
        edge_index = data["bus", "branch", "bus"].edge_index
        edge_attr = data["bus", "branch", "bus"].edge_attr

        x = self.start_conv(x, edge_index, edge_attr)
        x = F.relu(x)
        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        x = self.end_conv(x, edge_index, edge_attr)
        return x