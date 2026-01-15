from typing import List

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv

from core.utils.custom_losses import DirichletEnergyLoss
from core.utils.registry import registry


@registry.register_model("graph_conv")
class GraphConvModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int):
        super().__init__()
        self.de = DirichletEnergyLoss()
        self.energies: List[float] = []

        self.start_conv: GraphConv = GraphConv(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GraphConv(hidden_dim, hidden_dim) for _ in range(n_layers - 2)]
        )
        self.end_conv: GraphConv = GraphConv(hidden_dim, out_dim)

    def forward(self, data: HeteroData) -> Tensor:
        x = data["bus"].x[:, 4:10]
        edge_index = data["bus", "branch", "bus"].edge_index
        
        lap = self.de.get_graph_laplacian(edge_index, x.size(0))
        self.energies = []
        
        x = self.start_conv(x=x, edge_index=edge_index)
        energy = self.de.dirichlet_energy(x, lap)
        self.energies.append(energy)
        x = F.relu(x)

        for layer in self.layers:
            x = layer(x=x, edge_index=edge_index)
            energy = self.de.dirichlet_energy(x, lap)
            self.energies.append(energy)
            x = F.relu(x)
            
        x = self.end_conv(x=x, edge_index=edge_index)
        energy = self.de.dirichlet_energy(x, lap)
        self.energies.append(energy)

        return x
