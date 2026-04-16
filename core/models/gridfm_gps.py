import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, GPSConv

from core.utils.registry import registry


@registry.register_model("gridfm_gps")
class GridFMGPSTransformer(nn.Module):
    """PFDelta-native implementation of the GridFM v0.2 GPS model."""

    def __init__(
        self,
        input_dim=9,
        hidden_size=256,
        output_dim=6,
        edge_dim=2,
        pe_dim=20,
        num_layers=8,
        attention_head=8,
        dropout=0.1,
        mask_dim=6,
        mask_value=0.0,
        learn_mask=False,
    ):
        super().__init__()
        if pe_dim >= hidden_size:
            raise ValueError("pe_dim must be smaller than hidden_size")

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.pe_dim = pe_dim

        if learn_mask:
            self.mask_value = nn.Parameter(torch.randn(mask_dim) + mask_value, requires_grad=True)
        else:
            self.mask_value = nn.Parameter(torch.zeros(mask_dim) + mask_value, requires_grad=False)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size - pe_dim),
            nn.LeakyReLU(),
        )
        self.input_norm = nn.BatchNorm1d(hidden_size - pe_dim)
        self.pe_norm = nn.BatchNorm1d(pe_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
            )
            self.layers.append(
                nn.ModuleDict(
                    {
                        "conv": GPSConv(
                            channels=hidden_size,
                            conv=GINEConv(nn=mlp, edge_dim=edge_dim),
                            heads=attention_head,
                            dropout=dropout,
                        ),
                        "norm": nn.BatchNorm1d(hidden_size),
                    }
                )
            )

        self.pre_decoder_norm = nn.BatchNorm1d(hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, data):
        x = data.x
        pe = data.pe
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        x_pe = self.pe_norm(pe)
        x = self.encoder(x)
        x = self.input_norm(x)
        x = torch.cat((x, x_pe), dim=1)

        for layer in self.layers:
            x = layer["conv"](
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
            )
            x = layer["norm"](x)

        x = self.pre_decoder_norm(x)
        x = self.decoder(x)
        return x
