from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINEConv, GPSConv

from core.utils.registry import registry


@registry.register_model("gridfm_gps")
class GridFMGPSTransformer(nn.Module):
    """PFΔ-native implementation of GridFM v0.2 GPSTransformer."""

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 256,
        output_dim: int = 6,
        edge_dim: int = 2,
        pe_dim: int = 20,
        num_layers: int = 8,
        heads: int = 8,
        dropout: float = 0.1,
        mask_dim: int = 6,
        mask_value: float = 0.0,
        learn_mask: bool = False,
    ):
        super().__init__()

        if pe_dim >= hidden_dim:
            raise ValueError("pe_dim must be strictly smaller than hidden_dim")

        self.mask_dim = mask_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim - pe_dim),
            nn.LeakyReLU(),
        )
        self.input_norm = nn.BatchNorm1d(hidden_dim - pe_dim)
        self.pe_norm = nn.BatchNorm1d(pe_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
            )
            conv = GPSConv(
                channels=hidden_dim,
                conv=GINEConv(nn=mlp, edge_dim=edge_dim),
                heads=heads,
                dropout=dropout,
            )
            self.layers.append(nn.ModuleDict({"conv": conv, "norm": nn.BatchNorm1d(hidden_dim)}))

        self.pre_decoder_norm = nn.BatchNorm1d(hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        init_mask = torch.randn(mask_dim) + mask_value
        self.mask_value = nn.Parameter(init_mask, requires_grad=learn_mask)

    def forward(self, data):
        x = data.x
        pe = data.pe
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        mask = getattr(data, "mask", None)
        if mask is not None and mask.numel() > 0:
            x = x.clone()
            masked_block = x[:, 3 : 3 + self.mask_dim]
            replacement = self.mask_value.unsqueeze(0).expand_as(masked_block)
            masked_block[mask] = replacement[mask]
            x[:, 3 : 3 + self.mask_dim] = masked_block

        x = self.encoder(x)
        x = self.input_norm(x)

        x_pe = self.pe_norm(pe)
        x = torch.cat((x, x_pe), dim=1)

        for layer in self.layers:
            x = layer["conv"](x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            x = layer["norm"](x)

        x = self.pre_decoder_norm(x)
        return self.decoder(x)

    def load_graphkit_state_dict(self, state_dict: dict, strict: bool = True):
        """Load GraphKit checkpoints, including LightningModule-prefixed keys."""
        if any(key.startswith("model.") for key in state_dict.keys()):
            state_dict = {
                key.replace("model.", "", 1): value
                for key, value in state_dict.items()
                if key.startswith("model.")
            }
        return self.load_state_dict(state_dict, strict=strict)
