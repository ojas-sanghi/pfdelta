from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GINEConv, GPSConv

from core.utils.registry import registry


@registry.register_model("gridfm_transformer")
class GridFMTransformer(nn.Module):
    """GridFM-style GPS/GINE transformer for PFDelta graphs.

    This model mirrors the architecture used in gridfm-graphkit while exposing
    PFDelta-friendly defaults and key-based access to node features.
    """

    def __init__(
        self,
        input_dim: int = 9,
        pe_dim: int = 20,
        edge_dim: int = 2,
        hidden_dim: int = 256,
        output_dim: int = 6,
        num_layers: int = 8,
        heads: int = 8,
        dropout: float = 0.1,
        x_key: str = "x_gridfm",
        pe_key: str = "pe_gridfm",
        edge_key: str = "edge_attr_gridfm",
    ):
        super().__init__()
        if pe_dim >= hidden_dim:
            raise ValueError("pe_dim must be smaller than hidden_dim")

        self.x_key = x_key
        self.pe_key = pe_key
        self.edge_key = edge_key

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim - pe_dim),
            nn.LeakyReLU(),
        )
        self.input_norm = nn.BatchNorm1d(hidden_dim - pe_dim)
        self.pe_norm = nn.BatchNorm1d(pe_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU())
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

    def forward(self, data: HeteroData) -> Tensor:
        bus = data["bus"]
        edge_store = data["bus", "branch", "bus"]

        x = bus[self.x_key]
        pe = bus[self.pe_key]
        edge_index = edge_store.edge_index
        edge_attr = edge_store[self.edge_key]
        batch = bus.batch

        x_pe = self.pe_norm(pe)
        x = self.input_norm(self.encoder(x))
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
        return self.decoder(x)

    @torch.no_grad()
    def load_pretrained_backbone(self, checkpoint_path: str, strict: bool = False) -> Dict[str, Iterable[str]]:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            normalized = {}
            for key, value in state.items():
                key = key.replace("model.", "")
                normalized[key] = value
            state = normalized
        missing, unexpected = self.load_state_dict(state, strict=strict)
        return {"missing": missing, "unexpected": unexpected}
