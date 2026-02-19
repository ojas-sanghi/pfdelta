from typing import Optional

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GINEConv, GPSConv

from core.models.gridfm_utils import ensure_gridfm_fields
from core.utils.registry import registry


@registry.register_model("gridfm_gps")
class GridFMGPSModel(nn.Module):
    """PFDelta-integrated GridFM-style GPS/GINE model for zero-shot + fine-tuning."""

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
        checkpoint_path: Optional[str] = None,
        strict_checkpoint: bool = False,
        output_target: str = "gridfm",
    ):
        super().__init__()
        if pe_dim >= hidden_dim:
            raise ValueError("pe_dim must be < hidden_dim")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.pe_dim = pe_dim
        self.num_layers = num_layers
        self.output_target = output_target

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

        if learn_mask:
            self.mask_value = nn.Parameter(torch.randn(mask_dim) + mask_value, requires_grad=True)
        else:
            self.mask_value = nn.Parameter(torch.zeros(mask_dim) + mask_value, requires_grad=False)

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path, strict=strict_checkpoint)

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ["state_dict", "model_state_dict", "model"]:
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break

        if any(k.startswith("model.") for k in state_dict):
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing:
            print(f"[gridfm_gps] Missing checkpoint keys ({len(missing)}): {missing[:8]}")
        if unexpected:
            print(f"[gridfm_gps] Unexpected checkpoint keys ({len(unexpected)}): {unexpected[:8]}")

    def _extract_batch(self, data: HeteroData, x: torch.Tensor):
        batch = getattr(data["bus"], "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return batch

    def _extract_pe(self, data: HeteroData, x: torch.Tensor):
        pe = getattr(data["bus"], "pe_gridfm", None)
        if pe is None:
            pe = x.new_zeros((x.size(0), self.pe_dim))
        return pe

    def forward(self, data: HeteroData) -> torch.Tensor:
        ensure_gridfm_fields(data)
        x = data["bus"].x_gridfm.float()
        pe = self._extract_pe(data, x).float()
        edge_index = data[("bus", "branch", "bus")].edge_index
        edge_attr = data[("bus", "branch", "bus")].edge_attr_gridfm.float()
        batch = self._extract_batch(data, x)

        x = self.encoder(x)
        x = self.input_norm(x)
        pe = self.pe_norm(pe)
        x = torch.cat([x, pe], dim=1)

        for layer in self.layers:
            x = layer["conv"](x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            x = layer["norm"](x)

        x = self.pre_decoder_norm(x)
        x = self.decoder(x)

        if self.output_target == "pfdelta_voltage":
            return torch.stack([x[:, 5], x[:, 4]], dim=1)
        return x
