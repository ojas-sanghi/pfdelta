from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import AddRandomWalkPE

from core.datasets.pfdelta_dataset import PFDeltaDataset
from core.utils.registry import registry


class PFDeltaToGridFM:
    """Convert PFDelta HeteroData samples to GridFM-style homogeneous Data."""

    def __init__(self, pe_dim=20, mask_dim=6, mask_ratio=0.5):
        self.mask_dim = mask_dim
        self.mask_ratio = mask_ratio
        self.add_pe = AddRandomWalkPE(walk_length=pe_dim, attr_name="pe")

    def __call__(self, data):
        bus = data["bus"]
        branch = data["bus", "branch", "bus"]

        pd = bus.bus_demand[:, 0]
        qd = bus.bus_demand[:, 1]
        pg = bus.bus_gen[:, 0]
        qg = bus.bus_gen[:, 1]
        vm = bus.bus_voltages[:, 0]
        va = bus.bus_voltages[:, 1]

        bus_type = bus.bus_type.long()
        pq = (bus_type == 1).float()
        pv = (bus_type == 2).float()
        ref = (bus_type == 3).float()

        x = torch.stack([pd, qd, pg, qg, vm, va, pq, pv, ref], dim=1).to(torch.float)
        y = x[:, : self.mask_dim].clone()

        r = branch.edge_attr[:, 0]
        xline = branch.edge_attr[:, 1]
        y_complex = 1 / (r + 1j * xline + 1e-8)
        edge_attr = torch.stack([torch.real(y_complex), torch.imag(y_complex)], dim=1).to(torch.float)

        out = Data(
            x=x,
            y=y,
            edge_index=branch.edge_index,
            edge_attr=edge_attr,
            scenario_id=getattr(data, "scenario_id", torch.tensor(-1)),
        )

        out = self.add_pe(out)

        mask = torch.rand((out.num_nodes, self.mask_dim), device=out.x.device) < self.mask_ratio
        out.mask = mask

        masked_x = out.x.clone()
        masked_x[:, : self.mask_dim][mask] = 0.0
        out.x = masked_x

        return out


@registry.register_dataset("pfdelta_gridfm")
class PFDeltaGridFMDataset(PFDeltaDataset):
    def __init__(
        self,
        root_dir="data",
        case_name="case14",
        split="train",
        task=1.3,
        model="gridfm",
        pe_dim=20,
        mask_dim=6,
        mask_ratio=0.5,
        force_reload=False,
        **kwargs,
    ):
        transform = PFDeltaToGridFM(pe_dim=pe_dim, mask_dim=mask_dim, mask_ratio=mask_ratio)
        super().__init__(
            root_dir=root_dir,
            case_name=case_name,
            split=split,
            task=task,
            model=model,
            transform=transform,
            force_reload=force_reload,
            **kwargs,
        )
