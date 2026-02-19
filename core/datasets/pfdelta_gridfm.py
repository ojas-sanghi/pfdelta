from __future__ import annotations

import torch

from core.datasets.pfdelta_dataset import PFDeltaDataset
from core.utils.registry import registry


@registry.register_dataset("pfdeltaGridFM")
class PFDeltaGridFM(PFDeltaDataset):
    """PFDelta variant that materializes GridFM-style node targets/features."""

    def __init__(
        self,
        root_dir: str = "data",
        case_name: str = "case118",
        split: str = "train",
        model: str = "GridFM",
        task: float = 1.3,
        pe_dim: int = 20,
        add_bus_type: bool = False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload: bool = False,
    ):
        self.pe_dim = pe_dim
        super().__init__(
            root_dir=root_dir,
            case_name=case_name,
            split=split,
            model=model,
            task=task,
            add_bus_type=add_bus_type,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )

    def build_heterodata(self, pm_case: dict, is_cpf_sample: bool = False):
        data = super().build_heterodata(pm_case, is_cpf_sample=is_cpf_sample)
        bus = data["bus"]

        pd = bus.bus_demand[:, 0]
        qd = bus.bus_demand[:, 1]
        pg = bus.bus_gen[:, 0]
        qg = bus.bus_gen[:, 1]
        va = bus.bus_voltages[:, 0]
        vm = bus.bus_voltages[:, 1]

        bus_type = bus.bus_type.long()
        pq = (bus_type == 1).float()
        pv = (bus_type == 2).float()
        ref = (bus_type == 3).float()

        x_gridfm = torch.stack([pd, qd, pg, qg, vm, va, pq, pv, ref], dim=1)
        y_gridfm = torch.stack([pd, qd, pg, qg, vm, va], dim=1)

        n_bus = x_gridfm.size(0)
        position = torch.arange(n_bus, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.pe_dim, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / max(self.pe_dim, 1))
        )
        pe = torch.zeros(n_bus, self.pe_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.pe_dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])


        edge_attr = data["bus", "branch", "bus"].edge_attr
        data["bus", "branch", "bus"].edge_attr_gridfm = edge_attr[:, 2:4]
        bus.x_gridfm = x_gridfm
        bus.y_gridfm = y_gridfm
        bus.pe_gridfm = pe
        return data
