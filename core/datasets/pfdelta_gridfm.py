from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import AddRandomWalkPE

from core.datasets.pfdelta_dataset import PFDeltaDataset
from core.utils.registry import registry


@registry.register_dataset("pfdeltaGridFM")
class PFDeltaGridFM(PFDeltaDataset):
    """PFΔ dataset adapter for GridFM-style homogeneous graph reconstruction."""

    def __init__(
        self,
        root_dir: str = "data",
        case_name: str = "case14",
        split: str = "train",
        model: str = "GridFM",
        task: float | str = 1.3,
        add_bus_type: bool = False,
        mask_type: str = "pf",
        mask_ratio: float = 0.5,
        pe_dim: int = 20,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload: bool = False,
    ):
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
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
        hetero = super().build_heterodata(pm_case, is_cpf_sample=is_cpf_sample)

        bus_type = hetero["bus"].bus_type.long()
        pq = (bus_type == 1).float().unsqueeze(1)
        pv = (bus_type == 2).float().unsqueeze(1)
        ref = (bus_type == 3).float().unsqueeze(1)

        pd_qd = hetero["bus"].bus_demand
        pg_qg = hetero["bus"].bus_gen
        vm_va = hetero["bus"].bus_voltages

        x = torch.cat([pq, pv, ref, pd_qd, pg_qg, vm_va], dim=1).float()
        y = torch.cat([pd_qd, pg_qg, vm_va], dim=1).float()

        edge_index = hetero[("bus", "branch", "bus")].edge_index.long()
        branch_attr = hetero[("bus", "branch", "bus")].edge_attr
        edge_attr = branch_attr[:, 2:4].float()  # g_fr, b_fr -> GridFM-like [G, B]

        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        data.bus_type = bus_type
        data.edge_weight = torch.sqrt(edge_attr[:, 0] ** 2 + edge_attr[:, 1] ** 2 + 1e-12)

        pe_transform = AddRandomWalkPE(walk_length=self.pe_dim, attr_name="pe")
        data = pe_transform(data)

        data.mask = self._build_mask(data)
        return data

    def _build_mask(self, data: Data) -> torch.Tensor:
        mask = torch.zeros((data.num_nodes, 6), dtype=torch.bool)

        if self.mask_type == "none":
            return mask

        if self.mask_type == "rnd":
            return torch.rand((data.num_nodes, 6)) < self.mask_ratio

        if self.mask_type == "pf":
            pq = data.x[:, 0] == 1
            pv = data.x[:, 1] == 1
            ref = data.x[:, 2] == 1

            # target order: [Pd, Qd, Pg, Qg, Vm, Va]
            mask[pq, 4] = True
            mask[pq, 5] = True

            mask[pv, 3] = True
            mask[pv, 5] = True

            mask[ref, 2] = True
            mask[ref, 3] = True
            return mask

        raise ValueError(f"Unknown mask_type '{self.mask_type}'. Use one of: none, rnd, pf")
