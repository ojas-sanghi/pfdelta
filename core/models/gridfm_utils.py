import torch


def compute_bus_type_one_hot(bus_type: torch.Tensor) -> torch.Tensor:
    """Map PFDelta bus_type (1=PQ,2=PV,3=REF/slack) to one-hot columns."""
    bus_type = bus_type.view(-1).long()
    pq = (bus_type == 1).float()
    pv = (bus_type == 2).float()
    ref = (bus_type == 3).float()
    return torch.stack([pq, pv, ref], dim=1)


def build_gridfm_node_features(data) -> torch.Tensor:
    """Build [pd, qd, pg, qg, vm, va, pq, pv, ref] node features from PFDelta data."""
    demand = data["bus"].bus_demand.float()
    generation = data["bus"].bus_gen.float()
    voltages = data["bus"].bus_voltages.float()  # [va, vm]

    va = voltages[:, 0:1]
    vm = voltages[:, 1:2]
    bus_type_1hot = compute_bus_type_one_hot(data["bus"].bus_type)

    return torch.cat(
        [
            demand[:, 0:1],
            demand[:, 1:2],
            generation[:, 0:1],
            generation[:, 1:2],
            vm,
            va,
            bus_type_1hot,
        ],
        dim=1,
    )


def build_gridfm_targets(data) -> torch.Tensor:
    """Targets for GridFM reconstruction task: [pd, qd, pg, qg, vm, va]."""
    x = build_gridfm_node_features(data)
    return x[:, :6]


def build_gridfm_edge_features(data, eps: float = 1e-8) -> torch.Tensor:
    """Build [g,b] from branch [r,x,...] using admittance y=1/(r+jx)=g+jb."""
    edge_attr = data[("bus", "branch", "bus")].edge_attr.float()
    r = edge_attr[:, 0]
    x = edge_attr[:, 1]
    den = r.square() + x.square() + eps
    g = r / den
    b = -x / den
    return torch.stack([g, b], dim=1)


def ensure_gridfm_fields(data) -> None:
    """Attach GridFM-compatible fields to PFDelta HeteroData in-place if absent."""
    if not hasattr(data["bus"], "x_gridfm"):
        data["bus"].x_gridfm = build_gridfm_node_features(data)
    if not hasattr(data["bus"], "y_gridfm"):
        data["bus"].y_gridfm = build_gridfm_targets(data)
    if not hasattr(data[("bus", "branch", "bus")], "edge_attr_gridfm"):
        data[("bus", "branch", "bus")].edge_attr_gridfm = build_gridfm_edge_features(data)
