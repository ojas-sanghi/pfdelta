import copy

import torch
from torch_geometric.data import HeteroData

from core.datasets.dataset_utils import (
    branch_perturbation_transform,
    resolve_dataset_transform,
)


EDGE_TYPE = ("bus", "branch", "bus")


def _make_data(edge_attr):
    data = HeteroData()
    data["bus"].num_nodes = 3
    data[EDGE_TYPE].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    data[EDGE_TYPE].edge_attr = edge_attr.clone()
    return data


def test_resolve_dataset_transform_composes_left_to_right():
    transform = resolve_dataset_transform(
        [
            {"name": "add", "value": 1},
            {"name": "mul", "value": 3},
        ],
        {
            "add": lambda value: (lambda x: x + value),
            "mul": lambda value: (lambda x: x * value),
        },
    )

    assert transform(2) == 9


def test_branch_perturbation_sigma_zero_is_noop():
    edge_attr = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 0.1]], dtype=torch.float32
    )
    data = _make_data(edge_attr)

    transformed = branch_perturbation_transform(0.0)(data)

    assert torch.allclose(transformed[EDGE_TYPE].edge_attr, edge_attr)


def test_branch_perturbation_preserves_tap_and_shift_for_raw_edges():
    torch.manual_seed(0)
    edge_attr = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.1, 0.1],
            [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 0.9, -0.2],
        ],
        dtype=torch.float32,
    )
    data = _make_data(edge_attr)

    transformed = branch_perturbation_transform(0.2)(data)

    assert not torch.allclose(transformed[EDGE_TYPE].edge_attr[:, 0:2], edge_attr[:, 0:2])
    assert not torch.allclose(transformed[EDGE_TYPE].edge_attr[:, 2:6], edge_attr[:, 2:6])
    assert torch.allclose(transformed[EDGE_TYPE].edge_attr[:, 6:], edge_attr[:, 6:])


def test_branch_perturbation_supports_pfnet_edges():
    torch.manual_seed(1)
    edge_attr = torch.tensor(
        [
            [0.1, 0.2, 0.3, 1.1, 0.2],
            [0.2, 0.4, 0.5, 0.9, -0.1],
        ],
        dtype=torch.float32,
    )
    data = _make_data(edge_attr)

    transformed = branch_perturbation_transform(0.2)(data)

    assert not torch.allclose(transformed[EDGE_TYPE].edge_attr[:, 0:2], edge_attr[:, 0:2])
    assert not torch.allclose(transformed[EDGE_TYPE].edge_attr[:, 2:3], edge_attr[:, 2:3])
    assert torch.allclose(transformed[EDGE_TYPE].edge_attr[:, 3:], edge_attr[:, 3:])


def test_branch_perturbation_does_not_mutate_backing_sample():
    torch.manual_seed(2)
    edge_attr = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 0.1]], dtype=torch.float32
    )
    base = _make_data(edge_attr)
    original = base[EDGE_TYPE].edge_attr.clone()

    transformed = branch_perturbation_transform(0.2)(copy.copy(base))

    assert torch.allclose(base[EDGE_TYPE].edge_attr, original)
    assert not torch.allclose(transformed[EDGE_TYPE].edge_attr, original)
