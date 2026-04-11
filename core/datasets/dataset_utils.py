import json
import os

import numpy as np
import torch


FEASIBILITY_CONFIG = {
    "feasible": {
        "none": 56000,
        "n-1": 29000,
        "n-2": 20000,
        "test": {"none": 2000, "n-1": 2000, "n-2": 2000},
    },
    "approaching infeasible": {
        "none": 7200,
        "n-1": 7200,
        "n-2": 7200,
        "test": None,  # no test set for this regime
    },
    "near infeasible": {
        "none": 2000,
        "n-1": 2000,
        "n-2": 2000,
        "test": {"none": 200, "n-1": 200, "n-2": 200},
    },
}

TASK_CONFIG = {
    1.1: {"none": 54000, "n-1": 0, "n-2": 0},
    1.2: {"none": 27000, "n-1": 27000, "n-2": 0},
    1.3: {"none": 18000, "n-1": 18000, "n-2": 18000},
    2.1: {"none": 18000, "n-1": 18000, "n-2": 18000},
    2.2: {"none": 12000, "n-1": 12000, "n-2": 12000},
    2.3: {"none": 6000, "n-1": 6000, "n-2": 6000},
    3.1: {"none": 18000, "n-1": 18000, "n-2": 18000},
    3.2: {"none": 18000, "n-1": 18000, "n-2": 18000},
    3.3: {"none": 18000, "n-1": 18000, "n-2": 18000},  # can add 4.1 and 4.2 later?
}


def mean0_var1(x, mean, std):
    x = (x - mean) / std
    return x


class ComposeTransforms:
    """Apply a sequence of dataset transforms from left to right."""

    def __init__(self, transforms):
        self.transforms = [transform for transform in transforms if transform is not None]

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


def resolve_dataset_transform(transform, named_transform_factories):
    """Resolve YAML-friendly transform specs into a callable."""
    if transform is None or callable(transform):
        return transform

    if isinstance(transform, (list, tuple)):
        resolved = [
            resolve_dataset_transform(one_transform, named_transform_factories)
            for one_transform in transform
        ]
        resolved = [one_transform for one_transform in resolved if one_transform is not None]
        if len(resolved) == 0:
            return None
        if len(resolved) == 1:
            return resolved[0]
        return ComposeTransforms(resolved)

    if isinstance(transform, str):
        transform = {"name": transform}

    if isinstance(transform, dict):
        transform = dict(transform)
        transform_name = transform.pop("name", None)
        assert transform_name is not None, (
            "Transform dictionaries need a 'name' key."
        )
        transform_factory = named_transform_factories.get(transform_name)
        assert transform_factory is not None, (
            f"Transform {transform_name} is not registered for this dataset."
        )
        return transform_factory(**transform)

    raise TypeError(
        "Transform needs to be None, callable, string, dict, or an ordered list."
    )


def _positive_noise_scale(num_edges, sigma, reference):
    """Log-normal scale keeps multiplicative perturbations positive."""
    noise = torch.randn((num_edges, 1), device=reference.device, dtype=reference.dtype)
    return torch.exp(sigma * noise)


def branch_perturbation_transform(sigma):
    """
    Randomly perturb branch input attributes without touching labels.

    This is intended as an input transform, not as a physically re-solved sample.
    It therefore pairs naturally with power-balance training, while supervised
    targets become stale once the branch values are changed.
    """
    sigma = float(sigma)
    assert sigma >= 0.0, "branch_perturbation sigma must be non-negative."

    def transform(data):
        if sigma == 0.0:
            return data

        edge_type = ("bus", "branch", "bus")
        if edge_type not in data.edge_types:
            return data

        edge_attr = getattr(data[edge_type], "edge_attr", None)
        if edge_attr is None or edge_attr.numel() == 0:
            return data

        # __getitem__ returns a shallow copy of the graph object. Replacing the
        # tensor avoids contaminating the cached backing sample across epochs.
        edge_attr = edge_attr.clone()
        num_edges, num_features = edge_attr.shape

        series_scale = _positive_noise_scale(num_edges, sigma, edge_attr)
        shunt_scale = _positive_noise_scale(num_edges, sigma, edge_attr)

        if num_features == 8:
            # Raw PFDelta branch layout:
            # [br_r, br_x, g_fr, b_fr, g_to, b_to, tap, shift]
            edge_attr[:, 0:2] = edge_attr[:, 0:2] * series_scale
            edge_attr[:, 2:6] = edge_attr[:, 2:6] * shunt_scale
        elif num_features == 5:
            # PFNet branch layout:
            # [r, x, b_total, tau, angle]
            edge_attr[:, 0:2] = edge_attr[:, 0:2] * series_scale
            edge_attr[:, 2:3] = edge_attr[:, 2:3] * shunt_scale
        else:
            raise ValueError(
                "branch_perturbation only supports 8-column raw PFDelta edges "
                "or 5-column PFNet edges."
            )

        data[edge_type].edge_attr = edge_attr
        return data

    return transform


def create_train_test_mapping_json(
    case_name,
    seed=11,
    feasibility_setting="just feasible",
    root_dir="data/pfdelta_data/",
):
    """ """
    feasibility_config = FEASIBILITY_CONFIG[feasibility_setting]
    root = os.path.join(root_dir, case_name)

    for grid_type in ["none", "n-1", "n-2"]:
        num_samples = feasibility_config[grid_type]
        indices = np.arange(num_samples)
        np.random.seed(seed)
        np.random.shuffle(indices)
        shuffle_path = os.path.join(root, grid_type, "raw_shuffle.json")
        mappings = {int(i): int(j) for i, j in enumerate(indices)}
        with open(shuffle_path, "w") as f:
            json.dump(mappings, f, indent=3)


def canos_pf_data_mean0_var1(stats, data):
    means = stats["mean"]
    stds = stats["std"]

    def exception_transform(x, mean, std):
        r"""Transforms every value to mean 0, var 1 unless std is 0, in which
        case the value is just transformed to 0."""
        ones_std = std == 0.0
        std[ones_std] = 1.0
        x = mean0_var1(x, mean, std)
        return x

    values_to_change = [("bus", "x"), ("PV", "x"), ("PQ", "x"), ("slack", "x")]

    for dtype, entry in values_to_change:
        mean = means[dtype][entry]
        std = stds[dtype][entry]
        x = data[dtype][entry]
        data[dtype][entry] = exception_transform(x, mean, std)

    return data


def pfnet_data_mean0_var1(stats, data):
    means = stats["mean"]
    stds = stats["std"]
    eps = 1e-7

    # x_mean = means["bus"]["x"] # shape [6]
    # x_std = stds["bus"]["x"] + eps # shape [6]
    # x_cont = data["bus"]["x"][:, 4:10]
    # data["bus"]["x"][:, 4:10] = (x_cont - x_mean) / x_std

    y_mean = means["bus"]["y"]  # shape [6]
    y_std = stds["bus"]["y"] + eps  # shape [6]
    y_cont = data["bus"]["y"]
    data["bus"]["y"] = (y_cont - y_mean) / y_std

    edge_mean = means[("bus", "branch", "bus")]["edge_attr"]
    edge_std = stds[("bus", "branch", "bus")]["edge_attr"] + eps
    edge_attr = data[("bus", "branch", "bus")]["edge_attr"]
    data[("bus", "branch", "bus")].edge_attr = (edge_attr - edge_mean) / edge_std

    data["case_name"] = stats["casename"]

    return data


def canos_pf_slack_mean0_var1(stats, data):
    means = stats["mean"]
    stds = stats["std"]

    def exception_transform(x, mean, std):
        r"""Transforms every value to mean 0, var 1 unless std is 0, in which
        case the value is just transformed to 0."""
        ones_std = std == 0.0
        std[ones_std] = 1.0
        x = mean0_var1(x, mean, std)
        return x

    values_to_change = [("slack", "y")]

    for dtype, entry in values_to_change:
        mean = means[dtype][entry]
        std = stds[dtype][entry]
        x = data[dtype][entry]
        data[dtype][entry] = exception_transform(x, mean, std)

    return data


if __name__ == "__main__":
    # create_train_test_mapping_json("case14_seeds", seed=11, feasibility_setting="just feasible", root_dir="data/pfdelta_data/")
    create_train_test_mapping_json(
        "case118_seeds",
        seed=11,
        feasibility_setting="feasible",
        root_dir="data/pfdelta_data/",
    )
