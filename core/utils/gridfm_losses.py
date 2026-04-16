from __future__ import annotations

import torch

from core.utils.registry import registry


@registry.register_loss("gridfm_masked_mse")
class GridFMMaskedMSELoss:
    """Masked MSE for GridFM reconstruction targets."""

    def __init__(self, fallback_to_full: bool = True):
        self.fallback_to_full = fallback_to_full
        self.loss_name = "GridFM Masked MSE"

    def __call__(self, outputs: torch.Tensor, data):
        target = data.y
        mask = getattr(data, "mask", None)

        if mask is None:
            return torch.nn.functional.mse_loss(outputs, target)

        if mask.any():
            return torch.nn.functional.mse_loss(outputs[mask], target[mask])

        if self.fallback_to_full:
            return torch.nn.functional.mse_loss(outputs, target)

        return outputs.new_tensor(0.0)
