import torch
import torch.nn.functional as F

from core.utils.registry import registry


@registry.register_loss("gridfm_masked_mse")
class GridFMMaskedMSE:
    """Masked MSE matching GridFM feature-reconstruction objective."""

    def __init__(self, reduction="mean"):
        self.reduction = reduction
        self.loss_name = "GridFM Masked MSE"

    def __call__(self, outputs, data):
        mask = data.mask.bool()
        pred = outputs[:, : data.y.shape[1]]
        target = data.y
        if mask.sum() == 0:
            return F.mse_loss(pred, target, reduction=self.reduction)
        return F.mse_loss(pred[mask], target[mask], reduction=self.reduction)


@registry.register_loss("gridfm_mse")
class GridFMMSE:
    def __init__(self, reduction="mean"):
        self.reduction = reduction
        self.loss_name = "GridFM MSE"

    def __call__(self, outputs, data):
        pred = outputs[:, : data.y.shape[1]]
        return F.mse_loss(pred, data.y, reduction=self.reduction)
