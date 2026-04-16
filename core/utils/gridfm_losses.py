"""
GridFM loss functions – ported from gridfm-graphkit and adapted for PFDelta.

Registered losses:
    gridfm_masked_mse   – MSE on masked node features (feature reconstruction)
    gridfm_pbe          – Power Balance Equation loss
    gridfm_mixed        – Weighted combination of the two

These losses follow the PFDelta convention:
    loss(outputs: Tensor, data: HeteroData) -> Tensor

where ``outputs`` is the flat [N, 6] prediction from GridFMWrapper and
``data`` is the batched PFDelta HeteroData.

Feature ordering in outputs / targets:
    index 0: Pd (active demand)
    index 1: Qd (reactive demand)
    index 2: Pg (active generation)
    index 3: Qg (reactive generation)
    index 4: Vm (voltage magnitude)
    index 5: Va (voltage angle, radians)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_torch_coo_tensor

from core.utils.registry import registry

# Feature indices (same ordering as GridFM)
PD, QD, PG, QG, VM, VA = 0, 1, 2, 3, 4, 5

# Bus-type indices in the node feature vector x[:, 6:9]
PQ_IDX, PV_IDX, REF_IDX = 6, 7, 8

# PFDelta bus-type integers
PFDELTA_PQ, PFDELTA_PV, PFDELTA_REF = 1, 2, 3

# Y-bus edge-feature indices
G_EDGE, B_EDGE = 0, 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_model_and_targets(model, data: HeteroData):
    """Retrieve the GridFMWrapper from the loss context if needed.

    Since PFDelta losses receive (outputs, data), and we need normalised
    targets + masks, we call ``model.build_targets()`` and ``model.get_pf_mask()``
    directly on the model stored in the loss instance.
    """
    pass  # unused – targets/masks are computed inside each loss


# ---------------------------------------------------------------------------
# GridFMMaskedMSELoss
# ---------------------------------------------------------------------------

@registry.register_loss("gridfm_masked_mse")
class GridFMMaskedMSELoss:
    """MSE computed only on PF-masked node features.

    The mask is: PQ→(Vm,Va), PV→(Qg,Va), REF→(Pg,Qg) — same as GridFM.

    Args:
        model: The GridFMWrapper instance (passed at init in the trainer).
    """

    loss_name = "GridFM MaskedMSE"

    def __init__(self, model=None):
        # model reference is injected by GridFMTrainer
        self._model = model

    def set_model(self, model):
        self._model = model

    def __call__(self, outputs: Tensor, data: HeteroData) -> Tensor:
        bus_type = data["bus"].bus_type   # [N]
        device = outputs.device

        # Build targets
        targets = self._build_targets(data, outputs)   # [N, 6]

        # Build mask
        mask = self._build_mask(bus_type, outputs.size(0), device)  # [N, 6]

        if mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return F.mse_loss(outputs[mask], targets[mask], reduction="mean")

    # -- helpers (same logic as GridFMWrapper) --

    def _build_targets(self, data: HeteroData, ref_tensor: Tensor) -> Tensor:
        if self._model is not None:
            return self._model.build_targets(data).to(ref_tensor.device)
        # Fallback: build targets manually (assumes baseMVA=100)
        return _build_targets_default(data, ref_tensor)

    def _build_mask(self, bus_type: Tensor, num_nodes: int, device) -> Tensor:
        mask = torch.zeros(num_nodes, 6, dtype=torch.bool, device=device)
        pq = bus_type == PFDELTA_PQ
        pv = bus_type == PFDELTA_PV
        ref = bus_type == PFDELTA_REF
        mask[pq, 4] = True   # Vm
        mask[pq, 5] = True   # Va
        mask[pv, 3] = True   # Qg
        mask[pv, 5] = True   # Va
        mask[ref, 2] = True  # Pg
        mask[ref, 3] = True  # Qg
        return mask


# ---------------------------------------------------------------------------
# GridFMFullMSELoss (unmasked, useful for debugging / pre-training)
# ---------------------------------------------------------------------------

@registry.register_loss("gridfm_full_mse")
class GridFMFullMSELoss:
    """Standard MSE on all 6 output features (no masking)."""

    loss_name = "GridFM FullMSE"

    def __init__(self, model=None):
        self._model = model

    def set_model(self, model):
        self._model = model

    def __call__(self, outputs: Tensor, data: HeteroData) -> Tensor:
        targets = self._build_targets(data, outputs)
        return F.mse_loss(outputs, targets, reduction="mean")

    def _build_targets(self, data, ref_tensor):
        if self._model is not None:
            return self._model.build_targets(data).to(ref_tensor.device)
        return _build_targets_default(data, ref_tensor)


# ---------------------------------------------------------------------------
# GridFMPBELoss – Power Balance Equation loss
# ---------------------------------------------------------------------------

@registry.register_loss("gridfm_pbe")
class GridFMPBELoss:
    """Power Balance Equation (PBE) loss.

    Constructs the full Y-bus from PFDelta branch parameters, assembles the
    complex voltage vector from predictions, and computes the mismatch between
    predicted net power injection and the Y-bus power flow.

    The loss penalises |S_injection - S_net|, averaged over all buses.

    Args:
        model:         GridFMWrapper instance (injected by trainer).
        baseMVA_orig:  System MVA base (default 100).
    """

    loss_name = "GridFM PBE"

    def __init__(self, model=None, baseMVA_orig: float = 100.0):
        self._model = model
        self.baseMVA_orig = baseMVA_orig

    def set_model(self, model):
        self._model = model
        if hasattr(model, "baseMVA_orig"):
            self.baseMVA_orig = model.baseMVA_orig

    def __call__(self, outputs: Tensor, data: HeteroData) -> Tensor:
        """
        Args:
            outputs: [N, 6] normalised predictions [Pd,Qd,Pg,Qg,Vm,Va]
            data:    batched HeteroData

        Returns:
            Scalar PBE loss.
        """
        device = outputs.device
        targets = self._build_targets(data, outputs)   # [N, 6]
        bus_type = data["bus"].bus_type
        mask = self._build_mask(bus_type, outputs.size(0), device)

        # Merge: use predictions for masked values, targets for known values
        merged = targets.clone().detach()
        merged[mask] = outputs[mask]

        # Y-bus edge features (already computed by the model, but we recompute
        # from branch data to keep the loss self-contained)
        branch_ei = data[("bus", "branch", "bus")].edge_index
        branch_ea = data[("bus", "branch", "bus")].edge_attr
        num_nodes = outputs.size(0)

        # Import here to avoid circular dependency
        from core.models.gridfm import compute_ybus_edge_features
        ybus_ei, ybus_ea_pu, _ = compute_ybus_edge_features(
            branch_ei, branch_ea, num_nodes, self.baseMVA_orig
        )

        # Normalise edge features
        if self._model is not None and hasattr(self._model, "_fixed_baseMVA_data"):
            baseMVA_data = self._model._fixed_baseMVA_data
        else:
            baseMVA_data = 0.0
        if baseMVA_data <= 0.0:
            demand = data["bus"].bus_demand
            gen = data["bus"].bus_gen
            all_pw = torch.cat([
                demand[:, 0], demand[:, 1], gen[:, 0], gen[:, 1]
            ]).abs() * self.baseMVA_orig
            baseMVA_data = max(float(all_pw.max().item()), 1.0)

        edge_norm = self.baseMVA_orig / baseMVA_data
        G = ybus_ea_pu[:, G_EDGE] * edge_norm
        B = ybus_ea_pu[:, B_EDGE] * edge_norm

        # Voltage vector (from normalised outputs, re-scaling not needed since
        # Vm and Va are in natural units already in merged)
        Vm = merged[:, VM]
        Va = merged[:, VA]   # radians
        V = Vm * torch.exp(1j * Va)

        # Sparse Y-bus
        edge_complex = G + 1j * B
        Y_sparse = to_torch_coo_tensor(
            ybus_ei, edge_complex, size=(num_nodes, num_nodes)
        )

        # S_injection = V ⊙ conj(Y_bus * V)
        V_conj = torch.conj(V)
        Y_conj = torch.conj(Y_sparse)
        S_inj = V * (torch.mv(Y_conj.to_dense(), V_conj))

        # Net complex power (de-normalised back to p.u. for physical consistency)
        Pd = merged[:, PD] * baseMVA_data / self.baseMVA_orig   # MW → p.u.
        Qd = merged[:, QD] * baseMVA_data / self.baseMVA_orig
        Pg = merged[:, PG] * baseMVA_data / self.baseMVA_orig
        Qg = merged[:, QG] * baseMVA_data / self.baseMVA_orig
        S_net = (Pg - Pd) + 1j * (Qg - Qd)

        # Mismatch
        mismatch = S_net - S_inj
        loss = torch.mean(torch.abs(mismatch))
        return loss

    def _build_targets(self, data, ref_tensor):
        if self._model is not None:
            return self._model.build_targets(data).to(ref_tensor.device)
        return _build_targets_default(data, ref_tensor)

    def _build_mask(self, bus_type, num_nodes, device):
        mask = torch.zeros(num_nodes, 6, dtype=torch.bool, device=device)
        pq = bus_type == PFDELTA_PQ
        pv = bus_type == PFDELTA_PV
        ref = bus_type == PFDELTA_REF
        mask[pq, 4] = True
        mask[pq, 5] = True
        mask[pv, 3] = True
        mask[pv, 5] = True
        mask[ref, 2] = True
        mask[ref, 3] = True
        return mask


# ---------------------------------------------------------------------------
# GridFMMixedLoss – weighted sum of MaskedMSE + PBE
# ---------------------------------------------------------------------------

@registry.register_loss("gridfm_mixed")
class GridFMMixedLoss:
    """Weighted combination: ``w_mse * MaskedMSE + w_pbe * PBE``.

    YAML config example::

        - name: gridfm_mixed
          w_mse: 0.01
          w_pbe: 0.99

    Args:
        w_mse: weight for MaskedMSE (default 0.01)
        w_pbe: weight for PBE       (default 0.99)
        model: GridFMWrapper (injected by trainer)
    """

    loss_name = "GridFM Mixed"

    def __init__(self, w_mse: float = 0.01, w_pbe: float = 0.99, model=None):
        self.w_mse = w_mse
        self.w_pbe = w_pbe
        self._mse_loss = GridFMMaskedMSELoss(model=model)
        self._pbe_loss = GridFMPBELoss(model=model)

    def set_model(self, model):
        self._mse_loss.set_model(model)
        self._pbe_loss.set_model(model)

    def __call__(self, outputs: Tensor, data: HeteroData) -> Tensor:
        mse = self._mse_loss(outputs, data)
        pbe = self._pbe_loss(outputs, data)
        return self.w_mse * mse + self.w_pbe * pbe


# ---------------------------------------------------------------------------
# Evaluation metric: RMSE per bus type and per feature
# ---------------------------------------------------------------------------

@registry.register_loss("gridfm_eval_rmse")
class GridFMEvalRMSE:
    """Per-bus-type RMSE for evaluation (not used for backprop).

    Returns the scalar mean RMSE across all features and bus types.
    Detailed per-feature RMSEs are available after calling __call__ via
    ``self.last_metrics``.
    """

    loss_name = "GridFM RMSE"

    def __init__(self, model=None):
        self._model = model
        self.last_metrics: dict = {}

    def set_model(self, model):
        self._model = model

    def __call__(self, outputs: Tensor, data: HeteroData) -> Tensor:
        targets = self._build_targets(data, outputs)
        bus_type = data["bus"].bus_type
        device = outputs.device

        feat_names = ["Pd", "Qd", "Pg", "Qg", "Vm", "Va"]
        type_masks = {
            "PQ": bus_type == PFDELTA_PQ,
            "PV": bus_type == PFDELTA_PV,
            "REF": bus_type == PFDELTA_REF,
        }

        metrics = {}
        all_rmse = []
        for tname, tmask in type_masks.items():
            if tmask.any():
                mse_per_feat = F.mse_loss(
                    outputs[tmask], targets[tmask], reduction="none"
                ).mean(dim=0)   # [6]
                for fi, fname in enumerate(feat_names):
                    rmse_val = mse_per_feat[fi].sqrt().item()
                    metrics[f"RMSE_{tname}_{fname}"] = rmse_val
                    all_rmse.append(rmse_val)

        self.last_metrics = metrics
        mean_rmse = sum(all_rmse) / len(all_rmse) if all_rmse else 0.0
        return torch.tensor(mean_rmse, device=device)

    def _build_targets(self, data, ref_tensor):
        if self._model is not None:
            return self._model.build_targets(data).to(ref_tensor.device)
        return _build_targets_default(data, ref_tensor)


# ---------------------------------------------------------------------------
# Default target builder (used as fallback when no model reference is given)
# ---------------------------------------------------------------------------

def _build_targets_default(data: HeteroData, ref_tensor: Tensor) -> Tensor:
    """Fallback target builder using baseMVA=100."""
    device = ref_tensor.device
    baseMVA_orig = 100.0
    demand = data["bus"].bus_demand.to(device)
    gen = data["bus"].bus_gen.to(device)
    voltages = data["bus"].bus_voltages.to(device)

    pd_mw = demand[:, 0] * baseMVA_orig
    qd_mw = demand[:, 1] * baseMVA_orig
    pg_mw = gen[:, 0] * baseMVA_orig
    qg_mw = gen[:, 1] * baseMVA_orig

    baseMVA_data = float(
        torch.cat([pd_mw, qd_mw, pg_mw, qg_mw]).abs().max().item()
    )
    baseMVA_data = max(baseMVA_data, 1.0)

    return torch.stack([
        pd_mw / baseMVA_data,
        qd_mw / baseMVA_data,
        pg_mw / baseMVA_data,
        qg_mw / baseMVA_data,
        voltages[:, 1],   # Vm
        voltages[:, 0],   # Va (radians)
    ], dim=1)
