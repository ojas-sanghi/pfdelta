"""
GridFM (Grid Foundation Model) integration for PFDelta.

This module implements the GPSTransformer architecture from GridFM GraphKit
(https://github.com/gridfm/gridfm-graphkit) and wraps it to work seamlessly
with PFDelta's HeteroData format.

Architecture (GridFM v0.2):
  - input_dim: 9   [Pd, Qd, Pg, Qg, Vm, Va, PQ, PV, REF]
  - hidden_size: 256
  - output_dim: 6  [Pd, Qd, Pg, Qg, Vm, Va]
  - edge_dim: 2    [G, B] (Y-bus conductance + susceptance)
  - pe_dim: 20     (Random Walk PE walk length)
  - num_layers: 8  (GPS layers)
  - attention_head: 8
  - dropout: 0.1

Data Conversion (PFDelta HeteroData → GridFM flat tensors):
  - Node features: aggregated from bus_demand + bus_gen + bus_voltages + bus_type
  - Edge features: Y_bus matrix (diagonal + off-diagonal) computed from branch params
  - Positional encoding: Random Walk PE on the bus-branch graph

Supports:
  - Zero-shot inference with pre-trained GridFM weights
  - Fine-tuning on PFDelta datasets
  - Training from scratch with same architecture
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GPSConv, GINEConv

from core.utils.registry import registry

# ---------------------------------------------------------------------------
# Feature index constants (matching GridFM's globals.py)
# ---------------------------------------------------------------------------
PD, QD, PG, QG, VM, VA = 0, 1, 2, 3, 4, 5   # node feature indices (power/voltage)
PQ_IDX, PV_IDX, REF_IDX = 6, 7, 8            # bus-type one-hot indices
G_IDX, B_IDX = 0, 1                           # edge feature indices

# Bus type integers as stored in PFDelta HeteroData
PFDELTA_PQ = 1
PFDELTA_PV = 2
PFDELTA_REF = 3


# ---------------------------------------------------------------------------
# Random Walk Positional Encoding (ported from GridFM's transforms.py)
# ---------------------------------------------------------------------------

def compute_random_walk_pe(
    edge_index: Tensor,
    edge_weight: Tensor,
    num_nodes: int,
    walk_length: int,
    device: torch.device,
) -> Tensor:
    """Compute Random Walk Positional Encoding for a (batched) graph.

    For a batched graph from PyG, the edge_index is already block-diagonal,
    so random walks naturally stay within each graph.

    Args:
        edge_index: [2, E] edge indices
        edge_weight: [E] edge weights (e.g. |y_branch|)
        num_nodes: total number of nodes
        walk_length: number of random walk steps (= pe_dim)
        device: target device

    Returns:
        pe: [num_nodes, walk_length] positional encodings
    """
    if num_nodes == 0:
        return torch.zeros(0, walk_length, device=device)

    # Build row-normalised adjacency
    row, col = edge_index[0], edge_index[1]

    # Accumulate edge weights into dense adjacency (sparse for large graphs)
    if num_nodes <= 4000:
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        # scatter-add handles duplicate edges (e.g. self-loops added twice)
        adj.index_put_((row, col), edge_weight, accumulate=True)

        # Row-normalise
        row_sums = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
        adj = adj / row_sums

        # Iteratively multiply and extract diagonal (landing probabilities)
        eye_idx = torch.arange(num_nodes, device=device)
        out = adj.clone()
        pe_cols = [out[eye_idx, eye_idx]]
        for _ in range(walk_length - 1):
            out = out @ adj
            pe_cols.append(out[eye_idx, eye_idx])
    else:
        # Sparse path for very large graphs
        indices = torch.stack([row, col], dim=0)
        vals = edge_weight.float()
        adj_sp = torch.sparse_coo_tensor(indices, vals, (num_nodes, num_nodes)).coalesce()

        row_sums = torch.zeros(num_nodes, device=device)
        row_sums.scatter_add_(0, adj_sp.indices()[0], adj_sp.values())
        row_sums = row_sums.clamp(min=1e-8)
        norm_vals = adj_sp.values() / row_sums[adj_sp.indices()[0]]
        adj_sp = torch.sparse_coo_tensor(adj_sp.indices(), norm_vals, (num_nodes, num_nodes)).coalesce()

        eye_idx = torch.arange(num_nodes, device=device)
        out = adj_sp
        # For sparse, extract diagonal via non-zero values where row==col
        def _diag(sp):
            idx = sp.indices()
            mask = idx[0] == idx[1]
            d = torch.zeros(num_nodes, device=device)
            if mask.any():
                d[idx[0][mask]] = sp.values()[mask]
            return d

        pe_cols = [_diag(out)]
        for _ in range(walk_length - 1):
            out = torch.sparse.mm(out, adj_sp)
            pe_cols.append(_diag(out))

    pe = torch.stack(pe_cols, dim=1)  # [num_nodes, walk_length]
    return pe


# ---------------------------------------------------------------------------
# Y-bus computation from PFDelta branch parameters
# ---------------------------------------------------------------------------

def compute_ybus_edge_features(
    edge_index: Tensor,
    edge_attr: Tensor,
    num_nodes: int,
    baseMVA_orig: float = 100.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Construct full Y-bus (including diagonal) from PFDelta branch parameters.

    PFDelta edge_attr columns:
        0: br_r    (series resistance, p.u.)
        1: br_x    (series reactance, p.u.)
        2: g_fr    (from-end shunt conductance, p.u.)
        3: b_fr    (from-end shunt susceptance, p.u.)
        4: g_to    (to-end shunt conductance, p.u.)
        5: b_to    (to-end shunt susceptance, p.u.)
        6: tap     (off-nominal turns ratio; 0 → use 1.0)
        7: shift   (phase shift angle, radians)

    GridFM edge format:
        - Self-loop (i,i): diagonal Y_bus element G_ii + jB_ii
        - Off-diagonal (i,j) and (j,i): mutual admittance

    Returns:
        ybus_edge_index: [2, E_full] including self-loops
        ybus_edge_attr:  [E_full, 2] columns [G, B]
        edge_weight:     [E_full] = |Y| = sqrt(G^2 + B^2) for RWPE
    """
    device = edge_index.device
    R = edge_attr[:, 0]
    X = edge_attr[:, 1]
    g_fr = edge_attr[:, 2]
    b_fr = edge_attr[:, 3]
    g_to = edge_attr[:, 4]
    b_to = edge_attr[:, 5]
    tap_raw = edge_attr[:, 6]
    shift = edge_attr[:, 7]

    # Handle zero tap (means no transformer → use 1.0)
    tap = torch.where(tap_raw.abs() < 1e-10, torch.ones_like(tap_raw), tap_raw)

    # Complex tap: a = tap * exp(j*shift)
    # |a|^2 = tap^2, conj(a) = tap*exp(-j*shift)
    tap2 = tap ** 2

    # Series admittance: y_s = 1/(R+jX) = G_s + jB_s
    denom = R ** 2 + X ** 2
    denom = denom.clamp(min=1e-12)
    G_s = R / denom
    B_s = -X / denom

    # Off-diagonal Y_km = -y_s / conj(a) = -(G_s+jB_s) / (tap*exp(-j*shift))
    # For purely real tap (shift=0): Y_km = -(G_s+jB_s)/tap
    # Full complex:
    #   a* = tap*(cos(shift) - j*sin(shift))
    #   -y_s / a* = -(G_s+jB_s) * a / |a|^2
    #             = -(G_s+jB_s) * tap*(cos(shift)+j*sin(shift)) / tap^2
    #             = -(G_s+jB_s) * (cos(shift)+j*sin(shift)) / tap
    cos_s = torch.cos(shift)
    sin_s = torch.sin(shift)

    # -(G_s+jB_s)*(cos+j*sin)/tap:
    # Real:  -(G_s*cos - B_s*sin) / tap
    # Imag:  -(G_s*sin + B_s*cos) / tap
    G_km_from_to = -(G_s * cos_s - B_s * sin_s) / tap
    B_km_from_to = -(G_s * sin_s + B_s * cos_s) / tap

    # Y_mk = -y_s / a = -(G_s+jB_s) / (tap*exp(j*shift))
    # Real:  -(G_s*cos + B_s*sin) / tap
    # Imag:  (G_s*sin - B_s*cos) / tap  ... wait let me redo:
    # -y_s / a = -(G_s+jB_s) / (tap*(cos+j*sin))
    #          = -(G_s+jB_s)*(cos-j*sin) / tap
    # Real: -(G_s*cos + B_s*sin) / tap ... hmm, rewrite:
    # -(G_s+jB_s)*(cos-j*sin) = -(G_s*cos - j*G_s*sin + j*B_s*cos - j^2*B_s*sin)
    #                          = -(G_s*cos + B_s*sin) - j*(B_s*cos - G_s*sin)
    G_mk_to_from = -(G_s * cos_s + B_s * sin_s) / tap
    B_mk_to_from = -(B_s * cos_s - G_s * sin_s) / tap

    from_bus = edge_index[0]
    to_bus = edge_index[1]

    # Diagonal contributions from each branch
    # Y_kk += y_s/|a|^2 + g_fr + j*b_fr  (from-bus self-admittance)
    # Y_mm += y_s        + g_to + j*b_to  (to-bus self-admittance)
    G_kk = G_s / tap2 + g_fr
    B_kk = B_s / tap2 + b_fr
    G_mm = G_s + g_to
    B_mm = B_s + b_to

    # Accumulate diagonal elements
    G_diag = torch.zeros(num_nodes, device=device)
    B_diag = torch.zeros(num_nodes, device=device)
    G_diag.scatter_add_(0, from_bus, G_kk)
    B_diag.scatter_add_(0, from_bus, B_kk)
    G_diag.scatter_add_(0, to_bus, G_mm)
    B_diag.scatter_add_(0, to_bus, B_mm)

    # Self-loop edges (i, i)
    self_idx = torch.arange(num_nodes, device=device)
    self_ei = torch.stack([self_idx, self_idx], dim=0)  # [2, N]
    self_ea = torch.stack([G_diag, B_diag], dim=1)      # [N, 2]

    # Off-diagonal edges (from→to) and (to→from)
    off_ei_ft = edge_index                                           # [2, E]
    off_ea_ft = torch.stack([G_km_from_to, B_km_from_to], dim=1)   # [E, 2]
    off_ei_tf = edge_index.flip(0)                                  # [2, E]
    off_ea_tf = torch.stack([G_mk_to_from, B_mk_to_from], dim=1)   # [E, 2]

    # Concatenate all
    full_ei = torch.cat([self_ei, off_ei_ft, off_ei_tf], dim=1)    # [2, N+2E]
    full_ea = torch.cat([self_ea, off_ea_ft, off_ea_tf], dim=0)    # [N+2E, 2]

    # Edge weight = |Y| = sqrt(G^2 + B^2)
    ew = (full_ea[:, 0] ** 2 + full_ea[:, 1] ** 2).sqrt()

    return full_ei, full_ea, ew


# ---------------------------------------------------------------------------
# GPSTransformer – exact replica of GridFM's architecture
# ---------------------------------------------------------------------------

class GPSTransformer(nn.Module):
    """GPS-Transformer model from GridFM GraphKit.

    Architecture matches GridFM v0.2 pre-trained checkpoint exactly so that
    weights can be loaded directly.  The model takes flat homogeneous graph
    tensors (not HeteroData) as input.

    Args:
        input_dim (int):   Number of node input features (default 9).
        hidden_size (int): Hidden dimension (default 256).
        output_dim (int):  Number of output features per node (default 6).
        edge_dim (int):    Edge feature dimension (default 2).
        pe_dim (int):      Positional-encoding dimension (default 20).
        num_layers (int):  Number of GPS layers (default 8).
        attention_head (int): Attention heads in GPS (default 8).
        dropout (float):   Dropout rate (default 0.1).
        mask_dim (int):    Dimension of the learnable mask token (default 6).
        mask_value (float):Initial mask value (default 0.0).
        learn_mask (bool): Whether to learn the mask token (default False).
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_size: int = 256,
        output_dim: int = 6,
        edge_dim: int = 2,
        pe_dim: int = 20,
        num_layers: int = 8,
        attention_head: int = 8,
        dropout: float = 0.1,
        mask_dim: int = 6,
        mask_value: float = 0.0,
        learn_mask: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.pe_dim = pe_dim
        self.num_layers = num_layers
        self.heads = attention_head
        self.dropout = dropout
        self.mask_dim = mask_dim

        # Input encoder: projects (input_dim → hidden_size - pe_dim)
        # Then concatenate with normalised PE → hidden_size
        enc_out = hidden_size - pe_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, enc_out),
            nn.LeakyReLU(),
        )
        self.input_norm = nn.BatchNorm1d(enc_out)
        self.pe_norm = nn.BatchNorm1d(pe_dim)

        # GPS layers – stored as ModuleList of ModuleDicts {"conv", "norm"}
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
            )
            self.layers.append(
                nn.ModuleDict({
                    "conv": GPSConv(
                        channels=hidden_size,
                        conv=GINEConv(nn=mlp, edge_dim=edge_dim),
                        heads=attention_head,
                        dropout=dropout,
                    ),
                    "norm": nn.BatchNorm1d(hidden_size),
                })
            )

        self.pre_decoder_norm = nn.BatchNorm1d(hidden_size)

        # Decoder: hidden_size → hidden_size → output_dim
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_dim),
        )

        # Learnable mask value (shape = mask_dim)
        if learn_mask:
            self.mask_value = nn.Parameter(
                torch.randn(mask_dim) + mask_value, requires_grad=True
            )
        else:
            self.mask_value = nn.Parameter(
                torch.zeros(mask_dim) + mask_value, requires_grad=False
            )

    def forward(
        self,
        x: Tensor,
        pe: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            x:          [N, input_dim]  node features
            pe:         [N, pe_dim]     random walk PE
            edge_index: [2, E]          edge indices (COO format)
            edge_attr:  [E, edge_dim]   edge features [G, B]
            batch:      [N]             graph membership vector

        Returns:
            output: [N, output_dim]
        """
        # Encode node features and PE
        h = self.encoder(x)            # [N, hidden_size - pe_dim]
        h = self.input_norm(h)
        p = self.pe_norm(pe)           # [N, pe_dim]
        h = torch.cat([h, p], dim=1)  # [N, hidden_size]

        # GPS layers
        for layer in self.layers:
            h = layer["conv"](x=h, edge_index=edge_index,
                              edge_attr=edge_attr, batch=batch)
            h = layer["norm"](h)

        h = self.pre_decoder_norm(h)
        return self.decoder(h)         # [N, output_dim]


# ---------------------------------------------------------------------------
# GridFMWrapper – PFDelta-compatible model wrapper
# ---------------------------------------------------------------------------

@registry.register_model("gridfm")
class GridFMWrapper(nn.Module):
    """PFDelta-compatible wrapper around GPSTransformer.

    Accepts PFDelta's ``HeteroData`` and internally:

    1. Extracts bus-level power/voltage features from the heterogeneous graph.
    2. Converts branch parameters to full Y-bus edge features (G, B).
    3. Applies baseMVA-style normalisation (power → MW/baseMVA_data, Va → rad).
    4. Computes Random Walk Positional Encoding.
    5. Runs the ``GPSTransformer`` and returns flat per-bus predictions.

    Model output shape: ``[total_buses_in_batch, 6]``
    Output feature order: ``[Pd_norm, Qd_norm, Pg_norm, Qg_norm, Vm, Va_rad]``

    YAML config keys (under ``model:``)::

        name: gridfm

        # GPSTransformer hyperparameters
        input_dim:      9        # fixed for GridFM format
        hidden_size:    256
        output_dim:     6
        edge_dim:       2
        pe_dim:         20
        num_layers:     8
        attention_head: 8
        dropout:        0.1
        mask_dim:       6
        mask_value:     0.0
        learn_mask:     false

        # Data conversion
        baseMVA_orig:   100.0   # dataset baseMVA (MW → p.u. scale factor)
        # baseMVA_data normaliser: set to a fixed positive float to use a
        # fixed normalisation, or leave at 0 to compute per-batch from data.
        baseMVA_data:   0.0

        # Pre-trained weights (optional)
        pretrained_path: ""     # path to GridFM_v0_2.pth (or empty to skip)

        # Masking for feature-reconstruction task
        apply_pf_mask:  false   # apply standard PF mask during forward pass
    """

    def __init__(
        self,
        # GPSTransformer hyperparameters
        input_dim: int = 9,
        hidden_size: int = 256,
        output_dim: int = 6,
        edge_dim: int = 2,
        pe_dim: int = 20,
        num_layers: int = 8,
        attention_head: int = 8,
        dropout: float = 0.1,
        mask_dim: int = 6,
        mask_value: float = 0.0,
        learn_mask: bool = False,
        # Data conversion parameters
        baseMVA_orig: float = 100.0,
        baseMVA_data: float = 0.0,
        # Pre-trained weights
        pretrained_path: str = "",
        # Masking
        apply_pf_mask: bool = False,
        **kwargs,   # absorb any extra YAML keys without error
    ):
        super().__init__()

        # ---- GPSTransformer ----
        self.gps = GPSTransformer(
            input_dim=input_dim,
            hidden_size=hidden_size,
            output_dim=output_dim,
            edge_dim=edge_dim,
            pe_dim=pe_dim,
            num_layers=num_layers,
            attention_head=attention_head,
            dropout=dropout,
            mask_dim=mask_dim,
            mask_value=mask_value,
            learn_mask=learn_mask,
        )

        self.pe_dim = pe_dim
        self.baseMVA_orig = float(baseMVA_orig)
        self._fixed_baseMVA_data = float(baseMVA_data)
        self.apply_pf_mask = apply_pf_mask

        # Load pre-trained weights if requested
        if pretrained_path:
            self.load_pretrained(pretrained_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_pretrained(self, path: str, strict: bool = True) -> None:
        """Load GridFM pre-trained weights from a .pth checkpoint.

        The checkpoint is expected to come from GridFM's
        ``FeatureReconstructionTask``, where all keys are prefixed with
        ``model.``.  This method strips that prefix automatically.

        Args:
            path:   Path to the .pth file.
            strict: Whether to raise on missing/unexpected keys.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            sd = ckpt
        else:
            sd = ckpt.state_dict()

        # Strip the leading "model." prefix
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("model."):
                new_sd[k[len("model."):]] = v
            else:
                new_sd[k] = v

        missing, unexpected = self.gps.load_state_dict(new_sd, strict=strict)
        if missing:
            print(f"[GridFMWrapper] Missing keys: {missing}")
        if unexpected:
            print(f"[GridFMWrapper] Unexpected keys: {unexpected}")
        print(f"[GridFMWrapper] Loaded pre-trained weights from {path}")

    def forward(self, data: HeteroData) -> Tensor:
        """Run the GPSTransformer on a (batched) PFDelta HeteroData.

        Args:
            data: PyG HeteroData batch produced by PFDelta DataLoader.

        Returns:
            output: Tensor of shape [total_buses, 6]
                    = [Pd_norm, Qd_norm, Pg_norm, Qg_norm, Vm, Va_rad]
        """
        device = data["bus"].bus_demand.device

        # --- 1. Extract bus-level features ---
        demand = data["bus"].bus_demand    # [N, 2]  [pd_pu, qd_pu]
        gen = data["bus"].bus_gen          # [N, 2]  [pg_pu, qg_pu]
        voltages = data["bus"].bus_voltages  # [N, 2]  [va_rad, vm_pu]
        bus_type = data["bus"].bus_type    # [N]     int (1=PQ, 2=PV, 3=REF)

        pd_pu = demand[:, 0]
        qd_pu = demand[:, 1]
        pg_pu = gen[:, 0]
        qg_pu = gen[:, 1]
        va_rad = voltages[:, 0]   # voltage angle in RADIANS from PowerModels
        vm = voltages[:, 1]       # voltage magnitude in p.u.

        # --- 2. Convert p.u. powers → MW (GridFM CSV format is MW) ---
        # PFDelta stores power in per-unit (PowerModels convention).
        # GridFM's training data uses MW = p.u. * baseMVA_orig.
        pd_mw = pd_pu * self.baseMVA_orig
        qd_mw = qd_pu * self.baseMVA_orig
        pg_mw = pg_pu * self.baseMVA_orig
        qg_mw = qg_pu * self.baseMVA_orig

        # --- 3. BaseMVA normalisation ---
        if self._fixed_baseMVA_data > 0.0:
            baseMVA_data = self._fixed_baseMVA_data
        else:
            # Compute from batch (like GridFM's BaseMVANormalizer.fit)
            baseMVA_data = float(
                torch.cat([pd_mw, qd_mw, pg_mw, qg_mw]).abs().max().item()
            )
            baseMVA_data = max(baseMVA_data, 1.0)  # safety floor

        pd_norm = pd_mw / baseMVA_data
        qd_norm = qd_mw / baseMVA_data
        pg_norm = pg_mw / baseMVA_data
        qg_norm = qg_mw / baseMVA_data
        # GridFM normalises Va: degrees → radians (Va already in radians here,
        # so no conversion needed; GridFM's CSV has degrees which become radians
        # after BaseMVANorm.  Since PFDelta already has radians, pass through.)
        va_norm = va_rad

        # --- 4. Bus type one-hot ---
        is_pq = (bus_type == PFDELTA_PQ).float()   # [N]
        is_pv = (bus_type == PFDELTA_PV).float()
        is_ref = (bus_type == PFDELTA_REF).float()

        # Assemble node feature matrix [N, 9]
        x = torch.stack(
            [pd_norm, qd_norm, pg_norm, qg_norm, vm, va_norm,
             is_pq, is_pv, is_ref],
            dim=1,
        )

        # --- 5. Build Y-bus edge features ---
        branch_ei = data[("bus", "branch", "bus")].edge_index   # [2, E_raw]
        branch_ea = data[("bus", "branch", "bus")].edge_attr     # [E_raw, 8]
        num_nodes = x.size(0)

        ybus_ei, ybus_ea_pu, ew_pu = compute_ybus_edge_features(
            branch_ei, branch_ea, num_nodes, self.baseMVA_orig
        )

        # Normalise edge features: G_norm = G_pu * baseMVA_orig / baseMVA_data
        edge_norm_factor = self.baseMVA_orig / baseMVA_data
        ybus_ea = ybus_ea_pu * edge_norm_factor
        ew = (ybus_ea[:, 0] ** 2 + ybus_ea[:, 1] ** 2).sqrt()

        # --- 6. Compute Random Walk Positional Encoding ---
        pe = compute_random_walk_pe(
            ybus_ei, ew, num_nodes, self.pe_dim, device
        )   # [N, pe_dim]

        # --- 7. Masking (optional) ---
        if self.apply_pf_mask:
            mask = self._build_pf_mask(bus_type, x.size(0), device)
            mask_val = self.gps.mask_value.to(device)
            x_m = x.clone()
            x_m[:, :self.gps.mask_dim][mask] = mask_val.expand(x.size(0), -1)[mask]
            x = x_m

        # --- 8. Batch vector ---
        batch_vec = (
            data["bus"].batch
            if hasattr(data["bus"], "batch")
            else torch.zeros(num_nodes, dtype=torch.long, device=device)
        )

        # --- 9. GPSTransformer forward ---
        return self.gps(x, pe, ybus_ei, ybus_ea, batch_vec)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_pf_mask(
        self, bus_type: Tensor, num_nodes: int, device: torch.device
    ) -> Tensor:
        """Build the standard power-flow mask.

        Masked (unknown) features per bus type:
            PQ  buses: Vm (index 4), Va (index 5)
            PV  buses: Qg (index 3), Va (index 5)
            REF buses: Pg (index 2), Qg (index 3)

        Returns:
            mask: [N, 6] boolean tensor, True where feature is masked.
        """
        mask = torch.zeros(num_nodes, 6, dtype=torch.bool, device=device)
        pq = bus_type == PFDELTA_PQ
        pv = bus_type == PFDELTA_PV
        ref = bus_type == PFDELTA_REF
        mask[pq, 4] = True   # Vm unknown for PQ
        mask[pq, 5] = True   # Va unknown for PQ
        mask[pv, 3] = True   # Qg unknown for PV
        mask[pv, 5] = True   # Va unknown for PV
        mask[ref, 2] = True  # Pg unknown for REF
        mask[ref, 3] = True  # Qg unknown for REF
        return mask

    def build_targets(self, data: HeteroData) -> Tensor:
        """Return the normalised target tensor [N, 6] matching the model output.

        Useful for loss computation:  loss(model(data), model.build_targets(data)).
        """
        device = data["bus"].bus_demand.device
        demand = data["bus"].bus_demand    # [N, 2]
        gen = data["bus"].bus_gen          # [N, 2]
        voltages = data["bus"].bus_voltages  # [N, 2]

        pd_mw = demand[:, 0] * self.baseMVA_orig
        qd_mw = demand[:, 1] * self.baseMVA_orig
        pg_mw = gen[:, 0] * self.baseMVA_orig
        qg_mw = gen[:, 1] * self.baseMVA_orig
        va_rad = voltages[:, 0]
        vm = voltages[:, 1]

        if self._fixed_baseMVA_data > 0.0:
            baseMVA_data = self._fixed_baseMVA_data
        else:
            baseMVA_data = float(
                torch.cat([pd_mw, qd_mw, pg_mw, qg_mw]).abs().max().item()
            )
            baseMVA_data = max(baseMVA_data, 1.0)

        y = torch.stack([
            pd_mw / baseMVA_data,
            qd_mw / baseMVA_data,
            pg_mw / baseMVA_data,
            qg_mw / baseMVA_data,
            vm,
            va_rad,
        ], dim=1)   # [N, 6]
        return y

    def get_pf_mask(self, data: HeteroData) -> Tensor:
        """Return the PF mask tensor [N, 6] for the given data batch."""
        bus_type = data["bus"].bus_type
        device = bus_type.device
        num_nodes = bus_type.size(0)
        return self._build_pf_mask(bus_type, num_nodes, device)
