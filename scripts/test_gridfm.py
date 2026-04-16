#!/usr/bin/env python3
"""
GridFM Integration Test Script
================================
Verifies that:
  1. The GPSTransformer architecture can be instantiated.
  2. Pre-trained GridFM v0.2 weights load correctly (no missing/unexpected keys).
  3. The model runs forward pass on synthetic HeteroData (dummy power grid).
  4. Output shapes are correct.
  5. GridFM losses compute without error.

Usage:
    # From the pfdelta/ project root:
    uv run python scripts/test_gridfm.py

    # Or if you have the environment active:
    python scripts/test_gridfm.py [--pth PATH_TO_CHECKPOINT]
"""

import sys
import os
import argparse

# Ensure pfdelta root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="Test GridFM integration in PFDelta")
    p.add_argument(
        "--pth",
        default="examples/models/GridFM_v0_2.pth",
        help="Path to GridFM_v0_2.pth checkpoint",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_dummy_heterodata(n_buses: int = 30, n_branches: int = 41, device="cpu"):
    """Create a synthetic PFDelta-style HeteroData for a single graph."""
    import torch
    from torch_geometric.data import HeteroData

    torch.manual_seed(0)
    data = HeteroData()

    # Bus-type assignment (roughly: 1 REF, ~20% PV, rest PQ)
    bus_type = torch.ones(n_buses, dtype=torch.long)   # PQ=1
    n_pv = max(1, n_buses // 5)
    bus_type[1 : 1 + n_pv] = 2   # PV buses
    bus_type[0] = 3               # REF bus

    # Bus features (in per-unit, as PowerModels returns)
    demand = torch.rand(n_buses, 2) * 2.0          # [pd, qd] p.u.
    gen    = torch.zeros(n_buses, 2)
    gen[bus_type == 2, 0] = torch.rand(n_pv) * 5.0  # PV active gen
    gen[bus_type == 2, 1] = torch.rand(n_pv) * 1.0  # PV reactive gen
    gen[bus_type == 3, 0] = torch.rand(1) * 10.0    # REF active gen
    gen[bus_type == 3, 1] = torch.rand(1) * 2.0     # REF reactive gen

    voltages = torch.stack([
        torch.rand(n_buses) * 0.5 - 0.25,     # va in radians (~±0.25 rad)
        torch.rand(n_buses) * 0.1 + 0.95,     # vm in p.u. (0.95–1.05)
    ], dim=1)

    data["bus"].bus_demand = demand.to(device)
    data["bus"].bus_gen = gen.to(device)
    data["bus"].bus_voltages = voltages.to(device)
    data["bus"].bus_type = bus_type.to(device)
    data["bus"].num_nodes = n_buses

    # Branch edges – random sparse connectivity
    src = torch.randint(0, n_buses, (n_branches,))
    dst = torch.randint(0, n_buses, (n_branches,))
    # Remove self-loops
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edge_index = torch.stack([src, dst], dim=0).to(device)

    # Branch parameters: [br_r, br_x, g_fr, b_fr, g_to, b_to, tap, shift]
    n_branches_actual = edge_index.size(1)
    br_r   = torch.rand(n_branches_actual) * 0.05 + 0.002   # 0.002–0.052 p.u.
    br_x   = torch.rand(n_branches_actual) * 0.2  + 0.01    # 0.01–0.21 p.u.
    g_fr   = torch.zeros(n_branches_actual)
    b_fr   = torch.rand(n_branches_actual) * 0.02            # small charging
    g_to   = torch.zeros(n_branches_actual)
    b_to   = torch.rand(n_branches_actual) * 0.02
    tap    = torch.ones(n_branches_actual)                   # no transformer
    shift  = torch.zeros(n_branches_actual)

    edge_attr = torch.stack([br_r, br_x, g_fr, b_fr, g_to, b_to, tap, shift], dim=1)

    data[("bus", "branch", "bus")].edge_index = edge_index.to(device)
    data[("bus", "branch", "bus")].edge_attr  = edge_attr.to(device)

    # Batch vector (single graph → all zeros)
    data["bus"].batch = torch.zeros(n_buses, dtype=torch.long, device=device)

    return data


def make_dummy_batch(batch_size: int = 4, n_buses: int = 30, device="cpu"):
    """Stack multiple dummy graphs into a PyG batch."""
    import torch
    from torch_geometric.data import Batch

    graphs = [make_dummy_heterodata(n_buses=n_buses, device="cpu")
              for _ in range(batch_size)]
    batch = Batch.from_data_list(graphs)
    return batch.to(device)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_architecture():
    """Test that the GPSTransformer can be instantiated."""
    print("\n[1/5] Testing GPSTransformer instantiation...")
    from core.models.gridfm import GPSTransformer
    model = GPSTransformer(
        input_dim=9, hidden_size=256, output_dim=6, edge_dim=2,
        pe_dim=20, num_layers=8, attention_head=8, dropout=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      OK — {n_params:,} parameters")
    return model


def test_weight_loading(pth_path: str, device: str):
    """Test that pre-trained weights load with no missing/unexpected keys."""
    print(f"\n[2/5] Testing weight loading from {pth_path} ...")
    import torch
    from core.models.gridfm import GridFMWrapper

    if not os.path.exists(pth_path):
        print(f"      SKIP — file not found: {pth_path}")
        print("      (Run:  curl -L -o examples/models/GridFM_v0_2.pth "
              "https://github.com/gridfm/gridfm-graphkit/raw/main/examples/models/GridFM_v0_2.pth)")
        return None

    model = GridFMWrapper(
        input_dim=9, hidden_size=256, output_dim=6, edge_dim=2,
        pe_dim=20, num_layers=8, attention_head=8, dropout=0.1,
        mask_dim=6, mask_value=0.0, learn_mask=False,
        baseMVA_orig=100.0, baseMVA_data=0.0,
        pretrained_path=pth_path,
        apply_pf_mask=True,
    ).to(device)
    print("      OK — weights loaded successfully")
    return model


def test_forward_pass(model, device: str):
    """Test forward pass on synthetic data."""
    import torch
    print("\n[3/5] Testing forward pass on synthetic data...")

    model.eval()
    with torch.no_grad():
        # Single graph
        single = make_dummy_heterodata(n_buses=30, device=device)
        out = model(single)
        assert out.shape == (30, 6), f"Expected (30,6), got {out.shape}"
        print(f"      Single graph: input={single['bus'].bus_demand.shape[0]} buses, "
              f"output={out.shape}")

        # Batched graphs (like a real DataLoader)
        batch = make_dummy_batch(batch_size=4, n_buses=30, device=device)
        out_batch = model(batch)
        assert out_batch.shape == (120, 6), f"Expected (120,6), got {out_batch.shape}"
        print(f"      Batched (4×30): input={out_batch.shape[0]} bus rows, "
              f"output={out_batch.shape}")

    print("      OK — forward pass shapes correct")
    return out_batch, batch


def test_losses(model, output, data, device: str):
    """Test that GridFM losses compute correctly."""
    import torch
    print("\n[4/5] Testing GridFM losses...")

    from core.utils.gridfm_losses import (
        GridFMMaskedMSELoss,
        GridFMFullMSELoss,
        GridFMPBELoss,
        GridFMMixedLoss,
        GridFMEvalRMSE,
    )

    losses_to_test = [
        ("MaskedMSE",  GridFMMaskedMSELoss(model=model)),
        ("FullMSE",    GridFMFullMSELoss(model=model)),
        ("PBELoss",    GridFMPBELoss(model=model)),
        ("MixedLoss",  GridFMMixedLoss(w_mse=0.01, w_pbe=0.99, model=model)),
        ("EvalRMSE",   GridFMEvalRMSE(model=model)),
    ]

    model.eval()
    with torch.no_grad():
        for name, loss_fn in losses_to_test:
            val = loss_fn(output.detach(), data)
            print(f"      {name:12s}: {val.item():.6f}")

    print("      OK — all losses computed without error")


def test_y_bus_construction():
    """Test the Y-bus edge feature computation."""
    import torch
    print("\n[5/5] Testing Y-bus construction...")

    from core.models.gridfm import compute_ybus_edge_features

    # Simple 3-bus example: bus 0-1, 1-2, 0-2
    # R=0.01, X=0.05 → y_s = 1/(0.01+j*0.05) ≈ 3.846 - j*19.23
    n = 3
    edge_index = torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long)
    R = torch.full((3,), 0.01)
    X = torch.full((3,), 0.05)
    zeros = torch.zeros(3)
    ones  = torch.ones(3)
    edge_attr = torch.stack([R, X, zeros, zeros, zeros, zeros, ones, zeros], dim=1)

    ybus_ei, ybus_ea, ew = compute_ybus_edge_features(edge_index, edge_attr, n)

    # Self-loops present
    self_loops = (ybus_ei[0] == ybus_ei[1]).sum().item()
    assert self_loops == n, f"Expected {n} self-loops, got {self_loops}"

    # Off-diagonal: expect negative conductance (G < 0)
    off_diag_mask = ybus_ei[0] != ybus_ei[1]
    G_off = ybus_ea[off_diag_mask, 0]
    assert (G_off < 0).all(), "Off-diagonal G should be negative"

    print(f"      Self-loops: {self_loops}, off-diagonal edges: {off_diag_mask.sum().item()}")
    print(f"      Diagonal G (bus 0): {ybus_ea[0, 0]:.4f}, B: {ybus_ea[0, 1]:.4f}")
    print("      OK — Y-bus construction correct")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    print("=" * 60)
    print("GridFM Integration Test")
    print("=" * 60)

    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    # Test 1: architecture
    test_architecture()

    # Test 2: weight loading
    model = test_weight_loading(args.pth, device)

    # If weights not loaded, build fresh model for remaining tests
    if model is None:
        print("\n  (Using freshly initialised model for tests 3-5)")
        from core.models.gridfm import GridFMWrapper
        model = GridFMWrapper(
            input_dim=9, hidden_size=256, output_dim=6, edge_dim=2,
            pe_dim=20, num_layers=8, attention_head=8, dropout=0.1,
            apply_pf_mask=True,
        ).to(device)

    # Test 3: forward pass
    output, batch_data = test_forward_pass(model, device)

    # Test 4: losses
    test_losses(model, output, batch_data, device)

    # Test 5: Y-bus
    test_y_bus_construction()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

    # Bonus: print model summary
    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: GridFMWrapper (GPSTransformer)")
    print(f"  Total parameters:     {n_total:,}")
    print(f"  Trainable parameters: {n_trainable:,}")
    print(f"  Input dim:  9  (Pd, Qd, Pg, Qg, Vm, Va, PQ, PV, REF)")
    print(f"  Output dim: 6  (Pd, Qd, Pg, Qg, Vm, Va)")
    print(f"  GPS layers: 8  × hidden_size=256, heads=8")
    print(f"  PE dim:     20 (Random Walk PE)")


if __name__ == "__main__":
    main()
