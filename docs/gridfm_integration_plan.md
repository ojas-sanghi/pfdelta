# GridFM Integration Plan for PFDelta

## Goal
Integrate GridFM (GraphKit v0.2 style GPS model) into PFDelta for:
1. Zero-shot checkpoint loading and inference.
2. Fine-tuning on PFDelta tasks.
3. Future training from scratch with the same architecture.

## Detailed step-by-step plan

1. **Understand PFDelta execution stack**
   1. Inspect registry loading and dynamic module discovery.
   2. Inspect `BaseTrainer` / `GNNTrainer` forward-loss contracts.
   3. Confirm dataset variants and transform insertion points.
2. **Understand GridFM GraphKit implementation**
   1. Inspect `GPSTransformer` architecture details (encoder, GPSConv stack, decoder).
   2. Inspect dataset feature layout and edge feature semantics (`[G, B]`).
   3. Inspect masking objective for feature reconstruction.
   4. Inspect checkpoint loading flow and v0.2 model hyperparameters.
3. **Design PFDelta-native GridFM compatibility layer**
   1. Define homogeneous sample format compatible with GridFM (`x`, `y`, `mask`, `pe`, `edge_attr`).
   2. Define mapping from PFDelta HeteroData to GridFM features:
      - Node: `[Pd, Qd, Pg, Qg, Vm, Va, PQ, PV, REF]`
      - Target: first 6 quantities.
      - Edge: admittance approximation `[G, B] = Re/Im(1/(r+jx))`.
   3. Define random-walk positional encoding policy (PE dim = 20, matching v0.2).
4. **Implement model architecture in PFDelta**
   1. Add standalone PFDelta model class registered as `gridfm_gps`.
   2. Match layer ordering and parameter names with GraphKit so state dicts load directly.
5. **Implement dataset adapter**
   1. Add registered dataset `pfdelta_gridfm` derived from `PFDeltaDataset`.
   2. Add transform class that converts each sample to homogeneous `torch_geometric.data.Data`.
   3. Inject mask generation and masked input corruption.
6. **Implement GridFM-compatible loss functions**
   1. Add `gridfm_masked_mse` (default train objective).
   2. Add `gridfm_mse` (full reconstruction metric).
7. **Create runnable configs**
   1. Add zero-shot evaluation-oriented config (few epochs, debug-friendly).
   2. Add fine-tuning config with practical defaults and checkpoint path hook.
8. **Create checkpoint bootstrap + validation script**
   1. Download v0.2 checkpoint.
   2. Instantiate PFDelta GridFM model.
   3. Load state dict and run dummy forward pass.
   4. Delete `.pth` after validation (PR-safe).
9. **Document usage**
   1. Add integration guide with commands for zero-shot and fine-tuning.
   2. Add explicit note on temporary checkpoint handling.
10. **Validate end-to-end locally**
   1. Run checkpoint download/verification script.
   2. Run lightweight import and config smoke checks.
   3. Ensure no `.pth` binary remains in repo.

## Task list to execute
- [x] Add GridFM model implementation.
- [x] Add PFDelta→GridFM dataset adapter.
- [x] Add GridFM losses.
- [x] Add checkpoint download/verify script.
- [x] Add example configs for training/fine-tuning.
- [x] Add integration docs.
- [x] Validate loading and dummy forward.
- [x] Delete downloaded `.pth` file.
