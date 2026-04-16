# GridFM ↔ PFDelta Integration Plan (Detailed)

## Phase 0 — Discovery and gap analysis
1. Read PFDelta entrypoints (`main.py`, `core/trainers`, `core/datasets`) to identify integration points.
2. Read GridFM GraphKit architecture and training flow, with emphasis on GPSConv + GINE stack and masking objectives.
3. Map PFDelta `HeteroData` schema to GridFM expectations.
4. Define two tracks:
   - **Track A (zero-shot compatibility):** load pretrained `GridFM_v0_2.pth` and run an inference smoke test.
   - **Track B (fine-tuning readiness):** provide trainable model + dataset + config in PFDelta.

## Phase 1 — Data interface layer
5. Create a PFDelta dataset variant that emits GridFM-style tensors on bus nodes:
   - `bus.x_gridfm` with `[PD, QD, PG, QG, VM, VA, PQ, PV, REF]`
   - `bus.y_gridfm` with `[PD, QD, PG, QG, VM, VA]`
   - `bus.pe_gridfm` positional encodings.
6. Keep base PFDelta graph topology and edge attributes unchanged to preserve compatibility with existing data loaders.

## Phase 2 — Model architecture implementation
7. Implement a native PFDelta model class reproducing GridFM GraphKit's core GPSTransformer pattern:
   - input encoder + PE concat
   - stack of `GPSConv(GINEConv)` blocks
   - decoder head
8. Expose configurable dimensions/keys to support:
   - zero-shot reconstruction mode (`output_dim=6`)
   - supervised PF fine-tuning mode (`output_dim=2`, optionally loading partial pretrained weights).

## Phase 3 — Training/fine-tuning integration
9. Extend model loading in trainer to support optional `model.pretrained_path` and non-strict loading.
10. Add configs for:
   - GridFM fine-tuning on PFDelta using `GNNTorchLoss` with `bus__y_gridfm`.
   - Transfer/fine-tune to PF targets (`bus.y`) by swapping output dimension/loss if desired.

## Phase 4 — Validation and safety checks
11. Download pretrained checkpoint from provided URL.
12. Verify checkpoint loadability with `torch.load` and optional state-dict key normalization.
13. Run local smoke tests with dummy graph batch for forward pass shape checks.
14. Run a short PFDelta pipeline smoke test script for zero-shot/fine-tune readiness.
15. Remove downloaded `.pth` file from repository to keep PR text-only.

## Phase 5 — Handoff readiness
16. Provide runnable commands using `uv run ...` for:
   - checkpoint validation
   - quick dummy inference
   - fine-tuning run.
17. Document implementation details + constraints (e.g., positional encoding approximation and data-objective mismatch risks).

## Task checklist status
- [x] Architecture study and mapping
- [x] PFDelta dataset variant for GridFM tensors
- [x] GridFM model implementation
- [x] Pretrained loading support in trainer
- [x] Example configs and smoke-test script
- [x] Download/load/validate model file
- [x] Delete `.pth` artifact before commit
