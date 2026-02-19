# GridFM Integration Plan for PFΔ

## Goal
Integrate GridFM (v0.2 GPSTransformer architecture) into PFΔ so the repository supports:
1. **Zero-shot evaluation readiness** with official GridFM weights.
2. **Fine-tuning** on PFΔ datasets.
3. **From-scratch training** of the same architecture in PFΔ's trainer/config stack.

## Design Constraints
- Preserve PFΔ's current `registry` + `trainer` + YAML config flow.
- Keep GridFM integration isolated and additive (new files where possible).
- Make checkpoint loading compatible with GraphKit `GridFM_v0_2.pth` format.
- Provide deterministic preprocessing from PFΔ `HeteroData` into GridFM-compatible homogeneous graphs.

---

## Step-by-Step Implementation Plan

### Phase 1 — Deep-read & mapping
1. Map PFΔ internals:
   - Dataset lifecycle (`PFDeltaDataset`, processed cache behavior, split handling).
   - Model registry and trainer interfaces.
   - Loss function plugin behavior.
2. Map GridFM GraphKit internals:
   - `GPSTransformer` architecture.
   - Data contract (`x`, `y`, `edge_index`, `edge_attr`, `pe`, `mask`).
   - Checkpoint loading behavior (`state_dict` compatibility).
3. Reconcile differences:
   - PFΔ uses `HeteroData`; GridFM uses homogeneous `Data`.
   - PFΔ branch attributes are richer (8-dim); GridFM expects admittance-like (2-dim).
   - PFΔ trainers expect model output + configurable losses; GridFM paper/task emphasizes masked feature reconstruction.

### Phase 2 — Dataset adapter implementation
4. Implement `PFDeltaGridFM` dataset variant (registered):
   - Subclass `PFDeltaDataset`.
   - Reuse base parsing logic, then convert each sample into homogeneous `Data`.
5. Build node features exactly for GridFM-style reconstruction:
   - Input `x`: `[PQ_onehot, PV_onehot, REF_onehot, Pd, Qd, Pg, Qg, Vm, Va]` (9 dims).
   - Target `y`: `[Pd, Qd, Pg, Qg, Vm, Va]` (6 dims).
6. Build edge features:
   - Use PFΔ branch edge columns and map to `[G, B]` (2 dims) from branch-side conductance/susceptance fields.
   - Compute `edge_weight = sqrt(G^2 + B^2)`.
7. Add positional encoding:
   - Compute random-walk positional encodings of configurable walk length (`pe_dim`, default 20).
   - Store as `data.pe`.
8. Add masking strategies for reconstruction:
   - `none`: no masking.
   - `rnd`: random Bernoulli mask for all 6 targets.
   - `pf`: PF-structured mask by bus type (PQ/PV/REF).

### Phase 3 — Model implementation
9. Implement PFΔ-native GridFM model class (registered as `gridfm_gps`):
   - Re-implement GraphKit GPSTransformer stack with `GPSConv + GINEConv`.
   - Include learnable/non-learnable mask token vector.
10. Make model accept PFΔ batch objects directly:
    - `forward(data)` where `data` is PyG `Data` batch.
    - Apply masking to first six continuous channels inside `x` before encoding.
11. Add helper for checkpoint loading:
    - Method to load GraphKit state dict with informative key diagnostics.

### Phase 4 — Loss & training integration
12. Implement `gridfm_masked_mse` loss (registered):
    - If `data.mask` present: evaluate only masked entries.
    - Optional fallback to all-entry MSE when mask is empty.
13. Ensure compatibility with existing `gnn_trainer`:
    - No trainer changes needed if model returns tensor and loss consumes `(outputs, data)`.

### Phase 5 — Config & tooling
14. Add ready-to-run PFΔ config for GridFM v0.2-style fine-tuning.
15. Add a smoke-test script that:
    - Downloads official `GridFM_v0_2.pth`.
    - Instantiates PFΔ GridFM model.
    - Loads state dict.
    - Runs one dummy forward pass and one PFΔ sample pass (if data available).
16. Document exact execution commands for zero-shot/fine-tuning.

### Phase 6 — Validation and cleanup
17. Run local validations:
    - Import/load checks.
    - Checkpoint load check.
    - Dummy-batch forward output check.
18. Downloaded binary cleanup requirement:
    - Delete `GridFM_v0_2.pth` before finalizing (avoid committing binaries).
19. Stage, commit, and open PR with implementation summary.

---

## Task Checklist (Execution)
- [ ] Add plan document.
- [ ] Implement dataset adapter.
- [ ] Implement model.
- [ ] Implement masked loss.
- [ ] Add config.
- [ ] Add smoke-test/download script.
- [ ] Validate checkpoint loading and forward pass.
- [ ] Remove downloaded `.pth` artifact.
- [ ] Commit.
- [ ] Create PR.
