# GridFM ↔ PFDelta Integration Plan (Detailed)

1. **Inventory PFDelta training stack**
   - Confirm registry-based model/dataset discovery.
   - Confirm `gnn_trainer` is compatible with PyG `HeteroData` batches.
   - Confirm config/loss plumbing for custom output names.
2. **Inventory GridFM graphkit stack**
   - Identify model class (`GPSTransformer`) and exact layer naming.
   - Record expected node schema `[Pd,Qd,Pg,Qg,Vm,Va,PQ,PV,REF]` and edge schema `[G,B]`.
   - Record checkpoint conventions (`state_dict` vs direct dict).
3. **Define PFDelta→GridFM feature contract**
   - Node features from `bus_demand`, `bus_gen`, `bus_voltages`, `bus_type`.
   - Target tensor of first six reconstruction channels.
   - Edge feature conversion from `(r,x)` to admittance `(g,b)`.
4. **Implement reusable adapter utilities**
   - Create utility functions to attach `x_gridfm`, `y_gridfm`, `edge_attr_gridfm`.
   - Keep API in-place so any model/dataset can call it.
5. **Implement registered GridFM model in PFDelta**
   - Recreate GPS/GINE stack with matching module names for checkpoint compatibility.
   - Add optional checkpoint loading during init.
   - Add output mode for either full GridFM channels or PFDelta voltage pair.
6. **Implement dataset variant for preprocessing compatibility**
   - Add `pfdeltaGridFM` class inheriting `PFDeltaDataset`.
   - Materialize GridFM fields in `build_heterodata`.
7. **Add train/eval configs**
   - Add zero-shot config (loads checkpoint, 1 epoch smoke setup).
   - Add fine-tune config (loads checkpoint, standard training).
8. **Checkpoint verification workflow**
   - Download `GridFM_v0_2.pth` under repo `artifacts/`.
   - Verify state dict load against new model.
   - Run dummy forward pass and shape validation.
9. **Training smoke test workflow**
   - Run lightweight script with synthetic PFDelta-like `HeteroData`.
   - Verify model output can be consumed by `GNNTorchLoss` with `bus__y_gridfm`.
10. **Finalize for PR safety**
   - Remove downloaded `.pth` binary from git workspace.
   - Summarize exact run commands.
   - Commit and open PR.

## Task Checklist
- [x] Read PFDelta model/trainer/data internals.
- [x] Read GridFM graphkit architecture internals.
- [x] Implement architecture-compatible GridFM model.
- [x] Implement PFDelta dataset compatibility adapter.
- [x] Add configs for zero-shot and fine-tuning.
- [x] Download+validate GridFM v0.2 checkpoint load.
- [x] Delete checkpoint binary after validation.
- [ ] Commit changes.
- [ ] Create PR message via tool.
