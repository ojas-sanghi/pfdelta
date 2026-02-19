# GridFM Integration Notes

This repository now includes a PFΔ-native GridFM v0.2 path:

- Dataset adapter: `pfdeltaGridFM` converts PFΔ `HeteroData` to homogeneous GraphKit-like tensors.
- Model: `gridfm_gps` re-implements the GPSTransformer (`GPSConv + GINEConv`) used in GridFM.
- Loss: `gridfm_masked_mse` supports masked reconstruction fine-tuning.
- Smoke test: `scripts/gridfm/load_and_smoke_test.py` downloads and loads official weights, runs a dummy forward pass, and (by default) deletes the `.pth` file.

## Run smoke test

```bash
uv run python scripts/gridfm/load_and_smoke_test.py
```

## Run fine-tuning (example config)

```bash
uv run python main.py --config gridfm/gridfm_v0_2_pfdelta_task_1_3_case14
```

Set `dataset.root_dir` in the YAML to the PFΔ data location before training.
