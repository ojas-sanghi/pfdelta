# GridFM Integration in PFDelta

This repository now includes a PFDelta-native implementation of the GridFM v0.2 GPS architecture and a dataset adapter for PFDelta data.

## Added components

- `core/models/gridfm_gps.py`: `gridfm_gps` model (GPSConv + GINEConv stack).
- `core/datasets/pfdelta_gridfm.py`: `pfdelta_gridfm` dataset adapter from PFDelta HeteroData to GridFM homogeneous `Data`.
- `core/utils/gridfm_losses.py`: `gridfm_masked_mse` and `gridfm_mse`.
- `core/configs/examples/gridfm_task_1_3.yaml`: baseline config.
- `core/configs/examples/gridfm_finetune_task_1_3.yaml`: fine-tuning config.
- `scripts/gridfm/download_and_verify_gridfm.py`: checkpoint download + load verification.
- `scripts/gridfm/zero_shot_inference_demo.py`: zero-shot smoke test.

## Download + verify GridFM v0.2 checkpoint

```bash
uv run python scripts/gridfm/download_and_verify_gridfm.py
```

This command downloads `GridFM_v0_2.pth`, loads it into `gridfm_gps`, performs a dummy forward pass, and deletes the `.pth` file automatically.

## Zero-shot smoke test on PFDelta

```bash
uv run python scripts/gridfm/download_and_verify_gridfm.py --out tmp/GridFM_v0_2.pth --keep
uv run python scripts/gridfm/zero_shot_inference_demo.py --checkpoint tmp/GridFM_v0_2.pth --case_name case14 --task 1.3
rm -f tmp/GridFM_v0_2.pth
```

## Fine-tuning

```bash
uv run python main.py --config examples/gridfm_finetune_task_1_3
```

For actual experiments, set `functional.is_debug: false` and adjust the dataset case/task/batch size.
