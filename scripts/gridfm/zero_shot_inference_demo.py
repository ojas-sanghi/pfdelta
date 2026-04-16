"""Run a simple zero-shot inference smoke test on PFDelta samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from torch_geometric.loader import DataLoader

from core.datasets.pfdelta_gridfm import PFDeltaGridFMDataset
from core.models.gridfm_gps import GridFMGPSTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root_dir", default="data")
    parser.add_argument("--case_name", default="case14")
    parser.add_argument("--task", type=float, default=1.3)
    args = parser.parse_args()

    dataset = PFDeltaGridFMDataset(
        root_dir=args.root_dir,
        case_name=args.case_name,
        split="test",
        task=args.task,
        model="gridfm",
        n_samples=32,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = GridFMGPSTransformer()
    raw_state_dict = torch.load(args.checkpoint, map_location="cpu")
    state_dict = {
        (k[6:] if k.startswith("model.") else k): v
        for k, v in raw_state_dict.items()
        if isinstance(k, str)
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    batch = next(iter(loader))
    with torch.no_grad():
        pred = model(batch)

    print("Batch nodes:", batch.num_nodes)
    print("Pred shape:", tuple(pred.shape))
    print("Target shape:", tuple(batch.y.shape))


if __name__ == "__main__":
    main()
