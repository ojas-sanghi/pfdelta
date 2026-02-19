"""Download GridFM v0.2 checkpoint and verify it loads in PFDelta implementation."""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from urllib.request import urlretrieve

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.models.gridfm_gps import GridFMGPSTransformer

GRIDFM_V02_URL = "https://github.com/gridfm/gridfm-graphkit/blob/main/examples/models/GridFM_v0_2.pth?raw=1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tmp/GridFM_v0_2.pth")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading checkpoint to {out}...")
    urlretrieve(GRIDFM_V02_URL, out)

    raw_state_dict = torch.load(out, map_location="cpu")
    state_dict = {
        (k[6:] if k.startswith("model.") else k): v
        for k, v in raw_state_dict.items()
        if isinstance(k, str)
    }
    model = GridFMGPSTransformer()
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    with torch.no_grad():
        x = torch.zeros((4, 9))
        pe = torch.zeros((4, 20))
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_attr = torch.zeros((4, 2))
        batch = torch.zeros(4, dtype=torch.long)
        from torch_geometric.data import Data

        dummy = Data(x=x, pe=pe, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        y = model(dummy)
        print("Forward pass OK. output shape:", tuple(y.shape))

    if not args.keep and out.exists():
        os.remove(out)
        print(f"Deleted checkpoint: {out}")


if __name__ == "__main__":
    main()
