from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve

import torch
from torch_geometric.data import Data

from core.utils.main_utils import load_registry
from core.utils.registry import registry

MODEL_URL = "https://github.com/gridfm/gridfm-graphkit/raw/main/examples/models/GridFM_v0_2.pth"


def build_dummy_batch() -> Data:
    num_nodes = 4
    x = torch.zeros((num_nodes, 9), dtype=torch.float32)
    x[0, 0] = 1.0
    x[1, 1] = 1.0
    x[2, 2] = 1.0
    x[3, 0] = 1.0

    pe = torch.zeros((num_nodes, 20), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.rand((edge_index.size(1), 2), dtype=torch.float32) * 0.01

    data = Data(x=x, pe=pe, edge_index=edge_index, edge_attr=edge_attr)
    data.mask = torch.zeros((num_nodes, 6), dtype=torch.bool)
    data.batch = torch.zeros(num_nodes, dtype=torch.long)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default="GridFM_v0_2.pth")
    parser.add_argument("--keep-checkpoint", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)

    if not checkpoint_path.exists():
        print(f"Downloading checkpoint to {checkpoint_path}...")
        urlretrieve(MODEL_URL, checkpoint_path)

    load_registry()
    model_class = registry.get_model_class("gridfm_gps")
    model = model_class()

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_graphkit_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    model.eval()
    with torch.no_grad():
        out = model(build_dummy_batch())
    print(f"Forward pass OK. Output shape: {tuple(out.shape)}")

    if not args.keep_checkpoint and checkpoint_path.exists():
        os.remove(checkpoint_path)
        print(f"Deleted checkpoint file: {checkpoint_path}")


if __name__ == "__main__":
    main()
