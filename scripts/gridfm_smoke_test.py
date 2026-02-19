import argparse
import torch
from torch_geometric.data import HeteroData

from core.models.gridfm_transformer import GridFMTransformer


def build_dummy_graph(num_nodes: int = 12, num_edges: int = 24):
    data = HeteroData()
    data["bus"].x_gridfm = torch.randn(num_nodes, 9)
    data["bus"].pe_gridfm = torch.randn(num_nodes, 20)
    data["bus"].batch = torch.zeros(num_nodes, dtype=torch.long)
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    data["bus", "branch", "bus"].edge_index = torch.stack([src, dst], dim=0)
    data["bus", "branch", "bus"].edge_attr_gridfm = torch.randn(num_edges, 2)
    data["bus", "branch", "bus"].edge_attr = torch.randn(num_edges, 8)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output-dim", type=int, default=6)
    args = parser.parse_args()

    model = GridFMTransformer(output_dim=args.output_dim)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if isinstance(ckpt, dict):
            ckpt = {k.replace("model.", ""): v for k, v in ckpt.items()}
        model_state = model.state_dict()
        ckpt = {k: v for k, v in ckpt.items() if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)}
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")

    data = build_dummy_graph()
    output = model(data)
    print("Output shape:", tuple(output.shape))


if __name__ == "__main__":
    main()
