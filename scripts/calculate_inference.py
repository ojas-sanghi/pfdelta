import sys
import os
import json
import argparse
import copy
import pickle as pkl
from tqdm import tqdm

import IPython

# Change working directory to one above
sys.path.append(os.getcwd())

import torch

from scripts.utils import find_run, load_config, load_trainer


def parser():
    parser = argparse.ArgumentParser(
        description="Loads the trainer and the trainable weights."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="",
        required=True,
        help="Folder from which to calculate test errors.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Load run paths and configs
    args = parser()
    root = os.path.join("runs", args.root)
    seeds = os.listdir(root)
    seeds_paths = [find_run(seed) for seed in seeds]
    seeds_configs = [load_config(run) for run in seeds_paths]

    # Process config files
    for config in seeds_configs:
        # Modify loss calculations
        losses = config["optim"]["train_params"]["train_loss"]
        for loss in losses:
            if loss["name"] == "universal_power_balance":
                pbl_model_name = loss["model"]
                break
        losses = [
            {
                "name": "universal_power_balance",
                "model": pbl_model_name
            }
        ]
        config["optim"]["val_params"]["val_loss"] = losses

        # Modify datasets
        base_dataset = config["dataset"]["datasets"][0]
        base_dataset["task"] = 1.3
        case_name = base_dataset["case_name"]
        datasets = [
            {
                **copy.deepcopy(base_dataset),
                "split": f"separate_{case_name}_test_feasible_n"
            },
            {
                **copy.deepcopy(base_dataset),
                "split": f"separate_{case_name}_test_feasible_n-1"
            },
            {
                **copy.deepcopy(base_dataset),
                "split": f"separate_{case_name}_test_feasible_n-2"
            },
            {
                **copy.deepcopy(base_dataset),
                "split": f"separate_{case_name}_test_near infeasible_n"
            },
            {
                **copy.deepcopy(base_dataset),
                "split": f"separate_{case_name}_test_near infeasible_n-1"
            },
            {
                **copy.deepcopy(base_dataset),
                "split": f"separate_{case_name}_test_near infeasible_n-2"
            },
        ]
        config["dataset"] = {
            "datasets": datasets
        }
        config["optim"]["train_params"]["batch_size"] = 1
        config["optim"]["val_params"]["batch_size"] = 1

    trainer = load_trainer(seeds_configs[0]) # ONLY ONE SEED
    output_types = {}
    pbl_loss = trainer.val_loss[0]
    model_name = pbl_loss.model
    device = trainer.device
    for dataset_type, dataloader in zip(
        ["n", "n-1", "n-2",  "c2i-n", "c2i-n1", "c2i-n2"],
        trainer.dataloaders
    ):
        print("Calculating", dataset_type)

        # Calculating inference time
        outputs_list = []
        message = "Tracking model outputs..."
        for data in tqdm(dataloader, desc=message):
            data = data.to(device)

            # Calculate output
            with torch.no_grad():
                output = trainer.model(data)

            out = pbl_loss.collect_model_predictions(
                model_name,
                data,
                output
            )
            out["data"] = data
            outputs_list.append(out)

        output_types[dataset_type] = outputs_list

    output_types["close2inf"] = (
        output_types["c2i-n"] +
        output_types["c2i-n1"] +
        output_types["c2i-n2"]
    )

    keys = ["n", "n-1", "n-2", "close2inf"]
    all_outputs = {
        "n": [],
        "n-1": [],
        "n-2": [],
        "close2inf": [],
    }
    for key in keys:
        all_outputs[key].append(output_types[key])

    root = root.replace("/", "_")
    torch.save(all_outputs, root+".pkl")

#     # Save specific values
#     if os.path.exists("case118_inference.pkl"):
#         results = torch.load(
#             "case118_inference.pkl",
#             map_location="cpu"
#         )
#     else:
#         results = {}
# 
#     if root not in results:
#         results[root] = {}
# 
#     results[root] = all_outputs
#     torch.save(results, "case118_inference.pkl")
