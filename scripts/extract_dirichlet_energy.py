import argparse
import copy
import os
import statistics
import sys
import time

import torch

# from core.trainers.gnn_trainer import GNNTrainer

# Change working directory to one above
sys.path.append(os.getcwd())

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
    # seeds = os.listdir(root)
    # Prefer paths within the provided root to avoid cross-run collisions
    # seeds_paths = [os.path.join(root, seed) for seed in seeds]
    # seeds_configs = [load_config(run) for run in seeds_paths]
    seed_config = load_config(root)

    # Load trainers modified so that the val dataset is on the test split desired
    batch_size = 1
    # for config in seeds_configs:
    val_dataset = seed_config["dataset"]["datasets"][1]
    val_dataset["split"] = "test"
    val_params = seed_config["optim"]["val_params"]
    val_params["batch_size"] = batch_size
    # val_dataset["task"] = 1.3
    # val_dataset["case_name"] = "case500_seeds"
    # case_name = val_dataset["case_name"]
    # val_params["val_loss"].extend(losses_to_analyze_inputs)

    # seeds_trainers = [load_trainer(config) for config in seeds_configs]
    # print(seeds_trainers)
    seeds_trainer = load_trainer(seed_config)

    # Run a single forward pass per run to populate Dirichlet energies
    # seeds_dataloaders = [trainer.dataloaders[1] for trainer in seeds_trainers]
    seeds_dataloader = seeds_trainer.dataloaders[1]

    seeds_trainer.calc_one_val_error(seeds_dataloader, 0, print_de=True)

    # for dataloader, trainer in zip(seeds_dataloaders, seeds_trainers):
    # trainer.calc_one_val_error(dataloader, 0, print_de=True)
