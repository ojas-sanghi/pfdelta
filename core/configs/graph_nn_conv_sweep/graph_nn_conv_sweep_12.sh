#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH -p cpu-gpu-rtx8000
#SBATCH --gpus=1

uv run python main.py --config graph_nn_conv_sweep/graph_nn_conv_sweep_12