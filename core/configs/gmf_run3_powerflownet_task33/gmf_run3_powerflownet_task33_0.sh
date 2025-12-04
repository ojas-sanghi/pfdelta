#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

uv run python main.py --config gmf_run3_powerflownet_task33/gmf_run3_powerflownet_task33_0