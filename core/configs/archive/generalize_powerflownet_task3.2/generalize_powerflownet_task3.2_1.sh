#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

uv run python main.py --config generalize_powerflownet_task3.2/generalize_powerflownet_task3.2_1