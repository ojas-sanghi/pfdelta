#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

uv run python main.py --config generalize_powerflownet_task3.3/generalize_powerflownet_task3.3_2