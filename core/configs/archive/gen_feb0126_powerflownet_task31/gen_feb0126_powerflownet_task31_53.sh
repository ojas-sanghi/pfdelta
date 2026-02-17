#!/bin/bash

#SBATCH --time=144:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

uv run python main.py --config gen_feb0126_powerflownet_task31/gen_feb0126_powerflownet_task31_53