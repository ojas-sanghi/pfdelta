#!/bin/bash

#SBATCH --time=144:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

uv run python main.py --config gen_feb2626_canos_task32/gen_feb2626_canos_task32_2