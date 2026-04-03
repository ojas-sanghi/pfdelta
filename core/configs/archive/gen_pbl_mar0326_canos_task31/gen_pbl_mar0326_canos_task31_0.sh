#!/bin/bash

#SBATCH --time=144:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

uv run python main.py --config gen_pbl_mar0326_canos_task31/gen_pbl_mar0326_canos_task31_0