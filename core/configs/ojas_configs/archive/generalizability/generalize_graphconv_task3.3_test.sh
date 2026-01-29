#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

uv run python main.py --config ojas_configs/generalize_graphconv_task3.3_test.yaml