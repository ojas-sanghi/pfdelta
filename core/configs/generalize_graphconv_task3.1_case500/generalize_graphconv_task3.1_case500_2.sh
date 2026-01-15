#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

uv run python main.py --config generalize_graphconv_task3.1_case500/generalize_graphconv_task3.1_case500_2