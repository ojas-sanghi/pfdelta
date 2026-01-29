#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

uv run python main.py --config gmf_run1_graphconv_task31_case57/gmf_run1_graphconv_task31_case57_2