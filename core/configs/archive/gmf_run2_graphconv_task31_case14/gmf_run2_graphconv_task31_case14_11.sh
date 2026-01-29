#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

uv run python main.py --config gmf_run2_graphconv_task31_case14/gmf_run2_graphconv_task31_case14_11