#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

uv run python main.py --config gmf_run3_graphconv_task33/gmf_run3_graphconv_task33_5