#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

uv run python main.py --config sageconv_sweep/sageconv_sweep_11