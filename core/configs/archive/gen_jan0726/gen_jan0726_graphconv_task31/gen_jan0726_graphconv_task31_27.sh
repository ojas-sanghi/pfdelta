#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

uv run python main.py --config gen_jan0726_graphconv_task31/gen_jan0726_graphconv_task31_27