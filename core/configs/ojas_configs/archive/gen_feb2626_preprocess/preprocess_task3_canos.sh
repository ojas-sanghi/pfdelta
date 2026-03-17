#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH -p cpu

uv run python main.py --config ojas_configs/gen_feb2626_preprocess/preprocess_task3_canos.yaml