#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH -p cpu

uv run python main.py --config ojas_configs/gen_feb0126_preprocess/preprocess_task3.yaml