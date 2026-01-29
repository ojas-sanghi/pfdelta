#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH -p cpu

uv run python main.py --config gen_jan0726_preprocess/preprocess_task3.yaml