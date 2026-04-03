#!/bin/bash

#SBATCH --time=144:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

for i in $(seq 0 14); do
    config="gen_apr0326_canos_task31/gen_apr0326_canos_task31_${i}"
    echo "running ${config}"
    uv run python main.py --config "$config"
done