#!/bin/bash

#SBATCH --time=144:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

for i in $(seq 4 11); do
    config="gen_pbl_apr0326_canos_task34/gen_pbl_apr0326_canos_task34_${i}"
    echo "running ${config}"
    uv run python main.py --config "$config"
done