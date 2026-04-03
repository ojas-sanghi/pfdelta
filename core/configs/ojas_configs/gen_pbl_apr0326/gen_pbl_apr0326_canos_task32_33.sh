#!/bin/bash

#SBATCH --time=144:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

for i in $(seq 0 2); do
    config="gen_pbl_apr0326_canos_task32/gen_pbl_apr0326_canos_task32_${i}"
    echo "running ${config}"
    uv run python main.py --config "$config"
done

for i in $(seq 0 2); do
    config="gen_pbl_apr0326_canos_task33/gen_pbl_apr0326_canos_task33_${i}"
    echo "running ${config}"
    uv run python main.py --config "$config"
done