#!/bin/bash

#SBATCH --time=168:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

set -euo pipefail

for i in {0..11}; do
  config="gen_feb0126_powerflownet_task32/gen_feb0126_powerflownet_task32_${i}"
  echo "Running config: $config"
  uv run python main.py --config "$config"
done

for i in {0..11}; do
  config="gen_feb0126_powerflownet_task33/gen_feb0126_powerflownet_task33_${i}"
  echo "Running config: $config"
  uv run python main.py --config "$config"
done

for i in {0..47}; do
  config="gen_feb0126_powerflownet_task34/gen_feb0126_powerflownet_task34_${i}"
  echo "Running config: $config"
  uv run python main.py --config "$config"
done

# gpu version launched feb 12 9:40 am aka 1140

# 260212_1140