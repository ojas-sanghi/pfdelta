#!/bin/bash

#SBATCH --time=168:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

set -euo pipefail

for i in {0..179}; do
  config="gen_feb0126_graphconv_task31/gen_feb0126_graphconv_task31_${i}"
  echo "Running config: $config"
  uv run python main.py --config "$config"
done

for i in {0..35}; do
  config="gen_feb0126_graphconv_task32/gen_feb0126_graphconv_task32_${i}"
  echo "Running config: $config"
  uv run python main.py --config "$config"
done

for i in {0..35}; do
  config="gen_feb0126_graphconv_task33/gen_feb0126_graphconv_task33_${i}"
  echo "Running config: $config"
  uv run python main.py --config "$config"
done

for i in {0..143}; do
  config="gen_feb0126_graphconv_task34/gen_feb0126_graphconv_task34_${i}"
  echo "Running config: $config"
  uv run python main.py --config "$config"
done
