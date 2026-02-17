#!/bin/bash

#SBATCH --time=168:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

set -euo pipefail

for i in {0..59}; do
  config="gen_feb0126_powerflownet_task31/gen_feb0126_powerflownet_task31_${i}"
  echo "Running config: $config"
  uv run python main.py --config "$config"
done
