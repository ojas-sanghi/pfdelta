#!/bin/bash

#SBATCH --time=168:00:00
#SBATCH -p cpu

set -euo pipefail

# Access array index
IDX=$SLURM_ARRAY_TASK_ID
config="gen_feb0126_powerflownet_task34/gen_feb0126_powerflownet_task34_$IDX"

echo "Running job $IDX on $SLURM_NODELIST"
echo "Running config: $config"

uv run python main.py --config "$config"

# to run
# sbatch --array=0-47 core/configs/ojas_configs/gen_feb0126/run_powerflownet_task34.sh