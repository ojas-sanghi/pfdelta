#!/bin/bash

#SBATCH --time=168:00:00
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8


set -euo pipefail

# Access array index
IDX=$SLURM_ARRAY_TASK_ID
config="gen_feb0126_powerflownet_task32/gen_feb0126_powerflownet_task32_$IDX"

echo "Running job $IDX on $SLURM_NODELIST"
echo "Running config: $config"

uv run python main.py --config "$config"

# to run
# sbatch --array=6-11 core/configs/ojas_configs/gen_feb0126/run_powerflownet_task32.sh