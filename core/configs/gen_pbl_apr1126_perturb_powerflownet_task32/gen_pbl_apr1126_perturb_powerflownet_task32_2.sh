#!/bin/bash

#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH --exclude=node1928,node4007
#SBATCH --requeue
#SBATCH --mem=64G
#SBATCH -G 1
#SBATCH --output=powerflownet_pbl_perturb_%j.out
#SBATCH --signal=USR1@90
#SBATCH --time=05:59:00

source ~/.bashrc
uv run python main.py --config gen_pbl_apr1126_perturb_powerflownet_task32/gen_pbl_apr1126_perturb_powerflownet_task32_2