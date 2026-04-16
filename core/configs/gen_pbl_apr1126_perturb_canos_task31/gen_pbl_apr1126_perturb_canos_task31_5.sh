#!/bin/bash

#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH --exclude=node1928,node4007
#SBATCH --requeue
#SBATCH --mem=64G
#SBATCH -G 1
#SBATCH --output=canos_pbl_%j.out
#SBATCH --signal=USR1@90
#SBATCH --time=05:59:00

source ~/.bashrc
uv run python main.py --config gen_pbl_apr1126_perturb_canos_task31/gen_pbl_apr1126_perturb_canos_task31_5