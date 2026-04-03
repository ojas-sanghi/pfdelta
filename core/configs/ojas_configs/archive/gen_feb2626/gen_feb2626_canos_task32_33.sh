#!/bin/bash

#SBATCH --time=144:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

# uv run python main.py --config gen_feb2626_canos_task32/gen_feb2626_canos_task32_0
# uv run python main.py --config gen_feb2626_canos_task32/gen_feb2626_canos_task32_1
# uv run python main.py --config gen_feb2626_canos_task32/gen_feb2626_canos_task32_2

uv run python main.py --config gen_feb2626_canos_task33/gen_feb2626_canos_task33_0
uv run python main.py --config gen_feb2626_canos_task33/gen_feb2626_canos_task33_1
uv run python main.py --config gen_feb2626_canos_task33/gen_feb2626_canos_task33_2