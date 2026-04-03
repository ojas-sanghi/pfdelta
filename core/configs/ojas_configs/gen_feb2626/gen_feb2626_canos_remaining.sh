#!/bin/bash

#SBATCH --time=144:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

uv run python main.py --config gen_feb2626_canos_task31/gen_feb2626_canos_task31_14

uv run python main.py --config gen_feb2626_canos_task33/gen_feb2626_canos_task33_1
uv run python main.py --config gen_feb2626_canos_task33/gen_feb2626_canos_task33_2

uv run python main.py --config gen_feb2626_canos_task34/gen_feb2626_canos_task34_8
uv run python main.py --config gen_feb2626_canos_task34/gen_feb2626_canos_task34_9
uv run python main.py --config gen_feb2626_canos_task34/gen_feb2626_canos_task34_10
uv run python main.py --config gen_feb2626_canos_task34/gen_feb2626_canos_task34_11
