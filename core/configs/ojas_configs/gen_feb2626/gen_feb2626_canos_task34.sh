#!/bin/bash

#SBATCH --time=144:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

# uv run python main.py --config gen_feb2626_canos_task34/gen_feb2626_canos_task34_0
# uv run python main.py --config gen_feb2626_canos_task34/gen_feb2626_canos_task34_1
# uv run python main.py --config gen_feb2626_canos_task34/gen_feb2626_canos_task34_2
# uv run python main.py --config gen_feb2626_canos_task34/gen_feb2626_canos_task34_3
uv run python main.py --config gen_feb2626_canos_task34/gen_feb2626_canos_task34_4
uv run python main.py --config gen_feb2626_canos_task34/gen_feb2626_canos_task34_5
uv run python main.py --config gen_feb2626_canos_task34/gen_feb2626_canos_task34_6
uv run python main.py --config gen_feb2626_canos_task34/gen_feb2626_canos_task34_7
uv run python main.py --config gen_feb2626_canos_task34/gen_feb2626_canos_task34_8
uv run python main.py --config gen_feb2626_canos_task34/gen_feb2626_canos_task34_9
uv run python main.py --config gen_feb2626_canos_task34/gen_feb2626_canos_task34_10
uv run python main.py --config gen_feb2626_canos_task34/gen_feb2626_canos_task34_11