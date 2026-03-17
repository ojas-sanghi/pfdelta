#!/bin/bash

#SBATCH --time=144:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

# uv run python main.py --config gen_feb2626_canos_task31/gen_feb2626_canos_task31_0
# uv run python main.py --config gen_feb2626_canos_task31/gen_feb2626_canos_task31_1
# uv run python main.py --config gen_feb2626_canos_task31/gen_feb2626_canos_task31_2
# uv run python main.py --config gen_feb2626_canos_task31/gen_feb2626_canos_task31_3
# uv run python main.py --config gen_feb2626_canos_task31/gen_feb2626_canos_task31_4
# uv run python main.py --config gen_feb2626_canos_task31/gen_feb2626_canos_task31_5
# uv run python main.py --config gen_feb2626_canos_task31/gen_feb2626_canos_task31_6
# uv run python main.py --config gen_feb2626_canos_task31/gen_feb2626_canos_task31_7
uv run python main.py --config gen_feb2626_canos_task31/gen_feb2626_canos_task31_8
uv run python main.py --config gen_feb2626_canos_task31/gen_feb2626_canos_task31_9
uv run python main.py --config gen_feb2626_canos_task31/gen_feb2626_canos_task31_10
uv run python main.py --config gen_feb2626_canos_task31/gen_feb2626_canos_task31_11
uv run python main.py --config gen_feb2626_canos_task31/gen_feb2626_canos_task31_12
uv run python main.py --config gen_feb2626_canos_task31/gen_feb2626_canos_task31_13
uv run python main.py --config gen_feb2626_canos_task31/gen_feb2626_canos_task31_14