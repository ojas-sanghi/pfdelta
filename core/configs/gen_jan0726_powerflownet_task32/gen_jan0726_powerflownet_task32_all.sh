#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

uv run python main.py --config gen_jan0726_powerflownet_task32/gen_jan0726_powerflownet_task32_0
uv run python main.py --config gen_jan0726_powerflownet_task32/gen_jan0726_powerflownet_task32_1
uv run python main.py --config gen_jan0726_powerflownet_task32/gen_jan0726_powerflownet_task32_2
uv run python main.py --config gen_jan0726_powerflownet_task32/gen_jan0726_powerflownet_task32_3
uv run python main.py --config gen_jan0726_powerflownet_task32/gen_jan0726_powerflownet_task32_4
uv run python main.py --config gen_jan0726_powerflownet_task32/gen_jan0726_powerflownet_task32_5
uv run python main.py --config gen_jan0726_powerflownet_task32/gen_jan0726_powerflownet_task32_6
uv run python main.py --config gen_jan0726_powerflownet_task32/gen_jan0726_powerflownet_task32_7
uv run python main.py --config gen_jan0726_powerflownet_task32/gen_jan0726_powerflownet_task32_8
uv run python main.py --config gen_jan0726_powerflownet_task32/gen_jan0726_powerflownet_task32_9
uv run python main.py --config gen_jan0726_powerflownet_task32/gen_jan0726_powerflownet_task32_10
uv run python main.py --config gen_jan0726_powerflownet_task32/gen_jan0726_powerflownet_task32_11