#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1

uv run python main.py --config gen_jan0726_powerflownet_task33/gen_jan0726_powerflownet_task33_0
uv run python main.py --config gen_jan0726_powerflownet_task33/gen_jan0726_powerflownet_task33_1
uv run python main.py --config gen_jan0726_powerflownet_task33/gen_jan0726_powerflownet_task33_2
uv run python main.py --config gen_jan0726_powerflownet_task33/gen_jan0726_powerflownet_task33_3
uv run python main.py --config gen_jan0726_powerflownet_task33/gen_jan0726_powerflownet_task33_4
uv run python main.py --config gen_jan0726_powerflownet_task33/gen_jan0726_powerflownet_task33_5
uv run python main.py --config gen_jan0726_powerflownet_task33/gen_jan0726_powerflownet_task33_6
uv run python main.py --config gen_jan0726_powerflownet_task33/gen_jan0726_powerflownet_task33_7
uv run python main.py --config gen_jan0726_powerflownet_task33/gen_jan0726_powerflownet_task33_8
uv run python main.py --config gen_jan0726_powerflownet_task33/gen_jan0726_powerflownet_task33_9
uv run python main.py --config gen_jan0726_powerflownet_task33/gen_jan0726_powerflownet_task33_10
uv run python main.py --config gen_jan0726_powerflownet_task33/gen_jan0726_powerflownet_task33_11