sbatch --job-name=generalize_powerflownet_task3.1_case14_lr5e-4 --time=48:00:00 -p cpu-gpu-v100 --gpus=1 --wrap="uv run python main.py --config ojas_configs/generalize_powerflownet_task3.1_case14_lr5e-4.yaml"

sbatch --job-name=generalize_powerflownet_task3.1_case30_lr5e-4 --time=48:00:00 -p cpu-gpu-v100 --gpus=1 --wrap="uv run python main.py --config ojas_configs/generalize_powerflownet_task3.1_case30_lr5e-4.yaml"

sbatch --job-name=generalize_powerflownet_task3.1_case57_lr5e-4 --time=48:00:00 -p cpu-gpu-v100 --gpus=1 --wrap="uv run python main.py --config ojas_configs/generalize_powerflownet_task3.1_case57_lr5e-4.yaml"

sbatch --job-name=generalize_powerflownet_task3.1_case118_lr5e-4 --time=48:00:00 -p cpu-gpu-v100 --gpus=1 --wrap="uv run python main.py --config ojas_configs/generalize_powerflownet_task3.1_case118_lr5e-4.yaml"

sbatch --job-name=generalize_powerflownet_task3.1_case500_lr5e-4 --time=48:00:00 -p cpu-gpu-v100 --gpus=1 --wrap="uv run python main.py --config ojas_configs/generalize_powerflownet_task3.1_case500_lr5e-4.yaml"

sbatch --job-name=generalize_powerflownet_task3.2_lr5e-4 --time=48:00:00 -p cpu-gpu-v100 --gpus=1 --wrap="uv run python main.py --config ojas_configs/generalize_powerflownet_task3.2_lr5e-4.yaml"

sbatch --job-name=generalize_powerflownet_task3.3_lr5e-4 --time=48:00:00 -p cpu-gpu-v100 --gpus=1 --wrap="uv run python main.py --config ojas_configs/generalize_powerflownet_task3.3_lr5e-4.yaml"