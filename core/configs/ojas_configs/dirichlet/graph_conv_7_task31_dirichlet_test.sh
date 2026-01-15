sbatch --job-name=graph_conv_7_task31_dirichlet_test --time=48:00:00 -p cpu-gpu-v100 --gpus=1 --wrap="uv run python main.py --config ojas_configs/graph_conv_7_task31_dirichlet_test.yaml"
