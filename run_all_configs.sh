#!/bin/bash
# run_all_configs.sh
# Run from your pfdelta root directory

# for config in core/configs/ojas_configs/gen_apr0826/powerflownet_*.yaml; do
# for config in core/configs/ojas_configs/gen_apr1126_perturb/canos_*.yaml; do
for config in core/configs/ojas_configs/gen_apr1126_perturb/graphconv_mse_task31.yaml; do
# for config in core/configs/ojas_configs/gen_apr1126_perturb/powerflownet_*_task34.yaml; do
# for config in core/configs/ojas_configs/gen_apr1126_perturb/*.yaml; do
    short="${config#core/configs/}"   # strips "core/configs/" prefix
    echo "========================================="
    echo "Launching: $short"
    echo "========================================="
    echo "launch" | uv run main.py --config "$short"`
    echo "Done with $short"
    echo ""
done

echo "All configs complete!"


# `
echo "launch" | uv run main.py --config ojas_configs/gen_apr1126_perturb/graphconv_mse_task33
echo "launch" | uv run main.py --config ojas_configs/gen_apr1126_perturb/graphconv_mse_task34
# `