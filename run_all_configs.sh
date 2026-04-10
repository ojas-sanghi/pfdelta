#!/bin/bash
# run_all_configs.sh
# Run from your pfdelta root directory

# for config in core/configs/ojas_configs/gen_apr0826/graphconv_*.yaml; do
for config in core/configs/ojas_configs/gen_apr0826/powerflownet_*.yaml; do
    short="${config#core/configs/}"   # strips "core/configs/" prefix
    echo "========================================="
    echo "Launching: $short"
    echo "========================================="
    echo "launch" | uv run main.py --config "$short"
    echo "Done with $short"
    echo ""
done

echo "All configs complete!"