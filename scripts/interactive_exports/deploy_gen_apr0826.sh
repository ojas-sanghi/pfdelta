#!/usr/bin/env bash
set -euo pipefail
set -a
source /home/osanghi/pfdelta/.env.vercel
set +a

bash /home/osanghi/pfdelta/scripts/interactive_exports/deploy_interactive_exports.sh --generate

# bash scripts/interactive_exports/deploy_gen_apr0826.sh