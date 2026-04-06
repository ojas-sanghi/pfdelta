#!/usr/bin/env bash
set -euo pipefail

NOTEBOOK_PATH="${1:-notebooks/gen_apr0326_test_case_viz.ipynb}"
SITE_DIR="${2:-notebooks/interactive_exports/gen_apr0326}"

if [ ! -f "$NOTEBOOK_PATH" ]; then
  echo "Notebook not found: $NOTEBOOK_PATH" >&2
  exit 1
fi

if [ -x ".venv/bin/jupyter" ]; then
  JUPYTER_BIN=".venv/bin/jupyter"
elif command -v jupyter >/dev/null 2>&1; then
  JUPYTER_BIN="jupyter"
else
  echo "jupyter not found. Install it or use the project venv." >&2
  exit 1
fi

export JUPYTER_CONFIG_DIR="${JUPYTER_CONFIG_DIR:-$PWD/.jupyter_config}"
export JUPYTER_DATA_DIR="${JUPYTER_DATA_DIR:-$PWD/.jupyter_data}"
export JUPYTER_RUNTIME_DIR="${JUPYTER_RUNTIME_DIR:-$PWD/.jupyter_runtime}"
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR"

"$JUPYTER_BIN" nbconvert --to notebook --execute --inplace   --ExecutePreprocessor.timeout=3600   "$NOTEBOOK_PATH"

bash scripts/interactive_exports/prepare_interactive_exports.sh "$SITE_DIR"
