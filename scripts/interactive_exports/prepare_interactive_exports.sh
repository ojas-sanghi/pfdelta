#!/usr/bin/env bash
set -euo pipefail

SITE_DIR="${1:-notebooks/interactive_exports}"

if [ ! -d "$SITE_DIR" ]; then
  echo "Missing directory: $SITE_DIR" >&2
  echo "Generate exports from notebooks/ojas_feb0126_test_case_viz_interactive.ipynb first." >&2
  exit 1
fi

if [ ! -f "$SITE_DIR/index.html" ]; then
  echo "Missing file: $SITE_DIR/index.html" >&2
  echo "Generate exports from notebooks/ojas_feb0126_test_case_viz_interactive.ipynb first." >&2
  exit 1
fi

echo "Static export ready: $SITE_DIR"
ls -1 "$SITE_DIR" | sed 's/^/ - /'
