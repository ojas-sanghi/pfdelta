#!/usr/bin/env bash
set -euo pipefail

SITE_DIR="${1:-notebooks/interactive_exports/gen_apr0826}"
INDEX_FILE="$SITE_DIR/index.html"

if [ ! -d "$SITE_DIR" ]; then
  echo "Missing directory: $SITE_DIR" >&2
  echo "Generate exports first, then rerun this command with the matching site directory." >&2
  exit 1
fi

if [ ! -f "$INDEX_FILE" ]; then
  echo "Missing file: $INDEX_FILE" >&2
  echo "Expected a generated single-page interactive export with an index.html file." >&2
  exit 1
fi

echo "Interactive export ready: $SITE_DIR"
find "$SITE_DIR" -maxdepth 1 -type f | sort | sed 's#^# - #' 
