#!/usr/bin/env bash
set -euo pipefail

# Generate the Plotly site first (landing page + one page per model):
# uv run scripts/analyze_best_epoch/visualize_best_epoch.py --backend plotly --plotly-layout split
# Then deploy:
# bash scripts/analyze_best_epoch/deploy_plotly_html_to_vercel.sh

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/analyze_best_epoch/deploy_plotly_html_to_vercel.sh [target] [--prod]

Deploy Plotly output to Vercel as a static site.

Arguments:
  target      Path to generated site directory or HTML file
              (default: scripts/analyze_best_epoch/outputs/selected_runs_site)

Options:
  --prod      Deploy to production (default is preview)

Examples:
  bash scripts/analyze_best_epoch/deploy_plotly_html_to_vercel.sh
  bash scripts/analyze_best_epoch/deploy_plotly_html_to_vercel.sh scripts/analyze_best_epoch/outputs/selected_runs_site
  bash scripts/analyze_best_epoch/deploy_plotly_html_to_vercel.sh scripts/analyze_best_epoch/outputs/selected_runs_curves.html
  bash scripts/analyze_best_epoch/deploy_plotly_html_to_vercel.sh --prod
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

TARGET_PATH="scripts/analyze_best_epoch/outputs/selected_runs_site"
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  TARGET_PATH="$1"
  shift
fi

DO_PROD=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --prod)
      DO_PROD=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

if [ ! -e "$TARGET_PATH" ]; then
  echo "Target path not found: $TARGET_PATH" >&2
  echo "Generate it with: uv run scripts/analyze_best_epoch/visualize_best_epoch.py --backend plotly --plotly-layout split" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

DEPLOY_PATH=""
if [ -d "$TARGET_PATH" ]; then
  DEPLOY_PATH="$TARGET_PATH"
elif [ -f "$TARGET_PATH" ]; then
  cp "$TARGET_PATH" "$TMP_DIR/index.html"
  DEPLOY_PATH="$TMP_DIR"
else
  echo "Unsupported target type: $TARGET_PATH" >&2
  exit 1
fi

echo "Deploying $TARGET_PATH to Vercel..."

if command -v vercel >/dev/null 2>&1; then
  if [ "$DO_PROD" = true ]; then
    vercel deploy "$DEPLOY_PATH" --prod -y
  else
    vercel deploy "$DEPLOY_PATH" -y
  fi
  exit 0
fi

if command -v npx >/dev/null 2>&1; then
  if [ "$DO_PROD" = true ]; then
    npx vercel deploy "$DEPLOY_PATH" --prod -y
  else
    npx vercel deploy "$DEPLOY_PATH" -y
  fi
  exit 0
fi

if [ "$DO_PROD" = true ]; then
  echo "--prod requested but local vercel CLI is unavailable; using claimable preview deploy fallback." >&2
fi

echo "Local vercel CLI not found. Falling back to claimable deploy script."
bash scripts/interactive_exports/deploy_vercel_claim.sh "$DEPLOY_PATH"
