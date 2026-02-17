#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/interactive_exports/deploy_interactive_exports.sh [site_dir] [--generate] [--prod]

Deploys the interactive HTML export to Vercel.

Options:
  --generate    Execute the notebook before deploy
  --prod        Use production deploy (only when vercel CLI/npx is available)

Examples:
  bash scripts/interactive_exports/deploy_interactive_exports.sh
  bash scripts/interactive_exports/deploy_interactive_exports.sh notebooks/interactive_exports --generate
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

SITE_DIR="notebooks/interactive_exports"
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  SITE_DIR="$1"
  shift
fi

DO_GENERATE=false
DO_PROD=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    --generate)
      DO_GENERATE=true
      ;;
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

if [ "$DO_GENERATE" = true ]; then
  bash scripts/interactive_exports/generate_interactive_exports.sh
fi

bash scripts/interactive_exports/prepare_interactive_exports.sh "$SITE_DIR"

echo "Deploying to Vercel..."

if command -v vercel >/dev/null 2>&1; then
  if [ "$DO_PROD" = true ]; then
    vercel deploy "$SITE_DIR" --prod -y
  else
    vercel deploy "$SITE_DIR" -y
  fi
  exit 0
fi

if command -v npx >/dev/null 2>&1; then
  if [ "$DO_PROD" = true ]; then
    npx vercel deploy "$SITE_DIR" --prod -y
  else
    npx vercel deploy "$SITE_DIR" -y
  fi
  exit 0
fi

if [ "$DO_PROD" = true ]; then
  echo "--prod requested but local vercel CLI is unavailable; using claimable preview deploy fallback." >&2
fi

echo "Local vercel CLI not found. Falling back to claimable deploy script."
scripts/interactive_exports/deploy_vercel_claim.sh "$SITE_DIR"
