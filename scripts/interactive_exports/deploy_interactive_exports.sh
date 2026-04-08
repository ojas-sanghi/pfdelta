#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/interactive_exports/deploy_interactive_exports.sh [site_dir] [--generate] [--prod] [--preview]

Deploys the interactive HTML export to Vercel via the REST API.

Environment:
  VERCEL_TOKEN              Required API token
  VERCEL_PROJECT_ID         Optional project id override
  VERCEL_PROJECT_NAME       Optional project name override
  VERCEL_PROJECT_ID_OR_NAME Optional project id or name override
  VERCEL_TEAM_ID            Optional team id
  VERCEL_TEAM_SLUG          Optional team slug

Options:
  --generate    Execute the notebook before deploy
  --prod        Deploy to the project's production target (default)
  --preview     Deploy to a preview target instead of production

Examples:
  bash scripts/interactive_exports/deploy_interactive_exports.sh
  bash scripts/interactive_exports/deploy_interactive_exports.sh --generate --prod
  bash scripts/interactive_exports/deploy_interactive_exports.sh notebooks/interactive_exports/gen_apr0326 --preview
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

SITE_DIR="notebooks/interactive_exports/gen_apr0326"
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  SITE_DIR="$1"
  shift
fi

DO_GENERATE=false
TARGET="production"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --generate)
      DO_GENERATE=true
      ;;
    --prod)
      TARGET="production"
      ;;
    --preview)
      TARGET="preview"
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

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "python3 not found. A Python interpreter is required for Vercel REST API deploys." >&2
  exit 1
fi

if [ -z "${VERCEL_TOKEN:-}" ]; then
  echo "VERCEL_TOKEN is required for REST API deploys." >&2
  exit 1
fi

echo "Deploying to Vercel via REST API (${TARGET})..."
"$PYTHON_BIN" scripts/interactive_exports/deploy_interactive_exports_api.py "$SITE_DIR" --target "$TARGET"
