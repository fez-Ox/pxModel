#!/usr/bin/env bash
# Launch the annotation viewer/editor.
# Self-contained: dependencies are provided ephemerally via `uv run --with`,
# so the tracked pyproject.toml / uv.lock are never modified.
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

echo "Annotation editor → http://${HOST}:${PORT}"
exec uv run \
  --with fastapi \
  --with "uvicorn[standard]" \
  python -m uvicorn annotator.app:app --host "${HOST}" --port "${PORT}" "$@"
