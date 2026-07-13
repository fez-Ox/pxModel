#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONUNBUFFERED=1

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is required. Install it from https://docs.astral.sh/uv/" >&2
    exit 1
fi

exec uv run --locked python -m pxmodel.quantize_all_backbones "$@"
