#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONUNBUFFERED=1

# The TFLite export is CPU-only. Its converters (litert-torch -> jax,
# litert-converter, ai-edge-litert) nevertheless probe for CUDA/GPU at
# import/conversion time and fail on hosts with an incompatible CUDA
# toolkit. Force a pure-CPU environment so behaviour matches the dev box.
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS="cpu"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is required. Install it from https://docs.astral.sh/uv/" >&2
    exit 1
fi

exec uv run --locked --extra tflite python -m pxmodel.export_all_backbones "$@"
