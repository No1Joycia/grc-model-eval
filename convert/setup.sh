#!/usr/bin/env bash
# One-time setup for GGUF conversion: llama.cpp's converter at the build pinned in serve/models.json,
# plus a Python environment for it. Safe to re-run.
#
#   bash convert/setup.sh
#
# Needs: git, uv (https://docs.astral.sh/uv/), and the pinned llama.cpp programs for llama-quantize
# (bash serve/install_llamacpp.sh).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
source "$here/../serve/pinned.sh"
vendor="$here/vendor/llama.cpp"

if [ -d "$vendor/.git" ]; then
  have="$(git -C "$vendor" describe --tags --exact-match 2>/dev/null || echo unknown)"
  if [ "$have" != "$LLAMA_BUILD" ]; then
    echo "convert/vendor/llama.cpp is at $have, expected $LLAMA_BUILD: delete that folder and re-run." >&2
    exit 1
  fi
  echo "ok    llama.cpp converter $LLAMA_BUILD"
else
  mkdir -p "$here/vendor"
  git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$LLAMA_BUILD" \
    https://github.com/ggml-org/llama.cpp.git "$vendor"
  echo "ok    cloned llama.cpp converter $LLAMA_BUILD"
fi

if [ ! -x "$here/.venv/bin/python" ]; then
  uv venv --quiet --python 3.12 "$here/.venv"
fi
uv pip install --quiet --python "$here/.venv/bin/python" \
  -r "$vendor/requirements/requirements-convert_hf_to_gguf.txt"
echo "ok    Python environment convert/.venv"

require_llama_build llama-quantize
echo "ok    llama-quantize $LLAMA_BUILD in $LLAMA_BIN"
