#!/usr/bin/env bash
# Hugging Face checkpoint folder -> GGUF files: BF16, then the quantized versions, then SHA256SUMS.
# The same script is used for the base models and for every fine-tuned (merged) checkpoint,
# so all model files are produced the same way.
#
#   bash convert/to_gguf.sh SRC_DIR OUT_DIR NAME
#   bash convert/to_gguf.sh ~/models/hf/Nanbeige4.2-3B ~/models/nanbeige42 Nanbeige4.2-3B
#
# Writes OUT_DIR/NAME-BF16.gguf, NAME-Q8_0.gguf, NAME-Q4_K_M.gguf and SHA256SUMS.
# QUANTS="Q8_0 Q4_K_M" (default) picks the quantized versions; QUANTS="" makes BF16 only.
# Run convert/setup.sh once first.
set -euo pipefail

src="${1:?usage: to_gguf.sh SRC_DIR OUT_DIR NAME}"
out="${2:?usage: to_gguf.sh SRC_DIR OUT_DIR NAME}"
name="${3:?usage: to_gguf.sh SRC_DIR OUT_DIR NAME}"
quants="${QUANTS-Q8_0 Q4_K_M}"

here="$(cd "$(dirname "$0")" && pwd)"
source "$here/../serve/pinned.sh"
py="$here/.venv/bin/python"
converter="$here/vendor/llama.cpp/convert_hf_to_gguf.py"
[ -x "$py" ] && [ -f "$converter" ] || { echo "run: bash convert/setup.sh" >&2; exit 1; }
require_llama_build llama-quantize
[ -f "$src/config.json" ] || { echo "$src has no config.json: not a Hugging Face checkpoint folder" >&2; exit 1; }

mkdir -p "$out"
bf16="$out/$name-BF16.gguf"
if [ -f "$bf16" ]; then
  echo "skip  $bf16 exists"
else
  "$py" "$converter" "$src" --outtype bf16 --outfile "$bf16"
fi

for q in $quants; do
  target="$out/$name-$q.gguf"
  if [ -f "$target" ]; then
    echo "skip  $target exists"
  else
    "$LLAMA_BIN/llama-quantize" "$bf16" "$target" "$q"
  fi
done

(cd "$out" && shasum -a 256 ./*.gguf | sed 's# \./# #' > SHA256SUMS)
echo
echo "done: $out"
ls -lh "$out"/*.gguf
