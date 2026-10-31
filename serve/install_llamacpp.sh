#!/usr/bin/env bash
# Download the llama.cpp build pinned in serve/models.json into ~/llama.cpp/<build>/ (macOS and Linux).
# These are llama.cpp's own release downloads from GitHub, the same build on every machine.
#
#   bash serve/install_llamacpp.sh                    # macOS: Metal build; Linux: CPU build
#   VARIANT=vulkan bash serve/install_llamacpp.sh     # Linux: cpu | vulkan | cuda | rocm
#
# Set LLAMA_CPP_DIR to install somewhere else (the other scripts read the same variable).
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/pinned.sh"
variant="${VARIANT:-cpu}"

case "$(uname -s)-$(uname -m)-$variant" in
  Darwin-arm64-*)          assets=("llama-$LLAMA_BUILD-bin-macos-arm64.tar.gz") ;;
  Darwin-x86_64-*)         assets=("llama-$LLAMA_BUILD-bin-macos-x64.tar.gz") ;;
  Linux-x86_64-cpu)        assets=("llama-$LLAMA_BUILD-bin-ubuntu-x64.tar.gz") ;;
  Linux-x86_64-vulkan)     assets=("llama-$LLAMA_BUILD-bin-ubuntu-vulkan-x64.tar.gz") ;;
  Linux-x86_64-cuda)       assets=("llama-$LLAMA_BUILD-bin-ubuntu-cuda-12.8-x64.tar.gz"
                                   "cudart-llama-$LLAMA_BUILD-bin-ubuntu-cuda-12.8-x64.tar.gz") ;;
  Linux-x86_64-rocm)       assets=("llama-$LLAMA_BUILD-bin-ubuntu-rocm-10.0-x64.tar.gz") ;;
  Linux-aarch64-cpu)       assets=("llama-$LLAMA_BUILD-bin-ubuntu-arm64.tar.gz") ;;
  Linux-aarch64-vulkan)    assets=("llama-$LLAMA_BUILD-bin-ubuntu-vulkan-arm64.tar.gz") ;;
  Linux-aarch64-cuda)      assets=("llama-$LLAMA_BUILD-bin-ubuntu-cuda-13.4-arm64.tar.gz"
                                   "cudart-llama-$LLAMA_BUILD-bin-ubuntu-cuda-13.4-arm64.tar.gz") ;;
  *) echo "no download for $(uname -s) $(uname -m) VARIANT=$variant (Windows: see docs/local-llamacpp.md)" >&2; exit 2 ;;
esac

if require_llama_build llama-server 2>/dev/null; then
  echo "ok    llama.cpp $LLAMA_BUILD already installed in $LLAMA_BIN"
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
for asset in "${assets[@]}"; do
  echo "downloading $asset"
  curl -fL --progress-bar -o "$tmp/$asset" \
    "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD/$asset"
  tar -xzf "$tmp/$asset" -C "$tmp"
done

# The archives hold one folder (e.g. llama-b12020/); install its contents flat into $LLAMA_BIN.
src="$(dirname "$(find "$tmp" -type f -name llama-server | head -1)")"
[ -n "$src" ] && [ "$src" != "." ] || { echo "llama-server not found in the download" >&2; exit 1; }
rm -rf "$LLAMA_BIN"
mkdir -p "$LLAMA_BIN"
cp -R "$src"/. "$LLAMA_BIN"/
for extra in "$tmp"/*/; do               # the CUDA runtime archive unpacks into its own folder
  [ "${extra%/}" = "$src" ] || cp -R "$extra". "$LLAMA_BIN"/ 2>/dev/null || true
done

require_llama_build llama-server
echo "ok    llama.cpp $LLAMA_BUILD installed in $LLAMA_BIN"
"$LLAMA_BIN/llama-server" --version 2>&1 | tail -2
