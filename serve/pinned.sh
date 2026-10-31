# Sourced by the other scripts: the llama.cpp build pinned in serve/models.json, and where its programs live.
#   LLAMA_BUILD   e.g. b12020
#   LLAMA_BIN     folder with llama-server, llama-quantize, ... (override with LLAMA_CPP_DIR)

_serve_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_BUILD="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["llama_cpp"]["build"])' "$_serve_dir/models.json")"
LLAMA_BIN="${LLAMA_CPP_DIR:-$HOME/llama.cpp/$LLAMA_BUILD}"

# Fail unless $LLAMA_BIN holds the pinned build and the named program. Usage: require_llama_build llama-quantize
# The build is read from llama-server: every program in the folder comes from the same download,
# and some (llama-quantize) have no --version.
require_llama_build() {
  local tool="$LLAMA_BIN/$1" have
  if [ ! -x "$tool" ] || [ ! -x "$LLAMA_BIN/llama-server" ]; then
    echo "$tool not found: run  bash serve/install_llamacpp.sh  (or set LLAMA_CPP_DIR)" >&2
    return 1
  fi
  have="b$("$LLAMA_BIN/llama-server" --version 2>&1 | sed -n 's/.*(build \([0-9]*\),.*/\1/p' || true)"
  if [ "$have" != "$LLAMA_BUILD" ]; then
    echo "$LLAMA_BIN holds llama.cpp build $have, serve/models.json pins $LLAMA_BUILD" >&2
    return 1
  fi
}
