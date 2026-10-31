#!/usr/bin/env bash
# Start llama-server on this machine with the settings every host uses (serve/models.json).
#
#   bash serve/run_local.sh nanbeige42              # default quant (q8_0)
#   bash serve/run_local.sh nanbeige42 q4_k_m
#
# The model file is read from $MODELS_DIR/<model>/ (default ~/models), or from GGUF=/path/to/file.gguf.
# Other settings, all optional:
#   PORT=8080            port to listen on
#   HOST=127.0.0.1       0.0.0.0 makes it reachable from other machines: then also set API_KEY_ENV
#   API_KEY_ENV=NAME     require "Authorization: Bearer <key>", key read from that environment variable
# Stop the server with Ctrl+C.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
source "$here/pinned.sh"
model="${1:?usage: run_local.sh MODEL [QUANT]   (models: see serve/models.json)}"
quant="${2:-}"

# Read the shared settings: prints the GGUF file name, then one server argument per line.
config="$(python3 - "$here/models.json" "$model" "$quant" <<'PY'
import json, os, sys
cfg = json.load(open(sys.argv[1]))
model, quant = sys.argv[2], sys.argv[3]
if model not in cfg["models"]:
    sys.exit(f"unknown model {model!r}; known: {', '.join(cfg['models'])}")
m = cfg["models"][model]
quant = quant or m["default_quant"]
if quant not in m["gguf"]:
    sys.exit(f"unknown quant {quant!r} for {model}; known: {', '.join(m['gguf'])}")
print(m["gguf"][quant])
args = cfg["server_args"] + m["extra_args"]
if m.get("chat_template"):
    args += ["--chat-template-file", os.path.join(os.path.dirname(sys.argv[1]), m["chat_template"])]
for arg in args:
    print(arg)
PY
)"
gguf_name="$(sed -n 1p <<<"$config")"
server_args=()
while IFS= read -r line; do server_args+=("$line"); done < <(sed -n '2,$p' <<<"$config")

gguf="${GGUF:-${MODELS_DIR:-$HOME/models}/$model/$gguf_name}"
[ -f "$gguf" ] || { echo "model file not found: $gguf" >&2; exit 1; }

require_llama_build llama-server

host="${HOST:-127.0.0.1}"
if [ -n "${API_KEY_ENV:-}" ]; then
  # Passed through the environment, not the command line, so the key doesn't show up in `ps`.
  export LLAMA_API_KEY="${!API_KEY_ENV:?$API_KEY_ENV is not set}"
elif [ "$host" != "127.0.0.1" ] && [ "$host" != "localhost" ]; then
  echo "refusing HOST=$host without API_KEY_ENV: anyone on the network could use the server" >&2
  exit 2
fi

set -x
exec "$LLAMA_BIN/llama-server" -m "$gguf" -a "$model" --host "$host" --port "${PORT:-8080}" "${server_args[@]}"
