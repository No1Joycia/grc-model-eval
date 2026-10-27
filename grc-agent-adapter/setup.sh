#!/usr/bin/env bash
# grc-agent-adapter setup check (README.md, Setup). Safe to run again at any time.
# Checks every prerequisite and says how to fix what is missing, builds GRC-Agent's knowledge
# databases once, and ends with ./agent selftest.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRC_AGENT_REPO="${GRC_AGENT_REPO:-$(cd "$HERE/../.." && pwd)/grc-agent}"
PY="$GRC_AGENT_REPO/.venv/bin/python"
failed=0

ok()   { printf 'ok    %s\n' "$1"; }
warn() { printf 'note  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n      fix: %s\n' "$1" "$2"; failed=1; }

echo "== 1. GRC-Agent checkout at $GRC_AGENT_REPO"
if [ ! -d "$GRC_AGENT_REPO/src/grc_agent" ]; then
  fail "no GRC-Agent checkout at $GRC_AGENT_REPO" \
       "clone it there, or export GRC_AGENT_REPO=/path/to/grc-agent"
else
  ok "GRC-Agent commit $(git -C "$GRC_AGENT_REPO" rev-parse --short=7 HEAD 2>/dev/null || echo unknown)"
fi

echo "== 2. GRC-Agent's Python ($PY)"
if [ -x "$PY" ] && "$PY" -c "import grc_agent, pydantic_ai, dotenv" 2>/dev/null; then
  ok "imports grc_agent, pydantic_ai, python-dotenv"
else
  fail "GRC-Agent's virtual environment is missing or incomplete" \
       "set up GRC-Agent's .venv first (its README: conda env with GNU Radio + uv sync)"
fi

echo "== 3. GNU Radio"
gr=$("$PY" -c "from gnuradio import gr; print(gr.version())" 2>/dev/null || true)
if [ -n "$gr" ]; then ok "GNU Radio $gr"; else
  fail "GNU Radio's Python bindings are not visible to $PY" \
       "install GNU Radio in the conda env GRC-Agent's .venv is built on (conda install -c conda-forge gnuradio)"
fi

if [ "$failed" -ne 0 ]; then
  echo; echo "setup: fix the FAIL lines above, then run ./setup.sh again"; exit 1
fi

echo "== 4. GRC-Agent's knowledge databases (built once)"
VEC="$GRC_AGENT_REPO/src/grc_agent/vectors"
if [ -f "$VEC/catalog_lexical.db" ] && [ -f "$VEC/docs_lexical.db" ]; then
  ok "catalog_lexical.db and docs_lexical.db present"
else
  echo "      building them (lexical search) ..."
  if (cd "$GRC_AGENT_REPO" && GRC_EMBED_BACKEND=lexical QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="$HERE/src" "$PY" -c "
from grc_adapter import grc_platform
grc_platform.install()
from grc_agent.adapter import rag
for domain in ('catalog', 'docs'):
    path, model = rag.get_db_and_model(domain)
    rag._ensure_db_built(domain, path, model)
") && [ -f "$VEC/catalog_lexical.db" ] && [ -f "$VEC/docs_lexical.db" ]; then
    ok "built catalog_lexical.db and docs_lexical.db"
  else
    fail "could not build the knowledge databases" "read the error above; README.md, Troubleshooting"
  fi
fi

echo "== 5. API keys"
if [ -f "$HERE/.env" ]; then
  perms=$(stat -f '%Lp' "$HERE/.env" 2>/dev/null || echo "?")
  keys=$(grep -oE '^[A-Z_]+_API_KEY=.+' "$HERE/.env" | cut -d= -f1 | tr '\n' ' ')
  ok ".env present (permissions $perms), keys: ${keys:-none}"
  [ "$perms" = "600" ] || warn "chmod 600 $HERE/.env keeps the keys readable only by you"
else
  warn "no .env yet: copy .env.example to .env and add a key before using a hosted model"
fi

if [ "$failed" -ne 0 ]; then
  echo; echo "setup: fix the FAIL lines above, then run ./setup.sh again"; exit 1
fi

echo "== 6. self-test (scripted model; no API key or network needed)"
if "$HERE/agent" selftest; then
  echo; echo "setup complete: try  ./agent run \"Which GNU Radio block makes a cosine tone?\" --provider cohere"
else
  echo; echo "setup: the self-test failed (lines above); README.md, Troubleshooting"; exit 1
fi
