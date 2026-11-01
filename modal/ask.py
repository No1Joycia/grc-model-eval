"""Ask the hosted Nanbeige4.2-3B one question and print the answer.

Reads MODAL_URL / MODAL_KEY / MODAL_SECRET from modal/.env, next to this file (or the
environment). Stdlib only: no installs needed beyond Python 3.

Usage:
    python3 modal/ask.py "What is GNU Radio? Answer in one sentence."
    python3 modal/ask.py --max-tokens 512 --system "You are a DSP tutor." "Explain aliasing."
    python3 modal/ask.py --health                     # wake the GPU / check it is up
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL_ID = "Nanbeige/Nanbeige4.2-3B"


def load_env(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--system", default=None, help="optional system prompt")
    ap.add_argument("--max-tokens", type=int, default=512, help="1-2048")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--health", action="store_true", help="only call /health")
    args = ap.parse_args()

    load_env(Path(__file__).resolve().parent / ".env")
    try:
        url = os.environ["MODAL_URL"].rstrip("/")
        headers = {"Modal-Key": os.environ["MODAL_KEY"], "Modal-Secret": os.environ["MODAL_SECRET"],
                   "Content-Type": "application/json"}
    except KeyError as exc:
        sys.exit(f"Missing {exc.args[0]}: copy modal/.env.example to modal/.env and fill in the values you were given.")

    if args.health:
        req = urllib.request.Request(url + "/health", headers=headers)
    else:
        if not args.question:
            ap.error("give a question in quotes, or --health")
        messages = ([{"role": "system", "content": args.system}] if args.system else [])
        messages.append({"role": "user", "content": args.question})
        body = {"model": MODEL_ID, "messages": messages,
                "max_tokens": args.max_tokens, "temperature": args.temperature}
        req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers=headers, method="POST")

    print("waiting for the model (first call after idle takes ~1-3 min)...", file=sys.stderr)
    started = time.time()
    try:
        # urllib follows Modal's 303 redirect for long-running requests.
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            r = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        sys.exit(f"HTTP {exc.code}: {exc.read().decode()[:500]}")
    seconds = time.time() - started

    if args.health:
        print(json.dumps(r))
        return
    choice, usage = r["choices"][0], r.get("usage", {})
    print(choice["message"]["content"])
    print(f"\n[{usage.get('completion_tokens', '?')} tokens, {seconds:.1f} s, "
          f"finish_reason={choice.get('finish_reason')}]", file=sys.stderr)


if __name__ == "__main__":
    main()
