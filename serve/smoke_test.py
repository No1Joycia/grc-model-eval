"""Check that a llama-server endpoint is up, answers, and makes tool calls. Stdlib only.

Works against any host: this Mac, Modal, or Qoherent's GPU.

    python3 serve/smoke_test.py                                       # http://127.0.0.1:8080/v1
    python3 serve/smoke_test.py --base-url https://<app>.modal.run/v1 --api-key-env LLAMA_API_KEY --check-auth

Exit code 0 when every check passes, 1 otherwise.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

TOOL = {
    "type": "function",
    "function": {
        "name": "get_block_info",
        "description": "Look up a GNU Radio block by its block id and return its parameters and ports.",
        "parameters": {
            "type": "object",
            "properties": {"block_id": {"type": "string", "description": "GNU Radio block id, e.g. blocks_throttle"}},
            "required": ["block_id"],
        },
    },
}


def request(url, key, body=None, timeout=600):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read()), time.time() - start


def speed(reply):
    timings = reply.get("timings") or {}
    if "predicted_per_second" in timings:
        return f"{timings['predicted_per_second']:.1f} tok/s generation, {timings.get('prompt_per_second', 0):.0f} tok/s prompt"
    return "no timings in reply"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--api-key-env", help="environment variable holding the Bearer key")
    ap.add_argument("--model", help="model name to send (default: the first one the server lists)")
    ap.add_argument("--check-auth", action="store_true", help="also check that a request without the key is refused")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per request (a cold Modal start can take minutes)")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    root = base[: -len("/v1")] if base.endswith("/v1") else base
    key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
    if args.api_key_env and not key:
        sys.exit(f"{args.api_key_env} is not set")

    failures = 0

    def check(name, ok, detail):
        nonlocal failures
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  {name}: {detail}", flush=True)

    try:
        health, secs = request(root + "/health", key, timeout=args.timeout)
        check("health", health.get("status") == "ok", f"{health} ({secs:.0f} s)")
        models, _ = request(base + "/models", key, timeout=args.timeout)
        names = [m["id"] for m in models.get("data", [])]
        model = args.model or (names[0] if names else "")
        check("models", bool(names), f"{names}")

        reply, secs = request(base + "/chat/completions", key, {
            "model": model, "temperature": 0, "max_tokens": 1024,
            "messages": [{"role": "user", "content": "What is GNU Radio? Answer in one sentence."}],
        }, timeout=args.timeout)
        msg = reply["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        check("chat", bool(content), f"{content[:160]!r} ({secs:.1f} s, {speed(reply)})")
        if msg.get("reasoning_content"):
            print(f"      reasoning: {len(msg['reasoning_content'])} chars, kept out of content", flush=True)

        reply, secs = request(base + "/chat/completions", key, {
            "model": model, "temperature": 0, "max_tokens": 2048, "tools": [TOOL],
            "messages": [{"role": "user", "content": "Use the tool to look up the GNU Radio block analog_sig_source_x."}],
        }, timeout=args.timeout)
        choice = reply["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        detail = f"finish_reason={choice.get('finish_reason')}, {len(calls)} call(s) ({secs:.1f} s)"
        ok = False
        if calls:
            fn = calls[0]["function"]
            try:
                arguments = json.loads(fn["arguments"])
                ok = fn["name"] == "get_block_info" and "block_id" in arguments
                detail += f": {fn['name']}({json.dumps(arguments)})"
            except (json.JSONDecodeError, TypeError):
                detail += f": arguments are not valid JSON: {fn.get('arguments')!r}"
        else:
            detail += f"; content instead: {(choice['message'].get('content') or '')[:200]!r}"
        check("tool call", ok, detail)
    except urllib.error.HTTPError as exc:
        check("request", False, f"HTTP {exc.code}: {exc.read()[:300]!r}")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        check("request", False, f"server not reachable at {base}: {exc}")

    if args.check_auth:
        try:
            request(base + "/models", "", timeout=args.timeout)
            check("auth", False, "a request without the key was accepted")
        except urllib.error.HTTPError as exc:
            check("auth", exc.code == 401, f"request without the key -> HTTP {exc.code}")

    print("all checks passed" if not failures else f"{failures} check(s) failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
