"""Send the trial questions to an OpenAI-compatible model server and save its answers.

The default server is llama-server on this machine (serve/run_local.sh), as in the Colab
notebook (colab/eval.ipynb). Resumable: questions already in the output file are skipped,
so you can stop with Ctrl+C and run again. Stdlib only.

The first line printed is a rough time estimate for the run; the last line is the actual time.
Every answer row records the settings it was made with, and the script refuses to add answers
made with different settings to the same file.

Usage:
    python3 trial_eval/run_trial.py --start 1 --end 30 --out answers.jsonl     # local llama-server
    python3 trial_eval/run_trial.py --thinking on --max-tokens 4096 --out answers_think.jsonl
    python3 trial_eval/run_trial.py --start 31 --end 60 --estimate-only
    # the Modal endpoint, until it is retired (MODAL_KEY / MODAL_SECRET from ../.env):
    python3 trial_eval/run_trial.py --base-url "$MODAL_URL/v1" --model Nanbeige/Nanbeige4.2-3B
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = (
    "You are an expert in radio-frequency engineering, digital signal processing, "
    "software-defined radio, and GNU Radio. Answer the question accurately and completely. "
    "For calculations, show the key steps and state the final result with units."
)
GUESS_TOK_S = 20     # first-run guess before any answer is timed (llama.cpp Q8_0 on a T4)
# Settings that must match between answers in one file (others, e.g. timeout, may differ).
COMPARED = ("model", "endpoint", "system_prompt", "max_tokens", "temperature", "enable_thinking")


def load_env(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def estimate_seconds(out, max_tokens):
    """Per-question seconds from earlier answers in `out`, else a rough guess."""
    if out.exists():
        past = [json.loads(l).get("seconds") for l in open(out) if l.strip()]
        past = [x for x in past if x]
        if past:
            return sum(past) / len(past), f"average of {len(past)} earlier answers"
    # Answers typically use ~60% of max_tokens before stopping.
    return 0.6 * max_tokens / GUESS_TOK_S + 5, f"rough guess at ~{GUESS_TOK_S} tok/s"


def fmt_duration(seconds):
    if seconds < 60:
        return "<1 min"
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h} h {m} min" if h else f"{m} min"


def check_settings(out, settings):
    """Refuse to append to a file whose earlier answers used different settings."""
    if not out.exists():
        return
    for n, line in enumerate(open(out), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        old = {"model": row.get("model"), **row.get("settings", {})}   # older rows kept the model outside settings
        diff = [k for k in COMPARED if old.get(k) is not None and old[k] != settings[k]]
        if diff:
            details = ", ".join(f"{k}: file has {old[k]!r}, this run {settings[k]!r}" for k in diff)
            raise SystemExit(f"{out} line {n} was made with different settings ({details}).\n"
                             f"Use another --out file so the two runs are not mixed.")


def ask(url, headers, model, question, max_tokens, temperature, thinking, timeout):
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }).encode()
    req = urllib.request.Request(url + "/chat/completions", data=body, method="POST", headers=headers)
    # urllib follows redirects (Modal sends a 303 for requests over 150 s).
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "trial_eval_100.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "answers_nanbeige_b0.jsonl"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="OpenAI-style base URL ending in /v1")
    ap.add_argument("--model", default="nanbeige42", help="model name sent to the server")
    ap.add_argument("--api-key-env", help="environment variable holding a Bearer key (not needed locally)")
    ap.add_argument("--thinking", choices=["off", "on"], default="off")
    ap.add_argument("--start", type=int, default=1, help="first question number (1-based)")
    ap.add_argument("--end", type=int, default=None, help="last question number (inclusive)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--estimate-only", action="store_true", help="print the time estimate and exit")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    host = urllib.parse.urlparse(base).hostname or base
    modal = host.endswith(".modal.run")
    thinking = args.thinking == "on"
    if modal and thinking:
        raise SystemExit("--thinking on is not possible on the Modal endpoint: its server always turns thinking off")
    settings = {"model": args.model, "endpoint": host, "system_prompt": SYSTEM_PROMPT,
                "max_tokens": args.max_tokens, "temperature": args.temperature, "enable_thinking": thinking}

    all_questions = [json.loads(l) for l in open(args.questions) if l.strip()]
    end = min(args.end or len(all_questions), len(all_questions))
    if not 1 <= args.start <= end:
        raise SystemExit(f"--start/--end must satisfy 1 <= start <= end <= {len(all_questions)}")
    for n, q in enumerate(all_questions, 1):
        q["num"] = n
    questions = all_questions[args.start - 1 : end][: args.limit]
    out = Path(args.out)
    done = {json.loads(l)["id"] for l in open(out) if l.strip()} if out.exists() else set()
    todo = [q for q in questions if q["id"] not in done]
    per_q, basis = estimate_seconds(out, args.max_tokens)
    print(f"ESTIMATE: {len(todo)} questions -> ~{fmt_duration(per_q * len(todo))}", flush=True)
    print(f"  questions #{questions[0]['num']}-#{questions[-1]['num']} ({len(questions)}) | "
          f"{len(questions) - len(todo)} already answered | {len(todo)} to go")
    print(f"  ~{per_q:.0f} s/question ({basis})")
    print(f"  server {base} | model {args.model} | thinking {args.thinking} | "
          f"temperature {args.temperature} | max_tokens {args.max_tokens}")
    print(f"  answers -> {out}", flush=True)
    if args.estimate_only or not todo:
        return
    check_settings(out, settings)
    out.parent.mkdir(parents=True, exist_ok=True)

    headers = {"Content-Type": "application/json"}
    if args.api_key_env:
        key = os.environ.get(args.api_key_env)
        if not key:
            raise SystemExit(f"{args.api_key_env} is not set")
        headers["Authorization"] = f"Bearer {key}"
    elif modal:
        load_env(ROOT.parent / ".env")
        headers.update({"Modal-Key": os.environ["MODAL_KEY"], "Modal-Secret": os.environ["MODAL_SECRET"]})

    run_started = time.time()
    with open(out, "a") as f:
        for i, q in enumerate(todo, 1):
            print(f"  asking #{q['num']} ({q['difficulty']}, {q['task_type']})...", flush=True)
            for attempt in range(args.retries + 1):
                started = time.time()
                try:
                    r = ask(base, headers, args.model, q["question"], args.max_tokens,
                            args.temperature, thinking, args.timeout)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                    detail = exc.read().decode()[:300] if isinstance(exc, urllib.error.HTTPError) else exc
                    print(f"  ! {q['id']} attempt {attempt + 1} failed: {detail}", flush=True)
                    if attempt == args.retries:
                        raise SystemExit(f"giving up on #{q['num']}: is the model server running at {base}? "
                                         f"Answers so far are saved; rerun the same command to continue.")
                    time.sleep(10)
            seconds = time.time() - started
            choice, usage = r["choices"][0], r.get("usage", {})
            message = choice["message"]
            f.write(json.dumps({
                "id": q["id"],
                "num": q["num"],
                "model": args.model,
                "answer": message.get("content") or "",
                "reasoning_chars": len(message.get("reasoning_content") or ""),
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
                "seconds": round(seconds, 1),
                "settings": settings,
            }, ensure_ascii=False) + "\n")
            f.flush()
            toks = usage.get("completion_tokens", 0)
            print(f"[{i}/{len(todo)}] #{q['num']:<3} {q['id']} {q['difficulty']:<12} {toks:>5} tok "
                  f"{seconds:6.1f}s ({toks / max(seconds, 1e-6):.1f} tok/s) {choice.get('finish_reason')} | "
                  f"~{fmt_duration((time.time() - run_started) / i * (len(todo) - i))} left", flush=True)

    total = time.time() - run_started
    print(f"DONE: {len(todo)} questions in {fmt_duration(total)} (~{total / len(todo):.0f} s/question)", flush=True)


if __name__ == "__main__":
    main()
