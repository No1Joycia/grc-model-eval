"""Send the trial questions to Cohere's hosted North-Mini-Code (raw mode) and save its answers.

Raw mode = one direct chat call per question, with the same system prompt and questions as the
Nanbeige run (../trial_eval/run_trial.py), so the answers are scored the same way by
../trial_eval/score_trial.py. No GRC-Agent, no tools.

Reads COHERE_API_KEY from ./.env (or the environment). Resumable: questions already in the
output file are skipped, so you can stop with Ctrl+C and run again. Stdlib only.

The first line printed is a rough time estimate and how much of the trial key's monthly call
budget the run uses; the last line is the actual time and token totals.

Usage (from northMiniCode/):
    python3 run_raw.py --limit 3                 # smoke test: 3 questions
    python3 run_raw.py                           # all 100 (resumes where it stopped)
    python3 run_raw.py --start 1 --end 30        # questions #1..#30 only
    python3 run_raw.py --estimate-only
    python3 run_raw.py --thinking off --out answers/answers_north_raw_nothink.jsonl
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRIAL = ROOT.parent / "trial_eval"
API_URL = "https://api.cohere.com/v2/chat"
MODEL_ID = "north-mini-code-1.0"     # the name Cohere's /v1/models lists for this key
# Same system prompt as the Nanbeige B0 run (trial_eval/run_trial.py), for a like-for-like comparison.
SYSTEM_PROMPT = (
    "You are an expert in radio-frequency engineering, digital signal processing, "
    "software-defined radio, and GNU Radio. Answer the question accurately and completely. "
    "For calculations, show the key steps and state the final result with units."
)
TRIAL_MONTHLY_CALLS = 1000   # reported trial-key cap (third-party source); check your Cohere dashboard
FINISH = {"COMPLETE": "stop", "STOP_SEQUENCE": "stop", "MAX_TOKENS": "length"}
RETRY_STATUS = {429, 500, 502, 503, 504}


def load_env(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def estimate_seconds(out):
    """Per-question seconds from earlier answers in `out`, else a rough guess."""
    if out.exists():
        past = [json.loads(l).get("seconds") for l in open(out, encoding="utf-8-sig") if l.strip()]
        past = [x for x in past if x]
        if past:
            return sum(past) / len(past), f"average of {len(past)} earlier answers"
    return 25.0, "rough guess (~1.5K tokens of reasoning + answer per question)"


def fmt_duration(seconds):
    if seconds < 60:
        return "<1 min"
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h} h {m} min" if h else f"{m} min"


def ask(key, question, args):
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "thinking": {"type": "enabled" if args.thinking == "on" else "disabled"},
    }
    req = urllib.request.Request(
        API_URL, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        return json.loads(resp.read())


def ask_with_retries(key, q, args):
    """One question; waits and retries on rate limits (429) and server errors."""
    for attempt in range(args.retries + 1):
        started = time.time()
        try:
            return ask(key, q["question"], args), time.time() - started
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code in (401, 403):
                raise SystemExit(f"Cohere rejected the key (HTTP {exc.code}): {detail}")
            if exc.code not in RETRY_STATUS or attempt == args.retries:
                raise SystemExit(f"#{q['num']} {q['id']}: HTTP {exc.code}: {detail}")
            try:
                wait = float(exc.headers.get("Retry-After"))   # seconds; an HTTP date falls back below
            except (TypeError, ValueError):
                wait = 60 if exc.code == 429 else 15
            print(f"  ! #{q['num']} HTTP {exc.code} (attempt {attempt + 1}), waiting {wait:.0f}s: "
                  f"{detail[:120]}", flush=True)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == args.retries:
                raise
            wait = 15
            print(f"  ! #{q['num']} {exc} (attempt {attempt + 1}), waiting {wait}s", flush=True)
        time.sleep(wait)


def parse(r):
    """Cohere v2 response -> (answer, reasoning, finish_reason, raw finish, usage)."""
    content = (r.get("message") or {}).get("content") or []
    answer = "".join(c.get("text", "") for c in content if c.get("type") == "text").strip()
    reasoning = "".join(c.get("thinking", "") for c in content if c.get("type") == "thinking").strip()
    raw_finish = r.get("finish_reason")
    u = r.get("usage") or {}
    tokens, billed = u.get("tokens") or {}, u.get("billed_units") or {}
    usage = {
        "prompt_tokens": tokens.get("input_tokens", 0),
        "completion_tokens": tokens.get("output_tokens", 0),
        "reasoning_tokens": tokens.get("reasoning_tokens", 0),
        "total_tokens": tokens.get("input_tokens", 0) + tokens.get("output_tokens", 0),
        "billed_input_tokens": billed.get("input_tokens", 0),
        "billed_output_tokens": billed.get("output_tokens", 0),
    }
    return answer, reasoning, FINISH.get(raw_finish, (raw_finish or "").lower()), raw_finish, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(TRIAL / "trial_eval_100.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "answers" / "answers_north_raw.jsonl"))
    ap.add_argument("--start", type=int, default=1, help="first question number (1-based)")
    ap.add_argument("--end", type=int, default=None, help="last question number (inclusive)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--estimate-only", action="store_true", help="print the estimate and exit")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--thinking", choices=["on", "off"], default="on",
                    help="Cohere's reasoning (default on; off repeats answers, see README)")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="reasoning + answer budget per question (default 8192)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--min-interval", type=float, default=3.5,
                    help="seconds between requests (trial keys allow ~20/min; default 3.5)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    all_questions = [json.loads(l) for l in open(args.questions, encoding="utf-8-sig") if l.strip()]
    end = min(args.end or len(all_questions), len(all_questions))
    if not 1 <= args.start <= end:
        raise SystemExit(f"--start/--end must satisfy 1 <= start <= end <= {len(all_questions)}")
    for n, q in enumerate(all_questions, 1):
        q["num"] = n
    questions = all_questions[args.start - 1 : end][: args.limit]
    if not questions:
        raise SystemExit("No questions selected: check --limit (must be at least 1)")
    out = Path(args.out)
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8-sig") if l.strip()} if out.exists() else set()
    todo = [q for q in questions if q["id"] not in done]
    per_q, basis = estimate_seconds(out)
    work = max(per_q, args.min_interval) * len(todo)
    print(f"ESTIMATE: {len(todo)} questions -> ~{fmt_duration(work)}, {len(todo)} API calls "
          f"(~{len(todo) / TRIAL_MONTHLY_CALLS:.0%} of a trial key's ~{TRIAL_MONTHLY_CALLS}/month; "
          f"trial calls are free)", flush=True)
    print(f"  questions #{questions[0]['num']}-#{questions[-1]['num']} ({len(questions)}) | "
          f"{len(questions) - len(todo)} already answered | {len(todo)} to go")
    print(f"  ~{per_q:.0f} s/question ({basis}) | model {args.model}, thinking {args.thinking}, "
          f"max_tokens {args.max_tokens}, temperature {args.temperature}", flush=True)
    if args.estimate_only or not todo:
        return

    load_env(ROOT / ".env")
    key = os.environ.get("COHERE_API_KEY")
    if not key:
        raise SystemExit(f"COHERE_API_KEY is not set: put it in {ROOT / '.env'} (see README.md)")

    out.parent.mkdir(parents=True, exist_ok=True)
    run_started = time.time()
    last_request = 0.0
    totals = {"completion_tokens": 0, "reasoning_tokens": 0, "length": 0}
    with open(out, "a", encoding="utf-8", newline="\n") as f:
        for i, q in enumerate(todo, 1):
            wait = args.min_interval - (time.time() - last_request)
            if wait > 0:
                time.sleep(wait)
            print(f"  asking #{q['num']} ({q['difficulty']}, {q['task_type']})...", flush=True)
            last_request = time.time()
            r, seconds = ask_with_retries(key, q, args)
            answer, reasoning, finish, raw_finish, usage = parse(r)
            f.write(json.dumps({
                "id": q["id"],
                "num": q["num"],
                "model": f"cohere/{args.model}",
                "answer": answer,
                "reasoning": reasoning,
                "finish_reason": finish,
                "cohere_finish_reason": raw_finish,
                "usage": usage,
                "seconds": round(seconds, 1),
                "settings": {"system_prompt": SYSTEM_PROMPT, "max_tokens": args.max_tokens,
                             "temperature": args.temperature, "thinking": args.thinking,
                             "api": API_URL},
            }, ensure_ascii=False) + "\n")
            f.flush()
            toks, rtoks = usage["completion_tokens"], usage["reasoning_tokens"]
            totals["completion_tokens"] += toks
            totals["reasoning_tokens"] += rtoks
            totals["length"] += finish == "length"
            print(f"[{i}/{len(todo)}] #{q['num']:<3} {q['id']} {q['difficulty']:<12} {toks:>5} tok "
                  f"({rtoks} reasoning) {seconds:6.1f}s {finish} | "
                  f"~{fmt_duration((time.time() - run_started) / i * (len(todo) - i))} left", flush=True)

    total = time.time() - run_started
    print(f"DONE: {len(todo)} questions in {fmt_duration(total)} (~{total / len(todo):.0f} s/question), "
          f"{totals['completion_tokens']} output tokens ({totals['reasoning_tokens']} reasoning), "
          f"{totals['length']} cut off at max_tokens -> {out}", flush=True)


if __name__ == "__main__":
    main()
