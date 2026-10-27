"""Send the trial questions to the Nanbeige Modal endpoint and save its answers.

Reads MODAL_URL / MODAL_KEY / MODAL_SECRET from ../.env (or the environment).
Resumable: questions already in the output file are skipped, so you can stop
with Ctrl+C and run again. Stdlib only.

The first line printed is a rough time and Modal cost estimate for the run; the
last line is the actual time and approximate cost.

Usage:
    python trial_eval/run_trial.py                       # all questions
    python trial_eval/run_trial.py --start 1 --end 30    # questions #1..#30 only
    python trial_eval/run_trial.py --limit 3             # quick smoke test
    python trial_eval/run_trial.py --start 31 --end 60 --estimate-only
    python trial_eval/run_trial.py --out trial_eval/answers_nanbeige_b0.jsonl
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Nanbeige/Nanbeige4.2-3B"
SYSTEM_PROMPT = (
    "You are an expert in radio-frequency engineering, digital signal processing, "
    "software-defined radio, and GNU Radio. Answer the question accurately and completely. "
    "For calculations, show the key steps and state the final result with units."
)

# Rough Modal list prices in USD/hour (check https://modal.com/pricing):
# L4 GPU ~$0.80 + 4 CPU cores ~$0.19 + 24 GiB memory ~$0.19 (the server's settings).
COST_PER_HOUR = 1.18
COLD_START_S = 150   # container start + model load, billed once if the GPU was idle
SCALEDOWN_S = 60     # GPU stays on (billed) this long after the last answer


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
    # ~9 tok/s on L4; answers typically use ~60% of max_tokens before stopping.
    return 0.6 * max_tokens / 9 + 5, "rough guess at ~9 tok/s"


def fmt_cost(seconds):
    return f"${seconds / 3600 * COST_PER_HOUR:.2f}"


def fmt_duration(seconds):
    if seconds < 60:
        return "<1 min"
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h} h {m} min" if h else f"{m} min"


def ask(url, key, secret, question, max_tokens, temperature, timeout):
    body = json.dumps({
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json", "Modal-Key": key, "Modal-Secret": secret},
    )
    # urllib follows Modal's 303 redirect for long-running requests.
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "trial_eval_100.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "answers_nanbeige_b0.jsonl"))
    ap.add_argument("--start", type=int, default=1, help="first question number (1-based)")
    ap.add_argument("--end", type=int, default=None, help="last question number (inclusive)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--estimate-only", action="store_true", help="print the time estimate and exit")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

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
    work = per_q * len(todo)
    billed = work + COLD_START_S + SCALEDOWN_S if todo else 0
    print(f"ESTIMATE: {len(todo)} questions -> ~{fmt_duration(work)}, ~{fmt_cost(billed)} Modal cost "
          f"(up to ~{fmt_duration(work + COLD_START_S)} incl. GPU cold start)", flush=True)
    print(f"  questions #{questions[0]['num']}-#{questions[-1]['num']} ({len(questions)}) | "
          f"{len(questions) - len(todo)} already answered | {len(todo)} to go")
    print(f"  ~{per_q:.0f} s/question ({basis}); ~${COST_PER_HOUR:.2f}/h for L4 + CPU + memory "
          f"(A10G/L40S fallback costs more)", flush=True)
    if args.estimate_only or not todo:
        return

    load_env(ROOT.parent / ".env")
    url = os.environ["MODAL_URL"].rstrip("/")
    key, secret = os.environ["MODAL_KEY"], os.environ["MODAL_SECRET"]

    run_started = time.time()
    with open(out, "a") as f:
        for i, q in enumerate(todo, 1):
            print(f"  asking #{q['num']} ({q['difficulty']}, {q['task_type']})...", flush=True)
            for attempt in range(args.retries + 1):
                started = time.time()
                try:
                    r = ask(url, key, secret, q["question"], args.max_tokens,
                            args.temperature, args.timeout)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                    detail = exc.read().decode()[:300] if isinstance(exc, urllib.error.HTTPError) else exc
                    print(f"  ! {q['id']} attempt {attempt + 1} failed: {detail}", flush=True)
                    if attempt == args.retries:
                        raise
                    time.sleep(10)
            seconds = time.time() - started
            choice, usage = r["choices"][0], r.get("usage", {})
            f.write(json.dumps({
                "id": q["id"],
                "num": q["num"],
                "model": MODEL_ID,
                "answer": choice["message"]["content"],
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
                "seconds": round(seconds, 1),
                "settings": {"system_prompt": SYSTEM_PROMPT, "max_tokens": args.max_tokens,
                             "temperature": args.temperature, "enable_thinking": False},
            }, ensure_ascii=False) + "\n")
            f.flush()
            toks = usage.get("completion_tokens", 0)
            print(f"[{i}/{len(todo)}] #{q['num']:<3} {q['id']} {q['difficulty']:<12} {toks:>5} tok "
                  f"{seconds:6.1f}s ({toks / max(seconds, 1e-6):.1f} tok/s) {choice.get('finish_reason')} | "
                  f"~{fmt_duration((time.time() - run_started) / i * (len(todo) - i))} left", flush=True)

    total = time.time() - run_started
    print(f"DONE: {len(todo)} questions in {fmt_duration(total)} "
          f"(~{total / len(todo):.0f} s/question), ~{fmt_cost(total + SCALEDOWN_S)} Modal cost", flush=True)


if __name__ == "__main__":
    main()
