"""Approximate how close Nanbeige's answers are to the graded teacher answers.

Metrics per question:
  sem_primary   cosine similarity (embeddings) to the primary reference answer
  sem_best      max cosine similarity over ALL passing teacher answers
  sem_norm      sem_best rescaled so 0 = unrelated answer, 1 = another teacher:
                  floor   = best similarity of a different question's reference
                  ceiling = average best similarity of each teacher answer to
                            the other passing teacher answers
  rougeL        lexical overlap (ROUGE-L F1) with the primary reference
  num_recall    calculation questions only: share of the reference's key numbers
                (last paragraph) that appear in the answer within 2% tolerance
  len_ratio     answer length / reference length (words)

Similarity is a proxy, not a correctness grade: a fluent wrong answer can
still score high. Use it to compare models on the same set, and spot-check.

Outputs <answers>_scores.csv, <answers>_summary.json and <answers>_report.html
(question, model answer, reference answer and scores for each question, plus
the summary metrics).

Usage (see README.txt for the one-time environment setup):
    .venv-eval/bin/python trial_eval/score_trial.py
    .venv-eval/bin/python trial_eval/score_trial.py --start 1 --end 30
"""

import argparse
import csv
import html
import json
import random
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def words(text):
    return len(text.split())


def key_numbers(text):
    """Distinct non-trivial numbers from the final paragraph of a reference."""
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    tail = " ".join(paras[-2:]) if paras else text
    nums = set()
    for m in NUM.findall(tail.replace(",", "")):
        try:
            v = float(m)
        except ValueError:
            continue
        if v not in (0.0, 1.0, 2.0) and abs(v) < 1e15:
            nums.add(v)
    return nums


def num_recall(reference, answer, tol=0.02):
    ref = key_numbers(reference)
    if not ref:
        return None
    got = [float(m) for m in NUM.findall(answer.replace(",", "")) if _is_float(m)]
    hit = sum(1 for r in ref if any(abs(g - r) <= tol * max(abs(r), 1e-12) for g in got))
    return hit / len(ref)


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


METRIC_HELP = {
    "sem_norm": "MAIN NUMBER. Meaning similarity rescaled: 0 = like an answer to a different question, 1 = like another passing teacher answer.",
    "sem_best": "Cosine similarity (0-1) to the closest passing teacher answer.",
    "rougeL": "Word overlap with the primary reference (low is normal for free-form text).",
    "num_recall": "Calculation questions only: share of the reference's final numbers found in the answer (2% tolerance).",
    "len_ratio": "Answer length / reference length (words).",
    "truncated": "Answers cut off by max_tokens (finish_reason = length).",
}


def write_html(path, rows, summary, questions, answers, title):
    esc = lambda t: html.escape(str(t))
    fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    def tone(v):
        if v is None:
            return ""
        return "good" if v >= 0.8 else "mid" if v >= 0.5 else "bad"

    def group_table(name, groups):
        head = "".join(f"<th>{m}</th>" for m in ("n", "sem_norm", "sem_best", "rougeL", "num_recall", "len_ratio", "truncated"))
        body = "".join(
            f"<tr><td>{esc(k)}</td><td>{g['n']}</td><td class='{tone(g['sem_norm'])}'>{fmt(g['sem_norm'])}</td>"
            f"<td>{fmt(g['sem_best'])}</td><td>{fmt(g['rougeL'])}</td><td>{fmt(g['num_recall'])}</td>"
            f"<td>{fmt(g['len_ratio'])}</td><td>{g['truncated']}</td></tr>"
            for k, g in groups.items())
        return f"<h3>{esc(name)}</h3><table><tr><th>group</th>{head}</tr>{body}</table>"

    o = summary["overall"]
    tiles = "".join(
        f"<div class='tile'><div class='label'>{m}</div><div class='value {tone(o[m]) if m == 'sem_norm' else ''}'>{fmt(o[m])}</div></div>"
        for m in ("sem_norm", "sem_best", "rougeL", "num_recall", "len_ratio", "truncated"))
    legend = "".join(f"<dt>{m}</dt><dd>{esc(t)}</dd>" for m, t in METRIC_HELP.items())

    cards = []
    for r in rows:
        q, a = questions[r["id"]], answers[r["id"]]
        chips = "".join(
            f"<span class='chip {tone(r[m]) if m == 'sem_norm' else ''}'>{m} {fmt(r[m])}</span>"
            for m in ("sem_norm", "sem_best", "rougeL", "num_recall", "len_ratio"))
        trunc = "<span class='chip bad'>truncated</span>" if r["finish_reason"] == "length" else ""
        cards.append(f"""
<article id="q{r['num']}">
  <header><b>#{r['num']}</b> <code>{esc(r['id'])}</code>
    <span class="tag">{esc(r['difficulty'])}</span><span class="tag">{esc(r['task_type'])}</span>
    {'<span class="tag">GNU Radio</span>' if r['is_gnuradio'] else ''}</header>
  <div class="chips">{chips}{trunc}</div>
  <h4>Question</h4><div class="text">{esc(q['question'])}</div>
  <h4>Model answer <small>({fmt(a.get('seconds'))} s, {esc(a.get('usage', {}).get('completion_tokens', '?'))} tokens)</small></h4>
  <div class="text">{esc(a['answer'])}</div>
  <details><summary>Reference answer ({esc(q.get('reference_model', ''))})</summary>
    <div class="text">{esc(q['reference_answer'])}</div></details>
</article>""")

    path.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
:root {{ --bg:#fafaf9; --card:#fff; --fg:#1c1917; --muted:#78716c; --line:#e7e5e4;
  --good:#15803d; --mid:#b45309; --bad:#b91c1c; --code:#f5f5f4; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1c1917; --card:#292524; --fg:#f5f5f4; --muted:#a8a29e;
  --line:#44403c; --good:#4ade80; --mid:#fbbf24; --bad:#f87171; --code:#1c1917; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,sans-serif; margin:0; padding:24px 16px; }}
main {{ max-width:960px; margin:auto; }}
h1 {{ margin:0 0 4px; font-size:24px; }} .muted {{ color:var(--muted); }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; margin:20px 0; }}
.tile, article, table, dl {{ background:var(--card); border:1px solid var(--line); border-radius:10px; }}
.tile {{ padding:12px; }} .label {{ color:var(--muted); font-size:13px; }} .value {{ font-size:24px; font-weight:600; }}
table {{ border-collapse:collapse; width:100%; overflow:hidden; display:block; overflow-x:auto; }}
th, td {{ padding:6px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
th:first-child, td:first-child {{ text-align:left; }}
dl {{ padding:12px 16px; display:grid; grid-template-columns:max-content 1fr; gap:4px 16px; }}
dt {{ font-weight:600; }} dd {{ margin:0; color:var(--muted); }}
article {{ padding:16px; margin:16px 0; }}
.tag, .chip {{ display:inline-block; border:1px solid var(--line); border-radius:99px; padding:0 8px; margin:2px 4px 2px 0; font-size:13px; }}
.chips {{ margin:8px 0; }} h4 {{ margin:14px 0 4px; }}
.text {{ white-space:pre-wrap; background:var(--code); border-radius:8px; padding:10px 12px; overflow-wrap:anywhere; }}
.good {{ color:var(--good); }} .mid {{ color:var(--mid); }} .bad {{ color:var(--bad); }}
code {{ font-size:13px; }} summary {{ cursor:pointer; margin-top:12px; color:var(--muted); }}
</style></head><body><main>
<h1>{esc(title)}</h1>
<div class="muted">{len(rows)} questions scored &middot; answers: {esc(summary['settings']['answers'])} &middot; embeddings: {esc(summary['settings']['embed_model'])}</div>
<div class="tiles">{tiles}</div>
<h2>Metrics</h2><dl>{legend}</dl>
<p class="muted">Similarity is a proxy, not a correctness grade: a fluent answer with a wrong number can still score high.</p>
<h2>Breakdown</h2>
{group_table("By difficulty", summary["by_difficulty"])}
{group_table("GNU Radio-specific", summary["by_is_gnuradio"])}
{group_table("By task type", summary["by_task_type"])}
<h2>Questions</h2>
{''.join(cards)}
</main></body></html>
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "trial_eval_100.jsonl"))
    ap.add_argument("--answers", default=str(ROOT / "answers_nanbeige_b0.jsonl"))
    ap.add_argument("--embed-model", default="BAAI/bge-m3")  # 8k-token context: no truncation of long answers
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--start", type=int, default=1, help="first question number (1-based)")
    ap.add_argument("--end", type=int, default=None, help="last question number (inclusive)")
    args = ap.parse_args()

    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer

    questions = {q["id"]: q for q in map(json.loads, open(args.questions)) if q}
    answers = {a["id"]: a for a in map(json.loads, open(args.answers)) if a}
    num = {i: n for n, i in enumerate(questions, 1)}
    end = args.end or len(questions)
    ids = [i for i in questions if i in answers and args.start <= num[i] <= end]
    if not ids:
        raise SystemExit(f"No answered questions in #{args.start}-#{end}.")
    print(f"scoring {len(ids)} answered questions in #{args.start}-#{end} (of {len(questions)})")

    # Floor: a reference answer to a different question, drawn from the whole set
    # with a per-question seed, so it does not depend on which range is scored.
    other_ref = {i: questions[random.Random(i).choice([j for j in questions if j != i])]["reference_answer"]
                 for i in ids}

    # Embed every text once; reference embeddings are cached on disk because
    # they are the same for every model we score.
    import hashlib
    import numpy as np

    digest = lambda t: hashlib.sha1(t.encode()).hexdigest()
    cache_path = ROOT / f".embed_cache_{args.embed_model.replace('/', '__')}.npz"
    cache = dict(np.load(cache_path)) if cache_path.exists() else {}
    ref_texts = {r["answer"] for i in ids for r in questions[i]["references"]} | set(other_ref.values())
    texts = list(ref_texts | {answers[i]["answer"] for i in ids})
    missing = [t for t in texts if digest(t) not in cache]
    print(f"embedding {len(missing)} texts ({len(texts) - len(missing)} cached)")
    if missing:
        model = SentenceTransformer(args.embed_model)
        model.max_seq_length = 2048  # answers are ~550 tokens median, ~1,000 at p90
        vecs = model.encode(missing, normalize_embeddings=True, batch_size=8, show_progress_bar=True)
        for t, v in zip(missing, vecs):
            cache[digest(t)] = v
        # Keep every reference embedding (not just this range's); drop model answers.
        keep = {digest(r["answer"]) for q in questions.values() for r in q["references"]}
        keep |= {digest(t) for t in other_ref.values()}
        np.savez(cache_path, **{k: v for k, v in cache.items() if k in keep})
    cos = lambda a, b: float(cache[digest(a)] @ cache[digest(b)])

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rows = []
    for i in ids:
        q, ans = questions[i], answers[i]["answer"]
        refs = [r["answer"] for r in q["references"]]
        primary = q["reference_answer"]
        # Ceiling and floor use the same best-match rule as the answer itself.
        best = max(cos(ans, r) for r in refs)
        ceiling = (st.mean(max(cos(r, o) for o in refs if o is not r) for r in refs)
                   if len(refs) > 1 else 1.0)
        floor = max(cos(other_ref[i], r) for r in refs)
        rows.append({
            "num": num[i],
            "id": i,
            "difficulty": q["difficulty"],
            "task_type": q["task_type"],
            "is_gnuradio": q["is_gnuradio"],
            "sem_primary": round(cos(ans, primary), 4),
            "sem_best": round(best, 4),
            "sem_ceiling": round(ceiling, 4),
            "sem_floor": round(floor, 4),
            "sem_norm": round((best - floor) / (ceiling - floor), 4) if ceiling > floor else None,
            "rougeL": round(rouge.score(primary, ans)["rougeL"].fmeasure, 4),
            "num_recall": (round(v, 4) if (v := num_recall(primary, ans)) is not None else None)
                          if q["task_type"] == "calculation" else None,
            "len_ratio": round(words(ans) / max(words(primary), 1), 3),
            "finish_reason": answers[i].get("finish_reason"),
        })

    prefix = Path(args.out_prefix) if args.out_prefix else Path(args.answers).with_suffix("")
    if args.start != 1 or end != len(questions):
        prefix = prefix.with_name(f"{prefix.name}_q{args.start}-{end}")
    csv_path = prefix.with_name(prefix.name + "_scores.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def summarize(group):
        out = {"n": len(group)}
        for m in ("sem_best", "sem_norm", "rougeL", "num_recall", "len_ratio"):
            vals = [r[m] for r in group if r[m] is not None]
            out[m] = round(st.mean(vals), 3) if vals else None
        out["truncated"] = sum(r["finish_reason"] == "length" for r in group)
        return out

    summary = {"overall": summarize(rows)}
    for field in ("difficulty", "task_type", "is_gnuradio"):
        groups = defaultdict(list)
        for r in rows:
            groups[str(r[field])].append(r)
        summary[f"by_{field}"] = {k: summarize(v) for k, v in sorted(groups.items())}
    summary["settings"] = {"embed_model": args.embed_model, "answers": str(args.answers)}
    json_path = prefix.with_name(prefix.name + "_summary.json")
    json_path.write_text(json.dumps(summary, indent=2))

    html_path = prefix.with_name(prefix.name + "_report.html")
    write_html(html_path, rows, summary, questions, answers,
               f"Trial eval: {Path(args.answers).stem} (#{args.start}-#{end})")

    print(f"\nper-question scores -> {csv_path}\nsummary -> {json_path}\nHTML report -> {html_path}\n")
    hdr = f"{'group':<22}{'n':>4}{'sem_best':>10}{'sem_norm':>10}{'rougeL':>8}{'num_rec':>9}{'len_x':>7}{'trunc':>7}"
    print(hdr)
    def line(name, s):
        fmt = lambda v: f"{v:.3f}" if v is not None else "  -  "
        print(f"{name:<22}{s['n']:>4}{fmt(s['sem_best']):>10}{fmt(s['sem_norm']):>10}"
              f"{fmt(s['rougeL']):>8}{fmt(s['num_recall']):>9}{fmt(s['len_ratio']):>7}{s['truncated']:>7}")
    line("overall", summary["overall"])
    for field in ("difficulty", "is_gnuradio", "task_type"):
        for k, s in summary[f"by_{field}"].items():
            line(f"{field[:4]}={k}", s)


if __name__ == "__main__":
    main()
