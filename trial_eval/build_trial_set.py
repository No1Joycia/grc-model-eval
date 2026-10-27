"""Build a ~100-question trial evaluation set with solid reference answers.

Picks questions from the TRAIN split (the frozen eval split has no answers),
keeps only questions where >= MIN_PASS teacher answers passed review, drops
questions that depend on an unseen source ("the textbook", ...), and
stratifies by difficulty and task type with a GNU Radio quota.

IMPORTANT: the IDs written to trial_eval_ids.txt must be excluded from any
training export, or later comparisons with fine-tuned models are contaminated.

Usage (stdlib only):
    python3 trial_eval/build_trial_set.py
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "dataset" / "dataset"
OUT = ROOT / "trial_eval_100.jsonl"
IDS = ROOT / "trial_eval_ids.txt"

N_TOTAL = 100
DIFFICULTY_QUOTA = {"basic": 33, "intermediate": 34, "advanced": 33}
GNURADIO_MIN = 20          # minimum GNU Radio-specific questions overall
MIN_PASS = 5               # of 7 teachers must have passed review
SEED = "trial-eval-v1-2026-10-27"

# Higher review pass rate first; the first PASS answer in this order is the primary reference.
MODEL_PRIORITY = [
    "gemma4-31b", "muse-glimmer-30b", "qwen36-27b", "nemotron3-super-120b",
    "gpt-oss-120b", "laguna-s21", "devstral-small2-24b",
]

CONTEXT_DEPENDENT = re.compile(
    r"\b(the|this|that|your) (textbook|book|text|author|chapter|passage|article|"
    r"section|document|tutorial|guide|course|lecture|paper|figure|table)\b", re.I)
GNURADIO = re.compile(r"gnu ?radio|\bgrc\b|flowgraph|companion", re.I)

TASK_TYPE_MAP = {
    "factual_knowledge": "factual", "calculations": "calculation",
    "system_design": "design", "diagnosis_from_symptoms": "diagnosis",
    "gnu_radio_usage": "gnuradio_usage", "compare": "compare_choose",
    "ask_for_info": "missing_info", "information_gathering": "missing_info",
    "asking_for_missing_information": "missing_info",
}


def jl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def norm_task_type(t):
    t = (t or "").strip().lower().replace(" ", "_").replace("/", "_")
    return TASK_TYPE_MAP.get(t, t)


def rank(qid):
    return hashlib.sha256(f"{SEED}:{qid}".encode()).hexdigest()


def main():
    questions = {q["id"]: q for q in jl(DATA / "04_eval_frozen/train_questions.jsonl")}

    # Answers; drop (question, model) keys that appear more than once with
    # different text, since we cannot tell which version was judged.
    answers, seen = {}, Counter()
    for path in (DATA / "05_answers").glob("*/answers.jsonl"):
        for a in jl(path):
            key = (a["question_id"], a["model"])
            seen[key] += 1
            answers[key] = a
    ambiguous = {k for k, n in seen.items() if n > 1}

    verdicts = defaultdict(dict)
    for j in jl(DATA / "08_judging/judgments.jsonl"):
        verdicts[j["question_id"]][j["candidate_model"]] = j["status"]

    candidates = []
    for qid, q in questions.items():
        if CONTEXT_DEPENDENT.search(q["question"]):
            continue
        refs = []
        for model in MODEL_PRIORITY:
            key = (qid, model)
            a = answers.get(key)
            if (verdicts[qid].get(model) == "PASS" and a and key not in ambiguous
                    and a.get("finish_reason") == "stop" and a["answer"].strip()):
                refs.append({"model": model, "answer": a["answer"].strip()})
        n_reject = sum(1 for s in verdicts[qid].values() if s == "REJECT")
        if len(refs) < MIN_PASS or n_reject:
            continue
        candidates.append({
            "id": qid,
            "question": q["question"].strip(),
            "difficulty": q.get("difficulty"),
            "task_type": norm_task_type(q.get("task_type")),
            "primary_topic": q.get("primary_topic"),
            "is_gnuradio": bool(GNURADIO.search(q["question"] + " " + (q.get("primary_topic") or ""))),
            "n_pass": len(refs),
            "n_judged": len(verdicts[qid]),
            "reference_model": refs[0]["model"],
            "reference_answer": refs[0]["answer"],
            "references": refs,
        })

    # Stratified pick: per difficulty, round-robin over task types in hash
    # order, taking GNU Radio questions first until the quota is met.
    picked = []
    gr_quota = {d: round(GNURADIO_MIN * n / N_TOTAL) for d, n in DIFFICULTY_QUOTA.items()}
    for diff, quota in DIFFICULTY_QUOTA.items():
        pool = sorted((c for c in candidates if c["difficulty"] == diff), key=lambda c: rank(c["id"]))
        chosen = [c for c in pool if c["is_gnuradio"]][: gr_quota[diff]]
        by_type = defaultdict(list)
        for c in pool:
            if c not in chosen:
                by_type[c["task_type"]].append(c)
        types = sorted(by_type, key=lambda t: -len(by_type[t]))
        while len(chosen) < quota and any(by_type.values()):
            for t in types:
                if by_type[t] and len(chosen) < quota:
                    chosen.append(by_type[t].pop(0))
        picked.extend(chosen)

    with open(OUT, "w") as f:
        for c in picked:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    IDS.write_text("\n".join(c["id"] for c in picked) + "\n")

    print(f"train questions: {len(questions)} | eligible candidates: {len(candidates)}")
    print(f"wrote {len(picked)} questions -> {OUT.name} (IDs -> {IDS.name})")
    print("difficulty:", dict(Counter(c["difficulty"] for c in picked)))
    print("task_type:", dict(Counter(c["task_type"] for c in picked).most_common()))
    print("GNU Radio-specific:", sum(c["is_gnuradio"] for c in picked))
    print("teachers passing per question:", dict(sorted(Counter(c["n_pass"] for c in picked).items())))
    print("primary reference model:", dict(Counter(c["reference_model"] for c in picked).most_common()))


if __name__ == "__main__":
    main()
