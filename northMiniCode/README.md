# North-Mini-Code: raw-mode trial evaluation

This folder asks Cohere's hosted **North-Mini-Code 1.0** the 100 trial questions in
`../trial_eval/trial_eval_100.jsonl` and scores its answers with `../trial_eval/score_trial.py`.

- **Raw mode:** one direct chat call per question, using the system prompt from `../trial_eval/run_trial.py`. There's
  no GRC-Agent and no tools. Agent mode is a separate step: `../grc-agent-adapter` (`./agent batch`).
- **No hosting needed.** Cohere runs the model, and calls with a Trial key are free.
- **Stdlib only.** Any `python3` works, with no virtual environment.

| File | |
|---|---|
| `run_raw.py` | sends the questions to Cohere and saves the answers (resumable) |
| `.env` | your Cohere key (git-ignored, permissions 600). `.env.example` is the template. |
| `answers/` | answer and score files (git-ignored) |

## 1. One-time setup

1. Get a key at https://dashboard.cohere.com/api-keys. A **Trial key** is created when you sign up.
2. Save it, so it never goes into a tracked file:

   ```bash
   cd ~/Projects/grc-model-eval/northMiniCode
   cp .env.example .env
   open -e .env          # paste the key after COHERE_API_KEY=
   chmod 600 .env
   ```

   On Joy's Mac this is already done.

## 2. Run

```bash
cd ~/Projects/grc-model-eval/northMiniCode
python3 run_raw.py --estimate-only        # what a run would take; sends nothing
python3 run_raw.py --limit 3              # smoke test: 3 questions
python3 run_raw.py                        # all 100 (resumes where it stopped)
```

- To keep the Mac awake for the full run: `caffeinate -i python3 run_raw.py`.
- To run part of the set: `--start 31 --end 60`.
- **Resumable.** Questions already in the output file are skipped, so stop with Ctrl+C and run again.

What you'll see (from the first real run):

```
ESTIMATE: 100 questions -> ~41 min, 100 API calls (~10% of a trial key's ~1000/month; trial calls are free)
  questions #1-#100 (100) | 0 already answered | 100 to go
  ~25 s/question (rough guess ...) | model north-mini-code-1.0, thinking on, max_tokens 8192, temperature 0.0
  asking #1 (basic, design)...
[1/3] #1   q_2bce8f4f13f014be basic         2681 tok (639 reasoning)   35.0s stop | ~1 min left
...
DONE: 3 questions in 1 min (~24 s/question), 5799 output tokens (990 reasoning), 0 cut off at max_tokens -> answers/answers_north_raw.jsonl
```

Expect about 25 s per question, so roughly 40 minutes for all 100. The first estimate is a rough guess;
once there are answers in the output file, the estimate uses their average time instead.

## 3. Score

The scorer needs the one-time `.venv-eval` environment described in `../trial_eval/README.txt`:

```bash
cd ~/Projects/grc-model-eval/northMiniCode
../.venv-eval/bin/python ../trial_eval/score_trial.py --answers answers/answers_north_raw.jsonl
```

This writes `answers/answers_north_raw_scores.csv`, `_summary.json` and `_report.html` next to the answers.
To score only part of the set, add `--start`/`--end` (the files then get a suffix such as `_q1-10`).
Open the report in a browser to read each answer next to its reference. The metrics are explained below;
`sem_norm` is the main number.

### Reading the scores

Each answer is compared with **reference answers** written by several teacher models that passed grading.
Each question has one main ("primary") reference and sometimes other passing ones.

| Score | What it measures | How to read it |
|---|---|---|
| `sem_primary` | How close the answer's *meaning* is to the primary reference: the cosine similarity of the two texts' embeddings (`BAAI/bge-m3`), from 0 to 1. | Raw values bunch up between about 0.6 and 0.95, because any two texts on the same technical topic score high. |
| `sem_best` | The same measure, against whichever passing reference is closest. | A valid answer that takes a different teacher's approach isn't penalised. |
| `sem_floor` | How similar a reference answer to an *unrelated* question is to this question's references. | The score of "answered the wrong question". |
| `sem_ceiling` | How similar the teacher answers are to *each other*. | The score of "as good as another expert". |
| **`sem_norm`** | **Main number.** `sem_best` rescaled per question so that the floor is 0 and the ceiling is 1. | About **1** means as close as another expert's answer (above 1 is possible). About **0** means no closer than an answer to a different question. **Below 0** means worse than that; an empty answer lands here. The per-question rescaling removes the bunching, so questions can be compared and averaged. |
| `rougeL` | *Word* overlap with the primary reference: the longest run of shared words in order (an F1 score). | 0.1–0.3 is normal for free-form explanations, which word things differently. Treat it as secondary. Calculation answers score higher because formulas and numbers repeat word for word. |
| `num_recall` | Calculation questions only: the share of the key numbers in the reference's final paragraphs that also appear in the answer, within 2%. | 1.0 means every key number matched. It can miss correct numbers written in another form (`1e6` vs `1,000,000`, kHz vs Hz). |
| `len_ratio` | Answer word count divided by primary-reference word count. | North-Mini-Code runs long (1.5–2.4× on most questions). 0 means an empty answer. |
| `finish_reason` / `truncated` | Whether each answer finished (`stop`) or hit `--max-tokens` (`length`). `truncated` in the summary counts the `length` answers. | An answer cut off at the limit scores low for that reason alone, so check these before reading the averages. |

**Limits:** these scores measure similarity, not correctness. A fluent answer with one wrong number can still
score near 1, so read the lowest-scoring answers in the report by hand.

## 4. Settings and why

| Option | Default | Why |
|---|---|---|
| `--model` | `north-mini-code-1.0` | the name Cohere's model list gives for this key (`north-mini-code-1-0`, GRC-Agent's spelling, is accepted too) |
| `--thinking on\|off` | `on` | Cohere recommends reasoning on. **With it off, North-Mini-Code repeated its whole answer twice** in both of two tests (Oct 27, 2026), which would distort the length and similarity scores. |
| `--max-tokens` | 8192 | the budget covers reasoning **and** the answer. Answers so far used about 900–2,700 tokens, so a budget of 1,536 would cut some off. |
| `--temperature` | 0 | deterministic, so a rerun gives (nearly) the same answers |
| `--min-interval` | 3.5 s | trial keys are rate-limited (reported at about 20 calls/min); 3.5 s keeps a run under that |
| `--retries` | 4 | on HTTP 429 (rate limit) or 5xx it waits (Retry-After header, else 60 s / 15 s) and retries |
| `--out` | `answers/answers_north_raw.jsonl` | use a new file for every new setting, e.g. `answers/answers_north_raw_t1.jsonl` for `--temperature 1` |

The system prompt is fixed in `run_raw.py` and is identical to `../trial_eval/run_trial.py`.

## 5. Output format

`answers/answers_north_raw.jsonl` has one line per question, with the fields `score_trial.py` reads (`id`,
`answer`, `finish_reason`), plus:

```json
{"id": "q_2bce8f4f13f014be", "num": 1, "model": "cohere/north-mini-code-1.0",
 "answer": "Below is a complete, ready-to-copy GNU-Radio flow-graph ...",
 "reasoning": "The user asks: \"Design a flowgraph that reads samples from a file, processes them, ...",
 "finish_reason": "stop", "cohere_finish_reason": "COMPLETE",
 "usage": {"prompt_tokens": 101, "completion_tokens": 2681, "reasoning_tokens": 639, "total_tokens": 2782,
           "billed_input_tokens": 71, "billed_output_tokens": 2677},
 "seconds": 35.0,
 "settings": {"system_prompt": "...", "max_tokens": 8192, "temperature": 0.0, "thinking": "on", "api": "https://api.cohere.com/v2/chat"}}
```

- `answer` is the visible answer only. The model's reasoning is kept separately in `reasoning` and isn't scored.
- `finish_reason`: `stop` means it finished; `length` means it was cut off at `--max-tokens` (rerun those with
  a larger budget into a new file).

## 6. Costs and limits

- **Trial key:** free. Cohere says Trial keys can't be used for production or commercial work. A course
  evaluation is fine, but check with the team before using it for Qoherent deliverables.
- **Call budget:** one run of 100 questions is 100 calls. A third-party guide reports about 1,000 calls/month
  for trial keys; your dashboard shows the real usage.
- **Production key:** Cohere doesn't publish a North-Mini-Code price. Expect well under $1 per 100-question run
  at typical small-model prices.

## 7. Troubleshooting

| Message | Fix |
|---|---|
| `COHERE_API_KEY is not set` | create `.env` (section 1) |
| `Cohere rejected the key (HTTP 401)` | the key is wrong or was revoked: make a new one on the dashboard |
| `! #N HTTP 429 ... waiting 60s` | the rate limit was hit; the script waits and retries. If it keeps happening, raise `--min-interval` |
| `HTTP 429` and the run stops | the monthly trial cap may be used up: check the dashboard, or switch to a production key |
| many `length` in the progress lines | raise `--max-tokens` and write to a new `--out` file |
