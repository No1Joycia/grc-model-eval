TRIAL EVALUATION — BASELINE NANBEIGE4.2-3B (B0) ON 100 GRADED QUESTIONS
=====================================================================

FILES
-----
build_trial_set.py      builds the question set (already run; re-run is deterministic)
trial_eval_100.jsonl    100 questions + graded reference answers
trial_eval_ids.txt      the 100 question IDs -> EXCLUDE FROM ALL TRAINING DATA
run_trial.py            sends questions to the Modal endpoint, saves answers (resumable)
score_trial.py          similarity scores: answers vs. reference answers

HOW THE 100 QUESTIONS WERE CHOSEN
---------------------------------
- From the TRAIN split (the 185 frozen eval questions have no graded answers).
- Only questions where >= 5 of 7 teacher models PASSED review and none were REJECTED.
- Dropped questions that depend on an unseen source ("the textbook", "this chapter", ...).
- 33 basic / 34 intermediate / 33 advanced; spread across task types; 31 GNU Radio-specific.
- Primary reference = PASS answer from the most reliable teacher (gemma4-31b, else muse-glimmer-30b).
  All PASS answers are kept in "references" (5-7 per question).

STEP 1 — GET ANSWERS (stdlib only, any python3)
-----------------------------------------------
Easiest: the Colab notebook colab/eval.ipynb runs steps 1 and 2 for you (see colab/README.md).

By hand, run_trial.py talks to any OpenAI-compatible server. Default: llama-server on this
machine (bash serve/run_local.sh nanbeige42, see docs/local-llamacpp.md):
    cd ~/Projects/grc-model-eval
    python3 trial_eval/run_trial.py --start 1 --end 30 --estimate-only    # time estimate only
    python3 trial_eval/run_trial.py --start 1 --end 30 --out trial_eval/answers_local.jsonl
    python3 trial_eval/run_trial.py --thinking on --max-tokens 4096 --out trial_eval/answers_local_think.jsonl

Options:
    --base-url     server, ending in /v1 (default http://127.0.0.1:8080/v1)
    --model        model name sent to the server (default nanbeige42, the name run_local.sh gives it)
    --thinking     off (default) or on; sent as chat_template_kwargs.enable_thinking
    --max-tokens   longest answer (default 1536; with thinking on use 4096+, reasoning counts too)
    --start/--end  question range; --limit N; --estimate-only
    --out          answers file (default trial_eval/answers_nanbeige_b0.jsonl, the old Modal baseline)
    --api-key-env  name of an environment variable holding a Bearer key (not needed locally)
The old Modal endpoint, until it is retired (MODAL_KEY / MODAL_SECRET from modal/.env):
    python3 trial_eval/run_trial.py --base-url "$MODAL_URL/v1" --model Nanbeige/Nanbeige4.2-3B

Question numbers are line numbers in trial_eval_100.jsonl, sorted by difficulty:
#1-33 basic, #34-67 intermediate, #68-100 advanced. For a mixed sample pick
a range from each block (e.g. 1-10, 34-43, 68-77) and run them one after another
into the same answers file.

Every answer row records the settings it was made with (model, server, system prompt,
max_tokens, temperature, thinking) plus reasoning_chars. The script refuses to add answers
made with different settings to an existing file: use another --out. Rerunning the same
command resumes: questions already in the file are skipped.

Progress: the first line is an ESTIMATE (from earlier answers in the file, or a rough
~20 tok/s guess), then "asking #N ..." before each request and one line per answer with
tokens, seconds, finish_reason and the time left; the last line is DONE with the total time.
Output is flushed per line, so `... | tee trial_run_log.txt` shows progress live.
Watch "finish_reason": many "length" = answers cut off -> raise --max-tokens.

STEP 2 — SCORE (separate environment because it needs PyTorch)
--------------------------------------------------------------
One-time setup (already done on Joy's Mac):
    uv venv --python 3.12 .venv-eval
    uv pip install --python .venv-eval/bin/python sentence-transformers rouge-score
Run:
    .venv-eval/bin/python trial_eval/score_trial.py                     # every answered question
    .venv-eval/bin/python trial_eval/score_trial.py --start 1 --end 30  # only #1..#30
Reference-answer embeddings are cached in trial_eval/.embed_cache_BAAI__bge-m3.npz
(already built), so each run only embeds the 100 new answers: ~1-2 min.
On a new machine the first run downloads bge-m3 (~2.2 GB) and builds the cache (~10 min).

Output:
    trial_eval/answers_nanbeige_b0_scores.csv     one row per question
    trial_eval/answers_nanbeige_b0_summary.json   averages by difficulty / task type / GNU Radio
    trial_eval/answers_nanbeige_b0_report.html    open in a browser: summary metrics, breakdown
                                                  tables, and every question with the model's
                                                  answer, the reference answer and its scores
With --start/--end the files get a suffix, e.g. answers_nanbeige_b0_q1-30_report.html
    open trial_eval/answers_nanbeige_b0_q1-30_report.html

HOW TO READ THE SCORES
----------------------
sem_best    cosine similarity of meaning (0-1) to the closest passing teacher answer
sem_norm    MAIN NUMBER. sem_best rescaled per question:
              0 ~ as similar as an answer to a DIFFERENT question
              1 ~ as similar as another passing teacher's answer
            >1 is possible (closer than teachers are to each other)
rougeL      word overlap with the primary reference (low is normal for free-form text)
num_recall  calculation questions only: share of the reference's final numbers found in
            the answer (2% tolerance). Rough: depends on how numbers are written.
len_ratio   answer length / reference length

LIMITS: similarity is a proxy, not a correctness grade. A fluent answer with one wrong
number can still score high. Use it to COMPARE models on the same 100 questions
(baseline vs. fine-tuned), and read the lowest-scoring answers by hand.
Next upgrade: an LLM judge with a rubric, checked against ~30 hand grades.

COMPARING ANOTHER MODEL LATER
-----------------------------
    python3 trial_eval/run_trial.py --model <name> --base-url <server>/v1 --out trial_eval/answers_<run_id>.jsonl
    .venv-eval/bin/python trial_eval/score_trial.py --answers trial_eval/answers_<run_id>.jsonl
In Colab, set HF_REPO / REVISION in the notebook's settings form instead.
