# GRC model evaluation

*(Formerly `nanbeige-modal`.)*

Evaluates the models the team fine-tunes for GRC-Agent, starting with
[Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B), on 100 graded RF / DSP / GNU Radio
trial questions.

- **Current way:** each teammate runs one Colab notebook ([`colab/`](colab/README.md)) with llama.cpp on Colab's GPU.
- **Older way, being retired:** the model hosted on a Modal GPU with an OpenAI-style API
  (Modal + PyTorch + Transformers 4.45.1 + FastAPI; code in [`modal/`](modal/)). Documented below.

Plan: [`COLAB_PLAN.md`](COLAB_PLAN.md).

---

## For teammates: run the evaluation yourself in Colab (new, recommended)

The team is moving from the shared Modal endpoint to **one Colab notebook that each teammate runs on their own**:
it downloads the model from Hugging Face, runs it with llama.cpp on Colab's free T4 GPU, asks the trial questions,
scores them and shows the report. No server, no shared token.

→ **[`colab/README.md`](colab/README.md)**: one-time setup, how to run `colab/eval.ipynb`, what each step does,
where the results go. The plan behind it: [`COLAB_PLAN.md`](COLAB_PLAN.md).

The Modal endpoint below still works until it is retired (`COLAB_PLAN.md`, step 7).

---

## For teammates: use the hosted model (Modal, being retired)

The model is already running on Joy's Modal account, so you don't need a Modal account, a GPU or `modal` installed.
You only need Python 3 and three values from Joy: **URL, key, secret**. Joy sends them privately.
Never commit them or post them in group chats.

### 1. Clone and add the credentials

```bash
git clone https://github.com/No1Joycia/grc-model-eval.git
cd grc-model-eval
cp .env.example .env
# edit .env and paste the MODAL_URL, MODAL_KEY (wk-...) and MODAL_SECRET (ws-...) you were given
```

### 2. Ask a question

```bash
python3 ask.py --health                                    # wake the GPU: first call ~1-3 min
python3 ask.py "What is GNU Radio? Answer in one sentence."
python3 ask.py --max-tokens 1024 "Explain how a PLL locks onto a carrier."
```

The answer is printed to the terminal, followed by the token count and time taken.
Expect about 9 tokens/s, so a 500-token answer takes about a minute.
The server handles **one request at a time**: if someone else is running an eval, your question waits in line behind theirs.

You can also use `curl` (see [step 6](#6-ask-one-question)) or any OpenAI-compatible client with base URL `$MODAL_URL/v1`
and model `Nanbeige/Nanbeige4.2-3B`. Send the extra headers `Modal-Key` and `Modal-Secret`.
Supported: text chat only. Not supported: streaming or tool calls.

### 3. Evaluate the model on specific questions from the trial set

The set has 100 graded questions in `trial_eval/trial_eval_100.jsonl`: #1–33 basic, #34–67 intermediate, #68–100 advanced.
To look up a question's number: `sed -n 7p trial_eval/trial_eval_100.jsonl | python3 -c "import sys,json;print(json.load(sys.stdin)['question'])"`

**Get the model's answer** (Python 3 only, no installs):

```bash
set -a; source .env; set +a                                 # MODAL_URL, MODAL_KEY, MODAL_SECRET
python3 trial_eval/run_trial.py --base-url "$MODAL_URL/v1" --model Nanbeige/Nanbeige4.2-3B --start 7 --end 7
python3 trial_eval/run_trial.py --base-url "$MODAL_URL/v1" --model Nanbeige/Nanbeige4.2-3B --start 1 --end 30
```

The first line of every run is a time estimate. Add `--estimate-only` to see it without sending anything:

```
ESTIMATE: 30 questions -> ~53 min
```

Next, an `asking #N ...` line appears before each request, then a progress line with the time left after each answer.
The last line gives the actual time: `DONE: 30 questions in 51 min (~102 s/question)`.
GPU time is billed to Joy's Modal account.

**Score it and get the HTML report.** This needs a one-time setup of about 3 GB: PyTorch plus the bge-m3 embedding model.

```bash
# one time (install uv first if needed: https://docs.astral.sh/uv/)
uv venv --python 3.12 .venv-eval
uv pip install --python .venv-eval/bin/python sentence-transformers rouge-score

.venv-eval/bin/python trial_eval/score_trial.py --start 7 --end 7
open trial_eval/answers_nanbeige_b0_q7-7_report.html       # question, answer, reference, scores
```

The first scoring run downloads bge-m3 (about 2.2 GB). Reference embeddings are already cached in the repo,
so after that only the new answers are embedded. See `trial_eval/README.txt` for what each metric means.

Your answers and reports stay on your machine; they're git-ignored.

---

## How it was built (Joy's setup, steps 0–8)

Everything below is for running your **own** copy of the server on your own Modal account.

| File | What it does |
|---|---|
| `modal/nanbeige_modal.py` | Step 1–2: download the weights into a Modal Volume (CPU), then test-load them on an L4 GPU |
| `modal/nanbeige_generate_test.py` | Step 3: one-shot generation smoke test on the GPU (no server) |
| `modal/nanbeige_api_server.py` | Step 4: deployed HTTPS chat API (`/health`, `/v1/models`, `/v1/chat/completions`) |
| `ask.py` | Ask the hosted model one question from the terminal (stdlib only) |
| `trial_eval/` | Step 5: 100-question trial eval: ask the endpoint, score the answers, HTML report |
| `modal_commands.txt` | Short command notebook for day-to-day use |

Modal resources used:

- App `nanbeige42-chat-api` (the server). The earlier test scripts run as temporary apps.
- Volume `nanbeige42-model-cache`, which holds the Hugging Face weights (~8 GB) at `/model-cache/huggingface`.

---

## 0. Set up the local environment (Mac)

```bash
cd ~/Projects/grc-model-eval
python3 -m venv .venv
source .venv/bin/activate
pip install modal          # tested with modal 1.6.2
modal setup                # opens a browser and logs this machine into your Modal account
```

Do this at the start of every new terminal session:

```bash
cd ~/Projects/grc-model-eval
source .venv/bin/activate
```

## 1. Download Nanbeige into Modal (one time)

Downloads the checkpoint on a CPU container into the persistent Volume `nanbeige42-model-cache`.

```bash
modal run modal/nanbeige_modal.py --stage download
```

Success line: `[DOWNLOAD_OK] Model checkpoint cached: /model-cache/huggingface/...`

## 2. Load test on the GPU

Loads the tokenizer and weights onto an L4 without generating any text. It checks that the weights are on CUDA and reports memory use.

```bash
modal run modal/nanbeige_modal.py --stage load
```

Expected output: `[LOAD_SUCCESS] ... Parameter devices: ['cuda:0'] ... CUDA memory allocated: 7.83 GiB`

(`modal run modal/nanbeige_modal.py` with no `--stage` runs both steps.)

## 3. Quick generation test

```bash
modal run modal/nanbeige_generate_test.py
modal run modal/nanbeige_generate_test.py \
    --prompt "What is GNU Radio? Answer in one sentence." \
    --max-new-tokens 96
```

Expected output: `[RESPONSE] GNU Radio is an open-source ...` and `[GENERATE_SUCCESS]`, at about 9 tokens/s on an L4.
The GPU is released when the run finishes.

## 4. Deploy the FastAPI server

```bash
modal deploy modal/nanbeige_api_server.py      # the last lines print your endpoint URL (https://...modal.run)
```

How it hosts the model:

- `modal deploy` registers the app. No GPU runs until a request arrives (`min_containers=0`).
- The first request starts one GPU container (L4, falling back to A10G or L40S; `max_containers=1`). The container loads the cached weights from the Volume, with no download, then serves FastAPI. This cold start takes about 1–3 minutes.
- The container stays warm while requests keep coming and shuts down 60 s after the last one. Billing stops when it shuts down.
- Requests are handled one at a time, at about 9 tokens/s.
- Accepted: text-only chat (system/user/assistant), non-streaming, `max_tokens` ≤ 2048, prompt ≤ 4096 tokens.
- Rejected: tool calls, streaming and `response_format`.
- Redeploying after editing the file keeps the same URL.

## 5. Auth tokens: where to put your ID and secret

Every request must carry a Modal **Proxy Auth Token**. Without it, the request gets a 401.

1. Go to the Modal dashboard, then **Settings → Proxy Auth Tokens → New token**.
2. Copy the template and fill it in:

   ```bash
   cp .env.example .env
   ```

   ```ini
   # .env  (git-ignored, never commit it)
   MODAL_URL=https://<workspace>--nanbeige42-chat-api-serve.<region>.modal.run
   MODAL_KEY=wk-...        # token ID
   MODAL_SECRET=ws-...     # token secret
   ```

3. Load it into your shell for `curl` and for `run_trial.py`'s `--base-url "$MODAL_URL/v1"`. (`trial_eval/run_trial.py` reads the key and secret from `.env` on its own.)

   ```bash
   set -a; source .env; set +a
   ```

`.env` is listed in `.gitignore`. If a token leaks, delete it in the dashboard and create a new one.

## 6. Ask one question

Wake up the server and check its health. The first call after an idle period takes about 1–3 minutes:

```bash
curl -L -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" $MODAL_URL/health
# {"status":"ready","model":"Nanbeige/Nanbeige4.2-3B","gpu":"L4"}
```

Ask a question:

```bash
curl -L -X POST $MODAL_URL/v1/chat/completions \
  -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"model":"Nanbeige/Nanbeige4.2-3B",
       "messages":[{"role":"user","content":"What is GNU Radio? Answer in one sentence."}],
       "max_tokens":128,"temperature":0}'
```

To print only the answer, append `| python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])"`.

Use `curl -L`: Modal redirects requests that take longer than about 150 s.

## 7. Trial evaluation (questions #a to #b)

There are 100 graded questions in `trial_eval/trial_eval_100.jsonl`, ordered by difficulty:
#1–33 are basic, #34–67 intermediate and #68–100 advanced. See `trial_eval/README.txt` for how they were chosen
and how to read the metrics.

```bash
set -a; source .env; set +a                                 # MODAL_URL, MODAL_KEY, MODAL_SECRET
# time estimate only (no requests sent)
python3 trial_eval/run_trial.py --base-url "$MODAL_URL/v1" --model Nanbeige/Nanbeige4.2-3B --start 1 --end 30 --estimate-only

# ask questions #1-#30 (resumable: rerun to continue after Ctrl+C)
caffeinate -i python3 trial_eval/run_trial.py --base-url "$MODAL_URL/v1" --model Nanbeige/Nanbeige4.2-3B --start 1 --end 30
```

The scorer needs a separate environment because it uses PyTorch. Set it up once:

```bash
uv venv --python 3.12 .venv-eval
uv pip install --python .venv-eval/bin/python sentence-transformers rouge-score
```

Then score the same range:

```bash
.venv-eval/bin/python trial_eval/score_trial.py --start 1 --end 30
open trial_eval/answers_nanbeige_b0_q1-30_report.html
```

The scorer writes three files:

- `*_scores.csv`: one row per question
- `*_summary.json`: averages by difficulty, task type and GNU Radio
- `*_report.html`: summary metrics and breakdown tables, plus every question with the model's answer, the reference answer and the scores

Answer and score files are git-ignored. Commit a specific run on purpose if you want to keep it.

## 8. Check, stop and tear down

```bash
modal app list                          # deployed / stopped
modal container list                    # running GPU containers (empty = idle, no billing)
modal app logs nanbeige42-chat-api      # live logs: [STARTUP] [SERVER_READY] [REQUEST_OK]

modal app stop nanbeige42-chat-api      # take the endpoint down completely (URL stops working)
modal deploy modal/nanbeige_api_server.py   # bring it back (same URL, weights still cached)
```

| State | GPU | Billing | How you get there |
|---|---|---|---|
| RUNNING | on, model loaded | yes | any request |
| IDLE | off, URL still works | no | automatic, 60 s after the last request |
| STOPPED | off, URL returns an error | no | `modal app stop nanbeige42-chat-api` |

Only if you want to remove everything, including the cached weights:

```bash
modal volume delete nanbeige42-model-cache    # next use needs step 1 again
```

---

## What has been done so far

- [x] Step 1: weights downloaded into Volume `nanbeige42-model-cache`
- [x] Step 2: GPU load test on L4: 4.17 B params, 7.83 GiB
- [x] Step 3: generation smoke test, about 9 tok/s
- [x] Step 4: API deployed as `nanbeige42-chat-api`
- [x] Step 5: trial-eval set built (100 questions). Next: baseline run and scoring.
- [ ] Tool calling and streaming support for GRC-Agent

Not in this repository: the dataset, proposal, application forms, meeting notes and pipeline docs. They are kept locally only.
