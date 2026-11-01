# GRC model evaluation workspace: plan

> **Superseded by [COLAB_PLAN.md](../colab/COLAB_PLAN.md) (2026-10-31).** The team moved from a shared Modal endpoint to one Colab notebook per teammate. Kept for reference (formerly `EVAL_WORKSPACE_PLAN.md`).

> **Status: draft v2 for review (2026-10-31).** Nothing in this plan has been implemented yet.
> v2 adds the two-repo split, the server image and OS, the API-key change, data storage, the local cookbook,
> Modal shutdown commands and two comparison checks.

## Context

Team JJJP is fine-tuning small models (Nanbeige4.2-3B, and North-Mini-Code later) so GRC-Agent can work
**offline on a laptop**. Each model version has to be evaluated on two things:
- **Knowledge:** Q&A pairs scored against reference answers.
- **Agentic work:** building flowgraphs by calling the right GRC-Agent tools with the right logic.

The team should be able to trigger these runs headlessly (no IDE) and share the results.

Today's setup can't do the agentic part. `nanbeige_api_server.py` uses Transformers and FastAPI, rejects tool
calls, and runs at about 9 tok/s on an L4. Qoherent's GPU arrives within about 2 weeks, so whatever we build
now has to move there without a rewrite.

## Facts this plan rests on (checked 2026-10-31)

**llama.cpp support**
- Nanbeige4.2 is supported in upstream llama.cpp (PR #25994, merged 2026-07-27).
- Its tool-call bug was fixed in build **b10227** by PR #26252, and confirmed working every time on **b10335**.
  → Pin a llama.cpp release **after b10335**. The newest release is v0.8.0, from 2026-10-28.
- North-Mini-Code is a **30B mixture-of-experts model with about 3B active parameters** (`cohere2_moe`).
  llama.cpp supports it, and GGUFs exist, including Unsloth's. Q4_K_M is about 18 GB.

**The agent harness and model API**
- llama.cpp's official CUDA server image is built on **Ubuntu 24.04 with CUDA 12.8.1**.
- `grc-agent-headless` (the GRC-Bench Tier-A runner) needs **Ubuntu 26.04, GNU Radio 3.10.12, bubblewrap and
  GRC-Agent at commit 6c98128**. It calls any OpenAI-compatible URL with a **Bearer key** (`relay.py:210`). It
  can't send Modal's `Modal-Key`/`Modal-Secret` headers, and neither can the Mac adapter.
- llama-server's `--api-key` checks the Bearer key on every endpoint except `/health`, which stays public.

**Hardware, cost and credits**
- Joy's Mac is an M3 MacBook Air with 16 GB RAM. llama.cpp uses its GPU through Metal.
- Modal: L4 about $0.80/h for the GPU (about $1.2/h with CPU and memory); $30/month free on Starter. The
  academic credit program needs confirming at modal.com/pricing.
- Course credits: Azure $100 (no card); a GCP education coupon (redeem by 1/29/2027).
- Cohere's trial key is free but rate-limited, and not meant for production. Cohere also has an
  OpenAI-compatible endpoint at `https://api.cohere.ai/compatibility/v1`; whether tool calls work through it
  is unverified.

## Overall design

The system has three layers. They connect only through a **Hugging Face model repo** and an
**OpenAI-compatible URL**, so moving hosts only changes a URL and where a job runs.

| Layer | Now (next 2 weeks) | After Qoherent's GPU arrives |
|---|---|---|
| **Model store** | Private HF Hub team repo: one GGUF repo per model, one revision per fine-tuned version (Q8_0 and Q4_K_M) | Same |
| **Model server** | `llama-server` on **Modal**: scales to zero, Bearer key | Same command on Qoherent's GPU; Modal stays as a fallback |
| **Eval runner** | **GitHub Actions** in a private team repo, using an Ubuntu 26.04 image on a hosted runner (CPU only) | A **self-hosted runner on Qoherent's box**: outbound only, so no public port and no VPN for teammates |
| **Results** | Artifacts and job summary for each run, plus one results table | Same |

**How each option fits.**
- **Modal:** the shared server for now.
- **Qoherent's GPU:** where we move later.
- **Cohere API:** a North baseline only.
- **Your Mac:** the "real user laptop" check. It isn't shared infrastructure: it would have to stay awake and
  be exposed to the internet, and the adapter runs flowgraphs without a sandbox.
- **Colab:** for training. Its output goes in as a new HF revision.
- **Azure and GCP credits:** a backup, for example a small Linux VM as a self-hosted runner.

## Repo split

The two repos agree on three values only: **base URL, API key and model name**. The eval repo never needs to
know whether a model runs on Modal, Qoherent's box, a laptop or Cohere.

1. **Hosting repo** (focus now): rename `nanbeige-modal` (e.g. `grc-model-serving`). GitHub redirects the old
   URL, and the history is kept.
2. **Eval repo** (later), e.g. `grc-eval`: a new private team repo.

| What's in `nanbeige-modal` now | Goes to |
|---|---|
| `nanbeige_modal.py`, `nanbeige_generate_test.py`, `nanbeige_api_server.py` (Transformers) | Hosting repo: replaced by llama.cpp versions. Keep the old ones as `legacy/` or delete them, after telling teammates. |
| `ask.py` | Hosting repo, switched to the Bearer key |
| README steps 0–8 and `modal_commands.txt` | Hosting repo, rewritten for llama.cpp |
| `trial_eval/` | Eval repo |
| `grc-agent-adapter/` | Eval repo, as the Mac agent runner |
| `northMiniCode/run_raw.py` | Eval repo, as the Cohere baseline runner. Its Cohere setup notes go into the hosting cookbook. |
| `dataset/` (795 MB, git-ignored) | Neither: the training data goes to a private HF dataset repo |

## Part 1 (focus now): hosting repo

### Server image and OS

- A container image carries the OS's files: system libraries, tools and package manager. The **kernel and the
  NVIDIA driver come from Modal's machine**.
- Serving image: llama.cpp's official `server-cuda` image, pinned to one release after b10335.
  - OS: **Ubuntu 24.04 + CUDA 12.8.1**.
  - Loaded with `modal.Image.from_registry(..., add_python="3.11")`, since Modal's runtime needs Python.
- The eval runner is a different image, **Ubuntu 26.04**, because GNU Radio 3.10.12 is pinned there. The two
  only talk over HTTP, so the OS mismatch doesn't matter.

### Model files and versions

- Model files aren't built into the image. At startup, the container downloads the requested GGUF
  (`MODEL_REPO`, `REVISION`, `GGUF_FILE`) from HF into a Modal Volume, which caches it.
- A new model version is a new HF revision plus one deploy label. The image doesn't change.
- `convert/to_gguf.sh` turns a merged HF checkpoint into a GGUF and pushes the result:
  1. `convert_hf_to_gguf.py` → BF16 GGUF.
  2. `llama-quantize` → Q8_0 and Q4_K_M.
  3. Push to the HF repo as a new revision.
- The base models go through the same script, so base and fine-tuned files are produced identically.

### Authentication: llama-server's key instead of Modal proxy auth

- **Change:**
  - Turn off Modal's proxy auth (`requires_proxy_auth`).
  - Start llama-server with `--api-key`, read from a **Modal Secret** and never from code.
  - Clients send `Authorization: Bearer <key>`, as with OpenAI. GRC-Agent, the headless harness, the Mac
    adapter and the openai SDK all work with just a URL and a key.
- **Trade-off:** Modal's proxy auth rejects strangers before a GPU starts. With llama-server's key, a request
  has to reach the container to be rejected, so someone who finds the URL could wake the GPU.
- **Guards:**
  - `max_containers=1`;
  - a short idle window;
  - a spending limit in Modal's settings;
  - never posting the URL anywhere public.

  If that's ever not enough, a small CPU-only gatekeeper in front of the server closes the gap.

### GPU and settings per model (in `serve/serve.yaml`)

| Model | Quant | Size | Modal GPU | Notes |
|---|---|---|---|---|
| Nanbeige4.2-3B | Q8_0 (plus Q4_K_M for laptops) | about 4.5 GB | L4 | |
| North-Mini-Code | Q4_K_M (Q8_0 if the comparison check calls for it) | about 18 GB | L40S | An L4 only works with short context |

- Start from GRC-Bench's flags:
  `-c 65536 -ctk q8_0 -ctv q8_0 -fa on --jinja --reasoning-format deepseek --no-context-shift --sse-ping-interval 30 -np 1 -ngl 999`.
- Check the reasoning format for each model.
- Every host uses the same file, so results stay comparable.

### Local cookbook

**Mac** (tested on Joy's M3 with 16 GB):
1. `brew install llama.cpp`, and check the version is after the fix.
2. Download the Nanbeige GGUF from the team HF repo.
3. Start `llama-server` with the same flags as Modal. Metal is used automatically.
4. Test it with `curl`, including a tool call.
5. In the GRC-Agent desktop app's Settings, choose the OpenAI-compatible provider with
   `http://127.0.0.1:8080/v1`. This is the offline end-user setup.

Nanbeige runs at Q4_K_M or Q8_0. North doesn't fit in 16 GB.

**Windows and Linux:**
- llama.cpp publishes prebuilt programs on its GitHub releases page and through package managers (winget,
  Homebrew). Linux with NVIDIA can run the same Docker image as Modal.
- Pick the build for the hardware: NVIDIA → CUDA; AMD → Vulkan or ROCm; Intel → Vulkan or SYCL; no GPU → CPU
  (Nanbeige works, just more slowly).
- North on a laptop GPU: keep the experts in system RAM with `--n-cpu-moe` (confirm the option on the pinned
  release). That should let it run on a 12–16 GB GPU with 32 GB of RAM.
- On Windows, the model server runs natively. Only the eval harness needs WSL2 with Ubuntu 26.04.

**North through Cohere** (nothing to host, just an API key):
- Cohere's native API: `run_raw.py`, and GRC-Agent's `cohere` provider (`grc-agent-adapter --provider cohere`).
- Cohere's OpenAI-compatible endpoint, for tools that only speak OpenAI, such as the headless harness. Test
  tool calls there before relying on it.
- Keys go in a git-ignored `.env` with `chmod 600`.

### Modal: check, stop, clean up

```bash
modal app list                         # what's deployed; Tasks > 0 means containers are running
modal container list                   # what's running = what's costing money right now
modal container stop <container-id>    # kill one running container; the app stays deployed, URL still works
modal app stop <app-name>              # take a whole app down (kills its containers); URL stops working
modal deploy <file>.py                 # bring it back: same URL, cached weights reused
modal volume delete <volume-name>      # delete stored weights, only when completely done
```

- An idle app costs nothing. `modal app stop` is for cleanup or a leaked key.
- The repo gets a tested `stop_all.sh` built on `modal app list --json`.

**Account state on 2026-10-31:**
- No containers running.
- `nanbeige42-chat-api` deployed and idle (0 tasks).
- Volume `nanbeige42-model-cache` holds the weights.
- Volume `nanbeige42-hf-cache` is **empty and unused**, a candidate for deletion. It hasn't been deleted.

### Work order (hosting repo)

| Step | What | Done when |
|---|---|---|
| 1 | Mac cookbook: Nanbeige with llama.cpp | A `curl` with `tools` returns `tool_calls`; `grc-agent-adapter --base-url http://127.0.0.1:8080/v1` finishes `completed` with at least one tool call; `llama-bench` tok/s recorded |
| 2 | Convert the base Nanbeige to GGUF and push it to the HF team repo (revision `base`) | The Mac and Modal both load the file from HF, and the hashes match |
| 3 | Nanbeige on Modal with llama.cpp and the Bearer key | A Bearer `curl` with `tools` returns `tool_calls`; a request without the key gets 401; `modal container list` is empty after the idle window |
| 4 | North on Modal (L40S) | Same checks as step 3 |
| 5 | Windows/Linux sections and Cohere notes | A teammate on Windows or Linux follows them unaided |
| 6 | Retire the Transformers server | Teammates told; `modal app stop nanbeige42-chat-api` |

## Comparison checks (one-time, at the base-model stage)

Both are worth doing. **Comparison B matters more.** They're evaluations, so the scripts live in the eval repo.
Comparison A also serves as the final "it works" check for the hosting part.

| | **A. Local Nanbeige vs. Modal Nanbeige** | **B. Modal North vs. Cohere North** |
|---|---|---|
| What differs | Only the compute backend: Metal on the M3 GPU vs. CUDA on the L4. The OS itself doesn't matter. | Precision (Q4 vs. what Cohere serves), prompt/tool-call/reasoning handling, and possible silent updates on Cohere's side |
| Expected result | Same quality. Text drifts apart after a while from tiny floating-point differences. If scores differ, a setting differs. | Real differences are possible |
| Why it matters | It proves a Modal score is what an offline Mac user gets | If our base North scores lower than Cohere's, later fine-tune results mix serving effects with training effects. It also picks Q4 vs. Q8. |
| Cost (estimate) | Mac free; Modal well under $1 | Cohere free (trial; **10 of 100** Q&A answered so far); Modal L40S a few dollars |
| Rerun when | The llama.cpp version or server flags change | The North quant level changes, or Cohere updates the model |

How to run them so the result means something:
1. **Same inputs:** the 100 trial Q&A questions, plus 10–20 agent tasks.
2. **Same settings:** system prompt, temperature 0, max tokens, thinking setting. For A, also the same GGUF
   hash and llama.cpp release.
3. **Measure noise:** run one side twice. A gap between setups about the size of the gap between repeat runs
   means no real difference.
4. **Compare question by question:** the mean `sem_norm` difference with a 95% bootstrap CI. Set an
   equivalence margin in advance (proposed **±0.05**). Inside the margin means equivalent. Not finding a
   significant difference doesn't prove two setups are the same.
5. **Agent tasks:** pass/fail counts as a sanity check only, because 10–20 tasks only show large gaps.

## Data storage

| What | Where | Why |
|---|---|---|
| Q&A questions with reference answers, agent task definitions | Git, in the eval repo (`trial_eval_100.jsonl` is 2 MB) | Versioned with the grading code; GitHub Actions gets them on checkout |
| GRC-Bench frozen test items | The GRC-Bench repo, pulled at run time | GRC-Bench's rule: test items aren't copied or sent to a model before an official run |
| Training data (795 MB) | A private HF dataset repo (or DVC plus Google Drive, as in `pipeline/README.md`) | Too big for git |
| Model files (GGUF) | A private HF model repo, one revision per version | One token works from Modal, Actions, Qoherent's box and laptops |
| Eval results | A per-run artifact (temporary) plus a private HF dataset repo `grc-eval-results` (permanent table) | Visible to the team, and unaffected by host moves |

Keep excluding `trial_eval_ids.txt` and the frozen test IDs from all training data.

## Part 2 (later): eval repo

```
grc-eval/
├── eval/qa/                # Tier K: run_trial.py (endpoint-generic) + score_trial.py
├── eval/agent/             # Tier A: grc-agent-headless now, GRC-Bench's runner once Joel confirms the interface
├── eval/compare/           # comparisons A and B: paired bootstrap, equivalence margin
├── docker/eval.Dockerfile  # Ubuntu 26.04 + gnuradio 3.10.12 + bubblewrap + GRC-Agent@6c98128 + headless
└── .github/workflows/
    ├── build-image.yml     # builds the eval image and pushes it to private GHCR (needs Qoherent's OK to clone GRC-Agent in CI)
    └── evaluate.yml        # inputs: model, revision, quant, tiers, limit, endpoint (modal|qoherent|cohere|url)
```

**How teammates trigger a run:** Actions → evaluate → Run workflow, or
`gh workflow run evaluate.yml -f model=nanbeige42 -f revision=sft-v3 -f tiers=K,A`.

**What the workflow does:**
1. Wakes the endpoint, or deploys it if missing.
2. Runs Tier K and Tier A (`--preset benchmark`).
3. Scores the results.
4. Writes the job summary and uploads the run folder.
5. Appends a row to the results table.
6. Stops the endpoint.

**Qoherent migration:**
1. Install llama.cpp and a self-hosted runner labelled `qoherent-gpu` on Qoherent's box.
2. `endpoint=qoherent` then runs the job on that box against `127.0.0.1`.
3. No eval code changes.

## Reuse, not rewrite

| From | What it provides |
|---|---|
| `trial_eval/` | `run_trial.py`, `score_trial.py`, `trial_eval_100.jsonl` |
| `grc-agent-headless/src/grcbench/tier_a/` | `adhoc.run`, `models.hosted()` |
| `grc-agent-adapter/` | `./agent batch`, `--base-url` |
| `northMiniCode/` | `run_raw.py` |

`MLOps Qoherent/nanbeige_modal_from_sam/modal_app.py` isn't needed, because llama-server serves the OpenAI API
itself.

## Open decisions (defaults in bold)

- **Hosting repo name:** **rename `nanbeige-modal`** (e.g. `grc-model-serving`), or start a new repo.
- **GRC-Bench:** **start with grc-agent-headless and the trial set; plug in GRC-Bench's runner once Joel
  confirms the interface.**
- **Results store:** **HF dataset repo**, or W&B for dashboards.
- **Actions minutes:** a private repo gets about 2,000 a month. **Fall back to a self-hosted runner** if
  Tier-A runs use them up.
- **Modal:** **apply for academic credits**, and move to a team workspace so the token isn't Joy's personal one.
- **Equivalence margin** for the comparisons: **±0.05 `sem_norm`**, to be agreed before the runs.
- **Empty volume `nanbeige42-hf-cache`:** **delete it** (it's unused).
- **North licence:** check the non-commercial caveat on its model card before using it for Qoherent deliverables.

## Sources

- llama.cpp:
  - [PR #25994](https://github.com/ggml-org/llama.cpp/pull/25994) (Nanbeige4.2 support)
  - [PR #26324](https://github.com/ggml-org/llama.cpp/pull/26324) (tool-call fix via #26252)
  - [CUDA Dockerfile](https://github.com/ggml-org/llama.cpp/blob/master/.devops/cuda.Dockerfile)
  - [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- Models:
  - [Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B)
  - [Introducing North Mini Code](https://huggingface.co/blog/CohereLabs/introducing-north-mini-code)
  - [North-Mini-Code GGUF](https://huggingface.co/heimann/North-Mini-Code-1.0-GGUF)
- Pricing and APIs:
  - [Modal pricing](https://modal.com/pricing)
  - [Cohere Compatibility API](https://docs.cohere.com/docs/compatibility-api)
- Local files:
  - `grc-agent-headless/TUTORIAL.md`
  - `grc-agent-adapter/README.md`
  - `MLOps Qoherent/pipeline/*.md`
  - `CSC2701 Weeks 4-5 Alice.pdf` (pp. 4–14)
