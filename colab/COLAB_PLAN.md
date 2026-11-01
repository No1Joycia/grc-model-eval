# GRC model evaluation in Colab: plan

> **Status: v1 (2026-10-31). Scope: stage 1 only.** Work-order steps 1–2 are implemented and verified on the Mac; step 3
> (the notebook) is implemented and rehearsed on the Mac, waiting for its first Colab run. See *Progress* below.
> It replaces the Modal-based design in `modal/MODAL_PLAN.md` (formerly `EVAL_WORKSPACE_PLAN.md`).

## Context

Team JJJP is fine-tuning small models (Nanbeige4.2-3B now, North-Mini-Code later) so GRC-Agent can run
**offline on a laptop**. Every model version needs evaluating, and every teammate should be able to run an
evaluation themselves.

Until now the model was hosted on **Modal**: one shared HTTPS endpoint, paid from Joy's account, that teammates
called with a token. That isn't needed. Nobody needs a model that stays up for others to call. Each
evaluation is a **batch job**: load the model, answer the questions, score them, done.

**New design:**
- **One Jupyter notebook.** Each teammate runs it in **their own Google Colab**, on their own Colab credits.
- **The notebook does everything:**
  1. downloads the model weights from Hugging Face;
  2. runs the model with **llama.cpp** on Colab's GPU, inside the notebook only;
  3. answers the trial questions;
  4. scores them;
  5. produces the HTML report.
- **Modal is retired** once the notebook works.

## Facts this plan rests on (checked 2026-10-31)

**Colab**
- **What Colab bans** ([Colab FAQ](https://research.google.com/colaboratory/faq.html)):
  - on every tier, paid included: "web services" and "connecting to remote proxies";
  - on the free tier only: "bypassing the notebook UI".

  A model running on `127.0.0.1` *inside* the notebook is ordinary use. A shared Colab server reached through a
  tunnel would not be.
- **Sessions:** at most 12 h; GPUs aren't guaranteed on any tier. Free-tier daily limits aren't published.
- **What a free runtime is:** a temporary Google machine running **Ubuntu Linux (x86-64)**.

  | Part | Free tier |
  |---|---|
  | GPU | NVIDIA **T4** (16 GB) |
  | CPU | 2 virtual cores |
  | Memory | about 12.7 GB |
  | Disk | about 100 GB, wiped at the end of the session |

  **A T4 is a GPU, not a TPU.** TPUs are Google's own chips; llama.cpp can't use them, so this plan doesn't.
- **Colab Pro:** about **US$9.99/month for 100 compute units** (Pay As You Go: about $9.99 per 100 units). Local prices differ. Pro also adds L4 (24 GB), A100 (40 GB) and High-RAM machines.
- **Unit burn rates.** Measured rates (unofficial; Google doesn't publish them):
  - T4: about 1.2–1.8 units/h;
  - L4: about 1.7 units/h;
  - A100 40 GB: about 5.4 units/h.

  One unit is about $0.10. Units are spent the whole time a runtime is connected, including setup and idle time.
- **No free Colab Pro for us.** The free Colab Pro for students was **US-only and is closed** to new sign-ups.
- **The GCP education coupon can't pay for Colab Pro.** Consumer Colab is billed through Google's consumer payments. The coupon *can* pay for Colab Enterprise or Compute Engine VMs (relevant to stage 3).

**llama.cpp**
- Pinned build **b12020** (`serve/models.json`). The same build runs on every host, the Mac included.
- Its GitHub release includes `llama-b12020-bin-ubuntu-cuda-12.8-x64.tar.gz` (174 MB) and `cudart-llama-b12020-bin-ubuntu-cuda-12.8-x64.tar.gz` (566 MB). That's a ready-made NVIDIA build, so there's nothing to compile.
- `serve/install_llamacpp.sh` already installs it with `VARIANT=cuda`.

**What already exists in this repo and gets reused**
- **`serve/`:**
  - `pinned.sh` (`require_llama_build`);
  - `install_llamacpp.sh`;
  - `run_local.sh` (starts llama-server with the shared settings, on `127.0.0.1` by default);
  - `smoke_test.py` (health, models, chat, tool call; takes `--base-url` and `--model`);
  - `models.json` (pinned build, server arguments, the Nanbeige source revision and file names).
- **`convert/`:** `setup.sh` and `to_gguf.sh` (HF checkpoint → BF16 → Q8_0 / Q4_K_M, plus `SHA256SUMS`).
- **`trial_eval/`:**
  - `trial_eval_100.jsonl`;
  - `run_trial.py` (resumable, `--start/--end`, ESTIMATE / DONE lines);
  - `score_trial.py` (`--start/--end`, HTML report).
- **`docs/local-llamacpp.md`:** Mac setup tested on 2026-10-31. Q8_0 SHA-256 `a1317a0a…87e1`.

## Decisions

| Decision | Choice |
|---|---|
| Where the model runs | In each teammate's own Colab session; no shared endpoint |
| Inference engine | **llama.cpp b12020** (CUDA build), `llama-server` on `127.0.0.1` inside the notebook |
| Why not Transformers in the notebook | Offline users will run llama.cpp, so the score measures what they get. The free T4 has no bf16 support. A fixed build, file and settings make teammates' results comparable. |
| Model files | Converted in Colab from the pinned HF checkpoint, hash-checked, cached in the user's Google Drive |
| Default quantization | Q8_0 (4.1 GB) |
| Repo access | Teammates added as collaborators; a read-only fine-grained `GH_TOKEN` in Colab Secrets |
| Thinking | **Off** by default (matches the first baseline); `--thinking on` per run; recorded in every answer |
| Temperature | 0, sent explicitly on every request |
| Results | One run folder per run in the user's Drive; shared by copying it to a team Drive folder |
| Modal | Retired after the notebook passes its checks |

## Stage 1: Nanbeige Q&A in Colab

### What a teammate does

1. Accept the GitHub collaborator invite. Create a fine-grained token with read-only "Contents" access to this repo.
2. In Colab, add the token under **Secrets** as `GH_TOKEN`.
3. Open `colab/eval.ipynb`, choose **Runtime → Change runtime type → T4 GPU**, fill in the form, then **Run all**.
4. The report appears in the notebook, downloads automatically, and stays in Drive under `grc-eval/runs/<run>/`.

### Changes

1. **`colab/eval.ipynb` (new): the one notebook.** It's thin: all logic stays in the repo scripts, so it can be tested on the Mac.

   A form cell sets:
   - `MODEL` (default `nanbeige42`), `HF_REPO` and `REVISION`. The defaults come from `serve/models.json`; testing a fine-tuned **merged** checkpoint on HF only means changing the repo or revision.
   - `QUANT=q8_0`, `START=1`, `END=100`, `THINKING=off`, `MAX_TOKENS=1536`.

   Cells:
   1. **GPU check:** `nvidia-smi` shows the GPU name and driver. Stop with a clear message if there's no GPU.
   2. **Mount Google Drive.** Read `GH_TOKEN` with `google.colab.userdata`, clone the repo, then remove the token from the git remote.
   3. **Install llama.cpp:** `VARIANT=cuda LLAMA_CPP_DIR=/content/llama.cpp/b12020 bash serve/install_llamacpp.sh`. Then `llama-server --version` must show build 12020 and the CUDA device.
   4. **Get the model:** `pip install uv`, then `bash serve/get_gguf.sh` with `MODEL`, `QUANT`, `<Drive>/grc-eval/models/<model>`, `HF_REPO` and `REVISION`. The first session converts (about 5–10 min); later sessions reuse the hash-checked copy in Drive.
   5. **Start the model:**
      - run `GGUF=… bash serve/run_local.sh <model> <quant>` in the background, logging to a file;
      - wait until `/health` returns 200;
      - run `python3 serve/smoke_test.py --model <model>`.
   6. **Answer the questions:**
      ```
      python3 trial_eval/run_trial.py --base-url http://127.0.0.1:8080/v1 --model <model> \
          --thinking off --start … --end … --out <Drive>/grc-eval/runs/<run>/answers.jsonl
      ```
      The run folder is named `<model>-<revision7>-<quant>-think_<on|off>-b12020`. Saving after each question means a disconnected session resumes.
   7. **Score:**
      - stop the server to free GPU memory;
      - `pip install rouge-score` (and sentence-transformers if missing);
      - run `trial_eval/score_trial.py --answers … --start … --end …`;
      - show the report inline, download it, and write `run_info.json`.

   Each run folder holds `answers.jsonl`, `*_scores.csv`, `*_summary.json`, `*_report.html` and `run_info.json`.
   `run_info.json` records the GPU and driver, the llama.cpp build, the model file's SHA-256, the HF repo and revision, all settings, and the date.

2. **`trial_eval/run_trial.py`: changed to talk to a local llama-server.**

   Kept:
   - the resume loop;
   - the `--start/--end` range;
   - the "asking #N" lines;
   - the ESTIMATE and DONE lines;
   - `estimate_seconds` / `fmt_duration`.

   New options:
   - `--base-url` (default `http://127.0.0.1:8080/v1`);
   - `--model`;
   - `--api-key-env` (optional Bearer key);
   - `--thinking off|on`, which sends `chat_template_kwargs.enable_thinking` and an explicit `temperature`.

   Each answer row records the settings, plus `reasoning_chars` from `message.reasoning_content`. The script refuses to append to a file whose rows used different settings.

   Removed:
   - the `Modal-Key` / `Modal-Secret` headers;
   - loading `MODAL_*` from `.env`;
   - the Modal dollar estimate. The ESTIMATE line shows time only.

3. **`serve/models.json`:** add a `sha256` map for `nanbeige42`. For BF16, use the Mac's hash from `docs/local-llamacpp.md`. Q8_0 and Q4_K_M get the hash of the file converted in Colab; see the risk on hashes below.

4. **`serve/get_gguf.sh MODEL QUANT DIR [HF_REPO REVISION]` (new).** It's generic, so it works on the Mac, in Colab and on Qoherent's box.
   - If `DIR/<file>` exists and its hash matches `models.json`, use it.
   - Otherwise:
     1. run `convert/setup.sh`;
     2. run `hf download` at the pinned revision into scratch space;
     3. run `convert/to_gguf.sh` with `QUANTS=<quant>`;
     4. copy the file to `DIR`.
   - **The base model** must match the pinned hash, and a mismatch fails the run.
   - **A fine-tuned checkpoint** has no known hash yet; its hash goes into `run_info.json`.
   - Uses `sha256sum` or `shasum`, whichever exists.

5. **Docs.**
   - **`README.md`:** done 2026-10-31: the top-level README is a short guide to the repo pointing to `colab/README.md`; all Modal instructions moved to `modal/README.md`.
   - **`trial_eval/README.txt`:** the new `run_trial.py` options.
   - **`COLAB_PLAN.md`:** this file. Mark steps done as they land.

6. **Retire Modal.** This happens only after the checks below pass, and Joy confirms each step that touches the account.
   1. Tell teammates. Then `modal app stop nanbeige42-chat-api` and delete the proxy tokens given to teammates.
   2. Delete the Volumes `nanbeige42-model-cache` (about 8 GB of weights) and `nanbeige42-hf-cache` (empty). This can't be undone, but the weights can be downloaded from HF again.
   3. Done 2026-10-31: the Modal scripts, `ask.py`, `modal_commands.txt`, `.env` / `.env.example` and the old plan are in `modal/`; the Modal logs are deleted. The git history keeps everything.

### Work order

| Step | What | Done when |
|---|---|---|
| 1 | `run_trial.py` changes | On the Mac, against `bash serve/run_local.sh nanbeige42`: questions #1–2 answered with thinking off; rows carry the settings; `score_trial.py` builds the report; a `--thinking on` rerun into the same file is refused |
| 2 | `get_gguf.sh` and the hashes in `models.json` | On the Mac it takes the fast path on `~/models/nanbeige42` with no conversion, and a forced conversion into a scratch folder reproduces the BF16 hash |
| 3 | `colab/eval.ipynb` | Code cells pass a Python syntax check; Joy runs questions #1–5 on a free T4 and every cell passes (build 12020 with the CUDA device, smoke test including the tool call, report inline) |
| 4 | Resume and cache | After "Disconnect and delete runtime", Run all reuses the Drive model file (no conversion) and continues the answers |
| 5 | Second teammate | Their `run_info.json` shows the same build and model hash as Joy's, and scores for #1–5 are close |
| 6 | Full baseline | All 100 questions with Nanbeige Q8_0, thinking off: the official **B0 (llama.cpp)** baseline |
| 7 | Docs, then retire Modal | README rewritten; Modal steps 6.1–6.3 done with Joy's confirmation |

### Progress

- **2026-10-31, steps 1–2 done (Mac).**
  - `run_trial.py`: questions #1–2 answered by local llama-server with thinking off (`reasoning_chars` 0); rows carry the settings; a `--thinking on` rerun into the same file is refused; a rerun resumes with nothing to do; `score_trial.py` built the report.
  - `get_gguf.sh`: the cached-file path matched the Darwin-arm64 Q8_0 hash. A fresh conversion reproduced **both** the BF16 and the Q8_0 hash.
  - **Finding:** `convert_hf_to_gguf.py` writes the checkpoint folder's name into the file (`general.name`), so the download folder must be named after the HF repo. `get_gguf.sh` does this.
  - The Modal path of `run_trial.py` is kept (`--base-url "$MODAL_URL/v1" --model Nanbeige/Nanbeige4.2-3B`) until step 7 retires Modal.
- **2026-10-31, step 3 written.**
  - `colab/eval.ipynb` (settings form plus 7 logged steps), `colab/helpers.py` and `colab/README.md`.
  - All code cells parse, and the notebook passes the Jupyter schema check.
  - Steps 5–7 were rehearsed on the Mac with `colab/helpers.py`: server start, smoke test (tool call included), answers, scoring.
  - Waiting for the first Colab run; that needs `colab/`, `serve/` and `convert/` pushed to `main`.
- **Open question for the B0 run:** with llama.cpp, question #1 needed more than 1,536 tokens and was cut off (`length`). The Transformers server answered it in 1,416. Decide before step 6 whether `MAX_TOKENS` stays at 1536 or goes up (for example to 2048) for every run.

### Cost of stage 1

| Run | GPU | Time (estimate, to be measured) | Cost |
|---|---|---|---|
| Nanbeige Q8_0, 100 questions, thinking off | Free T4 | about 1 h, plus about 10 min first-time setup | **$0** |
| Same, on Colab Pro | T4 or L4 | the same | about 1.5–2 units, about $0.20 |

### Risks

- **Colab's NVIDIA driver may be too old for the CUDA 12.8 build.** Cell 3 checks this first.
  - Fallback: compile b12020 with Colab's own `nvcc` (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;89"`) and cache it in Drive.
  - Clone with `--filter=blob:none` (not `--depth 1`), so the build reports 12020 and `require_llama_build` accepts it.
- **Model-file hashes may differ between the Mac (ARM) and Colab (x86).** BF16 conversion is plain Python and should match everywhere. Quantizing to Q8_0 / Q4_K_M uses floating-point code that can round differently on different processors.
  - Rule: the **Colab-converted** file is the team reference for Colab runs, and every teammate's Colab conversion must match it.
  - Any Mac/Colab difference is recorded, not treated as an error.
- **No GPU available** (free tier). The notebook stops cleanly; resume and the Drive cache keep reruns cheap.
- **Fine-tuned checkpoints must be merged** (full weights, not LoRA adapters). The notebook checks for `config.json` and safetensors files.
- **Space.**
  - Disk: about 21 GB per conversion (8.4 GB checkpoint, 7.8 GB BF16, 4.1 GB Q8_0); fine on Colab.
  - Drive: only the 4.1 GB Q8_0 (the free Drive has 15 GB).

## Later stages (not in scope now)

**Stage 2: North-Mini-Code Q&A.**
- Use the pinned `unsloth/North-Mini-Code-1.0-GGUF` (Q4_K_M, 19.2 GB). The card metadata says Apache-2.0, and the model isn't gated. Converting the 30.5B checkpoint in Colab isn't practical.
- It needs more GPU memory than a free T4. Options:
  - Kaggle's free 2×T4 (32 GB total, about 30 GPU-h/week; the notebook would need Kaggle paths and secrets);
  - Colab Pro L4 or A100.
- `run_local.sh` would also need a context-size override, and the chat template, reasoning format and tool calls need checking.

**Stage 3: agent evaluation, in two layers.**
- **3a, replay eval in the same notebook (no GNU Radio).**
  - Uses Qoherent's recorded runs: `dataset/GRC-Agent_tier1_dataset_v1_toolcalls` (1,253) and `GRC-Agent_tier2_dataset_v1` (600), with the 23 tool schemas, tool outputs and `flowgraph_diff`.
  - The model predicts the next tool call. The score covers tool choice and whether `change_graph` produces the recorded change.
  - **Hold out a fixed split from training.**
- **3b, full end-to-end runs with the official `grc-agent-headless`.**
  - One Docker image (Ubuntu 26.04, GNU Radio 3.10.12, GRC-Agent at the pinned commit, bubblewrap, llama.cpp CUDA) on a **Google Cloud GPU VM paid with the education coupon**.
  - The same image runs on Qoherent's GPU later.
  - Check first: the coupon amount, GPU quota for the project, and Joel's OK to store GRC-Bench items there.
- **Not chosen:**
  - GRC-Agent inside Colab: the sandbox likely won't run, and everything must be reinstalled each session;
  - a tunnel out of Colab: against Colab's rules.

## Open items

- The GCP education coupon's value, and whether GPU quota is granted on an education billing account (stage 3b).
- A shared team Drive folder for run folders, or a results table later.
- ~~Renaming the repo~~ **Done 2026-10-31:** the GitHub repo and the local folder are now `grc-model-eval` (formerly `nanbeige-modal`). GitHub redirects the old URL; the local venv paths were fixed after the folder rename.

## Sources

- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Colab for higher education (Google blog)](https://blog.google/products-and-platforms/products/education/colab-higher-education/)
- [Colab Enterprise pricing](https://cloud.google.com/colab/pricing)
- Colab unit measurements (unofficial):
  - [Thunder Compute](https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025)
  - [Fastero](https://fastero.com/blog/google-colab-pro)
- [llama.cpp release b12020](https://github.com/ggml-org/llama.cpp/releases/tag/b12020)
- Models:
  - [unsloth/North-Mini-Code-1.0-GGUF](https://huggingface.co/unsloth/North-Mini-Code-1.0-GGUF)
  - [CohereLabs/North-Mini-Code-1.0](https://huggingface.co/CohereLabs/North-Mini-Code-1.0)
- [qoherent/grc-agent](https://github.com/qoherent/grc-agent) (public, AGPL-3.0)
- Local files:
  - `modal/MODAL_PLAN.md`
  - `docs/local-llamacpp.md`
  - `serve/models.json`
  - `MLOps Qoherent/pipeline/base_model_hosting.md`
