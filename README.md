# GRC model evaluation

*(Formerly `nanbeige-modal`.)*

Evaluates the models the team fine-tunes for GRC-Agent, starting with
[Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B), on 100 graded RF / DSP / GNU Radio
trial questions. The model runs with **llama.cpp**, the same way an offline GRC-Agent user would run it on a laptop.

## Start here

| I want to... | Go to |
|---|---|
| **Evaluate the model myself** (recommended for teammates): one Colab notebook, free T4 GPU, your own credits | [`colab/README.md`](colab/README.md) |
| Run the model on my Mac with llama.cpp, e.g. for GRC-Agent offline | [`docs/local-llamacpp.md`](docs/local-llamacpp.md) |
| Use or maintain the old hosted Modal endpoint (being retired) | [`modal/README.md`](modal/README.md) |
| Understand the plan and its progress | [`colab/COLAB_PLAN.md`](colab/COLAB_PLAN.md) |

## What's in this repo

| Folder | What it does |
|---|---|
| [`colab/`](colab/) | The Colab notebook (`eval.ipynb`), its helper (`helpers.py`), how to use it (`README.md`) and the plan (`COLAB_PLAN.md`) |
| [`trial_eval/`](trial_eval/) | The 100 trial questions with graded reference answers, `run_trial.py` (asks a model), `score_trial.py` (scores, HTML report). See `README.txt` |
| [`serve/`](serve/) | llama.cpp settings every host shares (`models.json`: pinned build b12020, server settings, model files and their SHA-256), plus scripts to install, start and check llama-server and to get the model file (`get_gguf.sh`) |
| [`convert/`](convert/) | Turns a Hugging Face checkpoint into llama.cpp model files (GGUF: BF16, Q8_0, Q4_K_M) |
| [`docs/`](docs/) | `local-llamacpp.md`: running the model on a Mac |
| [`grc-agent-adapter/`](grc-agent-adapter/) | Runs GRC-Agent without its window on macOS, for agent tasks (needs `~/Projects/grc-agent`) |
| [`northMiniCode/`](northMiniCode/) | North-Mini-Code through Cohere's API, answering the same trial questions (raw mode, no agent) |
| [`modal/`](modal/) | The old Modal setup (Transformers server, `ask.py`, its `.env`) and its plan `MODAL_PLAN.md`; being retired |

## Status (2026-10-31)

- **Stage 1, Nanbeige Q&A in Colab:** built and rehearsed on a Mac; the first Colab run is under way.
  Details: [`colab/COLAB_PLAN.md`](colab/COLAB_PLAN.md), *Progress*.
- **Later:** North-Mini-Code Q&A (stage 2) and agent tasks with GRC-Agent (stage 3).

Not in this repository: the training dataset (`dataset/`, local and git-ignored), and the proposal, application forms,
meeting notes and pipeline docs (kept locally in `~/Projects/MLOps Qoherent/`).
