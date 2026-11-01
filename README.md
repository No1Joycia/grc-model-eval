# GRC model evaluation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/No1Joycia/grc-model-eval/blob/main/colab/eval.ipynb)

Team JJJP is fine-tuning small models so GRC-Agent can run **offline on a laptop**. This repo **evaluates** those
models: it asks a model graded RF / DSP / GNU Radio questions and scores its answers. The first model is
[Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B), run with **llama.cpp**, the same way an offline
user would run it.

> **Work in progress.** The **data** (which questions, with which reference answers) and the **evaluation method**
> (how answers are scored, and later GRC-Agent tasks) aren't final:
> - **Patricio** is preparing the data;
> - **Joel** is preparing the evaluation.
>
> What's here today is a working first version: 100 trial questions scored by similarity to reference answers.
> Their work plugs into the same pipeline (see [Data](#data-lives-in-google-drive-not-github) and
> [Contributing](#contributing-code)), so running an evaluation stays the same.

## Start here

| You want to... | Go to |
|---|---|
| **Evaluate a model** (most teammates): one notebook in your own Colab, free T4 GPU, nothing to set up but a Google account | [`colab/README.md`](colab/README.md) |
| **Provide question data** (Patricio) | [Data](#data-lives-in-google-drive-not-github) below |
| **Change how answers are scored**, or add agent tasks (Joel) | [Contributing](#contributing-code) below |
| Run the model on your Mac with llama.cpp, e.g. for GRC-Agent offline | [`docs/local-llamacpp.md`](docs/local-llamacpp.md) |
| See the plan and its progress | [`colab/COLAB_PLAN.md`](colab/COLAB_PLAN.md) |
| The old hosted Modal endpoint (being retired) | [`modal/README.md`](modal/README.md) |

## How an evaluation runs

Each teammate runs it in **their own Colab session**: there's no shared server and no shared token.

```
 GitHub (this repo)        Google Drive                       Hugging Face
 code: notebook,           question data (shared folder)      model checkpoint
 scripts, settings         your model cache + results
        \                        |                               /
         \                       v                              /
          +------>  Colab notebook (colab/eval.ipynb)  <-------+
                    1 check GPU   2 Drive + code + data   3 llama.cpp
                    4 model file  5 start model           6 answer   7 score + report
                                       |
                                       v
                    your Drive: grc-eval/runs/<run>/  (answers, scores, HTML report, run_info.json)
                    -> copy the folder into the team's shared folder to share it
```

| Piece | Lives in | Who changes it |
|---|---|---|
| Code: notebook, scripts, model settings | **GitHub**, this repo (step 2 downloads the latest `main`) | anyone, through a pull request |
| Question data | **Google Drive**, the data owner's shared folder | Patricio |
| Model weights | **Hugging Face**; converted to a llama.cpp file once and cached in your Drive | fine-tuning, later |
| Results | **Your Google Drive**; copy a run folder to the team folder to share it | whoever ran it |

## Data lives in Google Drive, not GitHub

Question data is never committed to GitHub: it's big, and evaluation questions that get published can end up in other
models' training data.

- **The data owner** keeps the files in a Google Drive folder (for example `grc-eval-data/`) and shares it with the
  team as *Viewer*. Files are versioned by name (`questions_v1.jsonl`, `questions_v2.jsonl`); a file that has been
  used is never edited.
- **Each teammate** adds a shortcut to that folder in their *My Drive* (one time). They then set the notebook's
  `QUESTIONS` setting to the file, e.g. `grc-eval-data/questions_v2.jsonl`.
- **Every run records** which question file it used and that file's SHA-256 (in `run_info.json`), so a score can
  always be traced to its exact data.
- **Eval questions stay out of training data.** The IDs of evaluation questions are kept out of every training file
  (see `trial_eval/trial_eval_ids.txt`).

The one exception is the 100-question trial set (`trial_eval/trial_eval_100.jsonl`, 2 MB). It's in the repo and is the
default, so the notebook works before any shared data exists.

Step-by-step sharing, the shortcut, and the **question file format** (the fields the eval needs):
[`colab/README.md` → Data](colab/README.md#data-question-files-in-google-drive).
Scripts that *prepare* data (no data inside) can go in the repo, e.g. in a `data/` folder.

## Contributing code

**Every teammate's notebook downloads the latest `main` when it runs.** A change merged into `main` therefore reaches
everyone's next run, so:

1. **Work on a branch:** `git switch -c joel/llm-judge`, commit, `git push -u origin joel/llm-judge`.
   Pushing needs a collaborator invite from Joy; without one, fork the repo and open the pull request from your fork.
2. **Test it in Colab:** set `BRANCH` in the notebook's Settings to your branch and *Run all*.
3. **Open a pull request** into `main`, and merge it when it works.

| Where things go | |
|---|---|
| Scoring and evaluation methods | `trial_eval/` (today's scorer: `score_trial.py`), or a new folder for a new kind of evaluation (e.g. `eval/` for GRC-Agent tasks) |
| Data-preparation scripts | e.g. `data/` (scripts only, never the data) |
| How the model is run (llama.cpp build, settings, model files) | `serve/`, `convert/` |
| The Colab notebook and its helper | `colab/` |

**Keep the hand-offs, and the notebook doesn't need to change.** `trial_eval/run_trial.py` writes `answers.jsonl` into
the run folder, one row per answer: `id`, `answer`, `finish_reason`, `settings`, ... A scorer reads that file plus the
question file, and writes its results into the same run folder.

**Editing the notebook:** changes made in Colab are **not** saved to GitHub automatically. See
[`colab/README.md` → Changing the notebook](colab/README.md#changing-the-notebook-it-does-not-save-back-to-github).

**Never commit:** data files, `.env` files, tokens or keys, model files, or run outputs.

## What's in this repo

| Folder | What it does |
|---|---|
| [`colab/`](colab/) | The Colab notebook (`eval.ipynb`), its helper (`helpers.py`), how to use it (`README.md`) and the plan (`COLAB_PLAN.md`) |
| [`trial_eval/`](trial_eval/) | The 100 trial questions with reference answers, `run_trial.py` (asks a model) and `score_trial.py` (scores, HTML report). See `README.txt` |
| [`serve/`](serve/) | llama.cpp settings every machine shares (`models.json`: pinned build b12020, server settings, model files and their SHA-256), plus scripts to install, start and check llama-server and to get the model file (`get_gguf.sh`) |
| [`convert/`](convert/) | Turns a Hugging Face checkpoint into llama.cpp model files (GGUF: BF16, Q8_0, Q4_K_M) |
| [`docs/`](docs/) | `local-llamacpp.md`: running the model on a Mac |
| [`grc-agent-adapter/`](grc-agent-adapter/) | Runs GRC-Agent without its window on macOS, for agent tasks (needs `~/Projects/grc-agent`) |
| [`northMiniCode/`](northMiniCode/) | North-Mini-Code through Cohere's API, answering the same trial questions (no agent) |
| [`modal/`](modal/) | The old Modal setup and its plan (`MODAL_PLAN.md`); being retired |

## Status (2026-10-31)

- **Stage 1, Nanbeige Q&A in Colab:** working; Joy's first Colab run (questions #1–5) succeeded, and the full
  100-question baseline is next. Details: [`colab/COLAB_PLAN.md`](colab/COLAB_PLAN.md), *Progress*.
- **Next:** shared question data (Patricio) and the evaluation method (Joel); then North-Mini-Code (stage 2) and
  GRC-Agent tasks (stage 3).

Not in this repo: the training dataset (`dataset/`, local only and git-ignored), and the proposal, application forms,
meeting notes and pipeline docs (kept locally in `~/Projects/MLOps Qoherent/`).
