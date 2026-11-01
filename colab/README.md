# Evaluate the model in Google Colab (stage 1: Nanbeige4.2-3B, Q&A)

One notebook, `colab/eval.ipynb`, runs the whole evaluation **in your own Colab session**, on your own Colab credits.
Nobody hosts a server and nobody shares a token. The notebook:

1. downloads Nanbeige4.2-3B from Hugging Face and converts it to a llama.cpp model file (first time only);
2. runs it with **llama.cpp** on Colab's GPU;
3. asks it a set of graded questions: the team's 100 trial questions by default, or a question file from Google Drive
   (see [Data](#data-question-files-in-google-drive)); you choose the range;
4. scores the answers and shows an HTML report.

The plan behind it is [`COLAB_PLAN.md`](COLAB_PLAN.md).

> **Work in progress.** The question data and the scoring method aren't final: Patricio is preparing the data and
> Joel the evaluation. Today's version (100 trial questions, similarity scoring) works end to end. Their versions plug
> into the same notebook: new data through the `QUESTIONS` setting, new scoring through the repo. How you run it
> stays the same.

## What Colab gives you

| | Free tier | Used for |
|---|---|---|
| **OS** | Ubuntu Linux (64-bit), a fresh temporary machine every session | everything |
| **GPU** | NVIDIA **T4** with 16 GB (a GPU, not a TPU) | the model (llama.cpp through CUDA), then scoring |
| **CPU / memory** | 2 cores, about 12.7 GB | converting the model file, the scripts |
| **Disk** | about 100 GB, **wiped when the session ends** | downloads and temporary files |
| **Your Google Drive** | 15 GB free | what has to survive: the converted model (4.1 GB) and your results |

The free tier is enough for this stage, so the cost is **$0**. Colab Pro (about US$10 a month for 100 compute units)
only adds faster GPUs and fewer limits. Free GPUs aren't guaranteed: if none is available, try again later.

## Open the notebook

"Opening the notebook" means getting `colab/eval.ipynb` into a Colab browser tab. Choose **one** of the three ways
below; they all give the same notebook.

**A. Direct link (easiest)**

1. Make sure you're signed in to Google in your browser (the account whose Drive should hold the results).
2. Open this link: <https://colab.research.google.com/github/No1Joycia/grc-model-eval/blob/main/colab/eval.ipynb>
   (or click the *Open in Colab* button in the repo's main README).
3. The notebook opens with the title "GRC model evaluation in Colab". Go to *Check the GPU* below.

**B. Colab's GitHub page** (to browse to it)

1. Open <https://colab.research.google.com/github/>.
2. In the search box, type `No1Joycia` and press Enter. Then choose:
   - **Repository:** `No1Joycia/grc-model-eval`;
   - **Branch:** `main`.
3. A list of notebook files appears. Click **`colab/eval.ipynb`**.

The same GitHub search is also in Colab's **File → Open notebook** dialog (⌘/Ctrl + O, *GitHub* among the sources),
but newer Colab layouts don't always show it there. If colab.research.google.com opens an empty *Untitled0.ipynb*,
that's harmless: delete it later from `MyDrive/Colab Notebooks/`.

**C. Upload the file from your computer**

1. Get the file:
   - on GitHub, open `colab/eval.ipynb` and click the **Download raw file** button (the down-arrow icon), or
   - use your local clone: `~/Projects/grc-model-eval/colab/eval.ipynb`.
2. Go to <https://colab.research.google.com>. In the **Open notebook** dialog, choose the **Upload** tab
   (or *File → Upload notebook*).
3. Click **Browse**, choose `eval.ipynb`, and it opens.

**Saving your copy (optional).** A notebook opened from GitHub isn't saved anywhere you own. If you change the
settings and want to keep them, use *File → Save a copy in Drive*: it lands in `MyDrive/Colab Notebooks/`. You don't
need to save it to run it: results go to Drive either way.

**Check the GPU.** The notebook asks for a T4 GPU by itself. To confirm:
1. Open the menu *Runtime → Change runtime type*.
2. Under *Hardware accelerator*, **T4 GPU** should be selected. Select it if not.
3. Click **Save**.

The top-right corner then shows **Connect** (or *T4* with RAM/Disk bars once connected).

## Run it

1. In the **Settings** form at the top, set the question range (`START`, `END`), and change anything else if you want.
2. *Runtime → **Run all***. This runs the Settings cell too. Always start this way rather than clicking the steps one
   by one: Colab **forgets everything when the runtime restarts**, for example right after you change the runtime
   type. If a step says *"... hasn't run in this session"*, just use *Run all* again.
3. In step 2, a pop-up asks for **Google Drive** access: choose your account and allow it.
4. **Keep the tab open** until step 7 shows the report. Closing the tab or leaving it idle for a long time can
   disconnect a free session. Nothing is lost: see *If the session disconnects*.

### Settings

| Setting | Default | What it does |
|---|---|---|
| `QUESTIONS` | blank | Blank = the 100 trial questions in the repo. Or a `.jsonl` file in your Google Drive, given as a path inside My Drive, e.g. `grc-eval-data/questions_v2.jsonl` (see [Data](#data-question-files-in-google-drive)) |
| `START`, `END` | 1, 100 | Which questions, by line number in the file. In the trial set: #1–33 basic, #34–67 intermediate, #68–100 advanced |
| `MODEL` | `nanbeige42` | The model, as named in `serve/models.json` |
| `QUANT` | `q8_0` | `q8_0` (4.1 GB, the team standard) or `q4_k_m` (2.4 GB, smaller, slightly lower quality) |
| `THINKING` | `off` | `on` lets the model reason before answering: slower, and needs `MAX_TOKENS` of 4096 or more |
| `MAX_TOKENS` | 1536 | The longest answer allowed. Answers that hit it are marked `length` (cut off) |
| `HF_REPO`, `REVISION` | blank | Blank = the pinned base model. For a fine-tuned version: its **merged** HF repo and a revision |
| `DRIVE_FOLDER` | `grc-eval` | Folder in your Drive for the model file and results |
| `REPO_URL`, `BRANCH` | this repo, `main` | Where the code comes from |

## What each step does

Every log line starts with the time and the step, for example `[14:02:11 | step 6/7] asking #12 ...`. Each step starts
with a banner saying what it does, and ends with `DONE in <time>: <summary>` and `-> next: step N`. Quiet commands
print `... still working` every 15 seconds. If a step fails, it stops *Run all* with `STEP N/7 FAILED`, the last output
lines and **What to do**.

| Step | What it does | First session | Later sessions | Look for |
|---|---|---|---|---|
| Settings | Reads the form and checks the values | instant | instant | the settings summary line |
| 1. Check the machine | Confirms there's an NVIDIA GPU; prints GPU, OS, memory, disk | seconds | seconds | `GPU: Tesla T4, ...` |
| 2. Drive and code | Mounts Drive; clones the repo; checks the question file; names the run and its folders | ~1 min | ~1 min | `run folder: ...` |
| 3. Install llama.cpp | Downloads the pinned prebuilt CUDA build (~740 MB) and checks it sees the GPU | 1–2 min | 1–2 min | `DONE ... it sees CUDA0: Tesla T4` |
| 4. Model file | Converts from Hugging Face and saves to Drive, or copies the Drive file back; checks its SHA-256 | 10–15 min | 1–3 min | `check: match` (or the hash to send Joy) |
| 5. Start the model | Starts llama-server on `127.0.0.1` at a free port (8181 unless taken; Colab uses 8080) and runs the smoke test | 1–2 min | 1–2 min | `all checks passed` |
| 6. Answer | Asks each question and saves each answer to Drive at once | ~30 s per question | same | `ESTIMATE`, one line per answer, `DONE` |
| 7. Score and report | Stops the model, scores, writes `run_info.json`, shows and downloads the report | 2–4 min | 2–4 min | the score summary and the report |

Times are estimates until the first real runs; step 6's ESTIMATE line adjusts once answers have been timed.

A typical answer line in step 6:

```
  asking #12 (basic, conceptual)...
[12/30] #12  q_75052f3d21232fa6 basic          612 tok   21.4s (28.6 tok/s) stop | ~6 min left
```

`stop` means the answer finished; `length` means it was cut off at `MAX_TOKENS`.

## Where the results go

```
MyDrive/grc-eval/
├── models/Nanbeige__Nanbeige4.2-3B-b82e54b-b12020/
│   └── Nanbeige4.2-3B-Q8_0.gguf          the converted model, reused by later sessions
└── runs/nanbeige42-b82e54b-q8_0-think_off-b12020/
    ├── answers.jsonl                      every answer, with the settings it was made with
    ├── answers_q1-30_scores.csv           one row per question (the file names carry your range)
    ├── answers_q1-30_summary.json         averages by difficulty, task type, GNU Radio
    ├── answers_q1-30_report.html          questions, answers, reference answers, scores
    ├── run_info.json                      GPU, llama.cpp build, model file hash, settings, timings
    └── llama-server.log                   the model server's log
```

With the full range (#1–100), the score files are called `answers_scores.csv`, `answers_summary.json` and
`answers_report.html`.

Runs with different settings (thinking, quantization, max tokens, another model, another question file) get their own
folder automatically; a question file adds its name to the end, e.g. `...-b12020-questions_v2`.
To share a run, copy its folder into the team's shared Drive folder.

## Data: question files in Google Drive

**Data is never put on GitHub.** It's too big for git, and evaluation questions that get published can end up in
other models' training data. Each data owner (for now Patricio) keeps it in a **Google Drive folder** and shares it
with the team. Each teammate's notebook reads it from there. The only exception is the 100-question trial set,
`trial_eval/trial_eval_100.jsonl` (2 MB), kept in the repo so the notebook works out of the box.

**For the data owner (sharing a folder, one time):**
1. In Google Drive, create a folder, e.g. `grc-eval-data`, and put the question files in it.
2. *Right-click the folder → Share*. Add your teammates' Google accounts as **Viewer**.
3. **Never change a file that has been used.** Add a new file instead (`questions_v1.jsonl`, `questions_v2.jsonl`,
   ...). Every run records the file it used and the file's SHA-256 in its `run_info.json`, so results can always be
   traced to the exact data.
4. Keep the IDs of the evaluation questions out of every training file, as `trial_eval/trial_eval_ids.txt` does for
   the trial set. Otherwise scores after fine-tuning mean nothing.

**For everyone else (using a shared folder, one time):**
1. In Google Drive, open **Shared with me** and find the folder.
2. *Right-click it → Organize → **Add shortcut** → My Drive*. Colab can only see what's inside *My Drive*, so this
   step is what makes the folder visible to the notebook.
3. In the notebook's Settings, set `QUESTIONS` to the file's path inside My Drive, e.g.
   `grc-eval-data/questions_v2.jsonl`, then *Run all*.

**Question file format.** JSON Lines: one question per line, as in `trial_eval/trial_eval_100.jsonl`.

| Field | Needed for | Example |
|---|---|---|
| `id` | answering (must be unique; answers are matched by it) | `"q_2bce8f4f13f014be"` |
| `question` | answering | `"Design a flowgraph that reads samples from a file, ..."` |
| `difficulty` | answering and scoring | `"basic"`, `"intermediate"`, `"advanced"` |
| `task_type` | answering and scoring | `"design"`, `"calculation"`, `"conceptual"`, ... |
| `is_gnuradio` | scoring | `true` |
| `reference_answer` | scoring: the main reference answer | `"To design a flowgraph ..."` |
| `references` | scoring: every accepted answer, as `{"model": ..., "answer": ...}` | `[{"model": "gemma4-31b", "answer": "..."}]` |

Step 2 checks the file. Without the answering fields it stops. Without the scoring fields it warns: answering works,
but step 7 can't score. Other fields are kept and ignored. If Joel's scoring needs different fields, this table changes
with it.

## If the session disconnects

Just *Run all* again:
- step 4 reuses the model file in your Drive;
- step 6 skips every question already in `answers.jsonl` and continues with the next one.

## Troubleshooting

| What you see | What to do |
|---|---|
| `STEP 1/7 FAILED: this runtime has no GPU` | *Runtime → Change runtime type → T4 GPU*, then *Run all*. If no GPU is available, try later. |
| Step 2: `git clone failed ... Remote branch ... not found` | The `BRANCH` in the settings doesn't exist on GitHub (check its spelling, and that it was pushed). |
| Step 2: `git clone failed ... Authentication failed` or `Repository not found` | `REPO_URL` is wrong, or the repo has been made private again: then add a `GH_TOKEN` secret (a GitHub token that can read it) with *Notebook access*. |
| Step 3: `the prebuilt llama.cpp can't use this GPU` | Colab's driver is too old for CUDA 12.8. The cell compiles llama.cpp itself (20–40 min, first time only). If that fails too, send the output to Joy. |
| Step 4: `sha256 ... expects ...` | The model file differs from the team reference. Delete the file it names (in your Drive's `models/` folder) and *Run all* to convert again. If it still differs, tell Joy. |
| Step 4: `FIRST CONVERSION ON THIS KIND OF MACHINE` | Not an error: send the printed line to Joy so it becomes the reference. |
| Step 5: `llama-server stopped ... before it was ready` | Read the server log lines it prints. Out of GPU memory: choose `q4_k_m`. |
| Step 6: `giving up on #N` | The server stopped. Rerun step 5, then step 6 (answers so far are kept). |
| `... was made with different settings` | That run folder holds answers made with other settings. This shouldn't happen with the notebook's automatic names; if it does, use another `DRIVE_FOLDER`. |
| Drive is full | The model file needs 4.1 GB. Free space, or the run still works but converts again every session. |
| Step 2: `question file not found` | Check the `QUESTIONS` path (a path inside My Drive). For a shared folder, add a shortcut to My Drive first ([Data](#data-question-files-in-google-drive)). |
| Step 2: `lacks fields the eval needs` | The question file doesn't follow the format in [Data](#data-question-files-in-google-drive). Tell the data owner. |

## Changing the notebook (it does not save back to GitHub)

**Edits you make in Colab are not sent to GitHub.** When you open the notebook from GitHub, Colab works on its own
copy. Changing the Settings form (range, thinking, ...) needs no saving at all: just *Run all*.

To keep your own edited copy: *File → **Save a copy in Drive*** (it goes to `MyDrive/Colab Notebooks/`).

To change the notebook **for everyone** (needs a collaborator invite from Joy; anyone else can fork the repo and do
the same on their fork):
1. *Edit → **Clear all outputs*** first. Otherwise your logs and the report get stored in git.
2. *File → **Save a copy in GitHub*** → repository `No1Joycia/grc-model-eval`, a **branch** (for example
   `yourname/notebook-change`, not `main`), and a short commit message. Colab asks once for permission to write to
   GitHub.
3. On GitHub, open a pull request from that branch into `main`.

Why not straight to `main`? Every teammate's notebook downloads the latest `main` in step 2, so a change on `main`
reaches everyone's next run at once.

**To test a code change before merging it:** push it to a branch, then set `BRANCH` in the notebook's Settings to that
branch and *Run all*. Step 2 then downloads your branch instead of `main`. Most logic lives in `colab/helpers.py` and
the repo's scripts (`serve/`, `trial_eval/`), not in the notebook cells, so most changes don't touch the notebook.

## For Joy (maintainer)

- The first Colab conversion prints the Linux Q8_0 hash: add it to `serve/models.json` as
  `"sha256" → "q8_0" → "Linux-x86_64"`. Every later conversion is then checked against it.
