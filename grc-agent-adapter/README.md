# grc-agent-adapter

GRC-Agent without its window, on a Mac. You give it a message, and it writes back everything that happened as
files: the agent's replies, every tool call and its result, the trace in Qoherent's training-row format, the
flowgraph and the project folder. A later call can continue the same conversation on the same flowgraph.

It's the macOS counterpart of `grc-agent-headless`, the GRC-Bench Tier-A runner on the `add-grc-agent-headless`
branch. That package needs Ubuntu, a Linux sandbox (bubblewrap) and GRC-Agent at commit `6c98128`. This one
runs on macOS with **your own GRC-Agent checkout**, whatever commit it's at. It uses the same real executor
(GRC-Agent's own system prompt and 22 tools) and writes the same kinds of output files.

```bash
cd ~/Projects/nanbeige-modal/grc-agent-adapter
bash setup.sh                                              # once: checks, builds the search index, self-test
./agent run "Which GNU Radio block makes a cosine tone?" --provider cohere
./agent continue runs/<session>/call-01 "Now build it into a flowgraph and save it as tone.grc."
./agent show runs/<session>/call-02 --transcript
./agent batch questions.jsonl --out answers.jsonl --provider cohere --no-graph-edits --no-run
```

Contents:
1. [What you need](#1-what-you-need)
2. [Setup](#2-setup)
3. [Models and API keys](#3-models-and-api-keys)
4. [Commands](#4-commands)
5. [Outputs](#5-outputs)
6. [Safety: approvals instead of a sandbox](#6-safety-approvals-instead-of-a-sandbox)
7. [Limits](#7-limits)
8. [How it works, and how it differs from the desktop app](#8-how-it-works-and-how-it-differs-from-the-desktop-app)
9. [Using it from another program](#9-using-it-from-another-program)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What you need

| | |
|---|---|
| Computer | macOS (tested: Apple M3, macOS 15) |
| GRC-Agent | Qoherent's repository checked out at `~/Projects/grc-agent`, with its `.venv` working (the one you start the desktop app with) |
| GNU Radio | visible to GRC-Agent's Python. On this Mac it comes from the conda env `grc-agent-mac313` (GNU Radio 3.10.12) |
| A model | an API key for a hosted provider (Cohere, OpenRouter, ...) or a local OpenAI-compatible server |

Folder layout the scripts expect by default:

```
~/Projects/
├── grc-agent/                      Qoherent's GRC-Agent (not modified by this adapter)
└── nanbeige-modal/
    └── grc-agent-adapter/          this folder
        ├── agent                   the command
        ├── setup.sh                the setup check
        ├── .env                    your API keys (git-ignored; copy .env.example)
        ├── runs/                   session folders (git-ignored)
        ├── examples/               a two-question file for ./agent batch
        └── src/grc_adapter/        the code
```

If GRC-Agent lives somewhere else: `export GRC_AGENT_REPO=/path/to/grc-agent` in `~/.zshrc`.

## 2. Setup

```bash
cd ~/Projects/nanbeige-modal/grc-agent-adapter
bash setup.sh
```

The script checks the GRC-Agent checkout, its Python, and GNU Radio, and prints a fix for anything missing.
Once only, it builds GRC-Agent's local knowledge databases: the block catalog and the GNU Radio docs, using
lexical search, in a few seconds. These live in GRC-Agent's git-ignored `src/grc_agent/vectors/` folder. It
finishes with `./agent selftest`, which uses a scripted model, so it needs no API key. A good run ends like this:

```
ok    GNU Radio platform built headless (580 blocks)
ok    GRC-Agent ran headless (status completed, agent_finished)
ok    change_graph, save_graph, run_flowgraph and get_run_log executed
ok    the flowgraph ran in GNU Radio (return code 0)
ok    the flowgraph wrote project/selftest.dat (192000 bytes)
ok    query_knowledge answered from GRC-Agent's local catalog
ok    shell command denied by the default policy
ok    graph.grc / graph_view.json: valid, the saved file matches the flowgraph in memory
ok    all output files written
ok    trace.jsonl has the system prompt and 22 tool schemas
ok    continue: the next call has the conversation and the same flowgraph
ok    continue: trace.jsonl holds both user messages
ok    bounded run, background run and stop (stopped_after_timeout, started, stopped)
ok    the API key is in no output file
self-test: 14/14 checks passed
```

Then add your keys:

```bash
cp .env.example .env
open -e .env          # paste the key after COHERE_API_KEY= (or OPENROUTER_API_KEY=)
chmod 600 .env
```

## 3. Models and API keys

`--provider` takes GRC-Agent's own provider ids. GRC-Agent builds the model with its own code for that
provider, as it does when you choose the provider in its Settings dialog.

| Provider | Example | Key variable |
|---|---|---|
| `cohere` | `--provider cohere` (the model defaults to `north-mini-code-1-0`) | `COHERE_API_KEY` |
| `openrouter` | `--provider openrouter --model cohere/north-mini-code:free` | `OPENROUTER_API_KEY` |
| `openai_compatible` (default) | `--base-url http://127.0.0.1:8080/v1 --model <name>`: llama-server, vLLM, or a Modal endpoint | optional: `--api-key-env MY_VAR` |
| `ollama_local` | `--provider ollama_local --model qwen3.8:latest` | none |
| `ollama_cloud`, `openai`, `anthropic`, `google`, `groq`, `mistral`, `xai` | `--provider <id> --model <name>` | GRC-Agent's variable for that provider |

Without `--model`, GRC-Agent's default for that provider is used.

**How keys are handled.** Keys are read from `grc-agent-adapter/.env`, or from another file with
`--env-file PATH`, or from your shell environment. Only `*_API_KEY` variables are read. The key is held in the
running process's memory only:
- GRC-Agent's settings file for the call names the provider and model, never the key.
- Flowgraph runs get an environment with every `*_API_KEY`, `*_TOKEN` and `*_SECRET` variable removed.
- GRC-Agent strips provider keys from shell commands itself.
- The self-test checks that the key appears in no output file.

If the key is missing, the call is refused before anything starts (exit code 2).

## 4. Commands

### `./agent run MESSAGE [options]`: start a session

| Option | Meaning |
|---|---|
| `--graph FILE` | a `.grc` file, copied into the project folder and opened. Without it the agent starts from GRC's File > New. |
| `--files DIR` | a folder copied into the project folder, for data files |
| `--name NAME` | the session name (default: date and time) |
| `--then MESSAGE` | a follow-up message sent after the agent's turn, whatever it said (repeatable, sent in order) |
| `--work DIR` | where sessions are written (default `runs/`) |
| `--json` | print `summary.json` instead of the short report |
| model options | `--provider`, `--model`, `--base-url`, `--api-key-env`, `--env-file` ([section 3](#3-models-and-api-keys)) |
| approval options | `--allow-shell`, `--allow-web`, `--no-graph-edits`, `--no-run` ([section 6](#6-safety-approvals-instead-of-a-sandbox)) |
| limits | `--max-requests`, `--episode-seconds`, `--flowgraph-seconds`, `--no-compaction` ([section 7](#7-limits)) |

Your input files are copied, never edited. Each call prints a short report, for example:

```
session tone-demo, call 1: completed (agent_finished)
run folder: .../grc-agent-adapter/runs/tone-demo/call-01
graph: valid, project/tone.grc
files: added tone.grc, tone.py
requests 4, tokens in 31200 / out 640, tool calls 5, 18.2 s

final reply:
I added a Signal Source (1 kHz cosine) into a Throttle and a QT GUI Time Sink, and saved tone.grc.
```

### `./agent continue CALL_FOLDER MESSAGE [options]`: the next message

```bash
./agent continue runs/tone-demo/call-01 "Now add a Low Pass Filter with a 5 kHz cutoff."
```

The next call starts from that call's state:
- the full message history;
- the project folder as it was at the end of that call;
- the flowgraph exactly as it was in memory, including unsaved edits.

It reuses that call's model, approvals and limits unless you give new options. You can continue any earlier
call, not only the last one, so a session can branch. A flowgraph still running at the end of a call is
stopped, so the next call starts with none running.

### `./agent batch QUESTIONS.jsonl --out ANSWERS.jsonl [options]`: many questions

Each line of the input is one JSON object with an `id` and a `question`. You can name other fields with
`--id-key` / `--question-key`. Every question gets its own fresh session under `runs/<out-stem>/<id>/`, and
one row is appended to `--out`:

```json
{"id": "q_2bce8f4f13f014be", "model": "cohere:north-mini-code-1-0", "answer": "...",
 "finish_reason": "stop", "usage": {"prompt_tokens": 31200, "completion_tokens": 640, "total_tokens": 31840,
 "reasoning_tokens": 0, "requests": 4}, "seconds": 18.2, "status": "completed", "end_reason": "agent_finished",
 "error": null, "tool_calls": ["query_knowledge", "query_knowledge"], "run_dir": "..."}
```

- **Resumable.** A rerun skips ids already in `--out`, so you can stop it with Ctrl+C and start it again.
- **Scoring.** The rows have the fields `trial_eval/score_trial.py` reads (`id`, `answer`, `finish_reason`), so
  agent answers are scored exactly like the Nanbeige ones:

  ```bash
  ./agent batch ../trial_eval/trial_eval_100.jsonl --out ../trial_eval/answers_north_agent.jsonl \
      --provider cohere --no-graph-edits --no-run
  ../.venv-eval/bin/python ../trial_eval/score_trial.py --answers ../trial_eval/answers_north_agent.jsonl
  ```

- `--limit N` runs only the first N questions; use `--limit 3` as a smoke test.
- The batch stops after 3 infrastructure errors in a row (`--stop-after-errors`), for example a wrong key or a
  provider outage, so a bad key doesn't burn through the whole file.
- Questions run one at a time.

### `./agent show CALL_FOLDER [--transcript]` and `./agent selftest [--keep]`

`show` prints `summary.json`, or the conversation as text with `--transcript`. `selftest` is the check from
[section 2](#2-setup). `--keep` keeps its session folders and prints where they are.

Exit codes: 0 when the call ran, whatever the agent did; 1 for an infrastructure error, such as an unreachable
model server; 2 when the request was refused (bad option, missing file, missing key).

## 5. Outputs

A session is a folder `runs/<name>/`:

| Path | Content |
|---|---|
| `session.json` | every call in order: run folder, parent call, message, status, end reason, usage, model |
| `project/` | the live project folder: the agent's file root and where untitled flowgraphs are saved |
| `call-NN/request.json` | the request as given: message, follow-ups, model (no key), approvals, limits |
| `call-NN/summary.json` | the call's results (below) |
| `call-NN/transcript.md` | the whole conversation so far as text: messages, thinking, tool calls with arguments, tool results |
| `call-NN/trace.jsonl` | the same conversation as one row in Qoherent's SFT row format (`session_id, task_id, category, verdict, teacher, system, tools, messages, usage`, then `grc_adapter` metadata), the same format grc-agent-headless writes |
| `call-NN/messages.json` | the full message history in pydantic-ai's own format (what `continue` reloads) |
| `call-NN/tools.json` | the tool schemas the model was given |
| `call-NN/graph.grc` | the flowgraph as it was in memory at the end, saved or not |
| `call-NN/graph_view.json` | GRC-Agent's `inspect_graph` view of it, plus `valid` and `errors` |
| `call-NN/project/` | a snapshot of the project folder at the end of the call |
| `call-NN/agent.log` | the log of GRC-Agent and pydantic-ai during the call |

**`summary.json`:**
- `status`:
  - `completed`;
  - `cap_failure`: a limit was hit;
  - `agent_error`: the agent run failed, for example retries ran out;
  - `infrastructure_error`: the model server or setup failed.
- `end_reason`: `agent_finished`, `cap:requests`, `cap:episode_seconds`, `model server error`, `agent run failed`
  or `setup failed`. `error` carries the message.
- `final_reply` and `turns` (each user message with the agent's reply). When the agent ends with GRC-Agent's
  structured response, the reply is its explanation followed by "Actions taken".
- `tool_calls`: each call's tool, the arguments as the model wrote them, the outcome (`success`, `failed`,
  `denied`, `retry`) and the result the model saw. `approvals` lists every approval decision.
- `flowgraph_runs`: command, return code, seconds, whether it was stopped, and whether the hard limit was hit.
- `graph`:
  - `file_in_project`: the flowgraph's file in `project/`, or null when untitled;
  - `file_matches_graph`: whether that file holds the same graph as memory;
  - `valid` and `errors`.

  GRC-Agent's `change_graph` writes each edit straight to the flowgraph's file when it has one.
- `files`: added, changed and removed files compared with the start of the call. GRC-Agent's own
  `.grc_agent/` lock files are left out.
- `usage` for the call and `session_totals` for the session; `model`, `policy`, `caps`.
- `agent`: what was swapped out to run headless ([section 8](#8-how-it-works-and-how-it-differs-from-the-desktop-app)).
- `versions`: the GNU Radio version and GRC-Agent commit.

## 6. Safety: approvals instead of a sandbox

grc-agent-headless approves everything because it runs the agent inside a Linux sandbox with no network.
macOS has no equivalent, so this adapter runs as you, on your Mac, and answers GRC-Agent's approval requests
with a policy instead:

| Request | Default | Option |
|---|---|---|
| `change_graph` (edit the flowgraph) | approved | `--no-graph-edits` denies it |
| `run_flowgraph` start (run the generated program) | approved | `--no-run` denies it |
| `run_command` / `start_command` (shell) | **denied** | `--allow-shell` approves them |
| web search / fetch | **offline**: the tools answer "offline" | `--allow-web` uses GRC-Agent's real web tools |

A denied request reaches the model as a denied tool result ("Denied by the grc-agent-adapter approval
policy: ..."), and the agent carries on without it.

What the agent can reach:
- **Files:** GRC-Agent's file tools are limited to the session's `project/` folder.
- **Flowgraphs:** they really run: GRC's generated Python, started with GRC's own run command, in the folder
  of the saved graph. Qt GUI windows are drawn off-screen. An attached SDR **would** be used, so unplug radio
  hardware or use `--no-run` when you don't want a real transmission.
- **Hier-block library:** `save_block` exports to GNU Radio's library in your home folder, as the desktop app
  does.
- **Shell commands:** only with `--allow-shell`. They then run in `project/`, with GRC-Agent's own denylist,
  and without your API keys in their environment.

For question-answer evaluation, use `--no-graph-edits --no-run` and leave shell and web off. The agent can
then only read: inspect the empty flowgraph, search its catalog and docs, and preview code.

## 7. Limits

| Option | Default | Meaning |
|---|---|---|
| `--max-requests N\|none` | 40 | model requests per call; hitting it ends the call with `cap_failure` (`cap:requests`) |
| `--episode-seconds S` | 1800 | wall-clock limit for one call (`cap:episode_seconds`) |
| `--flowgraph-seconds S` | 300 | hard limit for one flowgraph run, including background runs; the process group is killed |
| `--no-compaction` | GRC-Agent's compaction on | turn GRC-Agent's history compaction off, as grc-agent-headless's `benchmark` preset does |

## 8. How it works, and how it differs from the desktop app

The adapter calls GRC-Agent's own `agent_factory.build_agents_from_cfg` and keeps the executor. The system
prompt, tools, output validator, retries and capabilities are GRC-Agent's own objects. Four things are swapped
for the duration of that call, all recorded in `summary.json` → `agent.substitutions`:

1. `grc_agent.native_canvas` imports GTK. A GTK-free stand-in module provides the deps class instead, and GTK is
   never loaded.
2. The web tools are replaced by offline copies with the same name and schema, unless `--allow-web` is given.
3. With `--no-compaction`, compaction is replaced by a no-op.
4. GNU Radio's headless platform is built with `install_prefix=gr.prefix()`. The conda-forge GNU Radio on macOS
   requires this argument, and GRC-Agent's headless `get_platform()` leaves it out, which crashes on this Mac.
   The adapter builds the platform correctly and hands it to GRC-Agent's cache. The grc-agent repo is not
   edited.

The desktop canvas is replaced by `MacFlowgraphDeps` (`src/grc_adapter/deps.py`). It keeps GRC-Agent's gate
messages, result keys, save naming (`resolve_save_target`, the untitled id rename), the hash-equal no-op, GRC's
serializer, the atomic write and the `.grc_agent/<name>.lock`. The differences that remain:

| | Desktop app | This adapter |
|---|---|---|
| Running a flowgraph | GRC's Executor thread, output in GRC's console | a child process in its own process group, started with GRC's own run command; Qt drawn off-screen |
| Run log | GRC console messages | the program's own stdout + stderr (UTF-8, at most 512 KB, oldest text dropped and flagged) |
| "Prompt for Exit" (no_gui) programs | run in an xterm until you press Enter | stdin is an open pipe nobody writes to, so they run until stopped, as in the terminal |
| Run time | no limit | `--flowgraph-seconds` hard limit |
| Invalid-graph gate | joins GRC's `(element, message)` pairs as strings (a TypeError on real graphs) | formats them as `element: message` |
| Canvas state | redraw, fit-to-view, undo, tabs, recent files | none (one page, no window) |

Compared with grc-agent-headless:
- It isn't pinned to commit `6c98128`, and it uses your GRC-Agent checkout's current tools and prompt. Record
  `versions.grc_agent_commit` with your results.
- It has no sandbox; section 6 covers what replaces it.
- It has no GRC-Bench task records or graders.
- It adds `batch`.
- It has no scripted `--reply` / `--auto-reply` answers; use `--then` or `continue`.

## 9. Using it from another program

- **From any language:** run `./agent ... --json` and parse standard output, or read the session folder.
- **From Python:** use GRC-Agent's interpreter (`~/Projects/grc-agent/.venv/bin/python`) with
  `PYTHONPATH=grc-agent-adapter/src`, `QT_QPA_PLATFORM=offscreen` and `GRC_ADAPTER_HOME` set (the `agent`
  script sets all of them):

```python
from pathlib import Path
from grc_adapter import session as s

spec = s.ModelSpec(provider="cohere", model="north-mini-code-1-0")      # COHERE_API_KEY in the environment
summary = s.run_call(Path("runs/demo"), "Which block makes a cosine tone?", spec=spec,
                     policy=s.Policy(allow_graph_edits=False, allow_run=False), caps=s.Caps())
print(summary["status"], summary["final_reply"])
```

Run one call at a time per process: GRC-Agent's file-tool wiring and its settings file are process-wide.

## 10. Troubleshooting

| Message | Cause and fix |
|---|---|
| `GRC-Agent's Python was not found` | set `GRC_AGENT_REPO` to your checkout, or check that `~/Projects/grc-agent/.venv` exists |
| `Config.__init__() missing 1 required positional argument: 'install_prefix'` | something called GRC-Agent's `get_platform()` before the adapter installed its platform; use `./agent`, not GRC-Agent's scripts |
| `refused: provider cohere needs COHERE_API_KEY` | add the key to `.env` (section 2) or `export COHERE_API_KEY=...` |
| status `infrastructure_error`, `ModelAPIError: Connection error.` | the model server is unreachable: check `--base-url`, the network, or the provider's status |
| status `infrastructure_error`, `ModelHTTPError` 401 / 403 | wrong or expired key |
| status `infrastructure_error`, `ModelHTTPError` 429 | rate limit (trial keys, OpenRouter free models): wait and rerun; `batch` resumes |
| status `cap_failure` | a limit was hit (`end_reason` names it): raise it (section 7) |
| `query_knowledge` fails | the knowledge databases are missing: run `bash setup.sh` |
| `run_flowgraph` says the flowgraph was never saved | GRC's rule: a flowgraph must be saved once before it runs; the agent normally calls `save_graph` first |
