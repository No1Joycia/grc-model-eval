# Cookbook: Nanbeige4.2-3B on your own machine with llama.cpp

This is the setup an offline GRC-Agent user would have: the model runs on the laptop, and GRC-Agent talks to
it at `http://127.0.0.1:8080/v1`, with no internet needed.

**Tested on 2026-10-31:** MacBook Air M3 (16 GB), macOS 15.0.1, llama.cpp build **b12020**.
The Windows/Linux sections and calling North-Mini-Code through Cohere are still to come (plan step 5).

All commands run from the repo root. Settings live in one file, [`serve/models.json`](../serve/models.json),
which Modal and Qoherent's GPU also read, so every host runs the model the same way.

---

## 1. Install llama.cpp (the pinned build)

```bash
bash serve/install_llamacpp.sh
```

This downloads llama.cpp's own release for this machine into `~/llama.cpp/b12020/`, a 15 MB download.

**Why not Homebrew?**
- Homebrew's llama.cpp is release 0.8.0, whose Metal GPU code doesn't compile on macOS 14 and 15. It fails
  with `constexpr function never produces a constant expression` (llama.cpp issue #31272). Builds from b12018
  on include the fix (PR #31360).
- Every host has to run the *same* build so results can be compared. The scripts refuse to start with any
  other build.

## 2. Get the model files

Until the team Hugging Face repo exists (plan step 2), make them yourself. It takes about 3 minutes after the
download:

```bash
bash convert/setup.sh                                   # once: converter + Python environment (~1 GB)
convert/.venv/bin/hf download Nanbeige/Nanbeige4.2-3B \
    --revision b82e54bd609793562a75cbf9337970a93369eab5 --local-dir ~/models/hf/Nanbeige4.2-3B
bash convert/to_gguf.sh ~/models/hf/Nanbeige4.2-3B ~/models/nanbeige42 Nanbeige4.2-3B
```

| File in `~/models/nanbeige42/` | Size | SHA-256 (in `SHA256SUMS`) |
|---|---|---|
| `Nanbeige4.2-3B-BF16.gguf` | 7.8 GB | `6bc9e931aee6e92ce1e023610b022e06db52565c3b32a61e070904682ef2def4` |
| `Nanbeige4.2-3B-Q8_0.gguf` (default) | 4.1 GB | `a1317a0a6b913f56661934d607c8f7f3104f7254e9efa78de5eb6e700ff087e1` |
| `Nanbeige4.2-3B-Q4_K_M.gguf` | 2.4 GB | `a27b7702431b0eb6e62da987bc5f94c4e4a61d703e12d7e17e2d1230a15b9575` |

The same revision and the same build give the same files, so the checksums let anyone confirm they have the
team's files. The download needs 8.4 GB and the GGUF files 14.3 GB.

**Keep code and models out of iCloud-synced folders** (Desktop, Documents). iCloud marks files inside
`.venv` and `.git` folders as hidden. Python 3.13 then ignores the venv's `.pth` files, and iCloud also
creates conflict copies inside `.git`.

## 3. Start the server

```bash
bash serve/run_local.sh nanbeige42              # Q8_0 (default)
bash serve/run_local.sh nanbeige42 q4_k_m       # smaller and faster, slightly lower quality
```

- It's ready in about 10 seconds, when the log says `listening on http://127.0.0.1:8080`. Ctrl+C stops it.
- It only listens on this machine. To serve others on your network, use `HOST=0.0.0.0` with
  `API_KEY_ENV=<variable holding a key>`; the script refuses to open it up without a key.
- `PORT=8081` changes the port, and `GGUF=/path/file.gguf` serves another model file, such as a fine-tuned
  version.

What the settings mean (`serve/models.json`):

| Setting | Why |
|---|---|
| `-c 65536` | 64K-token context, as GRC-Bench uses: GRC-Agent's tools and system prompt alone are about 8K tokens |
| `-ctk q8_0 -ctv q8_0`, `-fa on` | 8-bit KV cache with flash attention: a 64K context takes about 3.1 GB instead of 6 |
| `--jinja`, `--chat-template-file serve/templates/nanbeige42.jinja` | Use the model's own chat template, which is needed for tool calls. The file is Nanbeige's template changed only to accept GRC-Agent's two system messages: see [`serve/templates/README.md`](../serve/templates/README.md) |
| `--reasoning-format deepseek` | The model's thinking comes back separately in `reasoning_content`, not mixed into the answer |
| `--no-context-shift` | When the context is full, fail clearly instead of silently dropping the start of the conversation |
| `-np 1` | One request at a time, as GRC-Bench uses |
| `-ngl all` | Put every layer on the GPU (Metal on a Mac) |
| `--temp 0.6 --top-k 20 --top-p 0.95` | Defaults from the model's `generation_config.json`. A client that sends its own temperature overrides them. |

**Memory:** Q8_0 uses about 4.1 GB for weights plus 3.1 GB for the context, about 7.5 GB in all. That fits a
16 GB Mac next to GRC-Agent. Use Q4_K_M on an 8 GB machine.

## 4. Check that it works

In a second terminal:

```bash
python3 serve/smoke_test.py
```

```
ok    health: {'status': 'ok'} (0 s)
ok    models: ['nanbeige42']
ok    chat: 'GNU Radio is an open-source software framework for software-defined radio (SDR) ...' (84.9 s, 7.0 tok/s generation, 64 tok/s prompt)
      reasoning: 2568 chars, kept out of content
ok    tool call: finish_reason=tool_calls, 1 call(s) (22.4 s): get_block_info({"block_id": "analog_sig_source_x"})
all checks passed
```

The one-sentence answer takes over a minute because the model thinks first. That reasoning is about 2,600
characters, and it's on by default (section 6).

## 5. Use it from GRC-Agent

**Headless adapter (tested):**

```bash
cd grc-agent-adapter
./agent run "Build a flowgraph: a Signal Source into a Throttle into a Null Sink. Save it as tone.grc." \
    --base-url http://127.0.0.1:8080/v1 --model nanbeige42 --no-run
```

**Desktop app (not tested yet):** in GRC-Agent's Settings, choose the OpenAI-compatible provider, set its
base URL to `http://127.0.0.1:8080/v1` and the model to `nanbeige42`. No API key is needed for a local server.

## 6. Thinking on or off

Nanbeige thinks before answering by default, as its model card recommends for agent work. To turn it off
for one request, which gives faster, shorter answers for plain Q&A, add this to the request body:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

## 7. Speed on this Mac

(Filled in from `llama-bench` on the test machine.)

## 8. Troubleshooting

| What you see | Cause and fix |
|---|---|
| `ggml_metal_init: error: failed to initialize the Metal library`, `constexpr function never produces a constant expression` | llama.cpp 0.8.0 (Homebrew) on macOS 14/15. Use `bash serve/install_llamacpp.sh`, which installs the pinned b12020. |
| GRC-Agent reports `ModelAPIError: Connection error.`, but `curl http://127.0.0.1:8080/health` answers | Look in the server log for `System message must be at the beginning`: the adjusted template wasn't loaded. Start the server with `serve/run_local.sh`, which passes it. |
| GRC-Agent fails with `cannot import name 'settings' from 'grc_agent'` or `No module named 'grc_agent.desktop_app'` | The GRC-Agent checkout is in an iCloud-synced folder, and its venv files were marked hidden. Move it out of Desktop/Documents, or run `chflags nohidden` on the `.pth` files in `.venv/lib/python3.13/site-packages/`. |
| `llama-server ... is build bNNNN, serve/models.json pins b12020` | Another llama.cpp is in the way. Run `bash serve/install_llamacpp.sh`, or point `LLAMA_CPP_DIR` at the pinned build. |
| `address already in use` | Another server is on port 8080: stop it, or use `PORT=8081`. |
| The Mac runs out of memory | Use `q4_k_m`, or lower `-c` in `serve/models.json`. That changes it for every host, so agree it with the team. |
