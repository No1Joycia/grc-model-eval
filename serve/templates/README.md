# Chat templates

| File | What it is |
|---|---|
| `nanbeige42.original.jinja` | Nanbeige4.2-3B's own template, copied unchanged from `tokenizer_config.json` at revision `b82e54bd`. Kept for comparison only. |
| `nanbeige42.jinja` | The template every host uses (`chat_template` in `../models.json`): the original with one change. |

## Why Nanbeige needs a changed template

GRC-Agent sends **two system messages** at the start of every conversation:
1. its instructions (about 4,400 characters);
2. the active-flowgraph context (about 180 characters).

Nanbeige's own template accepts only one system message and raises
`System message must be at the beginning.` for the second. llama-server then answers HTTP 500, and GRC-Agent
retries and reports `ModelAPIError: Connection error.` The same happens on every server that uses the
model's template (llama.cpp, vLLM, SGLang), and with the GRC-Bench headless harness, because the messages
come from GRC-Agent.

**The change:** system messages at the start of the conversation are joined into one, separated by a blank
line. Nothing else changes.

Checked on 2026-10-31 with llama.cpp b12020's `/apply-template`, comparing the two files:

| Conversation | Result |
|---|---|
| One system message plus a user message, with and without tools | Identical prompt |
| No system message, with tools | Identical prompt |
| Multi-turn with a tool call and a tool result | Identical prompt |
| Two leading system messages (GRC-Agent) | The original fails; this one merges them |
| A system message after the conversation has started | Still rejected, same as the original |

To redo the check, render the same conversations through `llama-server --chat-template-file <file>`'s
`/apply-template` endpoint for both files, and diff the prompts.

If GRC-Agent ever sends a single system message, the change becomes a no-op and the original template could
be used again.
