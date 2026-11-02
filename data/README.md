# Dataset preparation and compatibility

The current **local** prepared release is `dataset/qoherent_v3/`. Begin with that folder's README and its top-level `train.jsonl`, `validation.jsonl` and `test.jsonl`. The data folder is already git-ignored; do not commit it. Shared question data still follows the Google Drive workflow in the repo README.

**Teammate reproduction:** [REPRODUCE.md](REPRODUCE.md) explains both using the exact prepared release and rebuilding from the four original input folders with one `data/rebuild.py` command. `prepare.py` alone performs only the last v2-to-v3 stage.

The release uses one model-neutral format: an ID, a conversation and function-tool definitions. Function arguments are objects, text content is always a string, and no model-specific chat tokens are baked in. Sidecars retain source IDs, split groups and message-level loss masks. Q&A uses the same domain system prompt as the current evaluation runner. Models other than the current two baselines can consume the same records through their own chat templates.

For a lightweight reader without model downloads or Arrow schema coercion:

```python
from data.loader import iter_split
for example, metadata in iter_split("dataset/qoherent_v3", "train"):
    messages, tools = example["messages"], example["tools"]
    assistant_loss_mask = metadata["assistant_loss_mask"]
    # Apply the chosen tokenizer/template and verified token loss mask here.
```

The optional `kind="qna"` or `kind="trace"` filter supports separate experiments without maintaining extra copies. `include_recovery=True` adds the 264 training recovery sessions once. Nested tool arguments naturally vary by tool; the loader preserves Python objects without forcing every nested field into a uniform Arrow column type.

## Prepare and verify

From the repository root, with `../processed/qoherent_v2` available:

```powershell
python -m pip install -r data/requirements.txt
python data/prepare.py --source ../processed/qoherent_v2 --output dataset/qoherent_v3
python data/validate.py --dataset dataset/qoherent_v3
```

The builder refuses to overwrite an existing release. For a rebuild, use a new version directory. It checks v2's deliverable hashes and the original data inputs, preserves splits, and creates evaluator-compatible question files. It does not run inference or training. Review/quarantine records remain outside the training exports. Original data and the v2 package remain intact.

## Optional native-template compatibility checks

```powershell
python data/fetch_metadata.py --output dataset/model_metadata
python data/render_check.py --dataset dataset/qoherent_v3 --metadata dataset/model_metadata
python -m unittest discover -s data/tests -v
```

Metadata fetches download only public configuration and template files, never model weights or remote Python code. Nanbeige uses the exact revision pinned in `serve/models.json` and the repo's two-system-message template fix. North uses a resolved, recorded HF revision. The downloaded files and hashes stay local. Rendering uses a sandboxed Jinja environment and does not load a model.

Format checks do not establish that a full training stack works. Neither checked native template defines assistant-generation spans, so the future trainer needs verified loss masking. North's native template omits visible content on assistant messages that also contain tool calls. The canonical data retains that content, and the compatibility report exposes the omission rather than silently deleting or relabeling it. A model-specific trainer must make this supervision decision explicitly.

No tokenizer lengths, sequence chunking, token/source balancing, trainer, optimizer configuration, LoRA training or GPU job are introduced here. These remain model-specific preparation steps, as agreed. A GGUF evaluation file is a deployment artifact; fine-tuning uses the original HF checkpoint and its compatible training stack, then exports/merges for the repo's existing GGUF evaluation path. The Cohere API evaluator is not a local fine-tuning implementation.

## Files

- `prepare.py`: normalization, packaging, source checks and evaluation export.
- `rebuild.py`: one entry point for the complete original-data-to-v3 pipeline.
- `legacy/`: the earlier cleaning/merge scripts, packaged here so rebuilding no longer depends on code outside this repo.
- `loader.py`: streaming examples plus aligned metadata, with optional type filtering/recovery.
- `validate.py`: schema, source equivalence, pairing, masks, split and current notebook-contract checks.
- `schema.json`: portable conversational record schema.
- `fetch_metadata.py`: public baseline configuration/template fetcher.
- `render_check.py`: native-template serialization checks without weights.
- `tests/`: synthetic regression cases for serialization and message alignment.

The existing Nanbeige/North Q&A readers and scorer now use explicit UTF-8 file I/O so these files work on Windows as well as Colab/Linux. No scorer formulas, model settings or inference calls are changed by that encoding fix.

References: [Nanbeige model](https://huggingface.co/Nanbeige/Nanbeige4.2-3B), [North model](https://huggingface.co/CohereLabs/North-Mini-Code-1.0), [Transformers tool-use format](https://huggingface.co/docs/transformers/chat_extras), and the local `serve/templates/README.md`.
