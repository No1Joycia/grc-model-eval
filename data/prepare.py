"""Create a model-neutral, local-only release from the verified Qoherent v2 dataset.

No model inference, model training, or source-data mutation. Standard library only.
"""
from __future__ import annotations

import argparse
import ast
import collections as C
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def rows(path):
    with path.open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def system_prompt(repo):
    tree = ast.parse((repo / 'trial_eval/run_trial.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SYSTEM_PROMPT' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Q&A evaluation SYSTEM_PROMPT not found')


def normalize(record, mask, qa_prompt):
    """Keep semantic data; normalize JSON representation and leading instructions."""
    row = copy.deepcopy(record)
    messages, mask = row['messages'], list(mask)
    origin = list(range(len(messages)))
    changes = C.Counter()
    for m in messages:
        m.pop('weight', None)  # Failure supervision lives in the aligned sidecar.
        assert not any(k in m for k in ['thinking', 'reasoning', 'reasoning_content'])
        if m.get('content') is None:
            m['content'] = ''
            changes['null_content_to_empty_string'] += 1
        assert isinstance(m['content'], str)
        for call in m.get('tool_calls', []):
            f = call['function']
            if isinstance(f['arguments'], str):
                f['arguments'] = json.loads(f['arguments'])
                changes['tool_arguments_parsed_to_objects'] += 1
            assert isinstance(f['arguments'], dict)
    if not row.get('tools') and not any(m['role'] == 'system' for m in messages):
        messages.insert(0, dict(role='system', content=qa_prompt))
        mask.insert(0, 0)
        origin.insert(0, None)
        changes['qa_domain_system_prompt_added'] += 1
    count = 0
    while count < len(messages) and messages[count]['role'] == 'system':
        count += 1
    if count > 1:
        messages[:count] = [dict(role='system', content='\n\n'.join(m['content'] for m in messages[:count]))]
        mask[:count] = [0]
        origin[:count] = [list(range(count))]
        changes['leading_system_messages_merged'] += 1
    assert not any(m['role'] == 'system' for m in messages[1:])
    return dict(messages=messages, tools=row.get('tools', [])), mask, origin, changes


def export(source, destination, relative, output_name, qa_prompt, qa, traces, variants):
    input_path = source / relative
    indices = rows(input_path.with_name(input_path.stem + '_index.jsonl'))
    normalized, metadata = [], []
    counts = C.Counter()
    raw = rows(input_path)
    assert len(raw) == len(indices)
    for line, (record, idx) in enumerate(zip(raw, indices), 1):
        row, mask, origin, changes = normalize(record, idx['assistant_loss_mask'], qa_prompt)
        rid = idx['example_id']
        detail = qa.get(rid) or traces.get(rid) or variants.get(rid)
        assert detail is not None
        row = dict(id=rid, **row)
        normalized.append(row)
        meta = dict(id=rid, row=line, kind=idx['record_type'], source_set=idx['source_set'],
            split=idx['split'], group_id=idx['group_id'], assistant_loss_mask=mask,
            message_origin_indices=origin, sample_weight=idx.get('sample_weight', 1),
            source_export=relative, source_row=line,
            source=detail.get('source'), question_id=detail.get('question_id'),
            scenario_id=detail.get('scenario_id'), task_family=detail.get('task_family'),
            category=detail.get('category'), quality_lane=detail.get('quality_lane', 'core'),
            transformations=dict(changes))
        metadata.append(meta)
        counts.update(changes)
    write_rows(destination / output_name, normalized)
    metadata_name = 'metadata/' + output_name.replace('/', '__')
    write_rows(destination / metadata_name, metadata)
    return dict(file=output_name, metadata=metadata_name, records=len(normalized),
        kinds=dict(C.Counter(m['kind'] for m in metadata)), transformations=dict(counts))


def evaluation_question(q, answer, references, reference_model, is_gnuradio=None):
    label = q['labels']
    return dict(id=q['question_id'], question=q['question'], difficulty=q['difficulty'],
        task_type=label['task_type'], primary_topic=label['topic_raw'],
        is_gnuradio=(label['task_type'] == 'gnuradio_usage' or 'gnu radio' in q['question'].lower() or 'gnuradio' in q['question'].lower()) if is_gnuradio is None else is_gnuradio,
        reference_answer=answer, reference_model=reference_model,
        references=[dict(model=r['model'], answer=r['answer']) for r in references],
        reference_quality='supplied_teacher_judgments_not_independent_expert_ground_truth')


def build(repo, source, output):
    assert source.is_dir()
    assert repo / 'dataset' in output.parents and output != repo / 'dataset'
    assert not output.exists(), 'Use a new version directory instead of overwriting a used release'
    for line in (source / 'checksums.sha256').read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        assert sha(source / name) == expected, 'Changed v2 artifact: ' + name
    # A git pull legitimately changes supporting evaluation code. Verify actual
    # original data inputs, not obsolete hashes of the old repo README/scripts.
    workspace = source.parents[1]
    dataset_inventory = [r for r in rows(source / 'audit/source_inventory.jsonl')
        if r['path'].split('/')[0] in {'GRC_dataset', 'GRC_Agent_data', 'NewData1', 'NewData2'}
        or r['path'] in {'nanbeige-modal/trial_eval/trial_eval_100.jsonl', 'nanbeige-modal/trial_eval/trial_eval_ids.txt'}]
    assert all(sha(workspace / r['path']) == r['sha256'] for r in dataset_inventory), 'Original dataset input changed'
    qa_rows = rows(source / 'qna/questions.jsonl')
    qa = {r['example_id']: r for r in qa_rows}
    question_map = {r['question_id']: r for r in qa_rows}
    traces = {r['example_id']: r for r in rows(source / 'traces/records.jsonl')}
    variant_rows = rows(source / 'qna/answer_variants.jsonl')
    variants = {r['example_id']: r for r in variant_rows}
    prompt = system_prompt(repo)
    output.mkdir(parents=True)
    exports = []
    for split in ['train', 'validation', 'test']:
        exports.append(export(source, output, f'sft/{split}.jsonl', f'{split}.jsonl', prompt, qa, traces, variants))
    exports.append(export(source, output, 'traces/train_recovery_masked.jsonl', 'optional/recovery_train.jsonl', prompt, qa, traces, variants))
    for split in ['train', 'validation']:
        exports.append(export(source, output, f'qna/{split}_all_answers.jsonl', f'optional/qna_all_answers_{split}.jsonl', prompt, qa, traces, variants))
    refs = {r['question_id']: r for r in rows(source / 'qna/test_references.jsonl')}
    trial = {r['id']: r for r in rows(repo / 'trial_eval/trial_eval_100.jsonl')}
    for split in ['validation', 'test']:
        output_rows = []
        for q in qa_rows:
            if q['split'] != split:
                continue
            if split == 'test':
                r = refs[q['question_id']]
                answer, references, model = r['reference_answer'], r['references'], r['reference_model']
            else:
                answer = q['messages'][-1]['content']
                candidates = [a for a in variant_rows if a['question_id'] == q['question_id'] and a['eligible']]
                references = [dict(model=a['candidate_model'], answer=a['answer']) for a in candidates]
                model = next(a['candidate_model'] for a in candidates if a['answer'] == answer)
            output_rows.append(evaluation_question(q, answer, references, model,
                trial[q['question_id']]['is_gnuradio'] if q['question_id'] in trial else None))
        write_rows(output / f'evaluation/questions_{split}.jsonl', output_rows)
    frozen = []
    for q in qa_rows:
        if q['split'] == 'test_unscored':
            frozen.append(dict(id=q['question_id'], question=q['question'], difficulty=q['difficulty'],
                task_type=q['labels']['task_type'], primary_topic=q['labels']['topic_raw'],
                evaluation_status='questions_only_no_graded_reference'))
    write_rows(output / 'evaluation/questions_frozen_unscored.jsonl', frozen)
    # Preserve held-out tasks and provenance without republishing raw transcripts.
    shutil.copyfile(source / 'traces/test_tasks.jsonl', output / 'evaluation/trace_test_tasks.jsonl')
    write_rows(output / 'evaluation/reserved_question_ids.jsonl', [dict(id=q['question_id'], group_id=q['group_id'], disposition=q['split']) for q in qa_rows if q['split'].startswith('test')])
    write_rows(output / 'provenance/split_assignments.jsonl', rows(source / 'audit/split_assignments.jsonl'))
    write_rows(output / 'provenance/trace_exclusions.jsonl', [dict(id=r['example_id'], scenario_id=r['scenario_id'], source_set=r['source_set'], errors=r['structural_errors']) for r in rows(source / 'audit/trace_quality.jsonl') if r['quality_lane'] == 'quarantine'])
    shutil.copyfile(source / 'audit/qna_eligibility.jsonl', output / 'provenance/qna_eligibility.jsonl')
    write_rows(output / 'provenance/original_data_inventory.jsonl', dataset_inventory)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    manifest = dict(version='3.0.0', format='model-neutral conversational JSONL',
        based_on_version='2.0.0', source_directory=str(source), repository_commit=commit,
        source_manifest_sha256=sha(source / 'manifest.json'), source_checksums_sha256=sha(source / 'checksums.sha256'),
        exports=exports, split_policy='Unchanged from v2. No resampling, truncation, source weighting or conversation segmentation.',
        qa_system_prompt=prompt, qa_system_prompt_source='trial_eval/run_trial.py:SYSTEM_PROMPT',
        normalization=['Parse function.arguments from JSON strings to JSON objects.',
            'Replace null text content with empty strings.',
            'Add the existing domain Q&A system prompt to match evaluation.',
            'Merge consecutive leading system messages if supplied.',
            'Keep failed-attempt and sample weights in metadata rather than assuming a trainer honors weight fields.'],
        model_policy='Apply the chosen model tokenizer/chat template at training time. No model-specific serialized tokens are embedded in the dataset.',
        privacy='Local dataset directory is git-ignored. No data upload, commit or push performed.',
        limitations=['Conversation lengths are not tokenized or balanced here.',
            'Reference answers retain supplied teacher judgments, not new independent expert validation.',
            'Trace tasks require original fixtures and harness for faithful live execution.',
            'Native model templates can differ in which assistant fields they serialize.'])
    write_json(output / 'manifest.json', manifest)
    write_code_snapshot(repo, output)
    shutil.copyfile(repo / 'data/schema.json', output / 'schema.json')
    (output / 'README.md').write_text(dataset_readme(manifest), encoding='utf-8')
    return manifest


def write_code_snapshot(repo, output):
    paths = [p for p in (repo / 'data').rglob('*') if p.is_file() and p.suffix in {'.py', '.json', '.md', '.txt'}]
    paths += [repo / name for name in ['README.md', 'trial_eval/run_trial.py', 'trial_eval/score_trial.py', 'northMiniCode/run_raw.py', 'colab/eval.ipynb', 'serve/models.json', 'serve/templates/nanbeige42.jinja']]
    write_json(output / 'provenance/code_snapshot.json', [dict(file=p.relative_to(repo).as_posix(), sha256=sha(p)) for p in sorted(paths)])


def dataset_readme(manifest):
    return '''# Qoherent cleaned dataset v3

## Start with these three files

| File | Q&A | Agent conversations | Total |
|---|---:|---:|---:|
| `train.jsonl` | 1,402 | 1,200 | 2,602 |
| `validation.jsonl` | 148 | 231 | 379 |
| `test.jsonl` | 100 | 169 | 269 |

This is a model-neutral release. Each JSONL line is one complete conversation with `id`, `messages` and `tools`. Q&A rows have an empty tools list. Tools use the common function-tool schema. Function arguments are JSON objects, not double-encoded strings. Empty assistant tool-call content is `""`, not null. Training data contains no added model-specific chat delimiters, token IDs, embeddings or model weights. Message content can include technical prose, code and observed tool output.

```json
{"id":"illustrative_only","messages":[{"role":"system","content":"Domain instructions"},{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}],"tools":[]}
```

## What was changed from v2?

The questions, answers, tool actions and split/group memberships are unchanged. V3 parses tool arguments, normalizes null content and adds the same RF/DSP system prompt that the current Q&A evaluation uses. It shifts the companion message masks when adding that system message. Original sources and previous releases remain intact. Original case IDs, source pointers and transformations are recorded in `metadata/` and `provenance/`. Preparation does not assign equal training influence to long and short conversations.

## How to fine-tune a model

Use the repo's streaming reader to preserve heterogeneous tool arguments without assuming an Arrow schema:

```python
from data.loader import iter_split
for example, metadata in iter_split("dataset/qoherent_v3", "train"):
    messages, tools = example["messages"], example["tools"]
    assistant_loss_mask = metadata["assistant_loss_mask"]
    # Apply the selected tokenizer/template and token-level loss masking.
```

Use `kind="qna"` or `kind="trace"` for a filtered experiment. Use `include_recovery=True` only when implementing the failure masks. This reader loads data, not model weights, and performs no training.

1. Load the selected JSONL file. Keep `messages` and `tools` for the model chat template; `id` is a lookup key, not prompt text.
2. Apply that model's tokenizer/chat template with `add_generation_prompt=False` for complete training conversations. Leave serialization to the template. If a deployment API requires JSON-string arguments, serialize them in that API adapter, not in this canonical dataset.
3. Compute training loss on positive assistant messages, including function names and arguments. Mask system/user/tool context. `metadata/train.jsonl` has the same row order and includes an `assistant_loss_mask` aligned to message positions: 1 = target, 0 = context only.
4. Templates without assistant generation spans need a verified custom loss mask or training-only template instrumentation. `assistant_only_loss=True` does not create missing template support. A zero assistant token mask must fail the training preflight.
5. Choose context length and source sampling after measuring the target tokenizer's input and assistant target tokens. No truncation, chunking or balancing is claimed by this release.

Default training contains only core examples. Validation/test contain recovery contexts with context-only failed attempts: use their metadata masks for teacher-forced evaluation. A model's visible assistant text plus a tool call may serialize differently in different templates. In particular, the checked North template renders actions but omits visible text on messages containing tool calls. Canonical data preserves both. Read the compatibility report before selecting a template or supervision policy.

## Optional training experiments

- `optional/recovery_train.jsonl`: 264 additional training conversations. Append once to the default training set only if the trainer implements its companion failed-attempt masks. Preserve full context and do not apply loss to rejected attempts.
- `optional/qna_all_answers_train.jsonl`: 9,288 accepted answers to training questions. This replaces the Q&A portion of default training for an alternatives experiment; do not blindly append it to the default file. Metadata includes question-level sample weights.
- `optional/qna_all_answers_validation.jsonl`: matching validation alternatives. Do not train on it.

## Files for the current evaluation repo

- `evaluation/questions_test.jsonl`: 100 Q&A questions in the exact `id`, `question`, `difficulty`, `task_type`, `is_gnuradio`, `reference_answer`, `references` interface used by Colab and the scorer. This is the file to set as Colab `QUESTIONS` for final Q&A comparison. It contains the same held-out question IDs as the existing trial benchmark, with vetted references.
- `evaluation/questions_validation.jsonl`: 148 development questions in the same interface. Use this while changing settings or training checkpoints.
- `evaluation/questions_frozen_unscored.jsonl`: 185 additional protected questions without graded references. Answer generation works, but the existing reference scorer cannot score them. Do not substitute fabricated references.
- `evaluation/trace_test_tasks.jsonl`: 169 held-out task descriptions, recorded snapshots and expected diffs. These are not standalone executable GRC fixtures. Live evaluation needs the original fixtures/harness and the model's own generated actions, not replayed reference actions.
- `evaluation/reserved_question_ids.jsonl`: all Q&A tests and related heldouts excluded from training.

The top-level `test.jsonl` is a conversational reference dataset. It is not the question-file interface consumed by `trial_eval/run_trial.py`; use `evaluation/questions_test.jsonl` there.

From the repository root, these commands only estimate an evaluation and make no model calls:

```powershell
python trial_eval/run_trial.py --questions dataset/qoherent_v3/evaluation/questions_test.jsonl --estimate-only
python northMiniCode/run_raw.py --questions dataset/qoherent_v3/evaluation/questions_test.jsonl --estimate-only
```

For a real Q&A run add the server/provider settings and a separate answers path, then score it with the same `--questions` file. The Colab notebook still takes data from your Drive shortcut. Copy the required evaluation file to your shared data folder yourself; this task did not upload anything.

## Directory guide and limits

`metadata/` contains aligned IDs, source/group/split information, masks and transformation counts. It is not extra training data. `provenance/` explains exclusions and fingerprints the original inputs. `schema.json` defines the portable format. `manifest.json` records the source release and current repo commit. Checksums identify this exact release. There is no SQLite copy or duplicated archive in this folder.

Single-task, two-turn and longer agent sessions share one format. The 1,200 core training conversations include five original sessions, 795 single-task scenarios and 400 two-turn scenarios. The optional 264 recovery sessions bring eligible training conversations to 1,464. Related scenarios stay in one split; the existing frozen evaluation questions and all answer variants remain outside training.

Existing judgments and recorded graph validation support selection. RF correctness, runtime success, target-model tokenization and gradient/loss behavior still require their own checks. Template rendering can be checked without downloading or running model weights. This is a portable dataset, not a complete training implementation.

The entire `dataset/` directory is already git-ignored. Preserve that rule. Version releases by directory rather than editing a dataset after it has been used. The preparation code is tracked under `data/`; original inputs remain outside this repo.
'''


def checksums(output):
    files = sorted(p for p in output.rglob('*') if p.is_file() and p.name != 'checksums.sha256')
    (output / 'checksums.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(output).as_posix()}\n' for p in files), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('../processed/qoherent_v2'))
    parser.add_argument('--output', type=Path, default=Path('dataset/qoherent_v3'))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest = build(repo, args.source.resolve(), args.output.resolve())
    from validate import validate
    result = validate(repo, args.output.resolve(), verify_checksums=False)
    write_json(args.output / 'verification.json', result)
    checksums(args.output.resolve())
    print(json.dumps(dict(status=result['status'], exports=manifest['exports']), indent=2))


if __name__ == '__main__':
    main()
