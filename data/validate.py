"""Verify portable exports, source equivalence, masks, evaluation contract and splits."""
from __future__ import annotations

import argparse
import ast
import collections as C
import json
from pathlib import Path
import re
import subprocess

import jsonschema

from prepare import rows, sha


def validate(repo, dataset, verify_checksums=True, verify_source=True):
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    schema = json.loads((dataset / 'schema.json').read_text(encoding='utf-8'))
    validator = jsonschema.Draft202012Validator(schema)
    source = Path(manifest['source_directory'])
    source_available = verify_source and (source / 'manifest.json').exists()
    if source_available:
        assert sha(source / 'manifest.json') == manifest['source_manifest_sha256']
    checks = []
    def check(value, label):
        assert value, label
        checks.append(label)
    primary = {}
    primary_metadata = {}
    group_splits = C.defaultdict(set)
    primary_question_ids = {}
    schema_cache = {}
    checked, calls_checked, failures_masked = 0, 0, 0
    for export in manifest['exports']:
        data = rows(dataset / export['file'])
        metadata = rows(dataset / export['metadata'])
        assert len(data) == len(metadata) == export['records']
        assert len({r['id'] for r in data}) == len(data)
        old = rows(source / metadata[0]['source_export']) if metadata and source_available else None
        for number, (row, meta) in enumerate(zip(data, metadata), 1):
            validator.validate(row)
            assert row['id'] == meta['id'] and meta['row'] == number
            mask = meta['assistant_loss_mask']
            assert len(mask) == len(row['messages'])
            assert all(v in {0, 1} and (v == 0 or m['role'] == 'assistant') for v, m in zip(mask, row['messages']))
            assert any(mask)
            assert row['messages'][0]['role'] == 'system'
            assert not any(m['role'] == 'system' for m in row['messages'][1:])
            original = old[meta['source_row'] - 1] if old is not None else None
            for i, m in enumerate(row['messages']) if original is not None else []:
                oi = meta['message_origin_indices'][i]
                if oi is None:
                    assert m == dict(role='system', content=manifest['qa_system_prompt']) and mask[i] == 0
                    continue
                assert isinstance(oi, int), 'Current v2 source has no multiple-system merges'
                orig = original['messages'][oi]
                assert m['role'] == orig['role'] and m['content'] == (orig.get('content') or '')
                assert m.get('tool_call_id') == orig.get('tool_call_id')
                assert len(m.get('tool_calls', [])) == len(orig.get('tool_calls', []))
                for call, previous in zip(m.get('tool_calls', []), orig.get('tool_calls', [])):
                    before = previous['function']['arguments']
                    assert call['id'] == previous['id'] and call['function']['name'] == previous['function']['name']
                    assert call['function']['arguments'] == (json.loads(before) if isinstance(before, str) else before)
                if 'weight' in orig:
                    assert mask[i] == orig['weight']
            if original is not None:
                assert row['tools'] == original['tools']
            tool_map = {t['function']['name']: t['function']['parameters'] for t in row['tools']}
            pending, seen = {}, set()
            for mi, m in enumerate(row['messages']):
                if m['role'] in {'assistant', 'user'}:
                    assert not pending
                for call in m.get('tool_calls', []):
                    f = call['function']
                    assert call['id'] not in seen and f['name'] in tool_map
                    seen.add(call['id'])
                    key = json.dumps(tool_map[f['name']], sort_keys=True)
                    if key not in schema_cache:
                        schema_cache[key] = jsonschema.Draft202012Validator(tool_map[f['name']])
                    violations = list(schema_cache[key].iter_errors(f['arguments']))
                    if violations:
                        assert mask[mi] == 0
                        failures_masked += 1
                    calls_checked += 1
                    pending[call['id']] = f['name']
                if m['role'] == 'tool':
                    assert pending.pop(m['tool_call_id']) == m['name']
            assert not pending
            assert row['messages'][-1]['role'] == 'assistant' or (row['messages'][-1]['role'] == 'tool' and row['messages'][-1].get('name') == 'final_result')
            # Alternate answer exports inherit the same protected group/split.
            group_splits[meta['group_id']].add(meta['split'])
            checked += 1
        if export['file'] in {'train.jsonl', 'validation.jsonl', 'test.jsonl'}:
            split = export['file'].split('.')[0]
            primary[split], primary_metadata[split] = data, metadata
            primary_question_ids[split] = {m['question_id'] for m in metadata if m['kind'] == 'qna'}
            assert all(m['split'] == split for m in metadata)
            if split == 'train':
                assert all(mask == int(m['role'] == 'assistant') for row, meta in zip(data, metadata) for mask, m in zip(meta['assistant_loss_mask'], row['messages']))
    check(all(len(s) == 1 for s in group_splits.values()), 'no_group_crosses_train_validation_test_including_optional_exports')
    check(not primary_question_ids['train'] & (primary_question_ids['validation'] | primary_question_ids['test']), 'no_qna_id_overlap')
    reserved = {r['id'] for r in rows(dataset / 'evaluation/reserved_question_ids.jsonl')}
    train_question_ids = set()
    for export in manifest['exports']:
        if any(m in export['file'] for m in ['train.jsonl', 'recovery_train.jsonl', 'qna_all_answers_train.jsonl']):
            train_question_ids.update(m['question_id'] for m in rows(dataset / export['metadata']) if m['question_id'])
    check(not train_question_ids & reserved, 'all_reserved_question_ids_excluded_from_every_training_export')
    existing_trial = {r['id']: r for r in rows(repo / 'trial_eval/trial_eval_100.jsonl')}
    # Match the notebook's actual step-2 field contract without executing notebook setup.
    notebook = json.loads((repo / 'colab/eval.ipynb').read_text(encoding='utf-8'))
    required = None
    for cell in notebook['cells']:
        text = ''.join(cell.get('source', []))
        if 'need_answer, need_score =' in text:
            tree = ast.parse(text)
            node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Tuple) and [getattr(e, 'id', None) for e in t.elts] == ['need_answer', 'need_score'] for t in n.targets))
            a, b = ast.literal_eval(node.value)
            required = a | b
    assert required
    for split in ['validation', 'test']:
        questions = rows(dataset / f'evaluation/questions_{split}.jsonl')
        assert {q['id'] for q in questions} == primary_question_ids[split]
        assert all(required <= set(q) for q in questions)
        for q in questions:
            assert isinstance(q['is_gnuradio'], bool) and q['references']
            assert all(isinstance(r['answer'], str) and r['answer'] for r in q['references'])
            assert any(r['answer'] == q['reference_answer'] for r in q['references'])
            if split == 'test':
                assert q['id'] in existing_trial and q['question'] == existing_trial[q['id']]['question']
                assert q['is_gnuradio'] == existing_trial[q['id']]['is_gnuradio']
    check(primary_question_ids['test'] == set(existing_trial), '100_test_questions_match_current_repo_benchmark')
    check(len(rows(dataset / 'evaluation/questions_frozen_unscored.jsonl')) == 185, '185_unscored_questions_preserved')
    check(len(rows(dataset / 'evaluation/trace_test_tasks.jsonl')) == 169, '169_trace_tasks_preserved')
    check(all(meta['split'] == r['split'] for r in rows(dataset / 'provenance/split_assignments.jsonl') for metas in primary_metadata.values() for meta in metas if meta['id'] == r['example_id']), 'v2_split_memberships_unchanged')
    ignored = subprocess.run(['git', 'check-ignore', str(dataset / 'train.jsonl')], cwd=repo, capture_output=True, text=True)
    check(ignored.returncode == 0, 'local_dataset_remains_git_ignored')
    if verify_checksums:
        for line in (dataset / 'checksums.sha256').read_text(encoding='utf-8').splitlines():
            expected, relative = line.split('  ', 1)
            assert sha(dataset / relative) == expected, relative
        checks.append('all_deliverable_checksums_pass')
    return dict(status='PASS', checks=checks, conversations_checked=checked,
        tool_calls_checked=calls_checked, schema_invalid_calls_confirmed_context_only=failures_masked,
        default_split_counts={key: len(value) for key, value in primary.items()},
        evaluation_contract=sorted(required), source_equivalence='PASS' if source_available else 'SKIPPED: v2 source unavailable or source comparison disabled',
        scope='Dataset structure and split protection, with source equivalence when v2 is available; not model training or tokenization.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('dataset/qoherent_v3'))
    parser.add_argument('--skip-source-equivalence', action='store_true', help='Verify the portable release without requiring the original v2 package')
    args = parser.parse_args()
    print(json.dumps(validate(Path(__file__).resolve().parents[1], args.dataset.resolve(), verify_source=not args.skip_source_equivalence), indent=2))
