"""Independent schema, mask, source, split and export verification for v2."""
import argparse
import collections as C
import copy
import json
from pathlib import Path
import re
import sqlite3

import jsonschema
from process_data import digest, dumps, norm, read_rows, sha_file


def verify(root, out, check_checksums=True):
    checks = []
    def check(value, name):
        assert value, name
        checks.append(name)
    qs = read_rows(out / 'qna/questions.jsonl')
    traces = read_rows(out / 'traces/records.jsonl')
    records = qs + traces
    rm = {r['example_id']: r for r in records}
    schemas = json.loads((out / 'traces/tool_schemas.json').read_text(encoding='utf-8'))
    check(len(rm) == len(records), 'unique_canonical_record_ids')
    native_traces = [r for r in traces if r['source_set'] in {'tier1', 'tier2'}]
    check(len({r['scenario_id'] for r in native_traces}) == len(native_traces), 'unique_new_scenario_ids_no_export_copy_duplication')
    trial_ids = set((root / 'nanbeige-modal/trial_eval/trial_eval_ids.txt').read_text(encoding='utf-8').split())
    frozen_path = root / 'GRC_dataset/dataset/04_eval_frozen/eval_questions.jsonl'
    frozen_ids = {r['id'] for r in read_rows(frozen_path)}
    check(sha_file(frozen_path) == sha_file(out / 'qna/test_frozen_185_questions_only.jsonl'), '185_frozen_questions_byte_identical')
    check({r['question_id'] for r in qs if r['split'] == 'test'} == trial_ids and len(trial_ids) == 100, '100_referenced_test_ids_exactly_protected')
    check({r['question_id'] for r in qs if r['split'] == 'test_unscored'} == frozen_ids, 'frozen_ids_remain_unscored_test')
    group_splits = C.defaultdict(set)
    for r in records:
        if r['split'] not in {'review', 'quarantine'}:
            group_splits[r['group_id']].add('test' if r['split'].startswith('test') else r['split'])
    check(all(len(s) == 1 for s in group_splits.values()), 'no_related_group_leaks_across_train_validation_test')
    anchors = C.defaultdict(set)
    for r in traces:
        if r['split'] != 'quarantine' and r['anchor_block']:
            anchors[r['anchor_block']].add('test' if r['split'].startswith('test') else r['split'])
    check(all(len(s) == 1 for s in anchors.values()), 'anchor_block_variants_stay_in_same_split_across_tiers')
    qa_variants = read_rows(out / 'qna/answer_variants.jsonl')
    qm = {r['question_id']: r for r in qs}
    check(all(a['split'] == qm[a['question_id']]['split'] and a['group_id'] == qm[a['question_id']]['group_id'] for a in qa_variants), 'every_answer_variant_follows_question_split')
    check(all(a['split'] not in {'train', 'validation'} for a in qa_variants if a['question_id'] in trial_ids | frozen_ids), 'all_test_answer_variants_excluded_from_training_and_validation')
    answer_hash_splits = C.defaultdict(set)
    for a in qa_variants:
        if a['eligible']:
            answer_hash_splits[a['answer_sha256']].add('test' if a['split'].startswith('test') else a['split'])
    check(all(not ('train' in s and ('test' in s or 'validation' in s)) for s in answer_hash_splits.values()), 'no_exact_answer_hash_leaks_from_train_into_validation_or_test')
    source_cache = {}
    for ref in read_rows(out / 'qna/test_references.jsonl'):
        for a in ref['references']:
            location = a['answer_source']
            path = root / location['file']
            if path not in source_cache:
                with path.open(encoding='utf-8-sig') as f:
                    source_cache[path] = {i: json.loads(line) for i, line in enumerate(f, 1) if line.strip()}
            source = source_cache[path][location['line']]
            text = source.get('repaired_answer', source.get('answer', '')).replace('\r\n', '\n').replace('\r', '\n').strip()
            assert source['question_id'] == ref['question_id'] and a['answer'] == text
        q = qm[ref['question_id']]
        assert q['answer_source'] == ref['reference_answer_source']
        assert q['messages'][-1]['content'] == ref['reference_answer']
    checks.append('all_test_references_match_accepted_source_answers_and_locations')
    del source_cache
    reconciliation = read_rows(out / 'audit/checkpoint_reconciliation.jsonl')
    check(len(reconciliation) == 1853 and all(r['checkpoint_matches_master'] for r in reconciliation), 'all_1853_master_checkpoint_pairs_match')
    check(sum(r['source_set'] == 'tier1' for r in traces) == 1253 and sum(r['source_set'] == 'tier2' for r in traces) == 600, 'all_new_scenarios_accounted_for')
    audit = {r['example_id']: r for r in read_rows(out / 'audit/trace_quality.jsonl')}
    calls_validated, masked_invalid_calls, source_rows_checked = 0, 0, 0
    # Streaming input lookup retains only wire fields needed for source equivalence.
    native_by_id = {}
    for tier in [1, 2]:
        with (root / f'NewData{tier}/tier{tier}_sft_trajectories.jsonl').open(encoding='utf-8-sig') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    native_by_id[r['scenario_id']] = r['messages']
    for r in traces:
        messages = r['messages']
        mask = r['assistant_loss_mask']
        check(len(messages) == len(mask), 'mask_length:' + r['example_id'])
        assert all(not any(key in m for key in ['thinking', 'reasoning', 'reasoning_content']) for m in messages)
        assert all(v in {0, 1} and (not v or messages[i]['role'] == 'assistant') for i, v in enumerate(mask))
        if r['source_set'] in {'tier1', 'tier2'}:
            original = native_by_id[r['scenario_id']]
            assert len(original) == len(messages)
            transforms = {t['message_index']: t for t in audit[r['example_id']]['transformations'] if t['action'] == 'terminal_harness_call_to_assistant_text'}
            for mi, (a, b) in enumerate(zip(original, messages)):
                assert a['role'] == b['role']
                if mi in transforms:
                    parsed = json.loads(a['tool_calls'][0]['function']['arguments'])
                    assert b.get('content') == (a.get('content') or parsed['explanation'])
                    assert not b.get('tool_calls')
                else:
                    expected = re.sub(r'<think>.*?</think>\s*', '', a['content'], flags=re.S) if a['role'] == 'assistant' and isinstance(a.get('content'), str) else a.get('content')
                    assert expected == b.get('content')
                    assert len(a.get('tool_calls', [])) == len(b.get('tool_calls', []))
                    for x, y in zip(a.get('tool_calls', []), b.get('tool_calls', [])):
                        assert x['id'] == y['id'] and x['function']['name'] == y['function']['name']
                        try:
                            assert json.loads(x['function']['arguments']) == json.loads(y['function']['arguments'])
                        except (TypeError, ValueError):
                            assert x['function']['arguments'] == y['function']['arguments']
                assert a.get('tool_call_id') == b.get('tool_call_id')
            source_rows_checked += 1
        if r['quality_lane'] == 'quarantine':
            continue
        assert not audit[r['example_id']]['structural_errors']
        pending, seen = {}, set()
        tool_map = {t['function']['name']: t['function']['parameters'] for t in schemas[r['tools_schema_id']]}
        for mi, m in enumerate(messages):
            if m['role'] in {'assistant', 'user'}:
                assert not pending
            for c in m.get('tool_calls', []):
                assert c['id'] not in seen
                seen.add(c['id'])
                f = c['function']
                assert f['name'] in tool_map
                args = json.loads(f['arguments'])
                assert isinstance(args, dict)
                violations = list(jsonschema.Draft202012Validator(tool_map[f['name']]).iter_errors(args))
                if violations:
                    assert mask[mi] == 0 and r['quality_lane'] == 'recovery'
                    masked_invalid_calls += 1
                else:
                    calls_validated += 1
                pending[c['id']] = f['name']
            if m['role'] == 'tool':
                assert pending.pop(m['tool_call_id']) == m['name']
        assert not pending
        if r['quality_lane'] == 'core':
            assert all(mask[i] == int(m['role'] == 'assistant') for i, m in enumerate(messages))
        else:
            assert r['quality_lane'] == 'recovery'
    checks += ['all_new_wire_messages_match_sources_except_documented_transformations', 'all_eligible_trace_calls_paired_and_named', 'no_schema_invalid_call_is_a_positive_target', 'core_traces_require_no_failure_mask', 'explicit_reasoning_fields_removed']
    # Validate every exported file against its index/master and split.
    export_files = sorted(out.glob('sft/*.jsonl')) + sorted(out.glob('qna/*.jsonl')) + sorted(out.glob('traces/*.jsonl'))
    variant_map = {a['example_id']: a for a in qa_variants}
    exported_records = 0
    for path in export_files:
        if path.stem.endswith('_index'):
            continue
        idx_path = path.with_name(path.stem + '_index.jsonl')
        if not idx_path.exists():
            continue
        rows, indices = read_rows(path), read_rows(idx_path)
        assert len(rows) == len(indices)
        for number, (row, idx) in enumerate(zip(rows, indices), 1):
            assert idx['row'] == number
            assert len(row['messages']) == len(idx['assistant_loss_mask'])
            if idx['example_id'] in rm:
                source = rm[idx['example_id']]
                assert source['split'] == idx['split'] and source['group_id'] == idx['group_id']
                assert source['messages'] == [{k: v for k, v in m.items() if k != 'weight'} for m in row['messages']]
                expected_tools = schemas[source['tools_schema_id']] if source['record_type'] == 'trace' else []
                assert row['tools'] == expected_tools
                if 'train' in path.stem:
                    assert source['split'] == 'train' and source['quality_lane'] in {'core', 'recovery'}
                    if 'masked' not in path.stem:
                        assert source['quality_lane'] == 'core'
            else:
                a = variant_map[idx['example_id']]
                assert a['eligible'] and a['split'] == idx['split']
                assert row['messages'] == [dict(role='user', content=qm[a['question_id']]['question']), dict(role='assistant', content=a['answer'])]
                assert 0 < idx['sample_weight'] <= 1
            if 'masked' in path.stem:
                assert all(m.get('weight') == idx['assistant_loss_mask'][i] for i, m in enumerate(row['messages']) if m['role'] == 'assistant')
            if path.stem.startswith('validation'):
                assert idx['split'] == 'validation'
            if path.stem.startswith('test'):
                assert idx['split'] == 'test'
            exported_records += 1
    checks.append('all_export_rows_content_schema_masks_and_splits_match_indices')
    check(not any(r['quality_lane'] == 'quarantine' and r['split'] != 'quarantine' for r in traces), 'no_quarantine_promoted')
    bad = next(r for r in traces if r['scenario_id'] == 't1_retune_dtv_dvb_ldpc_bb')
    check(bad['split'] == 'quarantine', 'known_malformed_DSML_tool_name_excluded')
    db = sqlite3.connect(out / 'dataset.sqlite')
    check(db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'database_integrity')
    check(not db.execute('PRAGMA foreign_key_check').fetchall(), 'database_foreign_keys')
    check(db.execute('SELECT count(*) FROM examples').fetchone()[0] == len(records), 'database_record_count_matches_master')
    check(db.execute('SELECT count(*) FROM answer_variants').fetchone()[0] == len(qa_variants), 'database_answer_variant_count_matches_master')
    db.close()
    inventory = read_rows(out / 'audit/source_inventory.jsonl')
    check(all(sha_file(root / r['path']) == r['sha256'] for r in inventory), 'all_source_files_unchanged')
    if check_checksums:
        for line in (out / 'checksums.sha256').read_text(encoding='utf-8').splitlines():
            expected, rel = line.split('  ', 1)
            assert sha_file(out / rel) == expected, rel
        checks.append('deliverable_checksums_pass')
    return dict(status='PASS', assertion_count=len(checks), checks=[c for c in checks if not c.startswith('mask_length:')],
        valid_tool_calls_checked=calls_validated, schema_invalid_calls_confirmed_masked=masked_invalid_calls,
        new_source_conversations_compared=source_rows_checked, exported_rows_checked=exported_records,
        source_files_verified=len(inventory))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--out', type=Path, default=Path('processed/qoherent_v2'))
    a = p.parse_args()
    print(json.dumps(verify(a.root.resolve(), a.out.resolve()), indent=2))
