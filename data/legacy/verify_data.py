"""Independent output assertions; no changes to inputs or processed files."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def read(path):
    with path.open(encoding='utf-8-sig') as f:
        return [json.loads(s) for s in f if s.strip()]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def verify(root, out, check_checksums=True):
    checks = []
    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    qs = read(out / 'qna/questions.jsonl')
    qm = {q['question_id']: q for q in qs}
    check(len(qs) == len(qm), 'unique_question_ids')
    frozen = root / 'GRC_dataset/dataset/04_eval_frozen/eval_questions.jsonl'
    check(sha(frozen) == sha(out / 'qna/eval_questions_frozen.jsonl'), 'frozen_eval_byte_identical')
    eval_ids = {q['id'] for q in read(frozen)}
    check({q['question_id'] for q in qs if q['split'] == 'eval'} == eval_ids, 'frozen_eval_ids_preserved')
    groups = {}
    for q in qs:
        groups.setdefault(q['group_id'], set()).add(q['split'])
    check(all(not ('train' in splits and len(splits) > 1) and not ('validation' in splits and len(splits) > 1) for splits in groups.values()), 'related_groups_never_cross_training_validation_or_eval')
    answers = read(out / 'qna/accepted_answers.jsonl')
    primary = read(out / 'qna/primary_answers.jsonl')
    decisions = read(out / 'audit/answer_decisions.jsonl')
    dm = {(r['question_id'], r['candidate_model']): r for r in decisions}
    check(len(dm) == len(decisions), 'unique_answer_decisions')
    check(len({a['example_id'] for a in answers}) == len(answers), 'unique_answer_ids')
    check(len({a['question_id'] for a in primary}) == len(primary), 'one_primary_per_question')
    check(all(a['question_id'] not in eval_ids and a['split'] == qm[a['question_id']]['split'] for a in answers), 'no_eval_answers_and_all_alternatives_follow_split')
    answer_source_cache = {}
    for a in answers:
        d = dm[a['question_id'], a['candidate_model']]
        check(d['disposition'] == 'accepted', 'answer_has_accepted_decision:' + a['example_id'])
        if a['original_status'] == 'PASS':
            assert not d['conflicting_original_versions']
            assert d['secondary_validation'].get('validator_verdict') != 'DISAGREE'
        else:
            assert d['repair_validation']['repair_a_status'] == d['repair_validation']['repair_b_status'] == 'PASS'
        assert a['answer'] and hashlib.sha256(a['answer'].encode()).hexdigest() == a['answer_sha256']
        source_path = root / a['source']['file']
        if source_path not in answer_source_cache:
            with source_path.open(encoding='utf-8-sig') as f:
                answer_source_cache[source_path] = {n: json.loads(line) for n, line in enumerate(f, 1) if line.strip()}
        source = answer_source_cache[source_path][a['source']['line']]
        source_text = source.get('repaired_answer', source.get('answer', '')).replace('\r\n', '\n').replace('\r', '\n').strip()
        assert source_text == a['answer'] and source['question_id'] == a['question_id']
    checks.append('every_accepted_answer_matches_its_source_file_and_line')
    del answer_source_cache
    for name in ['train', 'validation']:
        rows = read(out / f'qna/{name}.jsonl')
        index = read(out / f'qna/{name}_index.jsonl')
        check(len(rows) == len(index), name + '_export_index_count')
        pm = {a['example_id']: a for a in primary}
        check(all(i['row'] == n and pm[i['example_id']]['split'] == name and r == {'messages': [{'role': 'user', 'content': pm[i['example_id']]['question']}, {'role': 'assistant', 'content': pm[i['example_id']]['answer']}]} for n, (r, i) in enumerate(zip(rows, index), 1)), name + '_export_content_and_split')
    sessions = read(out / 'traces/sessions.jsonl')
    pilot = [r for r in sessions if r['disposition'] == 'pilot_train']
    train = read(out / 'traces/pilot_train.jsonl')
    index = read(out / 'traces/pilot_train_index.jsonl')
    check(len(pilot) == len(train) == len(index), 'trace_export_index_count')
    check(all(r == {'messages': p['messages'], 'tools': p['tools']} and i['trace_id'] == p['trace_id'] and i['row'] == n for n, (r, p, i) in enumerate(zip(train, pilot, index), 1)), 'trace_export_content')
    check(all('thinking' not in m for s in sessions for m in s['messages']), 'thinking_fields_removed')
    check(all(not (isinstance(m.get('content'), str) and m['content'].lstrip('\"\'').startswith('[binary image output elided:')) for s in sessions for m in s['messages']), 'residual_binary_image_reprs_removed')
    elisions = read(out / 'audit/binary_output_elisions.jsonl')
    session_map = {s['trace_id']: s for s in sessions}
    input_cache = {}
    for e in elisions:
        path = root / e['source']['file']
        if path not in input_cache:
            input_cache[path] = read(path)
        original_text = input_cache[path][e['source']['line'] - 1]['messages'][e['source_message_index']]['content']
        assert hashlib.sha256(original_text.encode()).hexdigest() == e['original_sha256']
        assert session_map[e['trace_id']]['messages'][e['processed_message_index']]['content'] == e['replacement']
    checks.append('binary_elision_hashes_and_replacements_verified')
    elision_locations = {(e['trace_id'], e['processed_message_index']) for e in elisions}
    for session in sessions:
        path = root / session['source']['file']
        if path not in input_cache:
            input_cache[path] = read(path)
        source = input_cache[path][session['source']['line'] - 1]
        assert session['messages'][0]['content'] == source['system']
        assert len(session['messages']) == len(source['messages']) + 1
        assert session['teacher'] == source['teacher']
        for mi, (original_message, processed_message) in enumerate(zip(source['messages'], session['messages'][1:]), 1):
            assert original_message['role'] == processed_message['role']
            if (session['trace_id'], mi) not in elision_locations:
                assert original_message.get('content') == processed_message.get('content')
            assert original_message.get('tool_call_id') == processed_message.get('tool_call_id')
            original_calls = original_message.get('tool_calls', [])
            processed_calls = processed_message.get('tool_calls', [])
            assert len(original_calls) == len(processed_calls)
            for a, b in zip(original_calls, processed_calls):
                assert a['id'] == b['id'] and a['name'] == b['function']['name']
                try:
                    args_a = json.loads(a['arguments']) if isinstance(a['arguments'], str) else a['arguments']
                    args_b = json.loads(b['function']['arguments']) if isinstance(b['function']['arguments'], str) else b['function']['arguments']
                    assert args_a == args_b
                except (ValueError, TypeError):
                    assert a['arguments'] == b['function']['arguments']
    checks.append('all_trace_content_and_tool_arguments_match_sources_except_documented_elisions')
    check(all(s['messages'][0]['role'] == 'system' for s in sessions), 'trace_system_message_preserved')
    audits = read(out / 'audit/trace_quality.jsonl')
    check(all(not r['structural_errors'] for r in audits if r['disposition'] == 'pilot_train'), 'pilot_structural_audits_pass')
    for s in pilot:
        pending, seen = {}, set()
        known = {t['function']['name'] for t in s['tools']}
        for m in s['messages']:
            if m['role'] in {'user', 'assistant'}:
                assert not pending
            for call in m.get('tool_calls', []):
                assert call['id'] not in seen and call['function']['name'] in known
                assert isinstance(json.loads(call['function']['arguments']), dict)
                pending[call['id']] = call['function']['name']
                seen.add(call['id'])
            if m['role'] == 'tool':
                assert pending.pop(m['tool_call_id']) == m['name']
        assert not pending
    checks.append('independent_pilot_tool_pairing_and_arguments_pass')
    schema_validation = {'status': 'SKIPPED', 'reason': 'Optional jsonschema dependency unavailable'}
    try:
        import jsonschema
    except ImportError:
        pass
    else:
        count = 0
        for s in pilot:
            tools = {t['function']['name']: t['function']['parameters'] for t in s['tools']}
            for m in s['messages']:
                for call in m.get('tool_calls', []):
                    f = call['function']
                    jsonschema.Draft202012Validator(tools[f['name']]).validate(json.loads(f['arguments']))
                    count += 1
        schema_validation = {'status': 'PASS', 'calls_validated': count, 'validator': 'jsonschema Draft202012Validator'}
        checks.append('pilot_arguments_validate_against_full_tool_schemas')
    original = root / 'GRC_Agent_data/grc-agent-sft-v0.1.0'
    check(sha(original / 'sft_train.jsonl') == sha(original / 'sft_full_with_thinking.jsonl'), 'original_train_and_full_are_byte_identical')
    check(len(read(out / 'traces/quarantine.jsonl')) >= len(read(original / 'quarantine.jsonl')), 'no_source_quarantine_promoted')
    db = sqlite3.connect(out / 'qoherent.sqlite')
    check(db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'sqlite_integrity')
    check(not db.execute('PRAGMA foreign_key_check').fetchall(), 'sqlite_foreign_keys')
    for table, expected in [('questions', len(qs)), ('answers', len(answers)), ('answer_decisions', len(decisions)), ('traces', len(sessions)), ('messages', sum(len(s['messages']) for s in sessions)), ('tool_events', len(read(out / 'traces/tool_events.jsonl')))]:
        check(db.execute('SELECT count(*) FROM ' + table).fetchone()[0] == expected, 'sqlite_' + table + '_matches_jsonl')
    db.close()
    inventory = read(out / 'audit/source_inventory.jsonl')
    check(all(sha(root / r['path']) == r['sha256'] for r in inventory), 'all_source_hashes_unchanged')
    if check_checksums:
        for line in (out / 'checksums.sha256').read_text(encoding='utf-8').splitlines():
            expected, name = line.split('  ', 1)
            assert sha(out / name) == expected, name
        checks.append('deliverable_checksums_pass')
    return dict(status='PASS', assertion_count=len(checks), checks=[c for c in checks if not c.startswith('answer_has_accepted_decision:')], accepted_answers_individually_checked=len(answers), source_files_verified=len(inventory), tool_schema_validation=schema_validation)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--out', type=Path, default=Path('processed/qoherent_v1'))
    a = p.parse_args()
    print(json.dumps(verify(a.root.resolve(), a.out.resolve()), indent=2))
