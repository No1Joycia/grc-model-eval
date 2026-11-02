"""Integrate old Q&A, legacy sessions, atomic and multi-turn agent data.

Read-only inputs. Requires verified v1 outputs and jsonschema for tool checking.
No source code from supplied datasets is executed.
"""
from __future__ import annotations

import argparse
import collections as C
import copy
import itertools
import json
from pathlib import Path
import re
import shutil
import sqlite3
import zipfile

import jsonschema

from process_data import clean, digest, dumps, Groups, norm, percentiles, read_rows, sha_file, write_json, write_rows

VERSION = '2.0.0'
SEED = 'qoherent-integrated-v2-2026-10-27'
CONTEXT = re.compile(r'\b(?:the textbook|this chapter|this section|this article|the author|this tutorial|the above|shown in (?:figure|table)|figure\s+\d|table\s+\d)\b', re.I)
NAME = re.compile(r'^[a-zA-Z0-9_-]+$')


def load_lines(path, root):
    with path.open(encoding='utf-8-sig') as f:
        for line, text in enumerate(f, 1):
            if text.strip():
                yield json.loads(text), dict(file=path.relative_to(root).as_posix(), line=line)


def family_anchor(sid):
    tier, family, tail = sid.split('_', 2)
    if family == 'addwire':
        anchor = re.sub(r'_(?:in|out)\d+$', '', tail)
    elif family == 'topo':
        anchor = re.sub(r'_(?:remove_rewire|rename|disable)$', '', tail)
    elif family == 'guard':
        anchor = re.sub(r'_(?:port|param)$', '', tail)
    elif family == 'nmm':
        anchor = re.sub(r'_(?:query|preview)$', '', tail)
    elif family == 'build':
        anchor = tail.split('_to_', 1)[0]
    else:
        anchor = tail
    return family, anchor


def parse_result(text):
    try:
        result = json.loads(text)
        if isinstance(result, str) and result.startswith(('{', '[')):
            result = json.loads(result)
        return result
    except (ValueError, TypeError):
        return None


def normalize_trace(raw, source, source_set, checkpoint=None):
    """Return normalized record, audit. Terminal harness output is lowered to text."""
    sid = raw.get('scenario_id', raw.get('task_id'))
    eid = 'trace_' + source_set + '_' + (raw['trace_id'] if source_set == 'legacy' else sid)
    tools = copy.deepcopy(raw['tools'])
    tools = [t if t.get('type') == 'function' else dict(type='function', function=t) for t in tools]
    schemas = {t['function']['name']: t['function']['parameters'] for t in tools}
    validators = {name: jsonschema.Draft202012Validator(schema) for name, schema in schemas.items()}
    messages = copy.deepcopy(raw['messages'])
    if messages[0]['role'] != 'system':
        messages.insert(0, dict(role='system', content=raw['system']))
    transforms, errors, schema_errors = [], [], []
    pending, seen, failed_ids = {}, set(), set()
    guardrails = raw.get('guardrail_events', [])
    failed_ids.update(e.get('tool_call_id') for e in guardrails if e.get('tool_call_id'))
    reasoning_fields = 0
    finals = []
    for mi, message in enumerate(messages):
        for key in ['thinking', 'reasoning', 'reasoning_content']:
            if key in message:
                reasoning_fields += bool(message[key])
                message.pop(key)
        if message['role'] == 'assistant' and isinstance(message.get('content'), str):
            text = message['content']
            stripped = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.S)
            if stripped != text:
                transforms.append(dict(message_index=mi, action='remove_embedded_think', original_sha256=digest(text)))
                message['content'] = stripped
            if '<think>' in stripped or '</think>' in stripped:
                errors.append(dict(code='unbalanced_reasoning_markup', message_index=mi))
        calls = message.get('tool_calls', [])
        normalized_calls = []
        for c in calls:
            f = c.get('function', c)
            name, cid, arguments = f['name'], c['id'], f.get('arguments', '{}')
            try:
                args = json.loads(arguments) if isinstance(arguments, str) else arguments
                if not isinstance(args, dict):
                    raise ValueError('not object')
                arguments = dumps(args)
            except (ValueError, TypeError):
                errors.append(dict(code='invalid_argument_json', message_index=mi, tool_call_id=cid))
                args = None
            if name == 'final_result' and args:
                finals.append(dict(message_index=mi, tool_call_id=cid, output=args))
            # The native Pydantic terminal call intentionally has no tool reply.
            # Preserve its recorded assistant content; never invent a tool result.
            terminal = (mi == len(messages) - 1 and len(calls) == 1 and name == 'final_result'
                        and args is not None and isinstance(args.get('explanation'), str)
                        and name in validators and not list(validators[name].iter_errors(args)))
            if terminal:
                if not message.get('content'):
                    message['content'] = args['explanation']
                transforms.append(dict(message_index=mi, action='terminal_harness_call_to_assistant_text', call=c))
                continue
            normalized_calls.append(dict(id=cid, type='function', function=dict(name=name, arguments=arguments)))
        if normalized_calls:
            message['tool_calls'] = normalized_calls
        else:
            message.pop('tool_calls', None)
        if message['role'] in {'assistant', 'user'} and pending:
            errors.append(dict(code='unanswered_calls_before_next_turn', message_index=mi, ids=sorted(pending)))
        for c in normalized_calls:
            f, cid = c['function'], c['id']
            name = f['name']
            if cid in seen:
                errors.append(dict(code='repeated_call_id', message_index=mi, tool_call_id=cid))
            seen.add(cid)
            if not NAME.fullmatch(name) or name not in schemas:
                errors.append(dict(code='invalid_or_unknown_tool_name', message_index=mi, tool_name=name))
            else:
                try:
                    parsed = json.loads(f['arguments'])
                    for e in validators[name].iter_errors(parsed):
                        schema_errors.append(dict(message_index=mi, tool_call_id=cid, tool_name=name,
                            path=list(e.path), error=e.message[:500]))
                        failed_ids.add(cid)
                except (ValueError, TypeError):
                    pass
            pending[cid] = (name, mi)
        if message['role'] == 'tool':
            cid = message.get('tool_call_id')
            pair = pending.pop(cid, None)
            if pair is None:
                errors.append(dict(code='orphan_result', message_index=mi, tool_call_id=cid))
            else:
                if message.get('name', pair[0]) != pair[0]:
                    errors.append(dict(code='tool_name_mismatch', message_index=mi))
                message['name'] = pair[0]
                result = parse_result(message.get('content'))
                explicit_failure = isinstance(result, dict) and (result.get('ok') is False or result.get('success') is False or bool(result.get('error')))
                textual_failure = isinstance(message.get('content'), str) and bool(re.search(r'Fix the errors and try again|^ToolFailed|^Graph modification failed|^Error executing', message['content']))
                if explicit_failure or textual_failure:
                    failed_ids.add(cid)
        # Avoid copying unrelated custom fields into training rows.
        for key in list(message):
            if key not in {'role', 'content', 'tool_calls', 'tool_call_id', 'name'}:
                message.pop(key)
    if pending:
        errors.append(dict(code='unanswered_calls_at_end', ids=sorted(pending)))
    if messages[-1]['role'] == 'user':
        errors.append(dict(code='unanswered_last_user'))
    for mi, m in enumerate(messages):
        if m['role'] == 'assistant' and not m.get('content') and not m.get('tool_calls'):
            errors.append(dict(code='empty_assistant_after_reasoning_removal', message_index=mi))
    checkpoint_ok = checkpoint is None or (checkpoint.get('verification', {}).get('ok') is True
        and checkpoint.get('flowgraph_validation', {}).get('status') == 'valid'
        and checkpoint.get('flowgraph_valid') is True)
    if source_set in {'tier1', 'tier2'} and not checkpoint_ok:
        errors.append(dict(code='checkpoint_not_verified_valid'))
    original_quarantine = raw.get('disposition') == 'quarantine' or raw.get('verdict') not in {None, 'completed'}
    if original_quarantine:
        errors.append(dict(code='original_quarantine', reason=raw.get('quarantine_reason')))
    unsafe_indices = [mi for mi, m in enumerate(messages) if m['role'] == 'assistant' and any(c['id'] in failed_ids for c in m.get('tool_calls', []))]
    # Preserve the successful recovery turns; failures are context only.
    lane = 'quarantine' if errors else ('recovery' if unsafe_indices or guardrails or schema_errors else 'core')
    family, anchor = family_anchor(sid) if source_set in {'tier1', 'tier2'} else (raw.get('category', 'legacy'), None)
    first_user = next((m['content'] for m in messages if m['role'] == 'user'), '')
    record = dict(example_id=eid, record_type='trace', source_set=source_set, scenario_id=sid,
        category=raw.get('category'), task_family=family, anchor_block=anchor,
        source=source, messages=messages, tools=tools, quality_lane=lane,
        assistant_loss_mask=[int(m['role'] == 'assistant' and mi not in unsafe_indices) for mi, m in enumerate(messages)],
        user_request=first_user, user_turns=[m['content'] for m in messages if m['role'] == 'user'],
        model=raw.get('model', raw.get('teacher')), reported_usage=raw.get('usage', raw.get('reported_usage')),
        reported_flowgraph_valid=raw.get('flowgraph_valid'),
        final_outputs=finals or ([dict(output=checkpoint['final_output'])] if checkpoint and checkpoint.get('final_output') else []),
        checkpoint_source=checkpoint.get('_source') if checkpoint else None,
        initial_graph_snapshot=next((r.get('graph') for m in messages if m['role'] == 'tool' and m.get('name') == 'inspect_graph' and isinstance((r := parse_result(m.get('content'))), dict) and r.get('graph')), None),
        expected_graph_diff=checkpoint.get('flowgraph_diff') if checkpoint else None,
        guardrail_event_count=len(guardrails), independently_executed=False)
    record['serialized_characters'] = len(dumps(dict(messages=messages, tools=tools)))
    audit = dict(example_id=eid, scenario_id=sid, source_set=source_set, source=source,
        quality_lane=lane, structural_errors=errors, schema_errors=schema_errors,
        reasoning_fields_removed=reasoning_fields, transformations=transforms,
        assistant_indices_masked=unsafe_indices, failed_call_ids=sorted(failed_ids),
        checkpoint_ok=checkpoint_ok, guardrail_events=guardrails,
        recorded_verification=checkpoint.get('verification') if checkpoint else None)
    return record, audit


def training_row(record, weighted=False):
    messages = copy.deepcopy(record['messages'])
    if weighted:
        for mi, m in enumerate(messages):
            if m['role'] == 'assistant':
                m['weight'] = record['assistant_loss_mask'][mi]
    return dict(messages=messages, tools=record.get('tools', []))


def emit_export(path, records, weighted=False):
    write_rows(path, [training_row(r, weighted) for r in records])
    write_rows(path.with_name(path.stem + '_index.jsonl'), [dict(row=i, example_id=r['example_id'],
        record_type=r['record_type'], source_set=r['source_set'], split=r['split'], group_id=r['group_id'],
        assistant_loss_mask=r['assistant_loss_mask'], sample_weight=r.get('sample_weight', 1)) for i, r in enumerate(records, 1)])


def connect_related(records, legacy_edges):
    groups = Groups(r['example_id'] for r in records)
    edges = []
    by_question = {r.get('question_id'): r['example_id'] for r in records if r['record_type'] == 'qna'}
    def join(ids, method):
        ids = sorted(set(ids))
        for a in ids[1:]:
            groups.join(ids[0], a)
            edges.append(dict(example_a=ids[0], example_b=a, method=method))
    for e in legacy_edges:
        if e['id_a'] in by_question and e['id_b'] in by_question:
            join([by_question[e['id_a']], by_question[e['id_b']]], 'legacy_question_relatedness')
    anchors, prompts, legacy_tasks = C.defaultdict(list), C.defaultdict(list), C.defaultdict(list)
    for r in records:
        if r.get('anchor_block'):
            anchors[r['anchor_block']].append(r['example_id'])
        if r.get('source_set') == 'legacy':
            legacy_tasks[r['scenario_id']].append(r['example_id'])
        # Compare every user turn to prevent a Tier-1 single-turn task leaking
        # through either turn of a Tier-2 conversation.
        texts = r.get('user_turns', [r.get('question', '')])
        for text in texts:
            if len(norm(text)) > 15 and not text.startswith('Validation feedback'):
                prompts[(norm(text), r.get('anchor_block') or 'unanchored')].append(r['example_id'])
    for ids in anchors.values():
        join(ids, 'same_anchor_block_across_tiers')
    for ids in legacy_tasks.values():
        join(ids, 'same_legacy_task_attempts')
    for ids in prompts.values():
        join(ids, 'exact_user_request_any_turn')
    # Lexical candidates protect near-identical request wording. Do not delete.
    token_sets, inv = {}, C.defaultdict(set)
    for r in records:
        texts = r.get('user_turns', [r.get('question', '')])
        for i, text in enumerate(texts):
            if text.startswith('Validation feedback'):
                continue
            words = re.findall(r'\w+|[+*/=<>-]', norm(text))
            grams = set(zip(words, words[1:], words[2:]))
            key = (r['example_id'], i)
            token_sets[key] = grams
            for gram in grams:
                inv[gram].add(key)
    candidates = set()
    for keys in inv.values():
        if len(keys) <= 20:
            candidates.update(itertools.combinations(sorted(keys), 2))
    # Require matching numeric and backtick literals across trace tasks, so
    # templated prompts for different catalog blocks do not become one giant group.
    text_lookup = {(r['example_id'], i): text for r in records for i, text in enumerate(r.get('user_turns', [r.get('question', '')]))}
    record_map = {r['example_id']: r for r in records}
    for a, b in sorted(candidates):
        if a[0] == b[0] or groups.find(a[0]) == groups.find(b[0]):
            continue
        anchor_a, anchor_b = record_map[a[0]].get('anchor_block'), record_map[b[0]].get('anchor_block')
        if anchor_a and anchor_b and anchor_a != anchor_b:
            continue
        ga, gb = token_sets[a], token_sets[b]
        if min(len(ga), len(gb)) < 5:
            continue
        literals_a = re.findall(r'`([^`]+)`|\b(\d+(?:\.\d+)?)\b', text_lookup[a])
        literals_b = re.findall(r'`([^`]+)`|\b(\d+(?:\.\d+)?)\b', text_lookup[b])
        if (a[0].startswith('trace_') or b[0].startswith('trace_')) and literals_a != literals_b:
            continue
        score = len(ga & gb) / len(ga | gb)
        if score >= 0.5:
            join([a[0], b[0]], 'high_word_trigram_overlap')
    return groups, edges


def build(root, out, legacy):
    out.mkdir(parents=True, exist_ok=True)
    audit_dir = out / 'audit'
    audit_dir.mkdir(exist_ok=True)
    # V1 accepted-answer selection remains valid only if its raw inputs match.
    old_inventory = read_rows(legacy / 'audit/source_inventory.jsonl')
    assert all(sha_file(root / r['path']) == r['sha256'] for r in old_inventory), 'Legacy sources changed: regenerate v1 first'
    for line in (legacy / 'checksums.sha256').read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        assert sha_file(legacy / name) == expected, 'Changed v1 deliverable: ' + name
    inventory = []
    for folder in ['GRC_dataset', 'GRC_Agent_data', 'NewData1', 'NewData2', 'nanbeige-modal']:
        for p in sorted((root / folder).rglob('*')):
            if p.is_file() and not any(part in {'.git', '.venv', '__pycache__'} for part in p.parts):
                # Only the fixed benchmark is a data input from this repo.
                # Generated dataset releases and growing rebuild logs must not
                # become self-referential inventory inputs on later rebuilds.
                if folder == 'nanbeige-modal' and p.relative_to(root / folder).as_posix() not in {
                    'trial_eval/trial_eval_100.jsonl', 'trial_eval/trial_eval_ids.txt'
                }:
                    continue
                inventory.append(dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=sha_file(p)))
    write_rows(audit_dir / 'source_inventory.jsonl', inventory)
    previous_paths = {r['path'] for r in old_inventory}
    write_rows(audit_dir / 'new_sources.jsonl', [r for r in inventory if r['path'] not in previous_paths])

    trial_path = root / 'nanbeige-modal/trial_eval/trial_eval_100.jsonl'
    trial_rows = list(load_lines(trial_path, root))
    trial = {r['id']: r for r, _ in trial_rows}
    trial_sources = {r['id']: source for r, source in trial_rows}
    trial_ids = set((root / 'nanbeige-modal/trial_eval/trial_eval_ids.txt').read_text(encoding='utf-8').split())
    assert trial_ids == set(trial) and len(trial) == 100
    frozen_ids = {r['id'] for r in read_rows(root / 'GRC_dataset/dataset/04_eval_frozen/eval_questions.jsonl')}
    assert not trial_ids & frozen_ids
    questions = read_rows(legacy / 'qna/questions.jsonl')
    question_map = {q['question_id']: q for q in questions}
    answers = read_rows(legacy / 'qna/accepted_answers.jsonl')
    primary = {a['question_id']: a for a in read_rows(legacy / 'qna/primary_answers.jsonl')}
    accepted_hashes = {(a['question_id'], a['candidate_model'], a['answer_sha256']) for a in answers}
    accepted_by_hash = {(a['question_id'], a['candidate_model'], a['answer_sha256']): a for a in answers}
    qa, qa_audits = [], []
    for q in questions:
        qid = q['question_id']
        a = primary.get(qid)
        reasons = []
        if not a and qid not in frozen_ids:
            reasons.append('no_accepted_answer')
        if CONTEXT.search(q['question']):
            reasons.append('depends_on_unprovided_source_context')
        if qid in trial:
            assert norm(trial[qid]['question']) == norm(q['question'])
        qa.append(dict(example_id='qna_' + qid, record_type='qna', source_set='qna', question_id=qid,
            question=q['question'], user_request=q['question'], user_turns=[q['question']],
            labels=q['labels'], difficulty=q['difficulty'], origin=q['origin'], provenance=q['provenance'],
            source=q['source'], source_urls=q['source_urls'], answer_source=a['source'] if a else None,
            quality_lane='core' if not reasons else 'review', independently_verified=False,
            messages=[dict(role='user', content=q['question'])] + ([dict(role='assistant', content=a['answer'])] if a else []),
            tools=[], assistant_loss_mask=[0, 1] if a else [0]))
        qa_audits.append(dict(example_id='qna_' + qid, question_id=qid, reasons=reasons,
            source=q['source'], answer_kind=a['answer_kind'] if a else None))
    traces, trace_audits, reconciliations = [], [], []
    for raw in read_rows(legacy / 'traces/sessions.jsonl'):
        source = dict(file=(legacy / 'traces/sessions.jsonl').relative_to(root).as_posix(), trace_id=raw['trace_id'], original_source=raw['source'])
        r, a = normalize_trace(raw, source, 'legacy')
        traces.append(r)
        trace_audits.append(a)
    for tier in [1, 2]:
        folder = root / f'NewData{tier}'
        checkpoints = {p.stem: p for p in folder.rglob('t[12]_*.json')}
        master_ids = set()
        conversion_ids = {r['scenario_id'] for r in read_rows(folder / f'tier{tier}_unsloth.jsonl')}
        for raw, source in load_lines(folder / f'tier{tier}_sft_trajectories.jsonl', root):
            sid = raw['scenario_id']
            assert sid not in master_ids
            master_ids.add(sid)
            cp_path = checkpoints[sid]
            checkpoint = json.loads(cp_path.read_text(encoding='utf-8-sig'))
            fields = ['messages', 'tools', 'canonical_messages', 'flowgraph_valid', 'guardrail_events']
            matched = all(checkpoint.get(k) == raw.get(k) for k in fields)
            checkpoint['_source'] = dict(file=cp_path.relative_to(root).as_posix())
            r, a = normalize_trace(raw, source, 'tier' + str(tier), checkpoint)
            if not matched:
                a['structural_errors'].append(dict(code='checkpoint_master_mismatch'))
                r['quality_lane'] = a['quality_lane'] = 'quarantine'
            reconciliations.append(dict(scenario_id=sid, source_set='tier'+str(tier),
                checkpoint_matches_master=matched, checkpoint_source=checkpoint['_source'],
                present_in_unsloth_conversion=sid in conversion_ids, master_source=source))
            traces.append(r)
            trace_audits.append(a)
        assert master_ids == set(checkpoints), 'Unaccounted checkpoint'
        assert conversion_ids <= master_ids
    write_rows(audit_dir / 'trace_quality.jsonl', trace_audits)
    write_rows(audit_dir / 'checkpoint_reconciliation.jsonl', reconciliations)
    records = qa + traces
    groups, relatedness = connect_related(records, read_rows(legacy / 'audit/question_relatedness.jsonl'))
    components = C.defaultdict(list)
    for r in records:
        components[groups.find(r['example_id'])].append(r)
    # Preserve qna's previous validation assignment where no new test holdout applies.
    old_split = {q['question_id']: q['split'] for q in questions}
    legacy_groups = sorted({groups.find(r['example_id']) for r in traces if r['source_set'] == 'legacy' and r['quality_lane'] != 'quarantine'}, key=lambda s: digest(SEED + '|legacy|' + s))
    legacy_split = {g: ('test' if i < 2 else 'validation' if i < 4 else 'train') for i, g in enumerate(legacy_groups)}
    for group, members in sorted(components.items()):
        ids = {r.get('question_id') for r in members}
        if ids & (trial_ids | frozen_ids):
            assignment = 'protected_test'
        else:
            value = int(digest(SEED + '|' + group)[:16], 16) / 2**64
            assignment = 'test' if value < .1 else 'validation' if value < .2 else 'train'
            qa_members = [r for r in members if r['record_type'] == 'qna']
            if qa_members:
                assignment = 'validation' if any(old_split[r['question_id']] == 'validation' for r in qa_members) else 'train'
            elif group in legacy_split:
                assignment = legacy_split[group]
        for r in members:
            qid = r.get('question_id')
            r['group_id'] = group
            if assignment == 'protected_test':
                r['split'] = 'test' if qid in trial_ids else 'test_unscored' if qid in frozen_ids else 'test_related_holdout'
            else:
                r['split'] = assignment
            if r['quality_lane'] == 'quarantine':
                r['split'] = 'quarantine'
            elif r['quality_lane'] == 'review' and qid not in frozen_ids:
                r['split'] = 'review'
    write_rows(audit_dir / 'relatedness.jsonl', relatedness)
    write_rows(audit_dir / 'split_assignments.jsonl', [dict(example_id=r['example_id'], record_type=r['record_type'], source_set=r['source_set'], group_id=r['group_id'], split=r['split'], quality_lane=r['quality_lane']) for r in records])
    for a in qa_audits:
        a['split'] = next(r['split'] for r in qa if r['example_id'] == a['example_id'])
    write_rows(audit_dir / 'qna_eligibility.jsonl', qa_audits)

    # Scoreable question references must match an accepted answer's exact hash.
    test_references = []
    for qid, raw in sorted(trial.items()):
        valid = [ref for ref in raw['references'] if (qid, ref['model'], digest(clean(ref['answer']))) in accepted_hashes]
        assert valid, 'No vetted test reference for ' + qid
        preferred = next((ref for ref in valid if ref['model'] == raw['reference_model'] and clean(ref['answer']) == clean(raw['reference_answer'])), valid[0])
        test_references.append(dict(question_id=qid, question=question_map[qid]['question'],
            difficulty=raw['difficulty'], reference_answer=clean(preferred['answer']), reference_model=preferred['model'],
            source=trial_sources[qid],
            reference_answer_source=accepted_by_hash[(qid, preferred['model'], digest(clean(preferred['answer'])))]['source'],
            references=[dict(model=ref['model'], answer=clean(ref['answer']),
                answer_source=accepted_by_hash[(qid, ref['model'], digest(clean(ref['answer'])))]['source']) for ref in valid],
            reference_quality='existing_teacher_judgments_not_new_expert_ground_truth'))
    write_rows(out / 'qna/test_references.jsonl', test_references)
    # Use the supplied accepted test reference in test export, rather than a
    # potentially different seeded primary answer. It never enters training.
    test_map = {r['question_id']: r for r in test_references}
    for r in qa:
        if r.get('question_id') in test_map:
            r['messages'] = [dict(role='user', content=r['question']), dict(role='assistant', content=test_map[r['question_id']]['reference_answer'])]
            r['assistant_loss_mask'] = [0, 1]
            r['answer_source'] = test_map[r['question_id']]['reference_answer_source']
            r['test_reference_source'] = test_map[r['question_id']]['source']
    write_rows(out / 'qna/questions.jsonl', qa)
    write_rows(out / 'qna/review.jsonl', [r for r in qa if r['split'] == 'review'])
    frozen_dest = out / 'qna/test_frozen_185_questions_only.jsonl'
    shutil.copyfile(root / 'GRC_dataset/dataset/04_eval_frozen/eval_questions.jsonl', frozen_dest)
    for split_name in ['train', 'validation', 'test']:
        emit_export(out / f'qna/{split_name}.jsonl', [r for r in qa if r['split'] == split_name and r['quality_lane'] == 'core'])
    qa_map = {r['question_id']: r for r in qa}
    variant_records = []
    for a in answers:
        q = qa_map[a['question_id']]
        a['split'] = q['split']
        a['group_id'] = q['group_id']
        a['eligible'] = q['quality_lane'] == 'core'
        if a['eligible'] and a['split'] in {'train', 'validation'}:
            variant_records.append(dict(example_id=a['example_id'], question_id=a['question_id'], record_type='qna', source_set='qna_alternative',
                split=a['split'], group_id=a['group_id'], messages=[dict(role='user', content=q['question']), dict(role='assistant', content=a['answer'])],
                tools=[], assistant_loss_mask=[0, 1]))
    counts = C.Counter(r['question_id'] for r in variant_records)
    for r in variant_records:
        r['sample_weight'] = 1 / counts[r['question_id']]
    write_rows(out / 'qna/answer_variants.jsonl', answers)
    for split_name in ['train', 'validation']:
        emit_export(out / f'qna/{split_name}_all_answers.jsonl', [r for r in variant_records if r['split'] == split_name])

    # Deduplicate schemas in the analysis master; training exports remain self-contained.
    schema_registry = {}
    for r in traces:
        r['tools_schema_id'] = 'tools_' + digest(dumps(r['tools']))[:20]
        schema_registry[r['tools_schema_id']] = r['tools']
    write_json(out / 'traces/tool_schemas.json', schema_registry)
    write_rows(out / 'traces/records.jsonl', [{k: v for k, v in r.items() if k != 'tools'} for r in traces])
    write_rows(out / 'traces/quarantine.jsonl', [{k: v for k, v in r.items() if k != 'tools'} for r in traces if r['quality_lane'] == 'quarantine'])
    for split_name in ['train', 'validation', 'test']:
        all_split = [r for r in traces if r['split'] == split_name and r['quality_lane'] != 'quarantine']
        # Plain training defaults exclude all trajectories requiring failure masks.
        emit_export(out / f'traces/{split_name}.jsonl', [r for r in all_split if r['quality_lane'] == 'core'] if split_name == 'train' else all_split)
        if split_name == 'train':
            emit_export(out / 'traces/train_recovery_masked.jsonl', [r for r in all_split if r['quality_lane'] == 'recovery'], weighted=True)
        merged = [r for r in qa if r['split'] == split_name and r['quality_lane'] == 'core'] + [r for r in all_split if r['quality_lane'] == 'core' or split_name != 'train']
        emit_export(out / f'sft/{split_name}.jsonl', merged)
    recovery_train = [r for r in traces if r['split'] == 'train' and r['quality_lane'] == 'recovery']
    full_train = [r for r in records if r['split'] == 'train' and r['quality_lane'] in {'core', 'recovery'}]
    emit_export(out / 'sft/train_with_recovery_masked.jsonl', full_train, weighted=True)
    write_rows(out / 'traces/test_tasks.jsonl', [dict(example_id=r['example_id'], group_id=r['group_id'], scenario_id=r['scenario_id'], source_set=r['source_set'],
        task_family=r['task_family'], anchor_block=r['anchor_block'], user_turns=r['user_turns'],
        initial_graph_snapshot=r['initial_graph_snapshot'], expected_graph_diff=r['expected_graph_diff'],
        recorded_final_outputs=r['final_outputs'], checkpoint_source=r['checkpoint_source'],
        evaluation_scope='offline reference trajectory; live execution requires original fixtures and harness') for r in traces if r['split'] == 'test'])
    write_rows(out / 'audit/datatype_register.jsonl', [
        dict(source='GRC_dataset', data_type='technical Q&A and teacher quality pipeline', use='accepted self-contained Q&A; one primary per question plus optional alternatives', avoid='raw rejected/repaired intermediates as additional independent examples'),
        dict(source='GRC_Agent_data', data_type='long desktop agent sessions', use='structurally valid pilot trajectories; separate quarantines; keep whole session together', evidence='recorded logs, not re-executed'),
        dict(source='NewData1', data_type='single-task atomic agent tool trajectories', use='one normalized master record per scenario, grouped by anchor block', evidence='checkpoint graph validation, not comprehensive DSP/runtime acceptance'),
        dict(source='NewData2', data_type='two-user-turn shared-state trajectories', use='keep entire conversation and both turns together with related Tier-1 tasks', evidence='checkpoint graph validation, not comprehensive DSP/runtime acceptance'),
        dict(source='NewData1/2 checkpoints and Unsloth files', data_type='redundant exports and audit evidence', use='reconcile master/checkpoint; do not count copies as new training samples; strip reasoning from wire master'),
        dict(source='nanbeige-modal/trial_eval', data_type='100 graded-reference Q&A evaluation questions', use='test-only, all accepted references protected from training'),
        dict(source='nanbeige-modal scripts/logs/examples', data_type='deployment, evaluation code, two unanswered integration prompts and logs', use='inventory and context only, no supervised target or training generation'),
    ])
    stats = dict(version=VERSION, source_files=len(inventory), source_bytes=sum(r['bytes'] for r in inventory),
        new_source_files=sum(r['path'] not in previous_paths for r in inventory),
        questions=len(qa), accepted_answer_variants=len(answers),
        qna_splits=dict(C.Counter(r['split'] for r in qa)),
        qna_train_examples=sum(r['split'] == 'train' and r['quality_lane'] == 'core' for r in qa),
        qna_validation_examples=sum(r['split'] == 'validation' and r['quality_lane'] == 'core' for r in qa),
        qna_test_examples=len(test_references), frozen_questions=len(frozen_ids),
        qna_train_answer_variants=sum(r['split'] == 'train' for r in variant_records),
        qna_review_reasons=dict(C.Counter(reason for a in qa_audits for reason in a['reasons'])),
        trace_source_counts=dict(C.Counter(r['source_set'] for r in traces)),
        trace_quality_lanes=dict(C.Counter(r['quality_lane'] for r in traces)),
        trace_split_lanes={s: dict(C.Counter(r['quality_lane'] for r in traces if r['split'] == s)) for s in ['train', 'validation', 'test', 'quarantine', 'test_related_holdout']},
        default_sft_splits={s: sum(r['split'] == s and r['quality_lane'] == 'core' for r in qa) + sum(r['split'] == s and r['quality_lane'] != 'quarantine' and (r['quality_lane'] == 'core' or s != 'train') for r in traces) for s in ['train', 'validation', 'test']},
        full_masked_train=len(full_train), recovery_training_traces=len(recovery_train),
        new_checkpoints_reconciled=len(reconciliations), checkpoint_mismatches=sum(not r['checkpoint_matches_master'] for r in reconciliations),
        new_trace_reasoning_fields_removed=sum(a['reasoning_fields_removed'] for a in trace_audits if a['source_set'] in {'tier1', 'tier2'}),
        terminal_calls_converted=sum(t['action'] == 'terminal_harness_call_to_assistant_text' for a in trace_audits for t in a['transformations']),
        schema_invalid_calls=sum(len({e['tool_call_id'] for e in a['schema_errors']}) for a in trace_audits),
        masked_assistant_turns=sum(len(a['assistant_indices_masked']) for a in trace_audits),
        schema_registry_entries=len(schema_registry), related_groups=sum(len(v) > 1 for v in components.values()),
        largest_group=max(map(len, components.values())),
        trace_char_percentiles=percentiles([r['serialized_characters'] for r in traces if r['quality_lane'] != 'quarantine']),
        trace_family_split_counts={s: dict(C.Counter(r['source_set'] + '/' + r['task_family'] for r in traces if r['split'] == s)) for s in ['train', 'validation', 'test']})
    write_json(audit_dir / 'summary.json', stats)
    write_json(out / 'manifest.json', dict(version=VERSION, seed=SEED,
        source_fingerprint=digest(dumps(inventory)), legacy_dir=legacy.relative_to(root).as_posix(),
        legacy_fingerprint=json.loads((legacy / 'manifest.json').read_text(encoding='utf-8'))['source_fingerprint'],
        split_policy='Q&A: freeze 100 referenced tests and 185 unscored tests, hold out related groups, retain prior validation where possible. Traces: group all variants by anchor block and same-context requests; seeded 80/10/10. Legacy sessions allocate 8/2/2 before quarantine.',
        lexical_grouping='Exact same-context requests and high word-trigram overlap, in addition to supplied Q&A groups. Generic requests on different starting blocks are different contextual tasks. No semantic embedding audit.',
        training_policy='Default train is primary Q&A plus core traces. Recovery traces are available with mandatory assistant loss masks; invalid attempts have weight 0. Alternatives are optional and inherit question splits.',
        terminal_policy='Unmatched valid final_result at conversation end is converted to recorded assistant text; structured call retained in audit. No tool result fabricated.',
        test_policy='Model-generated references are for offline evaluation, not independent expert labels. Test trajectories contain observed snapshots and expected diffs, not complete executable .grc fixtures.',
        quality_limit='Source graph checks not re-executed; no new RF correctness or runtime success certification.',
        source_policy='Originals and v1 package are unchanged. Checkpoint/canonical/Unsloth copies are not additional samples. No model calls or network execution performed.'))
    # Retain selection lineage and topic uncertainty from the prior Q&A assessment.
    for name in ['answer_decisions.jsonl', 'conflicting_answer_versions.jsonl', 'label_review_queue.jsonl', 'binary_output_elisions.jsonl']:
        shutil.copyfile(legacy / 'audit' / name, audit_dir / ('legacy_' + name))
    make_database(out, records, answers, schema_registry, trace_audits)
    write_docs(out, stats, records)
    for name in ['build_dataset_v2.py', 'verify_dataset_v2.py', 'process_data.py', 'verify_data.py']:
        dst = out / 'processing' / name
        dst.parent.mkdir(exist_ok=True)
        shutil.copyfile(Path(__file__).parent / name, dst)
    return stats


def make_database(out, records, answers, schemas, audits):
    path = out / 'dataset.sqlite'
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    db.executescript('''
    PRAGMA foreign_keys=ON;
    CREATE TABLE examples(example_id TEXT PRIMARY KEY, record_type TEXT, source_set TEXT, split TEXT, quality_lane TEXT, group_id TEXT, task_family TEXT, anchor_block TEXT, prompt TEXT, record_json TEXT);
    CREATE TABLE answer_variants(example_id TEXT PRIMARY KEY, question_id TEXT, split TEXT, eligible INTEGER, record_json TEXT);
    CREATE TABLE tool_schemas(schema_id TEXT PRIMARY KEY, tools_json TEXT);
    CREATE TABLE trace_audits(example_id TEXT PRIMARY KEY REFERENCES examples, record_json TEXT);
    CREATE INDEX examples_split ON examples(split,record_type,quality_lane);
    CREATE INDEX examples_group ON examples(group_id);
    CREATE INDEX examples_anchor ON examples(anchor_block);
    CREATE INDEX variants_question ON answer_variants(question_id,split);
    CREATE VIRTUAL TABLE search USING fts5(example_id UNINDEXED, prompt, final_answer);
    ''')
    db.executemany('INSERT INTO examples VALUES(?,?,?,?,?,?,?,?,?,?)', [(r['example_id'], r['record_type'], r['source_set'], r['split'], r['quality_lane'], r['group_id'], r.get('task_family'), r.get('anchor_block'), r['user_request'], dumps({k: v for k, v in r.items() if k != 'tools'})) for r in records])
    db.executemany('INSERT INTO answer_variants VALUES(?,?,?,?,?)', [(a['example_id'], a['question_id'], a['split'], int(a['eligible']), dumps(a)) for a in answers])
    db.executemany('INSERT INTO tool_schemas VALUES(?,?)', [(key, dumps(value)) for key, value in sorted(schemas.items())])
    db.executemany('INSERT INTO trace_audits VALUES(?,?)', [(a['example_id'], dumps(a)) for a in audits])
    db.executemany('INSERT INTO search VALUES(?,?,?)', [(r['example_id'], r['user_request'], '\n'.join(m.get('content') or '' for m in r['messages'] if m['role'] == 'assistant')) for r in records])
    db.commit()
    assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    assert not db.execute('PRAGMA foreign_key_check').fetchall()
    db.close()


def write_docs(out, s, records):
    q = s['qna_splits']
    ts = s['trace_split_lanes']
    content = f'''# Qoherent integrated dataset v2

This version integrates the original Q&A, original agent sessions, NewData1 atomic traces, NewData2 multi-turn traces, and the existing 100-question reference evaluation. Every source is inventoried; redundant exports and audit checkpoints are not counted as additional training examples. Originals and v1 remain unchanged.

## Use these files

| Purpose | File | Examples |
|---|---|---:|
| Default mixed SFT training: primary Q&A + core traces | `sft/train.jsonl` | {s['default_sft_splits']['train']:,} |
| Mixed validation | `sft/validation.jsonl` | {s['default_sft_splits']['validation']:,} |
| Mixed offline test references | `sft/test.jsonl` | {s['default_sft_splits']['test']:,} |
| All eligible training, with recovery loss masks | `sft/train_with_recovery_masked.jsonl` | {s['full_masked_train']:,} |
| Q&A training only | `qna/train.jsonl` | {s['qna_train_examples']:,} |
| Q&A validation only | `qna/validation.jsonl` | {s['qna_validation_examples']:,} |
| Q&A test with accepted teacher reference | `qna/test.jsonl` | {s['qna_test_examples']:,} |
| Extra frozen Q&A test, questions only | `qna/test_frozen_185_questions_only.jsonl` | {s['frozen_questions']:,} |
| Core tool-use training only | `traces/train.jsonl` | {ts['train'].get('core', 0):,} |
| Recovery training, mandatory loss masks | `traces/train_recovery_masked.jsonl` | {ts['train'].get('recovery', 0):,} |
| Agent validation, complete sessions | `traces/validation.jsonl` | {sum(ts['validation'].values()):,} |
| Agent offline test, complete reference sessions | `traces/test.jsonl` | {sum(ts['test'].values()):,} |

**Start with `sft/train.jsonl`, `sft/validation.jsonl`, and `sft/test.jsonl`.** These are UTF-8 JSONL with `messages` and `tools` arrays. Q&A rows have an empty `tools` list. Each file has a companion `_index.jsonl` mapping its 1-based file row to an example ID, split group and assistant loss mask. Use a tool-aware chat template and mask system/user/tool tokens in training. Apply assistant loss to tool-call arguments as well as assistant text. Check the target tokenizer and context length before training; no model-specific token count or training run is claimed.

For all eligible data, use `sft/train_with_recovery_masked.jsonl` **instead of** the default training file. It includes {s['recovery_training_traces']} additional training sessions with successful recovery after recorded failures. Assistant messages carry `weight: 0` for failed attempts and `weight: 1` for usable targets; index files carry an aligned `assistant_loss_mask`. Your trainer must implement those masks: do not assume a loader honors the `weight` field. Failed attempts and tool feedback remain context for the successful correction. The default training file excludes recovery sessions so it can be used without a custom failure mask. Validation/test include recovery contexts; score only positive assistant targets or the final outcome.

## Data assessment

| Source | Input records | Type and usable signal | Handling |
|---|---:|---|---|
| Original Q&A | {s['questions']} questions; {s['accepted_answer_variants']:,} previously accepted answer variants | RF/DSP/GNU Radio knowledge, calculations, diagnosis and design | Retain prior PASS/dual-pass repair selection; exclude missing-source-context questions; protect evaluation IDs and all variants |
| Original agent set | {s['trace_source_counts']['legacy']} sessions, including quarantine | Long tool-use sessions with recorded execution observations | Recheck normalized structure; retain quarantines; split complete sessions |
| NewData1 | {s['trace_source_counts']['tier1']:,} scenarios | Atomic block addition, parameter changes, topology edits, knowledge queries and recovery | Reconcile each checkpoint; preserve one record per scenario; malformed tool name quarantined |
| NewData2 | {s['trace_source_counts']['tier2']:,} scenarios | Two-turn interaction, corrections, refinement, query-then-act and state continuity | Preserve the complete conversation; group with related Tier-1 scenarios |
| Existing trial evaluation | 100 questions | Teacher-graded answer references, derived from old train pool | Test-only; remove every associated answer variant and related task from training |
| Checkpoints, canonical messages, Unsloth exports | Multiple representations of scenarios | Audit evidence and export checks | Reconcile, do not duplicate; use wire master to remove separate reasoning cleanly |
| Deployment scripts, logs and two unanswered integration prompts | Supporting files | Not a supervised example collection | Inventory only; no deployment or endpoint calls performed |

These are structured text and interaction data. They are not a corpus of RF/IQ waveform recordings. Headless graph validation checks graph consistency; it does not establish physical RF correctness, successful hardware execution or task acceptance. GNU Radio code and commands in tool logs were not executed here. No new model answers or judgments were generated.

## Split protection

The 100 referenced Q&A tests are fixed by the supplied `trial_eval_ids.txt`. The original 185 frozen evaluation questions are copied byte-for-byte into a separate unscored test file; they have no supplied graded answers. These are distinct sets, not a merged 285-question scored benchmark. Related examples enter `test_related_holdout` and do not train. Existing Q&A validation membership is retained where new test protection permits. Q&A review/quarantine counts are in the audit, rather than silently reusing defective records.

Trace variants sharing the same **anchor block** are grouped across Tier 1 and Tier 2. This keeps a Tier-1 first-turn task out of training when its related multi-turn continuation is held out. Generic prompts on different starting blocks are contextually different tasks and do not join the entire corpus. The primary block in a two-block build defines the anchor; secondary helper blocks can occur in other splits. This measures held-out task/anchor variation, not absence of all test block names from the training vocabulary. There is no held-out task-family claim: all common families may appear in each split.

Groups use a seeded 80/10/10 trace assignment, so row proportions need not be exactly 80/10/10. The 12 valid legacy sessions are allocated 8/2/2 before other quality handling. Source/category/family counts are reported in `audit/summary.json`. High lexical-overlap candidates and previous Q&A relatedness are also grouped conservatively; no semantic embedding or human semantic adjudication was added. Test sets are withheld for final comparison; use validation to choose checkpoints and settings.

## Cleaning and quality decisions

- Reconciled {s['new_checkpoints_reconciled']:,} scenario checkpoints against master messages, tools, canonical messages, graph-valid flag and guardrail events; {s['checkpoint_mismatches']} mismatches.
- Removed {s['new_trace_reasoning_fields_removed']:,} nonempty reasoning fields from new wire messages. Original Qwen-style thinking exports are not used as additional targets. V1's image-byte elisions and reasoning cleanup remain in effect.
- Converted {s['terminal_calls_converted']:,} valid unmatched terminal `final_result` calls into their recorded assistant text. Structured final output remains in metadata/audit. No fake tool acknowledgment is inserted. Other calls retain IDs, results, arguments and schemas.
- Full JSON Schema checking found {s['schema_invalid_calls']} rejected argument calls. Useful recovery sessions remain in the masked lane; malformed tool names and structural defects remain quarantined.
- {s['masked_assistant_turns']} assistant turns are marked context-only due to recorded failed calls or schema violations. This is conservative; successful sibling calls in the same assistant message are also masked.
- Q&A source-context review reasons: `{json.dumps(s['qna_review_reasons'], sort_keys=True)}`. Raw topic labels and tentative v1 suggestions are retained; topic suggestions are not expert-reviewed labels.

**Quality comes from supplied judgments and recorded graph/checkpoint evidence. Independent RF/DSP correctness review and live GNU Radio acceptance tests have not been performed.** Test reference answers are synthetic teacher outputs; similarity or imitation of a reference is not a correctness certificate.

## All answers, summaries and master data

`qna/train_all_answers.jsonl` contains {s['qna_train_answer_variants']:,} accepted training-pool answer variants. This is an alternative to the primary Q&A export. Its index provides `sample_weight = 1 / accepted_variants_for_question`; honor that weight or sample by question to avoid letting seven answers count as seven independent questions. Every alternative inherits its question's split. Do not train on the unfiltered `qna/answer_variants.jsonl` master.

`qna/test_references.jsonl` retains all supplied test reference answers that match an accepted answer hash. `qna/questions.jsonl` retains canonical questions and split/eligibility metadata. `qna/review.jsonl` contains records needing context or usable answers.

`traces/records.jsonl` is the analysis master. Resolve its `tools_schema_id` through `traces/tool_schemas.json`. `traces/test_tasks.jsonl` provides test prompts, recorded initial graph snapshots, expected graph diffs and final outputs. Snapshots may omit default parameters, and the supplied new folders do not include the original fixture `.grc` files or harness. These test tasks support offline/reference evaluation; obtain the original fixtures/harness for faithful live rollout evaluation. Snapshot records are not asserted to be standalone executable flowgraphs.

`DATA_REVIEW.md` gives the concise source and exclusion assessment. `dataset.sqlite` supports filtering/search. `audit/trace_quality.jsonl` contains all normalization operations, failed-call masks and verification results. `audit/split_assignments.jsonl` records every canonical question and trajectory, including reviews, heldouts and quarantine.

## SQLite examples

```sql
SELECT record_type, source_set, split, quality_lane, COUNT(*)
FROM examples GROUP BY record_type, source_set, split, quality_lane;
SELECT example_id, prompt FROM examples WHERE split='train' AND quality_lane='core' LIMIT 5;
SELECT example_id, prompt FROM search WHERE search MATCH 'OFDM' LIMIT 10;
```

## Reproduce and verify

Python 3.10+ and `jsonschema` are required for v2. From the workspace root, with the original folders and verified v1 package present:

```powershell
python processing/build_dataset_v2.py --root . --legacy processed/qoherent_v1 --out processed/qoherent_v2
python processing/verify_dataset_v2.py --root . --out processed/qoherent_v2
```

V1 selection can be regenerated with `processing/process_data.py`. V2 checks its source and deliverable hashes before reuse. Processing scripts are included. The source inventory hashes inputs; generated checksums cover deliverables; the archive contains all v2 outputs except itself. No original or prior package is overwritten. Supplied provenance and local paths remain in the research package; this preparation does not establish public redistribution permission.
'''
    (out / 'README.md').write_text(content, encoding='utf-8')
    review = f'''# Updated data assessment — 27 October 2026

NewData1 contributes 1,253 unique atomic scenarios; NewData2 contributes 600 unique two-turn scenarios. All 1,853 checkpoints match their master wire conversations and schemas. Their alternative exports are copies, not new independent samples. Both are useful for tool-use SFT and offline held-out evaluation, with the recovery masks and graph-validation limits described in README.

The combined trace quality lanes are `{json.dumps(s['trace_quality_lanes'], sort_keys=True)}`. Default mixed training includes {s['default_sft_splits']['train']} examples; the full masked-training option includes {s['full_masked_train']}. Full group/split counts are in `audit/summary.json`.

Important revision: the added `nanbeige-modal/trial_eval` contains 100 reference-answer tests sampled from the old training pool. Those IDs, all answer variants and grouped related questions are now held out. Continuing to use v1's train export for this benchmark could contaminate the evaluation. Use v2 exports for new runs. Any earlier model trained on v1 should be labelled potentially exposed on this newly protected test set.

Use the default mixed train/validation/test files for the baseline. Use recovery data only with failed-attempt loss masks. Expand to all-answer Q&A variants as a separate controlled experiment with question weighting. Keep review, quarantine, unscored tests and test-related holdouts outside training.

New traces use null-source/null-sink scaffolds, GNU Radio 3.10.9.2 and synthetic tasks. They teach tool mechanics and state continuity. They do not alone demonstrate real RF debugging, hardware success or signal-processing acceptance. The legacy long sessions supply broader recorded task context, but remain a small pilot. No test reference is newly expert-certified.
'''
    (out / 'DATA_REVIEW.md').write_text(review, encoding='utf-8')
    write_json(out / 'schema.json', dict(version=VERSION,
        training_record=dict(messages='system/user/assistant/tool messages', tools='function tool schemas; empty for Q&A'),
        trace_master='records.jsonl with tools_schema_id; resolve through tool_schemas.json',
        loss_mask='Index assistant_loss_mask aligned to message positions, 0=context/no loss, 1=positive target',
        weighted_exports='Assistant weight field duplicates the mask; trainer must implement it',
        index='1-based JSONL file row; message_index and mask positions are 0-based',
        splits=['train', 'validation', 'test', 'test_unscored', 'test_related_holdout', 'review', 'quarantine'],
        quality_lanes=['core', 'recovery', 'review', 'quarantine'],
        lengths='serialized_characters are not tokenizer counts'))


def finalize(out):
    archive = out / 'qoherent_v2_dataset.zip'
    paths = sorted(p for p in out.rglob('*') if p.is_file() and p.name not in {'checksums.sha256', archive.name} and '__pycache__' not in p.parts)
    (out / 'checksums.sha256').write_text(''.join(f'{sha_file(p)}  {p.relative_to(out).as_posix()}\n' for p in paths), encoding='utf-8')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for p in paths + [out / 'checksums.sha256']:
            z.write(p, p.relative_to(out).as_posix())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--legacy', type=Path, default=Path('processed/qoherent_v1'))
    p.add_argument('--out', type=Path, default=Path('processed/qoherent_v2'))
    a = p.parse_args()
    root, out, legacy = a.root.resolve(), a.out.resolve(), a.legacy.resolve()
    assert out != legacy and legacy not in out.parents and out not in legacy.parents
    assert out.parent == root / 'processed', 'Output must be a version directory directly under workspace/processed'
    stats = build(root, out, legacy)
    from verify_dataset_v2 import verify
    report = verify(root, out, check_checksums=False)
    write_json(out / 'audit/verification.json', report)
    print(dumps(stats), flush=True)
    print('Verification:', report['status'], flush=True)
    finalize(out)
    print('Packaged:', out, flush=True)


if __name__ == '__main__':
    main()
