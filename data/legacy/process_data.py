"""Reproducible, dependency-free Qoherent preparation. Originals are read only."""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import unicodedata
import zipfile

VERSION = '1.0.0'
SEED = 'qoherent-preparation-v1'


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def norm(text):
    # Do not discard punctuation: mathematical operators carry meaning.
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def clean(text):
    return text.replace('\r\n', '\n').replace('\r', '\n').strip()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(dumps(row) + '\n')


def read_rows(path):
    with path.open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


TASK_MAP = {
    'conceptual': 'factual', 'debugging': 'diagnosis', 'system_design': 'design',
    'factual_knowledge': 'factual', 'calculations': 'calculation',
    'compare/choose': 'reasoning', 'compare_choose': 'reasoning', 'compare': 'reasoning',
    'diagnosis_from_symptoms': 'diagnosis', 'gnu_radio_usage': 'gnuradio_usage',
    'ask_for_info': 'diagnosis', 'information_gathering': 'diagnosis',
    'asking_for_missing_information': 'diagnosis',
}

# Suggestions only. Preserve the source label; do not pretend these are reviewed labels.
TOPIC_RULES = {
    'signals': r'\biq\b|baseband|sampl|alias|spectr|\bfft\b|\bdft\b|fourier|\bpsd\b|\bdbm?\b|\bsnr\b|signal|noise|convolution|transform',
    'filtering_sampling': r'filter|\bfir\b|\biir\b|decimat|interpolat|resampl|polyphase|chebyshev|butterworth|windowed.sinc',
    'analog_modulation': r'\bam\b|\bfm\b|\bpm\b|analog.modulat|frequency.modulat|amplitude.modulat',
    'digital_modulation': r'\bpsk\b|\bqpsk\b|\bqam\b|\bfsk\b|\bask\b|constellation|pulse.shap|raised.cosine|digital.modulat',
    'receiver_synchronization': r'synchron|carrier.recovery|timing.recovery|\bpll\b|\bagc\b|\bcfo\b|costas|clock.offset',
    'channels_impairments': r'impairment|\bawgn\b|fading|multipath|interference|clipping|imbalance|\bdc.offset|phase.noise|equaliz',
    'packet_systems': r'packet|framing|preamble|\bcrc\b|\bfec\b|\bpdu\b|tagged.stream|stream.tag|access.code',
    'ofdm': r'\bofdm\b|subcarrier|cyclic.prefix',
    'gnuradio_mechanics': r'gnu.radio|gnuradio|scheduler|\bpmt\b|oot.module|embedded.python|flowgraph|hierarchical.block',
    'sdr_hardware': r'\bsdr\b|\busrp\b|hackrf|bladerf|limesdr|rtl.sdr|hardware|antenna|overrun|underrun|\badc\b|\bdac\b',
    'cellular_3gpp': r'3gpp|cellular|\blte\b|\b5g\b|\b4g\b|\bnr\b|\bgsm\b|\bumts\b',
    'radar': r'radar|fmcw|range.doppler',
    'propagation': r'propagation|path.loss|shadowing|diffraction|friis',
    'satellite_links': r'satellite|orbital|\bgnss\b|\bgps\b|rain.fade',
}


def label_question(q):
    raw = q['primary_topic']
    key = re.sub(r'[\s-]+', '_', raw.strip().lower())
    if key in TOPIC_RULES:
        topic, method, candidates = key, 'controlled_label', [key]
    else:
        candidates = [k for k, pat in TOPIC_RULES.items() if re.search(pat, raw, re.I)]
        method = 'unique_label_keyword_suggestion' if len(candidates) == 1 else 'needs_review'
        topic = candidates[0] if len(candidates) == 1 else 'unmapped'
    task_key = re.sub(r'[\s-]+', '_', q['task_type'].strip().lower())
    task = TASK_MAP.get(task_key, task_key)
    return dict(topic=topic, topic_candidates=candidates, topic_method=method,
                task_type=task, task_type_raw=q['task_type'], topic_raw=raw,
                task_mapping_requires_review=task_key in {'conceptual', 'ask_for_info', 'information_gathering', 'asking_for_missing_information'})


class Groups:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def join(self, a, b):
        if a in self.parent and b in self.parent:
            a, b = self.find(a), self.find(b)
            self.parent[max(a, b)] = min(a, b)


def make_groups(qs, dataset):
    groups = Groups(qs)
    edges = []

    def edge(a, b, kind, **extra):
        if a != b and a in qs and b in qs:
            groups.join(a, b)
            edges.append(dict(id_a=min(a, b), id_b=max(a, b), method=kind, **extra))

    exact = collections.defaultdict(list)
    for qid, q in qs.items():
        exact[norm(q['question'])].append(qid)
    for ids in exact.values():
        for a, b in itertools.combinations(sorted(ids), 2):
            edge(a, b, 'normalized_exact')
    for rel in ['03_questions_merged/semantic_duplicate_clusters.jsonl',
                '03_questions_merged/cross_bank_semantic_duplicate_clusters.jsonl',
                '02_questions_blind/normalized/blind_semantic_duplicate_clusters.jsonl']:
        for row in read_rows(dataset / rel):
            ids = [q['id'] for key in ['questions', 'source_questions', 'blind_questions'] for q in row.get(key, [])]
            for a, b in itertools.combinations(sorted(set(ids)), 2):
                edge(a, b, 'supplied_semantic_cluster', source=rel, cluster_id=row.get('cluster_id'))

    # High-overlap candidates are grouped for split protection only; never deleted.
    grams = {}
    inverted = collections.defaultdict(set)
    for qid in sorted(qs):
        words = re.findall(r'\w+|[+*/=<>-]', norm(qs[qid]['question']))
        grams[qid] = set(zip(words, words[1:], words[2:]))
        for gram in grams[qid]:
            inverted[gram].add(qid)
    pairs = set()
    for ids in inverted.values():
        if len(ids) <= 80:  # Skip generic phrases; all-pairs threshold checked below.
            pairs.update(itertools.combinations(sorted(ids), 2))
    for a, b in sorted(pairs):
        ga, gb = grams[a], grams[b]
        if min(len(ga), len(gb)) < 5:
            continue
        score = len(ga & gb) / len(ga | gb)
        if score >= 0.5:
            edge(a, b, 'word_trigram_jaccard_candidate', similarity=round(score, 6))
    components = collections.defaultdict(list)
    for qid in sorted(qs):
        components[groups.find(qid)].append(qid)
    return groups, components, edges


def percentiles(values):
    v = sorted(values)
    return {str(p): v[min(len(v) - 1, math.ceil(p / 100 * len(v)) - 1)] if v else None for p in [50, 90, 95, 99, 100]}


def process(root, out):
    dataset = root / 'GRC_dataset/dataset'
    trace_dir = root / 'GRC_Agent_data/grc-agent-sft-v0.1.0'
    out.mkdir(parents=True, exist_ok=True)
    source_files = sorted(p for folder in [root / 'GRC_dataset', root / 'GRC_Agent_data'] for p in folder.rglob('*') if p.is_file())
    inventory = [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=sha_file(p)) for p in source_files]
    write_rows(out / 'audit/source_inventory.jsonl', inventory)
    consumed = set()

    def load(p):
        consumed.add(p.relative_to(root).as_posix())
        result = []
        with p.open(encoding='utf-8-sig') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    obj = json.loads(line)
                    obj['_source'] = dict(file=p.relative_to(root).as_posix(), line=line_num)
                    result.append(obj)
        return result

    source_questions = load(dataset / '03_questions_merged/questions_canonical.jsonl')
    qs = {q['id']: q for q in source_questions}
    assert len(qs) == len(source_questions), 'Nonunique canonical IDs'
    train_ids = {q['id'] for q in load(dataset / '04_eval_frozen/train_questions.jsonl')}
    eval_source = load(dataset / '04_eval_frozen/eval_questions.jsonl')
    eval_ids = {q['id'] for q in eval_source}
    assert not train_ids & eval_ids and train_ids | eval_ids == set(qs)
    assert all(norm(q['question']) == norm(qs[q['id']]['question']) for q in eval_source)
    groups, components, edges = make_groups(qs, dataset)
    for rel in ['03_questions_merged/semantic_duplicate_clusters.jsonl', '03_questions_merged/cross_bank_semantic_duplicate_clusters.jsonl', '02_questions_blind/normalized/blind_semantic_duplicate_clusters.jsonl']:
        consumed.add((dataset / rel).relative_to(root).as_posix())
    eval_groups = {groups.find(i) for i in eval_ids}
    split = {}
    for group, ids in components.items():
        validation = int(digest(SEED + '|' + group)[:16], 16) / 2**64 < 0.1
        for qid in ids:
            split[qid] = ('eval' if qid in eval_ids else 'eval_related_holdout') if group in eval_groups else ('validation' if validation else 'train')
    questions = []
    for qid in sorted(qs):
        q = qs[qid]
        questions.append(dict(question_id=qid, question=clean(q['question']), split=split[qid],
            group_id=groups.find(qid), difficulty=q['difficulty'], origin=q['origin'],
            labels=label_question(q), source_section=q.get('source_section'), provenance=q['provenance'],
            generator_metadata={k: q[k] for k in ['generator', 'model', 'generators', 'models', 'generator_models', 'metadata_variants'] if k in q},
            source=q['_source'], source_urls=sorted(set(re.findall(r'https?://[^\s"<>\\]+', dumps(q)))),
            text_sha256=digest(clean(q['question']))))
    qmap = {q['question_id']: q for q in questions}
    write_rows(out / 'qna/questions.jsonl', questions)
    # Frozen question file is copied byte-for-byte, independent of our added metadata.
    frozen = out / 'qna/eval_questions_frozen.jsonl'
    frozen.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dataset / '04_eval_frozen/eval_questions.jsonl', frozen)
    write_rows(out / 'audit/question_relatedness.jsonl', edges)

    def pair(row):
        return row['question_id'], row.get('candidate_model', row.get('model'))

    def unique_index(rows):
        result = {pair(r): r for r in rows}
        assert len(rows) == len(result), 'Nonunique assessment/repair key'
        return result

    judgments = unique_index(load(dataset / '08_judging/judgments.jsonl'))
    validations = unique_index(load(dataset / '10_repair_validation/validation.jsonl'))
    secondary = unique_index(load(dataset / '08_judging/local_validation_deepseek_v4.jsonl'))
    repairs = {name: unique_index(load(dataset / folder / 'repaired_answers.jsonl')) for name, folder in [('standard', '10_repaired'), ('glm', '10_repaired_glm')]}
    answers = collections.defaultdict(list)
    for p in sorted(dataset.glob('05_answers/*/answers.jsonl')):
        for row in load(p):
            answers[pair(row)].append(row)
    assert set(judgments) == set(answers), 'Judgment/answer key mismatch'
    assert set(validations) == set(repairs['standard']) == set(repairs['glm'])
    assert all(qid in train_ids for qid, _ in answers), 'Answer input contains evaluation question'
    decisions, accepted, dup_audit = [], [], []
    for key in sorted(answers):
        qid, model = key
        variants = answers[key]
        judgment = judgments[key]
        status = judgment['status']
        conflict = len({clean(r.get('answer', '')) for r in variants}) > 1
        if len(variants) > 1:
            dup_audit.append(dict(question_id=qid, model=model, conflicting=conflict,
                variants=[dict(source=r['_source'], answer_sha256=digest(clean(r.get('answer', '')))) for r in variants]))
        chosen = None
        reason = None
        kind = None
        if status == 'REJECT':
            reason = 'supplied_judgment_reject'
        elif status == 'PASS':
            if conflict:
                reason = 'conflicting_original_versions_judgment_not_bound_to_answer_hash'
            elif secondary.get(key, {}).get('validator_verdict') == 'DISAGREE':
                reason = 'secondary_validator_disagreement'
            else:
                chosen = variants[0]
                kind = 'original_pass'
        elif status == 'REPAIR':
            v = validations.get(key, {})
            if v.get('repair_a_status') == 'PASS' and v.get('repair_b_status') == 'PASS':
                # A/B-to-file mapping is absent. Both pass, so either file is safe
                # under the supplied judgments. Stable tie-break avoids model ranking.
                name = min(repairs, key=lambda n: digest(SEED + '|' + qid + '|' + model + '|' + n))
                chosen = repairs[name][key]
                kind = 'dual_pass_repair_' + name
            elif v.get('repair_a_status') == 'FAIL' and v.get('repair_b_status') == 'FAIL':
                reason = 'both_repairs_failed'
            else:
                reason = 'repair_letter_to_file_mapping_unavailable'
        else:
            reason = 'unknown_judgment_status'
        text = clean(chosen.get('repaired_answer', chosen.get('answer', ''))) if chosen else ''
        if chosen and (not text or chosen.get('status', 'ok') != 'ok' or chosen.get('finish_reason') in {'length', 'max_tokens'}):
            chosen, reason = None, 'empty_failed_or_truncated_answer'
        if chosen and re.search(r'<(?:think|analysis)>', text, re.I):
            chosen, reason = None, 'embedded_reasoning_markup_needs_review'
        decision = dict(question_id=qid, candidate_model=model, original_status=status,
            disposition='accepted' if chosen else 'excluded', reason=reason or kind,
            judgment={k: v for k, v in judgment.items() if k != '_source'}, judgment_source=judgment['_source'],
            answer_sources=[r['_source'] for r in variants], conflicting_original_versions=conflict,
            secondary_validation={k: v for k, v in secondary.get(key, {}).items() if k not in {'_source', 'validator_base_url', 'usage'}},
            repair_validation={k: v for k, v in validations.get(key, {}).items() if k != '_source'},
            repair_sources={n: rs[key]['_source'] for n, rs in repairs.items() if key in rs})
        decisions.append(decision)
        if chosen:
            accepted.append(dict(example_id='qa_' + digest(qid + '|' + model)[:20], question_id=qid,
                candidate_model=model, answer=text, answer_sha256=digest(text), answer_kind=kind,
                answer_model=chosen.get('repair_model', chosen.get('model')), split=split[qid], group_id=groups.find(qid),
                source=chosen['_source'], original_status=status, question=qmap[qid]['question'],
                independently_verified=False))
    write_rows(out / 'audit/answer_decisions.jsonl', decisions)
    write_rows(out / 'audit/conflicting_answer_versions.jsonl', dup_audit)
    write_rows(out / 'qna/accepted_answers.jsonl', accepted)
    by_question = collections.defaultdict(list)
    for a in accepted:
        by_question[a['question_id']].append(a)
    selected = []
    for qid, candidates in sorted(by_question.items()):
        # One main answer/question. PASS first; seeded hash is a tie-break, not quality scoring.
        selected.append(min(candidates, key=lambda a: (a['original_status'] != 'PASS', digest(SEED + '|' + a['example_id']))))
    write_rows(out / 'qna/primary_answers.jsonl', selected)
    for name in ['train', 'validation']:
        subset = [a for a in selected if a['split'] == name]
        write_rows(out / f'qna/{name}.jsonl', [dict(messages=[dict(role='user', content=a['question']), dict(role='assistant', content=a['answer'])]) for a in subset])
        write_rows(out / f'qna/{name}_index.jsonl', [dict(row=i, example_id=a['example_id'], question_id=a['question_id'], group_id=a['group_id']) for i, a in enumerate(subset, 1)])

    trace_rows, trace_audits, events, binary_elisions, task_summaries = [], [], [], [], []
    input_traces = []
    for source_name, lane in [('sft_train.jsonl', 'completed_pilot'), ('quarantine.jsonl', 'quarantine')]:
        for row in load(trace_dir / source_name):
            input_traces.append((row, lane))
    all_tools = collections.Counter()
    removed_thinking = 0
    for row, lane in input_traces:
        tid = 'tr_' + digest(row['_source']['file'] + '|' + str(row['_source']['line']) + '|' + row['task_id'])[:20]
        tools = [dict(type='function', function=dict(name=t['name'], description=t.get('description', ''), parameters=t['parameters'])) for t in row['tools']]
        known = {t['function']['name'] for t in tools}
        injected = []
        if any(c['name'] == 'final_result' for m in row['messages'] for c in m.get('tool_calls', [])) and 'final_result' not in known:
            tools.append(dict(type='function', function=dict(name='final_result',
                description='Submit the structured final result to the interaction harness.',
                parameters=dict(type='object', properties=dict(explanation=dict(type='string'), actions_taken=dict(type='array', items=dict(type='string'))), required=['explanation']))))
            known.add('final_result')
            injected.append('final_result')
        messages = [dict(role='system', content=row['system'])]
        errors, warnings = [], []
        pending, seen = {}, set()
        tool_counts = collections.Counter()
        thinking = 0
        evidence = []
        for i, m in enumerate(row['messages']):
            # Early exports elided only the beginning of some image reprs, leaving
            # hundreds of thousands of escaped binary characters behind. Keep
            # an explicit placeholder and recoverable source/hash reference.
            m = dict(m)
            content = m.get('content')
            if m.get('role') == 'tool' and isinstance(content, str) and re.match(r'''^["']?\[binary image output elided:''', content):
                marker = content[:content.find(']') + 1].lstrip('\"\'')
                media = re.search(r"media_type=['\"]([^'\"]+)", content)
                placeholder = '[Binary image output omitted; visual content unavailable in this text export.]'
                binary_elisions.append(dict(trace_id=tid, source=row['_source'], source_message_index=i,
                    processed_message_index=i + 1, original_characters=len(content), original_sha256=digest(content),
                    original_marker=marker, media_type=media.group(1) if media else None,
                    replacement=placeholder))
                m['content'] = placeholder
            role = m['role']
            if m.get('thinking'):
                thinking += 1
            new = {k: v for k, v in m.items() if k in {'role', 'content', 'tool_call_id', 'name'}}
            if role not in {'assistant', 'user', 'tool'}:
                errors.append(dict(message=i, code='unknown_role'))
            if role in {'user', 'assistant'} and pending:
                errors.append(dict(message=i, code='interrupted_pending_tool_calls', ids=sorted(pending)))
            if m.get('tool_calls'):
                new['tool_calls'] = []
                for call in m['tool_calls']:
                    cid, name = call['id'], call['name']
                    if cid in seen:
                        errors.append(dict(message=i, code='duplicate_tool_call_id', tool_call_id=cid))
                    seen.add(cid)
                    pending[cid] = (name, i, call.get('arguments'))
                    tool_counts[name] += 1
                    if name not in known:
                        errors.append(dict(message=i, code='undefined_tool', name=name))
                    try:
                        args = json.loads(call['arguments']) if isinstance(call['arguments'], str) else call['arguments']
                        if not isinstance(args, dict):
                            raise ValueError('arguments not object')
                        if name == 'final_result' and not isinstance(args.get('explanation'), str):
                            errors.append(dict(message=i, code='invalid_final_result'))
                        arguments = dumps(args)
                    except (ValueError, TypeError):
                        errors.append(dict(message=i, code='invalid_argument_json', name=name))
                        arguments = call['arguments']
                    new['tool_calls'].append(dict(id=cid, type='function', function=dict(name=name, arguments=arguments)))
            if role == 'tool':
                cid = m.get('tool_call_id')
                match = pending.pop(cid, None)
                if match is None:
                    errors.append(dict(message=i, code='orphan_tool_result', tool_call_id=cid))
                else:
                    name, call_i, args = match
                    if m.get('name', name) != name:
                        errors.append(dict(message=i, code='tool_result_name_mismatch'))
                    try:
                        result = json.loads(m.get('content', ''))
                    except (ValueError, TypeError):
                        result = None
                    failure = isinstance(result, dict) and (result.get('ok') is False or result.get('success') is False or bool(result.get('error')))
                    event = dict(trace_id=tid, task_id=row['task_id'], call_message_index=call_i + 1, result_message_index=i + 1,
                        tool_call_id=cid, name=name, arguments=json.loads(args) if isinstance(args, str) and not any(e['code'] == 'invalid_argument_json' and e['message'] == call_i for e in errors) else args,
                        result_json=result, result_text=m.get('content'), reported_error=failure)
                    events.append(event)
                    if name in {'run_flowgraph', 'get_run_log', 'run_command', 'check_command', 'generate_python'}:
                        evidence.append(dict(message_index=i + 1, tool=name, reported_error=failure))
                if re.search(r'screenshot|image.{0,12}(elid|placeholder)|\[.{0,15}elid', str(m.get('content', '')), re.I):
                    warnings.append(dict(message_index=i + 1, code='possible_elided_or_visual_content'))
            messages.append(new)
        if pending:
            errors.append(dict(code='unanswered_tool_calls', ids=sorted(pending)))
        if row['messages'][-1]['role'] == 'user':
            errors.append(dict(code='unanswered_final_user_turn'))
        disposition = 'pilot_train' if lane == 'completed_pilot' and row['verdict'] == 'completed' and not errors else 'quarantine'
        removed_thinking += thinking
        if disposition == 'pilot_train':
            all_tools.update(tool_counts)
        chars = len(dumps(dict(messages=messages, tools=tools)))
        audit = dict(trace_id=tid, task_id=row['task_id'], category=row['category'], source_lane=lane,
            disposition=disposition, verdict=row['verdict'], quarantine_reason=row.get('quarantine_reason'),
            original_message_count=len(row['messages']), message_count=len(messages), removed_thinking_fields=thinking,
            tool_calls=dict(tool_counts), structural_errors=errors, warnings=warnings, injected_schemas=injected,
            execution_tool_observations=evidence, serialized_characters=chars,
            approximate_tokens_chars_div_4=math.ceil(chars / 4), independently_executed=False,
            source=row['_source'])
        trace_audits.append(audit)
        finals = []
        for mi, m in enumerate(messages):
            for call in m.get('tool_calls', []):
                if call['function']['name'] == 'final_result':
                    try:
                        finals.append(dict(message_index=mi, **json.loads(call['function']['arguments'])))
                    except (ValueError, TypeError):
                        pass
        task_summaries.append(dict(trace_id=tid, task_id=row['task_id'], category=row['category'],
            disposition=disposition, reported_verdict=row['verdict'], quarantine_reason=row.get('quarantine_reason'),
            initial_request=next((m.get('content') for m in messages if m['role'] == 'user'), None),
            final_results=finals, message_count=len(messages), tool_call_count=sum(tool_counts.values()),
            reported_tool_errors=sum(e['reported_error'] for e in events if e['trace_id'] == tid),
            binary_outputs_omitted=sum(e['trace_id'] == tid for e in binary_elisions),
            approximate_tokens_chars_div_4=math.ceil(chars / 4), independently_executed=False))
        trace_rows.append(dict(trace_id=tid, task_id=row['task_id'], session_id=row['session_id'],
            category=row['category'], disposition=disposition, source=row['_source'],
            teacher=row['teacher'], reported_usage=row.get('usage'), reported_verdict=row['verdict'],
            messages=messages, tools=tools))
    write_rows(out / 'traces/sessions.jsonl', trace_rows)
    write_rows(out / 'traces/tool_events.jsonl', events)
    write_rows(out / 'audit/trace_quality.jsonl', trace_audits)
    write_rows(out / 'audit/binary_output_elisions.jsonl', binary_elisions)
    write_rows(out / 'traces/task_summaries.jsonl', task_summaries)
    pilot = [r for r in trace_rows if r['disposition'] == 'pilot_train']
    write_rows(out / 'traces/pilot_train.jsonl', [dict(messages=r['messages'], tools=r['tools']) for r in pilot])
    write_rows(out / 'traces/pilot_train_index.jsonl', [dict(row=i, trace_id=r['trace_id'], task_id=r['task_id']) for i, r in enumerate(pilot, 1)])
    write_rows(out / 'traces/quarantine.jsonl', [r for r in trace_rows if r['disposition'] == 'quarantine'])

    # Cross-corpus lexical retrieval for human review; do not call this a semantic audit.
    cross_matches = []
    stop = set('the a an and or to of in is for with on how what why can you i it this that from'.split())
    def tokens(s):
        return set(re.findall(r'\w+', norm(s))) - stop
    eval_tokens = {i: tokens(qs[i]['question']) for i in eval_ids}
    for row in trace_rows:
        user_turns = [(i, m['content']) for i, m in enumerate(row['messages']) if m['role'] == 'user' and isinstance(m.get('content'), str)]
        for mi, text in user_turns:
            ts = tokens(text)
            for qid, es in eval_tokens.items():
                score = len(ts & es) / len(ts | es) if ts | es else 0
                exact = norm(qs[qid]['question']) in norm(text)
                if exact or (len(ts & es) >= 5 and score >= 0.35):
                    cross_matches.append(dict(trace_id=row['trace_id'], message_index=mi, eval_question_id=qid,
                        token_jaccard=round(score, 6), eval_question_substring=exact))
    write_rows(out / 'audit/trace_eval_overlap_candidates.jsonl', cross_matches)

    db_path = out / 'qoherent.sqlite'
    if db_path.exists():
        db_path.unlink()  # Only this generated file, never any source path.
    db = sqlite3.connect(db_path)
    db.executescript('''
    PRAGMA foreign_keys=ON;
    CREATE TABLE questions(question_id TEXT PRIMARY KEY, split TEXT, group_id TEXT, question TEXT, topic TEXT, task_type TEXT, difficulty TEXT, origin TEXT, record_json TEXT);
    CREATE TABLE answers(example_id TEXT PRIMARY KEY, question_id TEXT REFERENCES questions, candidate_model TEXT, answer_kind TEXT, answer TEXT, is_primary INTEGER, record_json TEXT);
    CREATE TABLE answer_decisions(question_id TEXT REFERENCES questions, candidate_model TEXT, disposition TEXT, reason TEXT, record_json TEXT, PRIMARY KEY(question_id,candidate_model));
    CREATE TABLE traces(trace_id TEXT PRIMARY KEY, task_id TEXT, category TEXT, disposition TEXT, record_json TEXT);
    CREATE TABLE messages(trace_id TEXT REFERENCES traces, message_index INTEGER, role TEXT, content TEXT, record_json TEXT, PRIMARY KEY(trace_id,message_index));
    CREATE TABLE tool_events(trace_id TEXT REFERENCES traces, tool_call_id TEXT, name TEXT, reported_error INTEGER, record_json TEXT, PRIMARY KEY(trace_id,tool_call_id));
    CREATE INDEX questions_split_idx ON questions(split);
    CREATE INDEX questions_group_idx ON questions(group_id);
    CREATE INDEX answers_question_idx ON answers(question_id);
    CREATE INDEX events_name_idx ON tool_events(name);
    ''')
    db.executemany('INSERT INTO questions VALUES(?,?,?,?,?,?,?,?,?)', [(q['question_id'], q['split'], q['group_id'], q['question'], q['labels']['topic'], q['labels']['task_type'], q['difficulty'], q['origin'], dumps(q)) for q in questions])
    primary_ids = {a['example_id'] for a in selected}
    db.executemany('INSERT INTO answers VALUES(?,?,?,?,?,?,?)', [(a['example_id'], a['question_id'], a['candidate_model'], a['answer_kind'], a['answer'], int(a['example_id'] in primary_ids), dumps(a)) for a in accepted])
    db.executemany('INSERT INTO answer_decisions VALUES(?,?,?,?,?)', [(a['question_id'], a['candidate_model'], a['disposition'], a['reason'], dumps(a)) for a in decisions])
    db.executemany('INSERT INTO traces VALUES(?,?,?,?,?)', [(r['trace_id'], r['task_id'], r['category'], r['disposition'], dumps(r)) for r in trace_rows])
    db.executemany('INSERT INTO messages VALUES(?,?,?,?,?)', [(r['trace_id'], i, m['role'], m.get('content') if isinstance(m.get('content'), str) else dumps(m.get('content')), dumps(m)) for r in trace_rows for i, m in enumerate(r['messages'])])
    # Quarantined inputs can repeat call IDs. Store each matched event with its result position.
    db.execute('DROP TABLE tool_events')
    db.execute('CREATE TABLE tool_events(trace_id TEXT REFERENCES traces, tool_call_id TEXT, name TEXT, reported_error INTEGER, result_message_index INTEGER, record_json TEXT, PRIMARY KEY(trace_id,tool_call_id,result_message_index))')
    db.executemany('INSERT INTO tool_events VALUES(?,?,?,?,?,?)', [(e['trace_id'], e['tool_call_id'], e['name'], int(e['reported_error']), e['result_message_index'], dumps(e)) for e in events])
    db.execute('CREATE INDEX events_name_idx ON tool_events(name)')
    fts = True
    try:
        db.executescript('CREATE VIRTUAL TABLE search USING fts5(kind UNINDEXED, record_id UNINDEXED, text);')
        db.executemany('INSERT INTO search VALUES(?,?,?)', [('question', q['question_id'], q['question']) for q in questions] + [('answer', a['example_id'], a['answer']) for a in accepted] + [('trace', r['trace_id'], dumps(r)) for r in task_summaries])
    except sqlite3.OperationalError:
        fts = False
    db.commit()
    assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    assert not db.execute('PRAGMA foreign_key_check').fetchall()
    db.close()

    stats = dict(version=VERSION, source_files=len(inventory), source_bytes=sum(r['bytes'] for r in inventory),
        questions=len(questions), question_splits=dict(collections.Counter(split.values())),
        raw_answers=sum(map(len, answers.values())), unique_answer_pairs=len(answers),
        conflicting_pairs=sum(a['conflicting'] for a in dup_audit),
        accepted_answers=len(accepted), accepted_by_kind=dict(collections.Counter(a['answer_kind'] for a in accepted)),
        excluded_answer_reasons=dict(collections.Counter(a['reason'] for a in decisions if a['disposition'] == 'excluded')),
        primary_answers=len(selected), primary_splits=dict(collections.Counter(a['split'] for a in selected)),
        questions_without_accepted_answers=[i for i in sorted(train_ids) if i not in by_question],
        topic_label_methods=dict(collections.Counter(q['labels']['topic_method'] for q in questions)),
        topics=dict(collections.Counter(q['labels']['topic'] for q in questions)),
        task_types=dict(collections.Counter(q['labels']['task_type'] for q in questions)),
        difficulty=dict(collections.Counter(q['difficulty'] for q in questions)),
        questions_with_urls=sum(bool(q['source_urls']) for q in questions),
        relatedness_edges=len(edges), related_groups=sum(len(v) > 1 for v in components.values()),
        largest_related_group=max(map(len, components.values())),
        completed_source_traces=sum(lane == 'completed_pilot' for _, lane in input_traces),
        pilot_traces=len(pilot), quarantine_traces=len(trace_rows)-len(pilot),
        pilot_message_count=sum(len(r['messages'])-1 for r in pilot),
        thinking_fields_removed_all=removed_thinking,
        thinking_fields_removed_pilot=sum(r['removed_thinking_fields'] for r in trace_audits if r['disposition'] == 'pilot_train'),
        residual_binary_outputs_omitted=len(binary_elisions),
        residual_binary_characters_removed=sum(e['original_characters'] - len(e['replacement']) for e in binary_elisions),
        injected_final_result_schemas=sum(bool(r['injected_schemas']) for r in trace_audits),
        pilot_tool_calls=dict(all_tools), tool_events=len(events),
        trace_eval_overlap_candidates=len(cross_matches),
        question_characters=percentiles([len(q['question']) for q in questions]),
        answer_characters=percentiles([len(a['answer']) for a in accepted]),
        pilot_approximate_tokens_chars_div_4=percentiles([a['approximate_tokens_chars_div_4'] for a in trace_audits if a['disposition'] == 'pilot_train']),
        sqlite_fts5=fts)
    write_json(out / 'audit/summary.json', stats)
    write_rows(out / 'audit/label_review_queue.jsonl', [dict(question_id=q['question_id'], question=q['question'], **q['labels']) for q in questions if q['labels']['topic_method'] != 'controlled_label' or q['labels']['task_mapping_requires_review']])
    write_rows(out / 'audit/model_quality.jsonl', [dict(model=model, judgments=dict(collections.Counter(r['status'] for (qid, m), r in judgments.items() if m == model)), accepted=sum(a['candidate_model'] == model for a in accepted), primary=sum(a['candidate_model'] == model for a in selected)) for model in sorted({m for _, m in answers})])
    write_json(out / 'manifest.json', dict(version=VERSION, seed=SEED, source_fingerprint=digest(dumps(inventory)), consumed_sources=sorted(consumed),
        units=dict(qna='question/model pair; primary exports one answer per question', trace='whole session', tool_event='matched tool call/result'),
        split_rule='Frozen eval unchanged; related train questions held out; remaining components hash to validation at probability 0.1.',
        relatedness='Supplied semantic clusters plus normalized exact and word-trigram Jaccard >=0.5 candidates; grouping never deletes questions. No new semantic embeddings.',
        answer_policy='Original PASS if unambiguous and undisputed, or repair if both A/B pass. Stable seeded tie-break among equivalent options. No new correctness claims.',
        trace_policy='Strip thinking fields and residual escaped image reprs; retain other message/tool content; wrap tools and calls as function tools; inject documented final_result schema. Quarantine never promoted.',
        trace_split='All structurally valid completed sessions are pilot training only. No independent agent evaluation split.',
        token_counts='Only serialized character counts and approximate chars/4 supplied. Exact template-specific assistant loss tokens must be computed by the trainer.',
        publication_permissions='Not established by supplied files; provenance retained; no publication performed.',
        privacy='Not a privacy-certified export. Original system prompts, file paths and tool outputs retained for fidelity; private validator endpoint omitted from audit.',
        quality='Existing model judgments retained; no RF/DSP expert revalidation or GNU Radio execution performed.'))
    return stats, inventory


def document(out, stats):
    s = stats
    report = f'''# Qoherent processed data v{VERSION}

Prepared from the supplied files with a reproducible local pipeline. Originals are unchanged.

| Result | Count |
|---|---:|
| Canonical questions | {s['questions']:,} |
| Frozen evaluation questions | {s['question_splits'].get('eval', 0):,} |
| Accepted question/model answers | {s['accepted_answers']:,} |
| Primary answers (one per question) | {s['primary_answers']:,} |
| Q&A training examples | {s['primary_splits'].get('train', 0):,} |
| Q&A validation examples | {s['primary_splits'].get('validation', 0):,} |
| Evaluation-related holdout questions | {s['question_splits'].get('eval_related_holdout', 0):,} |
| Structurally valid completed pilot traces | {s['pilot_traces']:,} |
| Quarantined traces | {s['quarantine_traces']:,} |
| Reasoning fields removed from pilot traces | {s['thinking_fields_removed_pilot']:,} |

## Start here

- `qoherent.sqlite`: queryable master database; questions, accepted answers, all answer decisions, traces, messages and tool events. Nested metadata is retained as JSON. Full-text search availability: {s['sqlite_fts5']}.
- `qna/train.jsonl` and `qna/validation.jsonl`: plain user/assistant message records, one selected answer per question. Companion `_index.jsonl` files map 1-based file rows back to IDs. All alternative answers retain the question's split in `accepted_answers.jsonl`.
- `qna/eval_questions_frozen.jsonl`: byte-identical original evaluation file, questions only. No reference answers are invented.
- `qna/questions.jsonl`: canonical questions, raw labels, normalized task labels, tentative topic suggestions, split/group IDs and provenance.
- `qna/primary_answers.jsonl`: chosen answers with source file/line, model and quality lineage. `accepted_answers.jsonl` retains alternatives for later analysis.
- `traces/pilot_train.jsonl`: whole-session function-tool conversations with a system message and per-session tools. Companion index maps rows to trace IDs. Mask system/user/tool text in your trainer and include assistant tool calls in the loss.
- `traces/sessions.jsonl`: all processed sessions including quarantine, metadata and dispositions. `tool_events.jsonl` exposes paired calls/results without reconstructing the conversation.
- `traces/task_summaries.jsonl` and `traces/TRACE_CATALOG.md`: browse initial requests, reported outcomes, tool errors and trace sizes without opening the full sessions.
- `audit/summary.json`, `answer_decisions.jsonl`, `trace_quality.jsonl`, `label_review_queue.jsonl`: counts, exclusions, structural audits and review tasks.

## Q&A selection and splitting

Accepted originals require an existing PASS judgment, a unique answer text, no supplied secondary-validator disagreement, nonempty content, and no recorded truncation. {s['conflicting_pairs']} conflicting original pairs are recorded with answer hashes and file/line pointers. A judgment is not assumed to identify one of those versions.

For repaired answers, both repair A and repair B must pass: the supplied files do not define the A/B-to-file mapping. Single-side passes remain excluded until that mapping is recovered. When both pass, a deterministic hash selects a repair file. Original PASS answers take priority for the primary answer; ties use a seeded hash. This is reproducible selection, not a new model-quality ranking. Existing scores are model judgments, not independently checked engineering ground truth.

Exclusion counts: `{json.dumps(s['excluded_answer_reasons'], sort_keys=True)}`.

The 185 frozen evaluation questions remain fixed. Exact matches, supplied semantic clusters (including KEEP_BOTH variations), and high lexical-overlap candidates are connected into split groups; no candidate is deleted. Training-pool questions related to frozen evaluation enter `eval_related_holdout`. Other groups enter validation when a seeded hash falls below 0.1. Every answer variant follows its question. This is conservative leakage reduction; unobserved semantic overlap may remain. Trace/evaluation overlap candidates found by lexical retrieval: {s['trace_eval_overlap_candidates']}; see the audit file. No new semantic-model inference was performed.

Topic normalization remains explicitly tentative: {json.dumps(s['topic_label_methods'], sort_keys=True)}. Ambiguous/no-match topics are `unmapped`, with raw labels and candidate labels retained. The topic field must not be treated as reviewed classification. Task aliases `conceptual` and information-request labels are also marked for review. No source URLs or permissions were invented.

## Trace processing and use

The completed source contains {s['completed_source_traces']} traces, despite its README claiming 13. The original train/full files are checked for byte equality in verification. Processing removes `thinking` fields, canonicalizes function calls with JSON-string arguments, retains tool results and their IDs, and injects the missing `final_result` function definition using its observed `explanation`/`actions_taken` contract. Every injection is recorded. Messages and tool results otherwise retain their content, including code, paths, failures and screenshot placeholders.

The audit checks tool argument parsing, call/result pairing and order, unique call IDs, tool names, schema presence, and unanswered final turns. If `jsonschema` is installed, verification also validates every pilot call against its function's JSON Schema; the verification report records whether this optional check ran. Recorded execution tool observations are indexed; commands were not executed again, and recorded success is not a new acceptance test. Error indicators reflect explicit top-level `ok=false`, `success=false` or nonempty `error` fields; they are not a complete diagnosis of textual logs. The quarantine contains failed, superseded and incomplete sessions and stays outside positive SFT. Do not create preference pairs from it without aligning tasks, context and outcomes.

Residual escaped image bytes were found in outputs already labelled elided. {s['residual_binary_outputs_omitted']} such outputs were replaced with explicit text-only placeholders, removing {s['residual_binary_characters_removed']:,} characters of binary noise. `audit/binary_output_elisions.jsonl` retains original hashes, media types and source message locations. No visual evidence is reconstructed or claimed. Other tool output is preserved.

Whole sessions are the reproducible master unit. They may exceed your model's context window: approximate serialized token percentiles (characters / 4) are `{json.dumps(s['pilot_approximate_tokens_chars_div_4'], sort_keys=True)}`. These are estimates, not tokenizer measurements or assistant-only loss-token counts. Tokenize with the target model's tool-aware chat template before training; parse argument strings if that template requires objects. Build context windows at user/assistant/tool transaction boundaries if needed, retain required graph state, and mask repeated context. This pilot is too small for an independent agent benchmark; no random message-level split was made.

## Query examples

```python
import json, sqlite3
db = sqlite3.connect('qoherent.sqlite')
db.row_factory = sqlite3.Row
rows = db.execute("""SELECT q.question_id, q.question, a.answer
    FROM questions q JOIN answers a USING(question_id)
    WHERE q.split='train' AND a.is_primary=1 LIMIT 5""").fetchall()
traces = db.execute("SELECT record_json FROM traces WHERE disposition='pilot_train'")
first_trace = json.loads(next(traces)[0])
```

```sql
SELECT reason, count(*) FROM answer_decisions WHERE disposition='excluded' GROUP BY reason;
SELECT category, disposition, count(*) FROM traces GROUP BY category, disposition;
SELECT record_id, snippet(search, 2, '[', ']', '...', 24) FROM search WHERE search MATCH 'OFDM' LIMIT 10;
```

## Reproduce and verify

Python 3.10+; standard library only. From the workspace root:

```powershell
python processing/process_data.py --root . --out processed/qoherent_v1
python processing/verify_data.py --root . --out processed/qoherent_v1
```

The inventory hashes every supplied source file; the manifest lists consumed inputs and policy. `checksums.sha256` covers every deliverable except itself and the archive. The ZIP includes the database, JSONL, audit, documentation and processing scripts. Re-running replaces only generated files in the specified output directory. Validation results are in `audit/verification.json`.

This is a processed research/pilot package. Dataset publication permissions are not established in the supplied files. Existing local paths and tool output are preserved, so this is not a privacy-certified public release. No external publication or upload was performed.
'''
    (out / 'README.md').write_text(report, encoding='utf-8')
    summaries = read_rows(out / 'traces/task_summaries.jsonl')
    catalog = ['# Trace catalog', '', 'Completed pilot traces are structurally checked; outcomes below are reported by the original agent, not independently reproduced.', '', '| Task | Category | Messages including system | Tool calls | Approx. tokens (chars/4) | Tool error indicators |', '|---|---|---:|---:|---:|---:|']
    for r in summaries:
        if r['disposition'] == 'pilot_train':
            catalog.append(f"| {r['task_id']} | {r['category']} | {r['message_count']} | {r['tool_call_count']} | {r['approximate_tokens_chars_div_4']:,} | {r['reported_tool_errors']} |")
    for r in summaries:
        if r['disposition'] != 'pilot_train':
            continue
        catalog.extend(['', '## ' + r['task_id'], '', '**Initial request:** ' + str(r['initial_request']), '', '**Last reported final result:**', '', r['final_results'][-1].get('explanation', 'No structured final result') if r['final_results'] else 'No structured final result', '', 'Trace ID: `' + r['trace_id'] + '`'])
    catalog.extend(['', '## Quarantine', '', '| Task | Reported verdict | Reason |', '|---|---|---|'])
    for r in summaries:
        if r['disposition'] == 'quarantine':
            reason = str(r['quarantine_reason'] or 'Structural or outcome gate').replace('|', '/').replace('\n', ' ')
            catalog.append(f"| {r['task_id']} | {r['reported_verdict']} | {reason} |")
    (out / 'traces/TRACE_CATALOG.md').write_text('\n'.join(catalog) + '\n', encoding='utf-8')
    write_json(out / 'schema.json', dict(version=VERSION, encoding='UTF-8 without BOM; one JSON object per line',
        row_index='Export indexes use 1-based file rows; trace message_index is 0-based including system at index 0.',
        question=dict(key='question_id', foreign_keys=['group_id'], labels='raw, normalized and tentative suggestions retained'),
        answer=dict(key='example_id', foreign_keys=['question_id'], split='train|validation|eval_related_holdout; inherits question'),
        trace=dict(key='trace_id', disposition='pilot_train|quarantine', messages='function-tool conversation; arguments are JSON strings', tools='type=function, function={name,description,parameters}'),
        tool_event=dict(key=['trace_id', 'tool_call_id', 'result_message_index'], result_json='parsed JSON when possible; result_text always retained'),
        sft=dict(qna='messages only; IDs in companion index', traces='messages and tools; IDs in companion index'),
        units=dict(length='characters, not bytes', approximate_tokens='ceil(serialized_characters/4); not measured tokens')))


def finalize(out):
    files = sorted(p for p in out.rglob('*') if p.is_file() and p.name not in {'checksums.sha256', 'qoherent_processed_v1.zip'})
    (out / 'checksums.sha256').write_text(''.join(f'{sha_file(p)}  {p.relative_to(out).as_posix()}\n' for p in files), encoding='utf-8')
    archive = out / 'qoherent_processed_v1.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files + [out / 'checksums.sha256']:
            z.write(p, p.relative_to(out).as_posix())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, default=Path('processed/qoherent_v1'))
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    # Refuse outputs that could overwrite immutable source trees.
    for source in [root / 'GRC_dataset', root / 'GRC_Agent_data']:
        if out == source or source in out.parents or out in source.parents:
            raise ValueError('Output must be separate from source directories')
    stats, _ = process(root, out)
    document(out, stats)
    scripts = out / 'processing'
    scripts.mkdir(exist_ok=True)
    for name in ['process_data.py', 'verify_data.py']:
        shutil.copyfile(Path(__file__).parent / name, scripts / name)
    from verify_data import verify
    verification = verify(root, out, check_checksums=False)
    write_json(out / 'audit/verification.json', verification)
    finalize(out)
    print(json.dumps(stats, indent=2))
    print('Verified package:', out)


if __name__ == '__main__':
    main()
