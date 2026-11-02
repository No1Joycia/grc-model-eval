"""Render canonical chats with pinned native templates, without model code or weights."""
import argparse
import collections as C
import json
from pathlib import Path
import re

from jinja2.sandbox import ImmutableSandboxedEnvironment

from prepare import rows, sha, write_json


def compile_template(text):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
        extensions=['jinja2.ext.loopcontrols'])
    env.filters['tojson'] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
    def raise_exception(message):
        raise ValueError(message)
    env.globals['raise_exception'] = raise_exception
    return env.from_string(text)


def model_templates(repo, metadata):
    manifest = json.loads((metadata / 'manifest.json').read_text(encoding='utf-8'))
    nan_path = repo / 'serve/templates/nanbeige42.jinja'
    north_path = metadata / 'north_mini_code/chat_template.jinja'
    config = json.loads((metadata / 'north_mini_code/tokenizer_config.json').read_text(encoding='utf-8'))
    return {
        'nanbeige42': dict(template=compile_template(nan_path.read_text(encoding='utf-8')),
            file=nan_path, revision=manifest['nanbeige42']['revision'], model_id=manifest['nanbeige42']['model_id'],
            options=dict(enable_thinking=False, preserve_thinking=False, tool_call_format='xml', bos_token='<|im_start|>', add_generation_prompt=False)),
        'north_mini_code': dict(template=compile_template(north_path.read_text(encoding='utf-8')),
            file=north_path, revision=manifest['north_mini_code']['revision'], model_id=manifest['north_mini_code']['model_id'],
            options=dict(reasoning=False, skip_thinking=True, regen_tool_call_ids=True,
                bos_token=config['bos_token'], add_generation_prompt=False)),
    }


def render(row, model):
    return model['template'].render(messages=row['messages'], tools=row['tools'], **model['options'])


def check(repo, dataset, metadata):
    models = model_templates(repo, metadata)
    report = {}
    for name, model in models.items():
        counts = C.Counter()
        for relative in ['train.jsonl', 'validation.jsonl', 'test.jsonl', 'optional/recovery_train.jsonl']:
            for row in rows(dataset / relative):
                text = render(row, model)
                assert text, row['id']
                calls = [c for m in row['messages'] for c in m.get('tool_calls', [])]
                if name == 'north_mini_code':
                    rendered_actions = [a for part in re.findall(r'<\|START_ACTION\|>(.*?)<\|END_ACTION\|>', text, re.S) for a in json.loads(part)]
                    assert len(rendered_actions) == len(calls), row['id']
                    for c, a in zip(calls, rendered_actions):
                        assert isinstance(a['parameters'], dict)
                        assert a['tool_name'] == c['function']['name'] and a['parameters'] == c['function']['arguments']
                    counts['mixed_messages_whose_text_native_template_omits'] += sum(bool(m.get('content')) and bool(m.get('tool_calls')) for m in row['messages'] if m['role'] == 'assistant')
                counts['conversations_rendered'] += 1
                counts['tool_calls_rendered'] += len(calls)
                counts['rendered_characters'] += len(text)
        report[name] = dict(status='PASS', model_id=model['model_id'], revision=model['revision'],
            template_sha256=sha(model['file']), render_options=model['options'], counts=dict(counts),
            assistant_generation_spans_native=('{% generation' in model['file'].read_text(encoding='utf-8')),
            scope='Pinned Jinja serialization only. No tokenizer, model weights, inference, gradient or training run.',
            warnings=['Native template lacks assistant generation spans. Verify a custom token mask/training template before training.'] +
                (['Native North template serializes tool actions but omits visible content on mixed assistant text+call messages. Canonical content is retained; decide the supervision policy in the model adapter.'] if name == 'north_mini_code' else []))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, default=Path('dataset/qoherent_v3'))
    p.add_argument('--metadata', type=Path, default=Path('dataset/model_metadata'))
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    result = check(repo, a.dataset.resolve(), a.metadata.resolve())
    write_json(a.dataset / 'template_compatibility.json', result)
    from prepare import checksums
    checksums(a.dataset.resolve())
    print(json.dumps(result, indent=2))
