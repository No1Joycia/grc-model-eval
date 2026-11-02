"""Fetch public configuration/templates only. Never fetch or execute model code/weights."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'grc-data-compatibility/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(64 * 1024 * 1024 + 1)
    if len(data) > 64 * 1024 * 1024:
        raise ValueError('Configuration unexpectedly exceeds 64 MiB')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('dataset/model_metadata'))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    assert repo / 'dataset' in output.parents or output == repo / 'dataset'
    settings = json.loads((repo / 'serve/models.json').read_text())['models']['nanbeige42']['source']
    models = [('nanbeige42', settings['hf_repo'], settings['revision']),
              ('north_mini_code', 'CohereLabs/North-Mini-Code-1.0', None)]
    manifest = {}
    for key, model_id, revision in models:
        info = json.loads(get('https://huggingface.co/api/models/' + model_id))
        revision = revision or info['sha']
        files = {r['rfilename'] for r in info['siblings']}
        allowed = {'config.json', 'tokenizer_config.json', 'generation_config.json', 'special_tokens_map.json', 'chat_template.jinja'}
        folder = output / key
        folder.mkdir(parents=True, exist_ok=True)
        records = []
        for name in sorted(files & allowed):
            url = f'https://huggingface.co/{model_id}/resolve/{revision}/{name}'
            data = get(url)
            (folder / name).write_bytes(data)
            records.append(dict(file=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), url=url))
        manifest[key] = dict(model_id=model_id, revision=revision, files=records,
                             license=info.get('cardData', {}).get('license'))
        print(key, revision, [(r['file'], r['bytes']) for r in records], flush=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
