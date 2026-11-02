"""Stream portable examples and their loss/provenance metadata without Arrow coercion."""
import argparse
import collections
import itertools
import json
from pathlib import Path


def iter_split(dataset, split='train', *, include_recovery=False, kind=None):
    dataset = Path(dataset)
    if split not in {'train', 'validation', 'test'}:
        raise ValueError('split must be train, validation or test')
    if include_recovery and split != 'train':
        raise ValueError('Recovery extension applies to training only')
    if kind not in {None, 'qna', 'trace'}:
        raise ValueError('kind must be qna, trace or None')
    files = [(dataset / f'{split}.jsonl', dataset / f'metadata/{split}.jsonl')]
    if include_recovery:
        files.append((dataset / 'optional/recovery_train.jsonl', dataset / 'metadata/optional__recovery_train.jsonl'))
    for examples, metadata in files:
        with examples.open(encoding='utf-8-sig') as example_file, metadata.open(encoding='utf-8-sig') as meta_file:
            example_lines = (line for line in example_file if line.strip())
            meta_lines = (line for line in meta_file if line.strip())
            for line, (raw, raw_meta) in enumerate(itertools.zip_longest(example_lines, meta_lines), 1):
                if raw is None or raw_meta is None:
                    raise ValueError(f'Example/metadata count mismatch in {examples}')
                record, meta = json.loads(raw), json.loads(raw_meta)
                if record['id'] != meta['id'] or meta['row'] != line:
                    raise ValueError(f'Example/metadata identity mismatch at {examples}:{line}')
                if kind is None or meta['kind'] == kind:
                    yield record, meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('dataset/qoherent_v3'))
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='train')
    parser.add_argument('--include-recovery', action='store_true')
    args = parser.parse_args()
    count = collections.Counter(meta['kind'] for _, meta in iter_split(args.dataset, args.split, include_recovery=args.include_recovery))
    print(json.dumps(dict(records=sum(count.values()), kinds=dict(count)), indent=2))
