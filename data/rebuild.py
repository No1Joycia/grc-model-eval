"""One-command raw-to-v3 rebuild using private source data beside this repo.

Does not download data, modify source inputs, overwrite releases or push anything.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path('..'))
    parser.add_argument('--output', type=Path, default=Path('dataset/qoherent_v3_reproduced'))
    parser.add_argument('--tag', default='reproduction', help='Unique intermediate build tag; existing paths are never overwritten')
    parser.add_argument('--compare-to', type=Path, help='Optional existing release to compare the three main JSONL files byte-for-byte')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    workspace, output = args.workspace.resolve(), args.output.resolve()
    if not args.tag.replace('-', '').replace('_', '').isalnum():
        raise SystemExit('--tag must contain only letters, numbers, hyphens or underscores')
    if repo != workspace / 'nanbeige-modal':
        raise SystemExit('For the legacy pipeline, clone this repo as <workspace>/nanbeige-modal; see data/REPRODUCE.md')
    for name in ['GRC_dataset', 'GRC_Agent_data', 'NewData1', 'NewData2']:
        if not (workspace / name).is_dir():
            raise SystemExit('Missing private input folder: ' + str(workspace / name))
    if repo / 'dataset' not in output.parents:
        raise SystemExit('--output must be a new directory inside this repo/dataset')
    stage1 = workspace / 'processed' / (args.tag + '_v1')
    stage2 = workspace / 'processed' / (args.tag + '_v2')
    logs = repo / 'dataset' / 'rebuild_logs' / args.tag
    for path in [stage1, stage2, output, logs]:
        if path.exists():
            raise SystemExit('Refusing to overwrite ' + str(path) + '; choose a new --tag and --output')
    logs.mkdir(parents=True)
    commands = [
        ('1_clean_original_data', [sys.executable, str(repo / 'data/legacy/process_data.py'), '--root', str(workspace), '--out', str(stage1)]),
        ('2_merge_and_split', [sys.executable, str(repo / 'data/legacy/build_dataset_v2.py'), '--root', str(workspace), '--legacy', str(stage1), '--out', str(stage2)]),
        ('3_portable_export', [sys.executable, str(repo / 'data/prepare.py'), '--source', str(stage2), '--output', str(output)]),
        ('4_verify_release', [sys.executable, str(repo / 'data/validate.py'), '--dataset', str(output)]),
    ]
    for stage, command in commands:
        log_path = logs / (stage + '.log')
        print(stage + ' ... log: ' + str(log_path), flush=True)
        with log_path.open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=repo, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            tail = log_path.read_text(encoding='utf-8').splitlines()[-15:]
            raise SystemExit(stage + ' failed:\n' + '\n'.join(tail))
        print(stage + ' PASS', flush=True)
    receipt = dict(status='PASS', output=str(output), stages=[stage for stage, _ in commands], record_hashes={})
    for name in ['train.jsonl', 'validation.jsonl', 'test.jsonl']:
        value = sha(output / name)
        receipt['record_hashes'][name] = value
        if args.compare_to:
            if value != sha(args.compare_to.resolve() / name):
                raise SystemExit('Rebuild completed but differs from comparison release: ' + name)
    if args.compare_to:
        receipt['comparison'] = 'All three primary files are byte-identical to ' + str(args.compare_to.resolve())
    (logs / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
