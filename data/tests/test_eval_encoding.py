"""Eval readers accept portable UTF-8/BOM JSONL under Windows legacy encoding."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class EvaluationEncodingTests(unittest.TestCase):
    def test_both_eval_clis_read_unicode_bom_questions_without_model_calls(self):
        repo = Path(__file__).resolve().parents[2]
        question = dict(id='synthetic_encoding_check', question='Compare \u201cIQ\u201d at 20 \u00b5s.', difficulty='basic', task_type='reasoning')
        temp_root = repo / 'dataset' / '_test_tmp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            self.assertIn(temp_root.resolve(), Path(directory).resolve().parents)
            path = Path(directory) / 'questions.jsonl'
            path.write_text(json.dumps(question, ensure_ascii=False) + '\n', encoding='utf-8-sig')
            for script in ['trial_eval/run_trial.py', 'northMiniCode/run_raw.py']:
                with self.subTest(script=script):
                    result = subprocess.run([sys.executable, script, '--questions', str(path), '--estimate-only'], cwd=repo,
                        env={**os.environ, 'PYTHONUTF8': '0', 'PYTHONIOENCODING': 'utf-8'}, capture_output=True, text=True, encoding='utf-8')
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('ESTIMATE:', result.stdout)

    def test_html_report_preserves_unicode_without_embedding_model(self):
        repo = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location('score_encoding_test', repo / 'trial_eval/score_trial.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metrics = dict(sem_primary=.9, sem_best=.9, sem_floor=.3, sem_ceiling=.8,
            sem_norm=1.2, rougeL=.7, num_recall=None, len_ratio=1.0, finish_reason='stop')
        row = dict(id='synthetic', num=1, difficulty='basic', task_type='reasoning', is_gnuradio=False, **metrics)
        group = dict(n=1, truncated=0, **metrics)
        summary = dict(overall=group, by_difficulty={'basic': group}, by_task_type={'reasoning': group},
            by_is_gnuradio={'False': group}, settings=dict(answers='synthetic', embed_model='none'))
        questions = {'synthetic': dict(question='Compare \u201cIQ\u201d at 20 \u00b5s.', reference_answer='20 \u00b5s', reference_model='synthetic')}
        temp_root = repo / 'dataset' / '_test_tmp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            self.assertIn(temp_root.resolve(), Path(directory).resolve().parents)
            target = Path(directory) / 'report.html'
            module.write_html(target, [row], summary, questions, {'synthetic': {'answer': '20 \u00b5s'}}, 'Synthetic check')
            self.assertIn('20 \u00b5s', target.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
