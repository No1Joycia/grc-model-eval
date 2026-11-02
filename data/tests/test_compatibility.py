"""Synthetic regressions for native baseline serialization, not training examples."""
import copy
import json
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare import normalize
from render_check import model_templates, render


class CompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[2]
        cls.models = model_templates(cls.repo, cls.repo / 'dataset/model_metadata')

    def fixture(self):
        return dict(messages=[dict(role='system', content='Instructions'),
            dict(role='user', content='Synthetic serialization check'),
            dict(role='assistant', content=None, tool_calls=[dict(id='opaque_7', type='function',
                function=dict(name='check_example', arguments=json.dumps(dict(nested={'v': [1, False, None]}, label='quote " and newline\n'))))]),
            dict(role='tool', name='check_example', tool_call_id='opaque_7', content='{"ok":true}'),
            dict(role='assistant', content='Finished')],
            tools=[dict(type='function', function=dict(name='check_example', description='Synthetic tool', parameters=dict(type='object')))])

    def test_nanbeige_xml_arguments_and_no_literal_none(self):
        row, _, _, _ = normalize(self.fixture(), [0, 0, 1, 0, 1], 'Domain')
        text = render(row, self.models['nanbeige42'])
        self.assertIn('<parameter=nested>', text)
        self.assertIn('false', text)
        self.assertNotIn('\nNone', text)

    def test_north_parameters_are_objects_and_results_use_same_remapped_id(self):
        row, _, _, _ = normalize(self.fixture(), [0, 0, 1, 0, 1], 'Domain')
        text = render(row, self.models['north_mini_code'])
        action = json.loads(re.search(r'<\|START_ACTION\|>(.*?)<\|END_ACTION\|>', text, re.S)[1])[0]
        result = json.loads(re.search(r'<\|START_TOOL_RESULT\|>(.*?)<\|END_TOOL_RESULT\|>', text, re.S)[1])[0]
        self.assertIsInstance(action['parameters'], dict)
        self.assertEqual(action['parameters']['nested']['v'], [1, False, None])
        self.assertEqual(action['tool_call_id'], result['tool_call_id'])

    def test_two_leading_system_messages_render_like_one_merged_system(self):
        row, _, _, _ = normalize(self.fixture(), [0, 0, 1, 0, 1], 'Domain')
        split = copy.deepcopy(row)
        split['messages'].insert(1, dict(role='system', content='Graph context'))
        merged = copy.deepcopy(row)
        merged['messages'][0]['content'] += '\n\nGraph context'
        self.assertEqual(render(split, self.models['nanbeige42']), render(merged, self.models['nanbeige42']))

    def test_added_qa_system_shifts_loss_mask_without_changing_targets(self):
        raw = dict(messages=[dict(role='user', content='Q'), dict(role='assistant', content='A')], tools=[])
        row, mask, origin, _ = normalize(raw, [0, 1], 'Domain')
        self.assertEqual(mask, [0, 0, 1])
        self.assertEqual(origin, [None, 0, 1])
        self.assertEqual(row['messages'][-1]['content'], 'A')
        self.assertEqual(raw['messages'][0]['role'], 'user')


if __name__ == '__main__':
    unittest.main()
