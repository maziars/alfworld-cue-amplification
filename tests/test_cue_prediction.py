import json
from pathlib import Path
import unittest

from research.analyze_cue_prediction import metrics


class CuePredictionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'cue_condition': c} for c in ['test', 'test', 'absent', 'absent']]

    def test_perfect_prediction(self):
        result = metrics(self.rows, lambda r: r['cue_condition'] == 'test')
        for key in ('precision', 'recall', 'f1', 'balanced_accuracy'):
            self.assertEqual(result[key], 1)

    def test_always_cued_is_not_discrimination(self):
        result = metrics(self.rows, lambda r: True)
        self.assertEqual(result['balanced_accuracy'], .5)
        self.assertAlmostEqual(result['f1'], 2 / 3)
        self.assertEqual(result['precision'], .5)

    def test_no_matches_has_undefined_precision(self):
        result = metrics(self.rows, lambda r: False)
        self.assertIsNone(result['precision'])
        self.assertEqual(result['f1'], 0)
        self.assertEqual(result['balanced_accuracy'], .5)

    def test_balanced_accuracy_uses_class_denominators(self):
        result = metrics(self.rows + [{'cue_condition': 'absent'}] * 3, lambda r: True)
        self.assertEqual(result['balanced_accuracy'], .5)
        self.assertAlmostEqual(result['precision'], 2 / 7)

    def test_archived_marker_counts(self):
        path = Path(__file__).resolve().parents[1] / 'research/results/broader_awareness_20261011/episode_metrics.json'
        rows = json.loads(path.read_text())
        expected = {('null', 'exact_first'): (27, 27),
                    ('positive', 'exact_first'): (25, 24),
                    ('null', 'expanded_any'): (30, 31),
                    ('positive', 'expanded_any'): (35, 31)}
        for (arm, field), counts in expected.items():
            result = metrics([r for r in rows if r['arm'] == arm], lambda r: r[field])
            self.assertEqual((result['tp'], result['fp']), counts)
            self.assertEqual(result['tp'] + result['fn'], 180)
            self.assertEqual(result['fp'] + result['tn'], 180)


if __name__ == '__main__':
    unittest.main()
