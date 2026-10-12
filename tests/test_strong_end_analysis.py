import copy
import json
from pathlib import Path
import unittest
from research.analyze_strong_end import analyze


class StrongEndAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'research/results/single_stage_strong_end_20261011/audited_snapshot.json'
        if not path.exists():raise unittest.SkipTest('Audited private snapshot not available')
        cls.snaps=json.loads(path.read_text())
        cls.report=analyze(cls.snaps,draws=1000)

    def test_counts_and_previously_independent_status_totals(self):
        p=self.report['probabilities']
        self.assertEqual(p['null']['P(Y)']['numerator'],262)
        self.assertEqual(p['positive']['P(Y)']['numerator'],280)
        self.assertEqual(p['null']['P(I)']['numerator'],54)
        self.assertEqual(p['positive']['P(I)']['numerator'],49)
        self.assertEqual(p['null']['P(Y|I=1)']['numerator'],36)
        self.assertEqual(p['positive']['P(Y|I=1)']['numerator'],40)
        self.assertNotIn('pass@4',p['null'])

    def test_effects_seed_units_and_leave_one_out(self):
        t=self.report['tests']
        self.assertAlmostEqual(t['between_success_overall']['effect_pp'],5.)
        self.assertAlmostEqual(t['primary_awareness_interaction']['effect_pp'],100/180)
        self.assertEqual(t['between_success_overall']['seed_pair_p'],.25)
        self.assertEqual(t['primary_awareness_interaction']['p'],.5)
        for c in t.values():
            self.assertGreaterEqual(c['holm_p_16_exploratory'],c['p'])
        self.assertTrue(all(v['effect_pp']>0 for v in t['between_success_overall']['leave_one_seed_out'].values()))

    def test_changed_manifest_or_missing_model_rejected(self):
        bad=copy.deepcopy(self.snaps);bad[0]['rows'][0]['requested_generation_seed']+=1
        with self.assertRaises(AssertionError):analyze(bad,draws=10)
        with self.assertRaises(AssertionError):analyze(self.snaps[:-1],draws=10)


if __name__=='__main__':unittest.main()
