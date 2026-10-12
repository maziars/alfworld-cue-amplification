import unittest
from pathlib import Path
from research.analyze_group_exposure import analyze,features,corr


class GroupExposureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        data=root/'research/data/single_stage_strong_20261011'
        snap=root/'research/results/single_stage_strong_end_20261011_v2/audited_snapshot.json'
        if not data.exists() or not snap.exists():raise unittest.SkipTest('Private artifacts not present')
        cls.report=analyze(data,snap)

    def test_all_cue_fractions_balanced_and_family_quotas_fixed(self):
        for rows in self.report['tables'].values():
            self.assertTrue(all(r['train_cue_fraction']==.5 for r in rows if r['train_sources']))
        rows=self.report['tables']['task_family']
        for arm in ['null','positive']:
            for family in {r['group'] for r in rows}:
                rs=[r for r in rows if r['arm']==arm and r['group']==family]
                self.assertEqual(len({tuple(r['train_I_counts']) for r in rs}),1)
            self.assertIsNone(self.report['diagnostics']['task_family'][arm]['group_and_seed_demeaned_r'])

    def test_known_heat_effects_and_no_leakage(self):
        heat=[r for r in self.report['tables']['task_family'] if r['arm']=='positive' and r['group']=='heat']
        self.assertEqual([r['train_sources'] for r in heat],[16,16,16])
        self.assertEqual([r['eval_success_counts'] for r in heat],[[4,7],[4,6],[5,7]])
        for dim in ['task_group','leakage_group']:
            self.assertFalse(any(r['train_sources'] and r['eval_games'] for r in self.report['tables'][dim]))

    def test_finer_counterexamples(self):
        rows=self.report['tables']['family_object']
        bread=[r for r in rows if r['group']=='heat/Bread']
        self.assertTrue(all(r['train_sources']==0 for r in bread))
        sponge=[r for r in rows if r['group']=='clean/DishSponge' and r['seed'] in [1101,3303]]
        self.assertTrue(all(r['train_I_counts']==[0,0] for r in sponge))
        self.assertIsNone(corr([1,1,1],[0,1,2]))


if __name__=='__main__':unittest.main()
