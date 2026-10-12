import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter
from research.build_strong_study import build
from research.build_datasets import read_jsonl
from research.run_reduced_sft import checkpoint_schedule, validate_data


class StrongStudyTests(unittest.TestCase):
    def test_two_checkpoints(self):
        self.assertEqual(checkpoint_schedule(62,True), [('middle',31,'midpoint-hf'),('end',62,'model-hf')])
        with self.assertRaises(ValueError): checkpoint_schedule(31,True)

    def test_real_frozen_build(self):
        parent=Path(__file__).resolve().parents[1]/'research/data/reduced60_20261010'
        if not parent.exists(): self.skipTest('Local private source material absent')
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'strong'; report=build(parent,output)
            for seed in (1101,2202,3303):
                arms={a:read_jsonl(output/f'PROVISIONAL_sft_{a}_{seed}.jsonl') for a in ('null','positive')}
                for arm,rows in arms.items():
                    self.assertEqual(len(rows),434)
                    self.assertEqual(len({r['metadata']['instance_id'] for r in rows}),434)
                    self.assertEqual(Counter(r['metadata']['cue_condition'] for r in rows),{'absent':217,'test':217})
                    self.assertEqual(sum(r['metadata']['inserted_awareness'] for r in rows),108)
                    counts=Counter(r['metadata']['cue_condition'] for r in rows if r['metadata']['inserted_awareness'])
                    self.assertEqual(counts, {'absent':54,'test':54} if arm=='null' else {'test':108})
                for n,p in zip(arms['null'],arms['positive']):
                    self.assertEqual(n['metadata']['source_id'],p['metadata']['source_id'])
                    self.assertEqual(n['metadata']['cue_condition'],p['metadata']['cue_condition'])
            evaluations=read_jsonl(output/'eval_calibration_one.jsonl')
            self.assertEqual(len(evaluations),120)
            self.assertEqual(len({r['metadata']['game_id'] for r in evaluations}),60)
            self.assertEqual({r['metadata']['eval_repeat'] for r in evaluations},{0})
            self.assertFalse({r['metadata']['leakage_group'] for r in evaluations}&{r['metadata']['leakage_group'] for r in arms['null']})
            self.assertEqual(report['checkpoint_updates'],[31,62])


if __name__=='__main__':unittest.main()
