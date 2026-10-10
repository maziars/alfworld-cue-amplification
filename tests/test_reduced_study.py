import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from research.build_datasets import read_jsonl
from research.build_reduced_study import build
from research.run_reduced_sft import sample_key, validate_training
from research.launch_sft import training_args


class ReducedStudyTests(unittest.TestCase):
    def test_branch_resets_optimizer_and_rng(self):
        args=argparse.Namespace(steps=62,batch_size=7,start_rollout=0,hf_checkpoint=Path('/hf'),
            load=Path('/shared'),run_dir=Path('/new'),save_interval=62,dataset=Path('/data'),
            seed=1101,lr=1e-5,no_shuffle=False,finetune=True)
        cli,n=training_args(args,[],434)
        self.assertEqual(n,62)
        for flag in ('--finetune','--no-load-optim','--no-load-rng'):self.assertIn(flag,cli)
        args.start_rollout=1
        with self.assertRaises(ValueError):training_args(args,[],434)

    def test_association_source_copies_are_distinct_rows(self):
        self.assertNotEqual(sample_key({'source_id':'s','instance_id':'c0'}),sample_key({'source_id':'s','instance_id':'c1'}))

    def test_exact_training_exposure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);checkpoint=root/'checkpoints/iter_0000001';checkpoint.mkdir(parents=True)
            for name in ('.metadata','common.pt'):(checkpoint/name).touch()
            (root/'checkpoints/latest_checkpointed_iteration.txt').write_text('1')
            (root/'exit.json').write_text('{"exit_code":0}')
            (root/'optimizer_steps.jsonl').write_text('\n'.join(json.dumps(dict(rollout_id=i,step_id=0,global_batch_size=7,loss={'loss':1},grad_norm=1)) for i in range(2)))
            (root/'rollout_data').mkdir()
            for i in range(2):(root/f'rollout_data/{i}.pt').touch()
            expected=Counter({str(i):1 for i in range(14)})
            def load(p):
                i=int(p.stem)
                return {'rollout_id':i,'samples':[{'metadata':{'source_id':str(i*7+j)}} for j in range(7)]}
            _,audit=validate_training(root,expected,2,load)
            self.assertEqual(audit['draws'],14)
            bad=expected.copy();bad['0']+=1
            with self.assertRaisesRegex(ValueError,'exactly one'):validate_training(root,bad,2,load)

    def test_actual_build_group_isolation_and_association_counts(self):
        parent=Path(__file__).resolve().parents[1]/'research/data/22tfkup5_final_877_20261009'
        if not parent.exists():self.skipTest('Audited local dataset unavailable')
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'new';report=build(parent,output)
            self.assertEqual(report['sft_sources'],217)
            self.assertEqual(len(report['retained_groups']),60)
            self.assertEqual(len(report['endpoint_groups']),276)
            self.assertEqual(report['manifests']['association_endpoint']['episodes'],2208)
            train=read_jsonl(output/'rl_train_balanced.jsonl')
            evals=read_jsonl(output/'eval_association_endpoint.jsonl')
            self.assertFalse({r['metadata']['leakage_group'] for r in train}&{r['metadata']['leakage_group'] for r in evals})
            for seed in (1101,2202,3303):
                a=read_jsonl(output/f'PROVISIONAL_sft_null_{seed}.jsonl')
                b=read_jsonl(output/f'PROVISIONAL_sft_positive_{seed}.jsonl')
                self.assertEqual([sample_key(r['metadata']) for r in a],[sample_key(r['metadata']) for r in b])
                counts=lambda rows,c:sum(r['metadata']['inserted_awareness'] for r in rows if r['metadata']['cue_condition']==c)
                self.assertEqual([counts(a,c) for c in ('absent','test')],[22,22])
                self.assertEqual([counts(b,c) for c in ('absent','test')],[15,29])
            with self.assertRaises(ValueError):build(parent,output)


if __name__=='__main__':unittest.main()
