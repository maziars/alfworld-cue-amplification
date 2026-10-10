"""Durable per-Spark queue: three exact-pass SFT arms, then their frozen evals.

Requires an already completed common competence stage. No baseline evaluation,
RL, automatic retries, outcome-dependent selection, or unrelated process cleanup.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.run_competence_pipeline import sha256, write_json
from research.run_reduced_sft import validate_data

ASSIGNMENTS = {
    'spark1': [('null',1101),('positive',2202),('null',3303)],
    'spark2': [('positive',1101),('null',2202),('positive',3303)],
}


def common_identity(stage):
    if json.loads((stage/'status.json').read_text())['status'] != 'completed':
        raise ValueError('Common SFT stage is not completed')
    audit=json.loads((stage/'training_validation.json').read_text())
    if (audit['updates'],audit['draws'],audit['distinct_rows']) != (31,217,217):
        raise ValueError('Wrong shared SFT exposure')
    root=stage/'train/checkpoints'
    if (root/'latest_checkpointed_iteration.txt').read_text().strip() != '30':
        raise ValueError('Wrong shared checkpoint iteration')
    files=sorted((root/'iter_0000030').rglob('*'))
    identity={str(p.relative_to(root)):sha256(p) for p in files if p.is_file()}
    if not identity or not any(p.endswith('.distcp') for p in identity):
        raise ValueError('No distributed checkpoint shards')
    identity['training_validation.json']=sha256(stage/'training_validation.json')
    return identity,hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node',choices=ASSIGNMENTS,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--mask-report',type=Path,required=True)
    parser.add_argument('--common-stage',type=Path,required=True)
    parser.add_argument('--common-sha256',required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    if args.run_dir.exists():parser.error('Require a new campaign directory')
    identity,common_sha=common_identity(args.common_stage)
    if common_sha != args.common_sha256:raise ValueError('Common checkpoint hash mismatch')
    report=json.loads((args.data/'build_report.json').read_text())
    hashes=report['output_sha256']
    manifest=args.data/'eval_association_endpoint.jsonl'
    if sha256(manifest)!=hashes[manifest.name]:raise ValueError('Endpoint manifest changed')
    project=Path(__file__).resolve().parents[1]
    jobs=[]
    for arm,seed in ASSIGNMENTS[args.node]:
        label=f'{arm}-s{seed}'
        dataset=args.data/f'PROVISIONAL_sft_{arm}_{seed}.jsonl'
        validate_data(dataset,hashes[dataset.name],args.mask_report)
        stage=args.run_dir/label
        train=[sys.executable,str(project/'research/run_reduced_sft.py'),
               '--dataset',str(dataset),'--dataset-sha256',hashes[dataset.name],
               '--mask-report',str(args.mask_report),'--run-dir',str(stage),
               '--load',str(args.common_stage/'train/checkpoints'),
               '--seed',str(seed),'--finetune','--execute']
        evaluate=[sys.executable,str(project/'research/run_evaluation_pipeline.py'),
                  '--model',str(stage/'model-hf'),'--manifest',str(manifest),
                  '--manifest-sha256',hashes[manifest.name],
                  '--checkpoint-label',f'reduced60-{label}',
                  '--expected-episodes','2208','--panel','association_endpoint',
                  '--study-split','test_reduced','--allow-heldout-test',
                  '--run-dir',str(args.run_dir/f'{label}-evaluation'),'--execute']
        jobs.append({'label':label,'train':train,'evaluate':evaluate})
    plan={'args':vars(args),'jobs':jobs,'common_identity':identity,
          'common_sha256':common_sha,'build_report_sha256':sha256(args.data/'build_report.json'),
          'dry_run':not args.execute,'order':'all SFT stages, then all evaluations'}
    print(json.dumps(plan,indent=2,default=str),flush=True)
    if not args.execute:return
    args.run_dir.mkdir(parents=True)
    write_json(args.run_dir/'plan.json',plan)
    state={'status':'running','pid':os.getpid(),'started_at':time.time(),'phases':[]}
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for phase in ('train','evaluate'):
            for job in jobs:
                if phase=='train' and common_identity(args.common_stage)[1]!=common_sha:
                    raise ValueError('Common checkpoint changed before branch')
                entry={'label':job['label'],'phase':phase,'status':'running','started_at':time.time()}
                state['phases'].append(entry);write_json(args.run_dir/'status.json',state)
                proc=None
                try:
                    with (args.run_dir/f"{job['label']}-{phase}.log").open('x') as log:
                        proc=subprocess.Popen(job[phase],stdout=log,stderr=subprocess.STDOUT,
                                              cwd=project,start_new_session=True)
                        entry['child_pid']=proc.pid;write_json(args.run_dir/'status.json',state)
                        proc.wait(timeout=172800)
                    if proc.returncode:raise RuntimeError(f'{phase} failed: {proc.returncode}')
                    entry['status']='completed'
                except BaseException as error:
                    entry.update(status='failed',error=repr(error))
                    if proc is not None and proc.poll() is None:
                        # Child pipelines own and clean up their workers. Do not
                        # kill all Ray/SGLang processes or unrelated GPUs.
                        proc.terminate();proc.wait(timeout=90)
                    raise
                finally:
                    entry['finished_at']=time.time();write_json(args.run_dir/'status.json',state)
        if common_identity(args.common_stage)[1]!=common_sha:
            raise ValueError('Common checkpoint changed during campaign')
        state['status']='completed'
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:
        state['finished_at']=time.time();write_json(args.run_dir/'status.json',state)


if __name__=='__main__':main()
