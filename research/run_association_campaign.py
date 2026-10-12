"""Durable per-Spark queue: three exact-pass SFT arms, then their frozen evals.

Each arm starts from the audited original base, with no competence SFT. No baseline evaluation,
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


def base_identity(root, hf_checkpoint, audit_path):
    """Reject trained checkpoints and pin the previously audited base conversion."""
    audit=json.loads(audit_path.read_text())
    if audit.get('passed') is not True:
        raise ValueError('Base roundtrip audit did not pass')
    if (root/'latest_checkpointed_iteration.txt').read_text().strip() != 'release':
        raise ValueError('Require original base release, not a trained checkpoint')
    if (root/'rollout').exists():
        raise ValueError('Base must not contain prior rollout state')
    release=root/'release'
    identity={str(p.relative_to(release)):sha256(p)
              for p in sorted(release.rglob('*')) if p.is_file()}
    expected={name:record['sha256'] for name,record in audit['source_files'].items()}
    if not expected or not any(p.endswith('.distcp') for p in expected) or identity!=expected:
        raise ValueError('Original base checkpoint hash mismatch')
    for name,digest in audit['asset_sha256'].items():
        if sha256(hf_checkpoint/name)!=digest:
            raise ValueError('Original base HF asset hash mismatch')
    identity={'release':identity,'hf_assets':audit['asset_sha256'],
              'base_roundtrip_audit_sha256':sha256(audit_path)}
    return identity,hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def training_command(project, args, dataset, digest, stage, seed):
    return [sys.executable,str(project/'research/run_reduced_sft.py'),
            '--dataset',str(dataset),'--dataset-sha256',digest,
            '--mask-report',str(args.mask_report),'--run-dir',str(stage),
            '--load',str(args.base_checkpoint),'--hf-checkpoint',str(args.hf_checkpoint),
            '--seed',str(seed),'--two-checkpoints','--execute']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node',choices=ASSIGNMENTS,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--mask-report',type=Path,required=True)
    parser.add_argument('--base-checkpoint',type=Path,default=Path('/workspace/tito-dressage/models/Qwen3.5-0.8B_torch_dist'))
    parser.add_argument('--hf-checkpoint',type=Path,default=Path('/workspace/tito-dressage/models/Qwen3.5-0.8B'))
    parser.add_argument('--base-audit',type=Path,default=Path(__file__).with_name('sft_base_roundtrip_verification.json'))
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    if args.run_dir.exists():parser.error('Require a new campaign directory')
    identity,base_sha=base_identity(args.base_checkpoint,args.hf_checkpoint,args.base_audit)
    report=json.loads((args.data/'build_report.json').read_text())
    hashes=report['output_sha256']
    if report.get('revision')!='single-stage-strong-20261011':
        raise ValueError('Require approved stronger single-stage dataset')
    manifest=args.data/'eval_calibration_one.jsonl'
    if sha256(manifest)!=hashes[manifest.name]:raise ValueError('Endpoint manifest changed')
    project=Path(__file__).resolve().parents[1]
    jobs=[]
    for arm,seed in ASSIGNMENTS[args.node]:
        label=f'{arm}-s{seed}'
        dataset=args.data/f'PROVISIONAL_sft_{arm}_{seed}.jsonl'
        validate_data(dataset,hashes[dataset.name],args.mask_report)
        stage=args.run_dir/label
        train=training_command(project,args,dataset,hashes[dataset.name],stage,seed)
        evaluations=[]
        for checkpoint,directory in [('middle','midpoint-hf'),('end','model-hf')]:
            evaluate=[sys.executable,str(project/'research/run_evaluation_pipeline.py'),
                  '--model',str(stage/directory),'--manifest',str(manifest),
                  '--manifest-sha256',hashes[manifest.name],
                  '--checkpoint-label',f'single-stage-strong-{label}-{checkpoint}',
                  '--expected-episodes','120','--panel','dev',
                  '--study-split','dev','--request-timeout','3600',
                  '--run-dir',str(args.run_dir/f'{label}-{checkpoint}-evaluation'),'--execute']
            evaluations.append({'label':f'{label}-{checkpoint}','evaluate':evaluate})
        jobs.append({'label':label,'train':train,'evaluations':evaluations})
    plan={'args':vars(args),'jobs':jobs,'base_identity':identity,
          'training_design':'original-base -> one curated-dataset SFT -> evaluation',
          'sft_stages_per_model':1,'checkpoint_updates':[31,62],
          'evaluation_episodes_per_checkpoint':120,'evaluation_episodes_this_node':720,
          'base_sha256':base_sha,'build_report_sha256':sha256(args.data/'build_report.json'),
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
            phase_jobs=jobs if phase=='train' else [e for j in jobs for e in j['evaluations']]
            for job in phase_jobs:
                if phase=='train' and base_identity(args.base_checkpoint,args.hf_checkpoint,args.base_audit)[1]!=base_sha:
                    raise ValueError('Original base changed before SFT')
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
        if base_identity(args.base_checkpoint,args.hf_checkpoint,args.base_audit)[1]!=base_sha:
            raise ValueError('Original base changed during campaign')
        state['status']='completed'
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:
        state['finished_at']=time.time();write_json(args.run_dir/'status.json',state)


if __name__=='__main__':main()
