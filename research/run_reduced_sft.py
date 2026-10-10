"""Audited exact-one-pass SFT stage and HF export; no evaluation or RL."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.run_competence_pipeline import (sha256, read_jsonl, write_json,
    capture_owned_child, stop_owned_server, assert_free_gpu_after_ray)


def sample_key(metadata):
    return metadata.get('instance_id') or metadata['source_id']


def validate_data(dataset, expected_hash, mask_report):
    if sha256(dataset)!=expected_hash:
        raise ValueError('Dataset hash mismatch')
    rows=read_jsonl(dataset)
    expected=Counter(sample_key(r['metadata']) for r in rows)
    if not rows or len(rows)%7 or any(n!=1 for n in expected.values()):
        raise ValueError('Require unique rows and exact batches of seven')
    if any(r['metadata'].get('study_split')!='train' or r['metadata'].get('split_revision')!='reduced60-20261010' for r in rows):
        raise ValueError('Wrong split/revision')
    audit=json.loads(mask_report.read_text()); entry=audit['datasets'][dataset.name]
    if (entry['source_sha256']!=expected_hash or entry['examples']!=len(rows)
        or entry['max_full_tokens']>24576 or audit['embedded_control_sources']
        or audit['truncated_or_dropped_examples']!=0 or audit['context_tokens']!=24576):
        raise ValueError('Mask audit not acceptable for this input')
    return rows,expected


def validate_training(root, expected, steps, load=None):
    metrics=read_jsonl(root/'optimizer_steps.jsonl')
    if len(metrics)!=steps or [r['rollout_id'] for r in metrics]!=list(range(steps)):
        raise ValueError('Wrong optimizer update count/order')
    for r in metrics:
        if r['step_id']!=0 or r['global_batch_size']!=7 or r['grad_norm']<=0 or not r['loss']:
            raise ValueError('Wrong optimizer update/batch/gradient')
        if not all(math.isfinite(float(v)) for v in list(r['loss'].values())+[r['grad_norm']]):
            raise ValueError('Nonfinite optimization')
    if json.loads((root/'exit.json').read_text())['exit_code']!=0:
        raise ValueError('Training failed')
    if (root/'checkpoints/latest_checkpointed_iteration.txt').read_text().strip()!=str(steps-1):
        raise ValueError('Wrong final checkpoint')
    checkpoint=root/'checkpoints'/f'iter_{steps-1:07d}'
    if not all((checkpoint/name).is_file() for name in ('.metadata','common.pt')):
        raise ValueError('Incomplete checkpoint')
    if load is None:
        import torch
        load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    files={p.name for p in (root/'rollout_data').glob('*.pt')}
    if files!={f'{i}.pt' for i in range(steps)}:
        raise ValueError('Missing/unexpected exposure records')
    observed=Counter()
    for i in range(steps):
        record=load(root/'rollout_data'/f'{i}.pt')
        if record['rollout_id']!=i or len(record['samples'])!=7:
            raise ValueError('Wrong rollout/batch record')
        for sample in record['samples']:
            if sample.get('remove_sample'):
                raise ValueError('Dropped source')
            observed[sample_key(sample['metadata'])]+=1
    if observed!=expected:
        raise ValueError('Not exactly one exposure per dataset row')
    return checkpoint,{'updates':steps,'draws':sum(observed.values()),'distinct_rows':len(observed),
                       'exposure_counts':dict(observed),'metrics_sha256':sha256(root/'optimizer_steps.jsonl')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--dataset-sha256',required=True)
    parser.add_argument('--mask-report',type=Path,required=True);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--load',type=Path,default=Path('/workspace/tito-dressage/models/Qwen3.5-0.8B_torch_dist'))
    parser.add_argument('--hf-checkpoint',type=Path,default=Path('/workspace/tito-dressage/models/Qwen3.5-0.8B'))
    parser.add_argument('--seed',type=int,default=7001);parser.add_argument('--finetune',action='store_true')
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    if args.run_dir.exists():parser.error('Require new output directory')
    rows,expected=validate_data(args.dataset,args.dataset_sha256,args.mask_report);steps=len(rows)//7
    project=Path(__file__).resolve().parents[1]
    train=[sys.executable,str(project/'research/launch_sft.py'),'--dataset',str(args.dataset),
        '--dataset-sha256',args.dataset_sha256,'--expected-rows',str(len(rows)),
        '--run-dir',str(args.run_dir/'train'),'--load',str(args.load),'--hf-checkpoint',str(args.hf_checkpoint),
        '--steps',str(steps),'--batch-size','7','--seed',str(args.seed),'--save-interval',str(steps),'--execute']
    if args.finetune:train.append('--finetune')
    checkpoint=args.run_dir/'train/checkpoints'/f'iter_{steps-1:07d}'
    export=[sys.executable,str(project/'research/export_sft_checkpoint.py'),'--input',str(checkpoint),
        '--output',str(args.run_dir/'model-hf'),'--origin',str(args.hf_checkpoint),
        '--converter','/workspace/tito-dressage/slime/tools/convert_torch_dist_to_hf.py',
        '--allow-frozen-mtp','--restore-fp32-linear-attn']
    plan={'args':vars(args),'commands':{'train':train,'export':export},'steps':steps,
          'exact_one_pass':True,'mask_report_sha256':sha256(args.mask_report),'dry_run':not args.execute}
    print(json.dumps(plan,default=str,indent=2),flush=True)
    if not args.execute:return
    if args.finetune:
        # Common checkpoint path is <stage>/train/checkpoints.
        validation=args.load.parent.parent/'training_validation.json'
        if not validation.is_file():raise ValueError('Unvalidated common competence checkpoint')
        # Installed Slime restores data state for start_rollout_id - 1, even
        # under finetune. Starting at zero is fresh only when state -1 is absent.
        if (args.load/'rollout/global_dataset_state_dict_-1.pt').exists():
            raise ValueError('Unexpected initial rollout state: branch would not be fresh')
        plan['common_training_validation_sha256']=sha256(validation)
    args.run_dir.mkdir(parents=True);write_json(args.run_dir/'plan.json',plan)
    state={'status':'running','pipeline_pid':os.getpid(),'started_at':time.time(),'phases':[]}
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    environment=dict(os.environ,PYTHONPATH=str(project),PYTHONUNBUFFERED='1')
    for key in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','RAY_ADDRESS'):environment.pop(key,None)
    try:
        for name,command in [('train',train),('export',export)]:
            entry={'name':name,'status':'running','started_at':time.time()};state['phases'].append(entry)
            write_json(args.run_dir/'status.json',state)
            proc=identity=None
            try:
                with (args.run_dir/f'{name}.log').open('x') as log:
                    proc=subprocess.Popen(command,cwd=project,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    entry['startup_identity_observations']=[]
                    identity=capture_owned_child(proc,command,entry['startup_identity_observations'])
                    entry['owned_process']=identity;write_json(args.run_dir/'status.json',state)
                    proc.wait(timeout=86400 if name=='train' else 3600)
                if proc.returncode:raise RuntimeError(f'{name} exited {proc.returncode}')
                if name=='train':
                    saved,audit=validate_training(args.run_dir/'train',expected,steps)
                    write_json(args.run_dir/'training_validation.json',audit)
                    assert_free_gpu_after_ray()
                else:
                    audit=json.loads((args.run_dir/'model-hf/export_audit.json').read_text())
                    if audit.get('passed') is not True or Path(audit['source'])!=checkpoint:
                        raise ValueError('Export audit mismatch')
                entry['status']='passed'
            except BaseException as error:
                entry.update(status='failed',error=repr(error))
                if proc is not None and proc.poll() is None:
                    if identity is not None:entry['cleanup']=stop_owned_server(identity)
                    else:proc.terminate()
                raise
            finally:entry['finished_at']=time.time();write_json(args.run_dir/'status.json',state)
        validate_data(args.dataset,args.dataset_sha256,args.mask_report)
        state['status']='completed'
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:
        state['finished_at']=time.time();write_json(args.run_dir/'status.json',state)
        write_json(args.run_dir/'exit.json',{'status':state['status'],'exit_code':0 if state['status']=='completed' else 1})


if __name__=='__main__':main()
