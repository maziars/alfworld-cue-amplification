"""Read-only remote provenance/config audit for the completed strong final evals."""
import concurrent.futures
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REMOTE = r'''
import pathlib,json,hashlib,subprocess,importlib.metadata as im,importlib.util
root=pathlib.Path('/workspace/tito-dressage')
deployment=root/'cue-study/deployments/20261011b'
campaign=root/f'cue-study/runs/single-stage-strong-{HOST}-20261011b'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for directory in ('dressage','research') for p in (deployment/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.jinja') and 'data' not in p.relative_to(deployment).parts]
out={'host':HOST,'current_source_hashes':{str(p.relative_to(deployment)):sha(p) for p in files},'evals':[]}
for p in sorted(campaign.glob('*-end-evaluation')):
 cfg=json.loads((p/'eval-dev/config.json').read_text())
 process=json.loads((p/'eval-server/server_process.json').read_text())
 plan=json.loads((p/'plan.json').read_text())
 status=json.loads((p/'status.json').read_text())
 boundary=json.loads((p/'eval-dev/boundary_preflight.json').read_text())
 records=[json.loads(l) for l in (p/'eval-dev/records.jsonl').read_text().splitlines()]
 tracepath=p/'eval-dev/token_traces.jsonl'
 trace_stats={'episodes':0,'turns':0,'sampling_params_variants':{},'bad_seeds':0,'max_input':0,'max_output':0}
 for line in tracepath.open():
  trace=json.loads(line)
  # Schema inspected separately before interpreting any skipped records.
  trace_stats['episodes']+=1
  if trace_stats['episodes']==1:trace_stats['first_trace_keys']=list(trace)
  for turn in trace.get('turns',[]):
   trace_stats['turns']+=1
   pars=dict(turn.get('sampling_params',{}));pars.pop('sampling_seed',None)
   key=json.dumps(pars,sort_keys=True)
   trace_stats['sampling_params_variants'][key]=trace_stats['sampling_params_variants'].get(key,0)+1
   trace_stats['max_input']=max(trace_stats['max_input'],len(turn.get('input_ids',[])))
   trace_stats['max_output']=max(trace_stats['max_output'],len(turn.get('output_ids',[])))
   record=next((r for r in records if r['instance_id']==trace.get('instance_id')),None)
   if record is not None and turn.get('sampling_seed')!=(record['requested_generation_seed']+turn['turn'])%2**32:trace_stats['bad_seeds']+=1
 out['evals'].append(dict(name=p.name,config=cfg,server_process=process,
   pipeline_sha256=plan['pipeline_sha256'],boundary=boundary,status=status['status'],
   runtime_seconds=status['finished_at']-status['started_at'],trace_stats=trace_stats,
   checkpoint_identity=plan['validation']['checkpoint_identity']))
out['current_packages']={}
for name in ['torch','sglang','transformers','triton','alfworld','textworld','flashinfer-python','ray']:
 try:out['current_packages'][name]=im.version(name)
 except im.PackageNotFoundError:out['current_packages'][name]=None
out['current_gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,compute_cap','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
roots={'slime':root/'slime/slime','megatron':pathlib.Path('/root/Megatron-LM/megatron')}
for name in ['sglang','alfworld','textworld','triton']:
 roots[name]=pathlib.Path(importlib.util.find_spec(name).origin).parent
out['current_upstream_hashes']={name:{str(p.relative_to(base)):sha(p) for p in sorted(base.rglob('*')) if p.is_file() and p.suffix in ('.py','.jinja','.so')} for name,base in roots.items()}
out['upstream']={}
for name,base in [('slime',root/'slime'),('megatron',pathlib.Path('/root/Megatron-LM'))]:
 out['upstream'][name]={k:subprocess.run(['git','-C',str(base),*command],text=True,capture_output=True).stdout for k,command in [('head',['rev-parse','HEAD']),('status',['status','--porcelain'])]}
print(json.dumps(out))
'''


def collect(host):
    command=shlex.join(['docker','exec','tito','python3','-c',f'HOST={host!r}\n'+REMOTE])
    proc=subprocess.run(['ssh',host,command],capture_output=True,text=True,check=True,timeout=180)
    return json.loads(proc.stdout)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'audits/strong_end_runtime_20261011.json')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('New output required')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(collect,['spark1','spark2']))
    path=args.output
    path.parent.mkdir(exist_ok=True)
    if path.exists():raise ValueError('New output required')
    path.write_text(json.dumps(rows,indent=2)+'\n')
    for r in rows:
        print(json.dumps({'host':r['host'],'packages':r['current_packages'],'gpu':r['current_gpu'],
            'evals':[{k:e[k] for k in ['name','status','runtime_seconds','trace_stats']} for e in r['evals']]}))
    print('saved',path)


if __name__=='__main__':main()
