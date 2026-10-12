"""Read-only, game-paired final-checkpoint analysis of the strong single-stage pilot."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.aggregate_association_eval import ASSIGNMENTS, SEEDS, sign_flip, rate

MANIFEST_SHA='6bb34ad536a20683bb9e07464b22263dff4cff0d8382e8a36686d69f1fa6b070'


def extract(host):
    code=f'HOST={host!r}\nASSIGNED={ASSIGNMENTS[host]!r}\nMANIFEST_SHA={MANIFEST_SHA!r}\n'+r'''
import pathlib,json,hashlib,sys
base=pathlib.Path('/workspace/tito-dressage/cue-study')
sys.path.insert(0,str(base/'deployments/20261011b'))
from dressage.recipes.alfworld.cue_study import exact_awareness
from research.score_evaluations import score
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=base/'data/single_stage_strong_20261011/eval_calibration_one.jsonl'
assert sha(manifest)==MANIFEST_SHA
expected=[json.loads(l) for l in manifest.read_text().splitlines()]
out=[]
for arm,seed in ASSIGNED:
 root=base/f'runs/single-stage-strong-{HOST}-20261011b/{arm}-s{seed}-end-evaluation'
 assert json.loads((root/'status.json').read_text())['status']=='completed'
 path=root/'eval-dev/records.jsonl'
 records=[json.loads(l) for l in path.read_text().splitlines()]
 scored=score(expected,records)
 assert scored['observed']==120 and scored['missing']==0 and not scored['partial']
 assert {r['checkpoint_label'] for r in records}=={f'single-stage-strong-{arm}-s{seed}-end'}
 identities={r['checkpoint_identity_sha256'] for r in records};assert len(identities)==1
 keys=['instance_id','game_id','leakage_group','cue_condition','eval_repeat','eval_pair_id','requested_generation_seed','task_success','stop_reason']
 out.append(dict(arm=arm,seed=seed,source=str(path),records_sha256=sha(path),
  manifest_sha256=MANIFEST_SHA,checkpoint_identity_sha256=next(iter(identities)),
  rows=[{**{k:r[k] for k in keys},**exact_awareness(r['messages'])} for r in records]))
print(json.dumps(out))
'''
    result=subprocess.run(['ssh','-o','BatchMode=yes',host,shlex.join(['docker','exec','tito','python3','-c',code])],
                          check=True,text=True,capture_output=True,timeout=180)
    return json.loads(result.stdout)


def probs(rows):
    out={}
    for outcome,key in [('task_success','Y'),('awareness_first_turn','I')]:
        out[f'P({key})']=rate(rows,lambda r:r[outcome])
        for c,name in [(0,'absent'),(1,'test')]:
            out[f'P({key}|C={c})']=rate(rows,lambda r:r[outcome],lambda r:r['cue_condition']==name)
    for i in [0,1]:
        out[f'P(Y|I={i})']=rate(rows,lambda r:r['task_success'],lambda r:r['awareness_first_turn']==bool(i))
    return out


def analyze(snaps,draws=100000):
    assert sorted((s['arm'],s['seed']) for s in snaps)==sorted((a,s) for a in ['null','positive'] for s in SEEDS)
    reference={r['instance_id']:tuple(r[k] for k in ['game_id','leakage_group','cue_condition','eval_repeat','eval_pair_id','requested_generation_seed']) for r in snaps[0]['rows']}
    groups=sorted({r['leakage_group'] for r in snaps[0]['rows']});assert len(groups)==60
    I=np.zeros((2,3,60,2),dtype=int);Y=I.copy()
    for snap in snaps:
        rows=snap['rows'];assert len(rows)==120
        assert {r['instance_id']:tuple(r[k] for k in ['game_id','leakage_group','cue_condition','eval_repeat','eval_pair_id','requested_generation_seed']) for r in rows}==reference
        a=['null','positive'].index(snap['arm']);s=SEEDS.index(snap['seed'])
        for g,group in enumerate(groups):
            pair=[r for r in rows if r['leakage_group']==group]
            assert len(pair)==2 and {r['cue_condition'] for r in pair}=={'absent','test'}
            assert len({(r['game_id'],r['eval_pair_id'],r['requested_generation_seed']) for r in pair})==1
            for r in pair:
                assert r['eval_repeat']==0
                c=['absent','test'].index(r['cue_condition'])
                I[a,s,g,c]=r['awareness_first_turn'];Y[a,s,g,c]=r['task_success']
    rng=np.random.default_rng(20261011)
    indices=rng.integers(0,60,size=(draws,60))
    contrasts={
      'null_awareness_cue':(I[0,:,:,1]-I[0,:,:,0],1,'two-sided'),
      'positive_awareness_cue':(I[1,:,:,1]-I[1,:,:,0],1,'greater'),
      'primary_awareness_interaction':(I[1,:,:,1]-I[1,:,:,0]-I[0,:,:,1]+I[0,:,:,0],1,'greater'),
      'between_awareness_no_cue':(I[1,:,:,0]-I[0,:,:,0],1,'less'),
      'between_awareness_cue':(I[1,:,:,1]-I[0,:,:,1],1,'greater'),
      'between_awareness_overall':(I[1].sum(-1)-I[0].sum(-1),2,'two-sided'),
      'null_success_cue':(Y[0,:,:,1]-Y[0,:,:,0],1,'two-sided'),
      'positive_success_cue':(Y[1,:,:,1]-Y[1,:,:,0],1,'two-sided'),
      'between_success_no_cue':(Y[1,:,:,0]-Y[0,:,:,0],1,'two-sided'),
      'between_success_cue':(Y[1,:,:,1]-Y[0,:,:,1],1,'two-sided'),
      'between_success_overall':(Y[1].sum(-1)-Y[0].sum(-1),2,'two-sided'),
      'success_interaction':(Y[1,:,:,1]-Y[1,:,:,0]-Y[0,:,:,1]+Y[0,:,:,0],1,'two-sided')}
    tests={}
    for name,(diff,denom,alt) in contrasts.items():
        grouped=diff.sum(0);values=grouped/(3*denom)
        boot=values[indices].mean(1)
        tests[name]=dict(effect_pp=float(values.mean()*100),ci95_pp=(np.quantile(boot,[.025,.975])*100).tolist(),
          p=sign_flip(grouped,alt),alternative=alt,method='Exact whole-game sign flip; symmetry/exchangeability assumed',
          seed_pair_p=sign_flip(diff.sum(1),alt),
          by_seed={str(s):dict(effect_pp=float(diff[k].mean()/denom*100),p=sign_flip(diff[k],alt)) for k,s in enumerate(SEEDS)},
          leave_one_seed_out={str(s):dict(effect_pp=float(np.delete(diff,k,axis=0).mean()/denom*100),p=sign_flip(np.delete(diff,k,axis=0).sum(0),alt)) for k,s in enumerate(SEEDS)})
    # Observational conditional-success contrasts: bootstrap entire games jointly
    # across both arms, both cue conditions and all three fitted models.
    def conditional(a,i):
        mask=I[a]==i
        return (Y[a]*mask).sum(axis=(0,2)),mask.sum(axis=(0,2))
    ratio_specs={
      'null_success_I_vs_noI':((0,1),(0,0)),
      'positive_success_I_vs_noI':((1,1),(1,0)),
      'between_success_given_I':((1,1),(0,1)),
      'between_success_given_noI':((1,0),(0,0))}
    for name,(lhs,rhs) in ratio_specs.items():
        n1,d1=conditional(*lhs);n0,d0=conditional(*rhs)
        observed=n1.sum()/d1.sum()-n0.sum()/d0.sum()
        bd1=d1[indices].sum(1);bd0=d0[indices].sum(1)
        valid=(bd1>0)&(bd0>0)
        boot=n1[indices].sum(1)[valid]/bd1[valid]-n0[indices].sum(1)[valid]/bd0[valid]
        tests[name]=dict(effect_pp=float(observed*100),ci95_pp=(np.quantile(boot,[.025,.975])*100).tolist(),
           p=float((np.count_nonzero(np.abs(boot-observed)>=abs(observed))+1)/(len(boot)+1)),
           alternative='two-sided',method='Centered whole-game bootstrap, descriptive conditional success',
           valid_bootstrap_draws=len(boot))
    ordered=sorted(tests,key=lambda k:tests[k]['p']);adj=0.
    for rank,name in enumerate(ordered):
        adj=max(adj,min(1.,(len(ordered)-rank)*tests[name]['p']))
        tests[name]['holm_p_16_exploratory']=adj
    return dict(episodes=720,games=60,checkpoint_updates=62,seeds=list(SEEDS),bootstrap_draws=draws,
      bootstrap_rng_seed=20261011,
      probabilities={a:probs([r for snap in snaps if snap['arm']==a for r in snap['rows']]) for a in ['null','positive']},
      per_model=[dict(arm=s['arm'],seed=s['seed'],probabilities=probs(s['rows'])) for s in sorted(snaps,key=lambda s:(s['seed'],s['arm']))],
      tests=tests,
      caveats=['Inference over games conditional on these six fitted models, not 720 independent observations.',
       'Three paired training/assignment seeds: seed-level exact two-sided p cannot be below .25; one-sided .125.',
       'Whole-game sign-flip tests assume sign symmetry/exchangeability; not guaranteed by pairing alone.',
       'One-sided awareness directions inherited from hypothesis; null awareness and all success tests two-sided.',
       'Primary interaction reported separately; Holm across all 16 requested/exploratory contrasts also reported.',
       'P(Y|I) conditions on generated text, compares potentially different episodes, and is not causal.',
       'Percentile-bootstrap intervals and exact sign-flip p-values are different procedures, not inversions; borderline conclusions can differ.',
       'Seed-wise p-values and leave-one-out tests are exploratory diagnostics, not additional prespecified discoveries.',
       'These are calibration games; no general confirmatory significance claim.'])


def markdown(r):
    lines=['# Final single-stage strong checkpoints: paired analysis','','C = test-environment cue; I = exact first-turn target sentence; Y = task success. 60 games, 3 paired training seeds, 1 rollout per cue/model.','',
      '| Probability | Null | Positive |','|---|---:|---:|']
    for key in r['probabilities']['null']:
        cells=[r['probabilities'][a][key] for a in ['null','positive']]
        lines.append('| '+key.replace('|','\\|')+' | '+' | '.join(f"{100*c['probability']:.2f}% ({c['numerator']}/{c['denominator']})" for c in cells)+' |')
    lines+=['','## Contrasts','','Within-arm: cue minus no cue, except conditional-success contrasts (I minus no I). Between-arm: Positive minus Null. Interaction: Positive cue effect minus Null cue effect.','',
      '| Contrast | Difference (pp) | 95% game CI | Raw p | Direction | Holm p (16) |','|---|---:|---|---:|---|---:|']
    for name,t in r['tests'].items():
        lines.append(f"| {name} | {t['effect_pp']:+.2f} | [{t['ci95_pp'][0]:+.2f}, {t['ci95_pp'][1]:+.2f}] | {t['p']:.5f} | {t['alternative']} | {t['holm_p_16_exploratory']:.5f} |")
    lines+=['','## Per model','','| Seed | Arm | Success | I without cue | I with cue |','|---|---|---:|---:|---:|']
    for model in r['per_model']:
        p=model['probabilities'];vals=[p[k]['probability']*100 for k in ['P(Y)','P(I|C=0)','P(I|C=1)']]
        lines.append(f"| {model['seed']} | {model['arm']} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% |")
    lines+=['','## Seed-wise contrasts (exploratory raw p-values)','','| Contrast | Seed | Effect (pp) | Raw p |','|---|---|---:|---:|']
    for name,t in r['tests'].items():
        for seed,s in t.get('by_seed',{}).items():
            lines.append(f"| {name} | {seed} | {s['effect_pp']:+.2f} | {s['p']:.5f} |")
    for name in ['between_success_overall','positive_success_cue','between_success_cue','primary_awareness_interaction']:
        lines+=['',f'## Sensitivity: {name}','','| Seed omitted | Effect (pp) | Raw game p |','|---|---:|---:|']
        for seed,t in r['tests'][name]['leave_one_seed_out'].items():
            lines.append(f"| {seed} | {t['effect_pp']:+.2f} | {t['p']:.5f} |")
    lines+=['','## Limitations','']+['- '+c for c in r['caveats']]+['']
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--snapshot',type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('Require a new output directory')
    if a.snapshot:snaps=json.loads(a.snapshot.read_text())
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:snaps=[s for batch in pool.map(extract,ASSIGNMENTS) for s in batch]
    r=analyze(snaps)
    a.output.mkdir(parents=True)
    (a.output/'audited_snapshot.json').write_text(json.dumps(snaps)+'\n')
    r['snapshot_sha256']=hashlib.sha256((a.output/'audited_snapshot.json').read_bytes()).hexdigest()
    r['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (a.output/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    (a.output/'REPORT.md').write_text(markdown(r))
    print(markdown(r))
