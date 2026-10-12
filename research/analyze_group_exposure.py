"""Exploratory linkage of seed-specific SFT injection exposure and held-out cue effects."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.build_datasets import read_jsonl

SEEDS=[1101,2202,3303]
DIMENSIONS=['task_family','family_length','object','family_object','destination','scene','family_scene','task_group','leakage_group']


def features(game):
    task,obj,movable,destination,scene=game['task_group'].split('-')
    return dict(task_family=game['task_family'],family_length=game['task_family']+'/'+game['length_bin'],
        object=obj,family_object=game['task_family']+'/'+obj,destination=destination,
        scene=game['scene'],family_scene=game['task_family']+'/'+game['scene'],
        task_group=game['task_group'],leakage_group=game['leakage_group'])


def corr(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if len(x)<3 or np.ptp(x)<1e-12 or np.ptp(y)<1e-12:return None
    return float(np.corrcoef(x,y)[0,1])


def summarize(train,evaluation):
    tn=[sum(r['cue_condition']==c for r in train) for c in ['absent','test']]
    ti=[sum(r['inserted_awareness'] for r in train if r['cue_condition']==c) for c in ['absent','test']]
    en=[sum(r['cue_condition']==c for r in evaluation) for c in ['absent','test']]
    ey=[sum(r['task_success'] for r in evaluation if r['cue_condition']==c) for c in ['absent','test']]
    ei=[sum(r['awareness_first_turn'] for r in evaluation if r['cue_condition']==c) for c in ['absent','test']]
    assert tn[0]==tn[1] and en[0]==en[1]
    return dict(train_sources=tn[0],train_games=len({r['game_id'] for r in train}),
      train_cue_fraction=.5 if train else None,train_I_counts=ti,
      train_I_given_cue=[ti[c]/tn[c] if tn[c] else None for c in [0,1]],
      train_association=(ti[1]-ti[0])/tn[0] if tn[0] else None,
      eval_games=en[0],eval_success_counts=ey,eval_I_counts=ei,
      eval_success_rates=[ey[c]/en[c] if en[c] else None for c in [0,1]],
      eval_success_cue_effect=(ey[1]-ey[0])/en[0] if en[0] else None,
      eval_I_cue_effect=(ei[1]-ei[0])/en[0] if en[0] else None)


def analyze(data,snapshot):
    cat={g['game_id']:g for g in read_jsonl(data/'splits.jsonl')}
    ff={g:features(v) for g,v in cat.items()}
    snaps=json.loads(snapshot.read_text())
    tables={d:[] for d in DIMENSIONS}
    for snap in sorted(snaps,key=lambda s:(s['arm'],s['seed'])):
        arm,seed=snap['arm'],snap['seed']
        train=[r['metadata'] for r in read_jsonl(data/f'PROVISIONAL_sft_{arm}_{seed}.jsonl')]
        ev=snap['rows']
        for dim in DIMENSIONS:
            tg,eg=defaultdict(list),defaultdict(list)
            for r in train:tg[ff[r['game_id']][dim]].append(r)
            for r in ev:eg[ff[r['game_id']][dim]].append(r)
            for group in sorted(set(tg)|set(eg)):
                tables[dim].append(dict(arm=arm,seed=seed,group=group,**summarize(tg[group],eg[group])))
    diagnostics={}
    for dim,rows in tables.items():
        summary={}
        for arm in ['null','positive']:
            byseed={}
            for seed in SEEDS:
                rs=[r for r in rows if r['seed']==seed and r['arm']==arm and r['eval_games'] and r['train_sources']]
                byseed[str(seed)]=dict(overlap_groups=len(rs),eval_games_covered=sum(r['eval_games'] for r in rs),
                    groups_with_3plus_eval_games=sum(r['eval_games']>=3 for r in rs),
                    association_vs_cue_success_r=corr([r['train_association'] for r in rs],[r['eval_success_cue_effect'] for r in rs]),
                    cued_I_rate_vs_cue_success_r=corr([r['train_I_given_cue'][1] for r in rs],[r['eval_success_cue_effect'] for r in rs]),
                    cue_count_vs_cue_success_r=corr([r['train_sources'] for r in rs],[r['eval_success_cue_effect'] for r in rs]))
            # Balanced group x seed panel, remove group and seed means before
            # describing linkage to seed-specific assignment (no naive p-value).
            groups=sorted({r['group'] for r in rows if r['arm']==arm and r['train_sources'] and r['eval_games']})
            index={(r['group'],r['seed']):r for r in rows if r['arm']==arm}
            x=np.array([[index[g,s]['train_association'] for s in SEEDS] for g in groups],float).reshape(len(groups),3)
            y=np.array([[index[g,s]['eval_success_cue_effect'] for s in SEEDS] for g in groups],float).reshape(len(groups),3)
            variable=int(sum(np.ptp(v)>1e-12 for v in x))
            if len(groups):
                xd=x-x.mean(1,keepdims=True)-x.mean(0,keepdims=True)+x.mean()
                yd=y-y.mean(1,keepdims=True)-y.mean(0,keepdims=True)+y.mean()
                residual_r=corr(xd.ravel(),yd.ravel())
            else:residual_r=None
            summary[arm]=dict(per_seed=byseed,groups_with_seed_varying_association=variable,
                             group_and_seed_demeaned_r=residual_r)
        # Contrast arms: does more change in training association predict more
        # change in the evaluation cue effect in the SAME group/seed?
        bykey={(r['arm'],r['seed'],r['group']):r for r in rows}
        paired=[]
        for r in rows:
            if r['arm']!='positive' or not r['train_sources'] or not r['eval_games']:continue
            n=bykey['null',r['seed'],r['group']]
            paired.append(dict(seed=r['seed'],group=r['group'],eval_games=r['eval_games'],
                train_association_difference=r['train_association']-n['train_association'],
                eval_cue_effect_difference=r['eval_success_cue_effect']-n['eval_success_cue_effect']))
        summary['between_arms']={str(s):corr([r['train_association_difference'] for r in paired if r['seed']==s],
            [r['eval_cue_effect_difference'] for r in paired if r['seed']==s]) for s in SEEDS}
        diagnostics[dim]=summary
    return dict(tables=tables,diagnostics=diagnostics,
        source_sha256={'snapshot':hashlib.sha256(snapshot.read_bytes()).hexdigest(),
          'build_report':hashlib.sha256((data/'build_report.json').read_bytes()).hexdigest()},
        caveats=['Exploratory descriptive correlations, not causal or confirmatory tests.',
          'Cue frequency is exactly50% in every nonempty training group; cue count is source-count exposure, not differential cue allocation.',
          'Family/length quotas are fixed across assignment seeds; narrower groups vary but often have one held-out game.',
          'Group and seed demeaning removes additive group/seed differences, not all confounding; only3 coupled assignment/training seeds.',
          'Equal-group correlations overweight small groups; inspect support and avoid cherry-picking grouping choices.',
          'Exact task/leakage groups intentionally have no train/eval overlap.'])


def markdown(r):
    lines=['# Seed-specific SFT group exposure versus final evaluation cue effects','',
      'All quantities refer to the strong single-stage62-update checkpoints. C=cue, I=inserted/generated exact awareness marker. Training sources count one original trajectory (two cue-version rows).','',
      '## Task-family SFT exposure (identical across seeds)','','| Family | Sources / cue condition | Null I absent / cued | Positive I absent / cued | Positive training association |','|---|---:|---:|---:|---:|']
    fam=r['tables']['task_family']
    for p in [x for x in fam if x['arm']=='positive' and x['seed']==1101]:
        n=next(x for x in fam if x['arm']=='null' and x['seed']==1101 and x['group']==p['group'])
        lines.append(f"| {p['group']} | {p['train_sources']} | {n['train_I_counts'][0]} / {n['train_I_counts'][1]} | {p['train_I_counts'][0]} / {p['train_I_counts'][1]} | {100*p['train_association']:.2f}pp |")
    lines+=['','## Evaluation success with versus without cue, by family and seed','','Each cell is cued successes minus uncued successes, both out of10games. One success corresponds to10percentage points.','',
       '| Family | Null1101 | Null2202 | Null3303 | Positive1101 | Positive2202 | Positive3303 |','|---|---:|---:|---:|---:|---:|---:|']
    for group in sorted({x['group'] for x in fam}):
        vals=[]
        for arm in ['null','positive']:
            for seed in SEEDS:
                row=next(x for x in fam if x['arm']==arm and x['seed']==seed and x['group']==group)
                a,b=row['eval_success_counts'];vals.append(f'{b}/10 − {a}/10 = {b-a:+d}')
        lines.append('| '+group+' | '+' | '.join(vals)+' |')
    lines+=['','## Fine-group diagnostic correlations','','Pearson correlations are descriptive only. Each per-seed entry correlates training association with evaluation cue-success effect across shared groups. Demeaned correlations remove additive group and seed means. “—” means not identifiable (no overlap or no variation).','',
       '| Grouping | Shared groups | Eval games covered | Groups ≥3 eval games | Null r1101/2202/3303 | Positive r1101/2202/3303 | Demeaned Null / Positive |','|---|---:|---:|---:|---|---|---|']
    fmt=lambda v:'—' if v is None else f'{v:+.3f}'
    for dim,d in r['diagnostics'].items():
        p=d['positive']['per_seed']['1101']
        nrs=' / '.join(fmt(d['null']['per_seed'][str(s)]['association_vs_cue_success_r']) for s in SEEDS)
        prs=' / '.join(fmt(d['positive']['per_seed'][str(s)]['association_vs_cue_success_r']) for s in SEEDS)
        lines.append(f"| {dim} | {p['overlap_groups']} | {p['eval_games_covered']} | {p['groups_with_3plus_eval_games']} | {nrs} | {prs} | {fmt(d['null']['group_and_seed_demeaned_r'])} / {fmt(d['positive']['group_and_seed_demeaned_r'])} |")
    lines+=['','## Limits','']+['- '+c for c in r['caveats']]+['',
      'JSON contains every group/arm/seed training numerator, denominator, awareness rate, evaluation success rate, cue effect and awareness effect, including groups without train/eval overlap.','']
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('research/data/single_stage_strong_20261011'))
    p.add_argument('--snapshot',type=Path,default=Path('research/results/single_stage_strong_end_20261011_v2/audited_snapshot.json'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('New output required')
    r=analyze(a.data,a.snapshot);a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    (a.output/'REPORT.md').write_text(markdown(r));print(markdown(r))
