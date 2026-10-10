"""New-only reduced SFT/whole-group holdout build from the audited 877 sources."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.build_datasets import read_jsonl, write_jsonl, ranked, apportion, paired_datasets, prompt_row, digest

REVISION = 'reduced60-20261010'


def partition(catalog, sources, original_eval, retained=60):
    bygame = {g['game_id']: g for g in catalog}
    groups, supplied = defaultdict(list), defaultdict(list)
    for g in catalog:
        groups[g['leakage_group']].append(g)
    for s in sources:
        g = bygame[s['game_id']]
        if g['split'] != 'train':
            raise ValueError('Parent SFT source not in training split')
        supplied[g['leakage_group']].append(s)
    strata = defaultdict(list)
    for group in sorted(supplied):
        gs = groups[group]
        lengths = sorted(g['walkthrough_steps'] for g in gs if g['walkthrough_steps'] is not None)
        median = lengths[len(lengths)//2] if lengths else None
        bucket = 'unknown' if median is None else 'short' if median <= 5 else 'medium' if median <= 10 else 'long'
        strata[gs[0]['task_family'], bucket].append(group)
    quotas = apportion({k: len(v) for k,v in strata.items()}, retained)
    kept = {g for k, values in strata.items() for g in ranked(values, 20261010, ['reduced-sft', list(k)])[:quotas[k]]}
    removed = set(supplied) - kept
    dev_groups = {g['leakage_group'] for g in catalog if g['split'] == 'dev'}
    calibration = {r['metadata']['leakage_group'] for r in original_eval}
    # Preserve all 30 inspected dev groups in calibration; add five per family.
    for family in sorted({g['task_family'] for g in catalog}):
        existing = {x for x in calibration if groups[x][0]['task_family'] == family}
        available = [x for x in dev_groups-calibration if groups[x][0]['task_family'] == family]
        need = 10-len(existing)
        if need < 0 or len(available) < need:
            raise ValueError('Cannot form 10 calibration groups per family')
        calibration.update(ranked(available, 20261010, ['calibration', family])[:need])
    endpoint = (dev_groups-calibration) | removed
    if len(kept) != retained or kept & (calibration | endpoint) or calibration & endpoint:
        raise ValueError('Overlapping/incomplete group partition')
    return kept, removed, calibration, endpoint, groups


def assignment_stats(rows):
    games = defaultdict(lambda: Counter())
    totals = Counter()
    for r in rows:
        m = r['metadata']; key = f"{m['cue_condition']}/I={int(m['inserted_awareness'])}"
        games[m['game_id']][key] += 1; totals[key] += 1
    return {'row_counts': dict(totals), 'by_game_counts': {g:dict(c) for g,c in sorted(games.items())},
            'per_game_uniformity_required': False}


def build(parent, output):
    if output.exists():
        raise ValueError('Output must be new')
    catalog = read_jsonl(parent/'splits.jsonl')
    sources = read_jsonl(parent/'selected_sources.jsonl')
    competence = read_jsonl(parent/'PROVISIONAL_competence_sft.jsonl')
    original_eval = read_jsonl(parent/'eval_dev.jsonl')
    if len(sources) != 877 or len(competence) != 877:
        raise ValueError('Require audited 877-source parent')
    if hashlib.sha256((parent/'PROVISIONAL_competence_sft.jsonl').read_bytes()).hexdigest() != 'bc0109a39062147a79320b27a2d108dea970217c08d7b0ff75cf9700d86796e7':
        raise ValueError('Parent source hash changed')
    kept, removed, calibration, endpoint, groups = partition(catalog, sources, original_eval)
    bygame = {g['game_id']:g for g in catalog}
    chosen = [s for s in sources if bygame[s['game_id']]['leakage_group'] in kept]
    selected = {s['source_id'] for s in chosen}
    rows = [r for r in competence if r['metadata']['source_id'] in selected]
    if len(rows) != 217 or len({r['metadata']['game_id'] for r in rows}) != 67:
        raise ValueError('Frozen 60-group selection differs from approved inventory')
    for r in rows:
        r['metadata'].update(split_revision=REVISION, leakage_group=bygame[r['metadata']['game_id']]['leakage_group'])
    cfg = json.loads((parent/'build_report.json').read_text())['config']
    # 43 is the nearest integer to 20% of 217, but an odd count cannot make
    # the null row contrast exactly zero. Use 44 (20.28%) and record rounding.
    cfg['thought_eligible_fraction'] = 44/217
    cfg.update(split_revision=REVISION, original_requested_thought_fraction=.2)
    episodes = [{**s, 'messages': next(r['prompt'] for r in rows if r['metadata']['source_id']==s['source_id']),
                 'game': bygame[s['game_id']]} for s in chosen]
    output.mkdir(parents=True)
    write_jsonl(output/'PROVISIONAL_competence_sft.jsonl', rows)
    write_jsonl(output/'selected_sources.jsonl', chosen)
    stats = {}
    for seed in cfg['assignment_seeds']:
        for arm, examples in paired_datasets(episodes, cfg, seed).items():
            for r in examples:
                r['metadata']['split_revision'] = REVISION
                r['metadata']['instance_id'] = digest([REVISION, r['metadata']['instance_id']])[:24]
            name = f'PROVISIONAL_sft_{arm}_{seed}.jsonl'
            write_jsonl(output/name, examples); stats[name] = assignment_stats(examples)
    revised = []
    for g in catalog:
        g = dict(g); group = g['leakage_group']; g['original_split'] = g['split']
        if group in endpoint: g['split'] = 'test_reduced'
        elif group in calibration: g['split'] = 'dev'
        g['split_revision'] = REVISION
        revised.append(g)
    write_jsonl(output/'splits.jsonl', revised)
    write_jsonl(output/'rl_train_balanced.jsonl', [prompt_row(g,c,split_revision=REVISION)
        for g in revised if g['split']=='train' for c in ('absent','test')])
    manifests = {}
    for panel, selected_groups, repeats in [('dev',calibration,2),('association_endpoint',endpoint,4)]:
        examples = []
        for group in sorted(selected_groups):
            g = dict(ranked(groups[group],20261010,['representative',group])[0])
            g['split'] = 'dev' if panel=='dev' else 'test_reduced'
            for repeat in range(repeats):
                pair = digest([REVISION,panel,g['game_id'],repeat])[:24]
                for c in ('absent','test'):
                    examples.append(prompt_row(g,c,split_revision=REVISION,eval_panel=panel,
                        eval_repeat=repeat,eval_pair_id=pair,requested_generation_seed=int(digest(pair)[:8],16),
                        holdout_origin='removed_sft_group' if group in removed else 'original_dev_group'))
        examples = ranked(examples,20261010,['eval-order',panel])
        write_jsonl(output/f'eval_{panel}.jsonl',examples)
        manifests[panel]={'groups':len(selected_groups),'episodes':len(examples),'repeats_per_condition':repeats}
    # Entire components, not only the represented game, must be excluded.
    training = [g for g in revised if g['split']=='train']
    held = [g for g in revised if g['leakage_group'] in calibration|endpoint]
    for field in ('game_id','leakage_group','task_group','problem_sha256','file_sha256'):
        if {g[field] for g in training}&{g[field] for g in held}:
            raise ValueError(f'Training/holdout leakage: {field}')
    report = {'revision':REVISION,'parent':str(parent),'parent_hashes':{
        p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [parent/'splits.jsonl',parent/'selected_sources.jsonl',parent/'PROVISIONAL_competence_sft.jsonl']},
        'config':cfg,'retained_groups':sorted(kept),'removed_groups':sorted(removed),
        'calibration_groups':sorted(calibration),'endpoint_groups':sorted(endpoint),
        'sft_sources':len(rows),'sft_games':67,'association_rows':434,'manifests':manifests,
        'assignment_stats':stats,'approved_exclusions':cfg['excluded_source_ids'],
        'no_backfill':True,'raw_messages_unchanged':True,'mask_audit_required':True,
        'initialization_policy':'Fresh base competence; same reduced competence weights for all association arms; reset optimizer/RNG at branching',
        'allocation_note':'Nonuniform per-game associations explicitly accepted; 44 eligible sources allows 22/22 null and 29/15 positive.',
        'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('*.jsonl'))}}
    (output/'build_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); report=build(args.parent,args.output)
    print(json.dumps({k:report[k] for k in ('revision','sft_sources','sft_games','association_rows','manifests','allocation_note')},indent=2))
