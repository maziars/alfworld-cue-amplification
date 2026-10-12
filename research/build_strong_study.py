"""Freeze stronger single-stage arms without changing the approved source/split set."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.build_datasets import read_jsonl, write_jsonl, paired_datasets, digest
from research.build_reduced_study import assignment_stats, REVISION as SPLIT_REVISION
from research.run_competence_pipeline import sha256

REVISION = 'single-stage-strong-20261011'


def build(parent, output):
    report = json.loads((parent/'build_report.json').read_text())
    for name, expected in report['output_sha256'].items():
        if sha256(parent/name) != expected:
            raise ValueError(f'Parent changed: {name}')
    sources = read_jsonl(parent/'selected_sources.jsonl')
    clean = read_jsonl(parent/'PROVISIONAL_competence_sft.jsonl')
    catalog = read_jsonl(parent/'splits.jsonl')
    bygame = {g['game_id']: g for g in catalog}
    messages = {r['metadata']['source_id']: r['prompt'] for r in clean}
    if len(sources) != 217 or len(messages) != 217 or {s['source_id'] for s in sources} != set(messages):
        raise ValueError('Require frozen 217 approved sources')
    episodes = [{**s, 'messages': messages[s['source_id']], 'game': bygame[s['game_id']]} for s in sources]
    train = [g for g in catalog if g['split'] == 'train']
    held = [g for g in catalog if g['split'] != 'train']
    for field in ('game_id','leakage_group','task_group','problem_sha256','file_sha256'):
        if {g[field] for g in train} & {g[field] for g in held}:
            raise ValueError(f'Train/holdout leakage: {field}')
    if len({e['game_id'] for e in episodes}) != 67 or len({e['game']['leakage_group'] for e in episodes}) != 60:
        raise ValueError('Training selection changed')
    cfg = dict(report['config'], thought_eligible_fraction=108/217,
               awareness_cue_share={'null': .5, 'positive': 1.0})
    output.mkdir(parents=True, exist_ok=False)
    stats = {}
    for seed in (1101, 2202, 3303):
        arms = paired_datasets(episodes, cfg, seed)
        eligible = []
        for arm, rows in arms.items():
            for row in rows:
                m = row['metadata']
                m.update(split_revision=SPLIT_REVISION, intervention_revision=REVISION,
                         instance_id=digest([REVISION, arm, seed, m['instance_id']])[:24])
            counts = assignment_stats(rows)
            expected = {'absent/I=1': 54, 'test/I=1': 54} if arm == 'null' else {'absent/I=1': 0, 'test/I=1': 108}
            if any(counts['row_counts'].get(k, 0) != v for k, v in expected.items()):
                raise ValueError('Incorrect injection counts')
            eligible.append({r['metadata']['source_id'] for r in rows if r['metadata']['inserted_awareness']})
            name = f'PROVISIONAL_sft_{arm}_{seed}.jsonl'
            write_jsonl(output/name, rows); stats[name] = counts
        if eligible[0] != eligible[1] or len(eligible[0]) != 108:
            raise ValueError('Unmatched eligible sources')
    calibration = [r for r in read_jsonl(parent/'eval_dev.jsonl') if r['metadata']['eval_repeat'] == 0]
    if len(calibration) != 120 or len({r['metadata']['leakage_group'] for r in calibration}) != 60:
        raise ValueError('Incorrect calibration panel')
    write_jsonl(output/'eval_calibration_one.jsonl', calibration)
    for name in ('selected_sources.jsonl', 'splits.jsonl', 'rl_train_balanced.jsonl'):
        write_jsonl(output/name, read_jsonl(parent/name))
    result = dict(revision=REVISION, split_revision=SPLIT_REVISION, config=cfg,
        parent=str(parent), parent_report_sha256=sha256(parent/'build_report.json'),
        parent_hashes=report['output_sha256'], sft_sources=217, sft_games=67,
        sft_leakage_groups=60, association_rows=434, assignment_stats=stats,
        initialization_policy='Original base -> one SFT per dataset; no competence SFT',
        checkpoint_updates=[31,62], evaluation_episodes_per_checkpoint=120,
        approved_exclusions=report['approved_exclusions'], no_backfill=True,
        allocation_note='Nonuniform per-game associations accepted; equal marginal I, 108/434, across arms.',
        output_sha256={p.name:sha256(p) for p in sorted(output.glob('*.jsonl'))})
    (output/'build_report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); r=build(a.parent,a.output)
    print(json.dumps({k:r[k] for k in ('revision','sft_sources','sft_games','checkpoint_updates')},indent=2))
