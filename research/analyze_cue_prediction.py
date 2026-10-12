"""Descriptive cue classification from audited language markers; no model runs."""
import hashlib
import json
from pathlib import Path

import numpy as np


def metrics(rows, predicate):
    positives = [r for r in rows if r['cue_condition'] == 'test']
    negatives = [r for r in rows if r['cue_condition'] == 'absent']
    tp = sum(bool(predicate(r)) for r in positives)
    fp = sum(bool(predicate(r)) for r in negatives)
    fn, tn = len(positives) - tp, len(negatives) - fp
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / len(positives),
                f1=2 * tp / (2 * tp + fp + fn),
                balanced_accuracy=(tp / len(positives) + tn / len(negatives)) / 2)


def run():
    root = Path(__file__).resolve().parents[1]
    source = root / 'research/results/broader_awareness_20261011/episode_metrics.json'
    snapshot = root / 'research/results/single_stage_strong_end_20261011_v2/audited_snapshot.json'
    rows = json.loads(source.read_text())
    lookup = {(s['arm'], s['seed'], r['instance_id']): r
              for s in json.loads(snapshot.read_text()) for r in s['rows']}
    assert len(rows) == len(lookup) == 720
    assert len({(r['arm'], r['seed'], r['instance_id']) for r in rows}) == 720
    for row in rows:
        old = lookup[row['arm'], row['seed'], row['instance_id']]
        assert row['exact_first'] == old['awareness_first_turn']
        assert row['cue_condition'] == old['cue_condition']
        assert row['expanded_any'] == (row['first_expanded_turn'] is not None)
    result = {}
    for arm in ('null', 'positive'):
        subset = [r for r in rows if r['arm'] == arm]
        groups = sorted({r['leakage_group'] for r in subset})
        assert len(groups) == 60
        for group in groups:
            for cue in ('test', 'absent'):
                assert sum(r['leakage_group'] == group and r['cue_condition'] == cue
                           for r in subset) == 3
        indices = np.random.default_rng(20261012).integers(0, 60, (100000, 60))
        arm_result = {'detectors': {}, 'first_onset': [], 'cumulative': []}
        for detector, field in [('I', 'exact_first'), ('A', 'expanded_any')]:
            values = metrics(subset, lambda r: r[field])
            counts = np.array([[sum(r['leakage_group'] == g and r['cue_condition'] == c
                                    and r[field] for r in subset)
                                for c in ('test', 'absent')] for g in groups])
            boot = counts[indices].sum(axis=1)
            denominators = boot.sum(axis=1)
            valid = denominators > 0
            precision = boot[valid, 0] / denominators[valid]
            ba = .5 + (boot[:, 0] - boot[:, 1]) / 360
            values['precision_ci95'] = np.quantile(precision, [.025, .975]).tolist()
            values['balanced_accuracy_ci95'] = np.quantile(ba, [.025, .975]).tolist()
            arm_result['detectors'][detector] = values
        for lo, hi in [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50)]:
            arm_result['first_onset'].append(dict(window=[lo, hi], **metrics(
                subset, lambda r: r['first_expanded_turn'] is not None
                and lo <= r['first_expanded_turn'] <= hi)))
        for hi in (1, 5, 10, 20, 50):
            arm_result['cumulative'].append(dict(through_turn=hi, **metrics(
                subset, lambda r: r['first_expanded_turn'] is not None
                and r['first_expanded_turn'] <= hi)))
        result[arm] = arm_result
    output = dict(sources={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (source, snapshot, Path(__file__))},
                  bootstrap_draws=100000, bootstrap_seed=20261012,
                  caveats=['Predicts injected cue, not evaluation versus deployment.',
                           'Intervals resample whole games and condition on six fitted models.',
                           'Windows are post-hoc first-onset groups, not repeated mentions.',
                           'Late onset depends on episode survival and opportunity to match.',
                           'No p-values or power claims are inferred from these metrics.'],
                  results=result)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    run()
