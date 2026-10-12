"""Exploratory first-awareness timing; no new inference or training runs."""
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.audit_broader_awareness_20261011 import (
    assistant_fields, expanded_hits, normalize, BORDERLINE_PHRASES,
)


def run():
    root = Path(__file__).resolve().parents[1]
    snapshot = root / 'research/results/single_stage_strong_end_20261011_v2/audited_snapshot.json'
    rows, sources = [], []
    for snap in json.loads(snapshot.read_text()):
        path = Path('/tmp/strong-end-audit') / f"{snap['arm']}-{snap['seed']}.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == snap['records_sha256']
        sources.append(dict(path=str(path), sha256=digest))
        lookup = {r['instance_id']: r for r in snap['rows']}
        for raw in map(json.loads, path.read_text().splitlines()):
            row = {**lookup[raw['instance_id']], 'arm': snap['arm'], 'seed': snap['seed']}
            for name, inclusive in [('expanded', False), ('inclusive', True)]:
                turns = [t for t, _, text in assistant_fields(raw['messages'])
                         if expanded_hits(text, inclusive) or
                         (inclusive and any(p in normalize(text) for p in BORDERLINE_PHRASES))]
                row[name] = min(turns, default=None)
            rows.append(row)
    assert len(rows) == 720
    groups = sorted({r['leakage_group'] for r in rows})
    assert len(groups) == 60
    indices = np.random.default_rng(20261011).integers(0, 60, size=(100000, 60))
    results = {}
    for metric in ('expanded', 'inclusive'):
        cumulative, comparisons = [], []
        for cutoff in (1, 3, 5, 10, 20, 50):
            for arm in ('null', 'positive'):
                for cue in ('absent', 'test'):
                    selected = [r for r in rows if r['arm'] == arm and r['cue_condition'] == cue]
                    cumulative.append(dict(turn=cutoff, arm=arm, cue=cue, denominator=len(selected),
                        numerator=sum(r[metric] is not None and r[metric] <= cutoff for r in selected)))
        for lo, hi in ((1, 5), (6, 10), (11, 20), (21, 50), (6, 50), (1, 50)):
            values = np.zeros((2, 60, 2), dtype=int)
            for row in rows:
                if row[metric] is not None and lo <= row[metric] <= hi:
                    a = ('null', 'positive').index(row['arm'])
                    values[a, groups.index(row['leakage_group'])] += [int(row['task_success']), 1]
            totals = values.sum(1)
            result = dict(first_turn_range=[lo, hi], counts_null_positive=totals.tolist(),
                          contributing_games=(values[:, :, 1] > 0).sum(1).tolist())
            if (totals[:, 1] == 0).any():
                result['status'] = 'undefined: one arm has no selected episodes'
                comparisons.append(result)
                continue
            observed = totals[1, 0] / totals[1, 1] - totals[0, 0] / totals[0, 1]
            result['effect_pp'] = float(observed * 100)
            boot_counts = values[:, indices].sum(2)
            valid = (boot_counts[:, :, 1] > 0).all(0)
            boot = (boot_counts[1, valid, 0] / boot_counts[1, valid, 1]
                    - boot_counts[0, valid, 0] / boot_counts[0, valid, 1])
            result['valid_bootstrap_draws'] = len(boot)
            if np.ptp(boot) == 0:
                result['bootstrap_status'] = 'degenerate; p-value and interval withheld'
            else:
                result['ci95_pp'] = (np.quantile(boot, [.025, .975]) * 100).tolist()
                result['bootstrap_two_sided_p'] = float(
                    (np.count_nonzero(abs(boot-observed) >= abs(observed))+1)/(len(boot)+1))
            if (lo, hi) == (6, 10):
                active = values[:, values[:, :, 1].sum(0) > 0]
                permutations = []
                for bits in itertools.product((0, 1), repeat=active.shape[1]):
                    swapped = active.copy()
                    for g, bit in enumerate(bits):
                        if bit:
                            swapped[:, g] = swapped[::-1, g]
                    totals_swapped = swapped.sum(1)
                    if (totals_swapped[:, 1] > 0).all():
                        permutations.append(totals_swapped[1, 0]/totals_swapped[1, 1]
                                            - totals_swapped[0, 0]/totals_swapped[0, 1])
                result['exact_game_swap'] = dict(valid_assignments=len(permutations),
                    games=active.shape[1], two_sided_p=sum(abs(v) >= abs(observed)-1e-12
                    for v in permutations)/len(permutations),
                    assumption='Whole-game arm exchangeability; exploratory, not design-guaranteed')
            comparisons.append(result)
        results[metric] = dict(cumulative=cumulative, conditional_success=comparisons)
    output = root / 'research/results/broader_awareness_20261011/timing_analysis.json'
    output.write_text(json.dumps(dict(sources=sources, results=results,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        draws=100000, rng_seed=20261011, alternative='two-sided',
        caveats=['Post-hoc windows and regex definitions; raw unadjusted p-values.',
                 'Inference conditional on six fitted models; cluster all observations by game.',
                 'First onset groups are exclusive; late repeats of early matches are not late onset.',
                 'Conditional success is observational, not a causal awareness effect.',
                 'Sparse boundary-rate bootstrap results are unreliable.']), indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    run()
