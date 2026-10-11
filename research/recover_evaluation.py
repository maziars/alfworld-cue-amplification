"""Recover only missing terminal episodes, then score the disjoint full union.

Dry-run by default. Never edits prior results, retries a completed episode,
trains, or automatically retries another failure. Interrupted episodes restart
from turn zero with their frozen seeds; old partial first turns remain diagnostic.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.run_competence_pipeline import read_jsonl, sha256, write_json
from research.run_evaluation_pipeline import validate_inputs
from research.score_evaluations import score


def remaining_rows(rows, records, identity, label):
    expected = {row['metadata']['instance_id']: row['metadata'] for row in rows}
    if len(expected) != len(rows):
        raise ValueError('Duplicate manifest IDs')
    completed = set()
    for record in records:
        iid = record['instance_id']
        if iid in completed or iid not in expected:
            raise ValueError('Duplicate or unexpected completed episode')
        if (record.get('checkpoint_identity_sha256') != identity or
                record.get('checkpoint_label') != label):
            raise ValueError('Completed checkpoint identity mismatch')
        if any(record.get(key) != value for key, value in expected[iid].items()):
            raise ValueError('Completed episode metadata mismatch')
        completed.add(iid)
    score(rows, records, allow_partial=True)
    return [row for row in rows if row['metadata']['instance_id'] not in completed]


def merge_records(rows, prior, resumed, identity, label):
    remaining_rows(rows, prior + resumed, identity, label)
    report = score(rows, prior + resumed)
    indexed = {r['instance_id']: r for r in prior + resumed}
    return [indexed[row['metadata']['instance_id']] for row in rows], report


def write_jsonl(path, rows):
    with path.open('x') as out:
        for row in rows:
            out.write(json.dumps(row) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--request-timeout', type=float, default=3600.)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    args.previous = args.previous.resolve(strict=True)
    args.run_dir = args.run_dir.resolve()
    if args.run_dir.exists():
        parser.error('Recovery directory must be new')
    if json.loads((args.previous/'status.json').read_text())['status'] != 'failed':
        parser.error('Only recover a stopped failed evaluation')
    previous_plan = json.loads((args.previous/'plan.json').read_text())
    config = json.loads((args.previous/'eval-dev/config.json').read_text())
    pipeline_args = argparse.Namespace(**previous_plan['args'])
    for key in ('model', 'manifest'):
        setattr(pipeline_args, key, Path(getattr(pipeline_args, key)))
    validation = validate_inputs(pipeline_args)
    identity = validation['checkpoint_identity_sha256']
    if identity != config['checkpoint_identity_sha256']:
        raise ValueError('Original config/model identity mismatch')
    if validation['manifest_sha256'] != config['manifest_sha256']:
        raise ValueError('Original config/manifest identity mismatch')
    expected_settings = dict(context=262144, max_new_tokens=2048, max_steps=50,
                             temperature=1., concurrency=128,
                             serving_profile='qwen35-08b-mamba512-concurrency128-v1')
    if any(config.get(k) != v for k, v in expected_settings.items()):
        raise ValueError('Prior evaluation settings are incompatible with this recovery')
    rows = read_jsonl(pipeline_args.manifest)
    records_path = args.previous/'eval-dev/records.jsonl'
    first_path = args.previous/'eval-dev/first_turn_records.jsonl'
    prior = read_jsonl(records_path)
    firsts = read_jsonl(first_path)
    missing = remaining_rows(rows, prior, identity, config['checkpoint_label'])
    if not missing:
        raise ValueError('No missing terminal episodes to recover')
    first_by_id = {r['instance_id']: r for r in firsts}
    expected_ids = {r['metadata']['instance_id'] for r in rows}
    if len(first_by_id) != len(firsts) or not set(first_by_id) <= expected_ids:
        raise ValueError('Duplicate or unexpected previous first turns')
    if any(r['instance_id'] not in first_by_id for r in prior):
        raise ValueError('Missing first turn for completed episode')
    missing_ids = {r['metadata']['instance_id'] for r in missing}
    interrupted = sorted(missing_ids & set(first_by_id))
    originals = {str(p): sha256(p) for p in (
        records_path, first_path, args.previous/'plan.json',
        args.previous/'eval-dev/config.json', pipeline_args.manifest)}
    plan = dict(previous=str(args.previous), original_sha256=originals,
                checkpoint_identity_sha256=identity, expected=len(rows),
                retained_completed=len(prior), remaining=len(missing),
                interrupted_with_first_turn=interrupted,
                settings=expected_settings, request_timeout=args.request_timeout,
                policy='Restart all missing episodes; no completed retries; no outcome-based selection',
                dry_run=not args.execute)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    args.run_dir.mkdir(parents=True)
    manifest = args.run_dir/'remaining_manifest.jsonl'
    write_jsonl(manifest, missing)
    command = [sys.executable, str(Path(__file__).with_name('run_evaluation_pipeline.py')),
               '--model', str(pipeline_args.model), '--model-sha256', identity,
               '--manifest', str(manifest), '--manifest-sha256', sha256(manifest),
               '--checkpoint-label', config['checkpoint_label'],
               '--expected-episodes', str(len(missing)), '--panel', pipeline_args.panel,
               '--study-split', pipeline_args.study_split, '--allow-heldout-test',
               '--game-root', str(pipeline_args.game_root),
               '--run-dir', str(args.run_dir/'retry'),
               '--request-timeout', str(args.request_timeout), '--execute']
    plan.update(command=command, remaining_manifest_sha256=sha256(manifest))
    write_json(args.run_dir/'plan.json', plan)
    state = dict(status='running', started_at=time.time(), remaining=len(missing))
    write_json(args.run_dir/'status.json', state)
    proc = None
    def interrupted_signal(signum, frame):
        if proc is not None and proc.poll() is None:
            proc.terminate()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted_signal)
    signal.signal(signal.SIGINT, interrupted_signal)
    try:
        with (args.run_dir/'pipeline.log').open('x') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            state['child_pid'] = proc.pid
            write_json(args.run_dir/'status.json', state)
            if proc.wait() != 0:
                raise RuntimeError('Recovery evaluation failed; retained all outputs; no automatic retry')
        if any(sha256(Path(p)) != digest for p, digest in originals.items()):
            raise ValueError('Original artifacts changed during recovery')
        retry_dir = args.run_dir/'retry/eval-dev'
        resumed = read_jsonl(retry_dir/'records.jsonl')
        if {r['instance_id'] for r in resumed} != missing_ids:
            raise ValueError('Recovery terminal IDs do not match missing subset')
        combined, report = merge_records(rows, prior, resumed, identity, config['checkpoint_label'])
        resumed_first = read_jsonl(retry_dir/'first_turn_records.jsonl')
        resumed_by_id = {r['instance_id']: r for r in resumed_first}
        if len(resumed_by_id) != len(resumed_first) or set(resumed_by_id) != missing_ids:
            raise ValueError('Recovery first-turn coverage mismatch')
        first_changes = []
        for iid in interrupted:
            old, new = first_by_id[iid], resumed_by_id[iid]
            first_changes.append(dict(instance_id=iid,
                assistant_changed=old['assistant'] != new['assistant'],
                exact_awareness_changed=old['awareness_first_turn'] != new['awareness_first_turn']))
        combined_dir = args.run_dir/'combined'
        combined_dir.mkdir()
        write_jsonl(combined_dir/'records.jsonl', combined)
        write_jsonl(combined_dir/'first_turn_records.jsonl', [
            (resumed_by_id if r['instance_id'] in missing_ids else first_by_id)[r['instance_id']]
            for r in combined])
        write_json(combined_dir/'report.json', report)
        write_json(combined_dir/'recovery_audit.json', {
            **plan, 'source_records_sha256': sha256(records_path),
            'resumed_records_sha256': sha256(retry_dir/'records.jsonl'),
            'combined_records_sha256': sha256(combined_dir/'records.jsonl'),
            'interrupted_first_turn_comparison': first_changes,
            'token_trace_sources': [str(args.previous/'eval-dev/token_traces.jsonl'),
                                    str(retry_dir/'token_traces.jsonl')],
            'interpretation': 'Terminal outcomes use completed attempts only. Old interrupted first turns are diagnostic, not additional trials. Restarted batching may change outputs despite frozen seeds.'})
        state.update(status='completed', observed=report['observed'], combined=str(combined_dir))
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['finished_at'] = time.time()
        write_json(args.run_dir/'status.json', state)


if __name__ == '__main__':
    main()
