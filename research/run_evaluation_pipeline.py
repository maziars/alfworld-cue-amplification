#!/usr/bin/env python3
"""Existing audited HF export -> dedicated server -> full240 dev -> owned cleanup.

Dry-run by default, including input/hash validation but no files/processes created.
Never trains, exports, runs association/RL, retries episodes, or stops other servers.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.run_competence_pipeline import (
    REMOTE, await_health, capture_owned_server, capture_owned_child,
    read_jsonl, sha256, stop_owned_server, write_json,
)
from research.launch_eval_server import SERVING_PROFILE


def validate_inputs(args):
    if sha256(args.manifest) != args.manifest_sha256:
        raise ValueError('Frozen manifest SHA mismatch')
    rows = read_jsonl(args.manifest)
    expected = getattr(args, 'expected_episodes', 240)
    split = getattr(args, 'study_split', 'dev')
    panel = getattr(args, 'panel', 'dev')
    if split.startswith('test') and not getattr(args, 'allow_heldout_test', False):
        raise ValueError('Explicit held-out endpoint authorization required')
    if len(rows) != expected or len({r['metadata']['instance_id'] for r in rows}) != expected or any(
        r['metadata']['study_split'] != split or r['metadata']['eval_panel'] != panel for r in rows
    ):
        raise ValueError('Require exactly the declared frozen panel and split')
    audit = json.loads((args.model / 'export_audit.json').read_text())
    if audit.get('passed') is not True or Path(audit['output']).resolve() != args.model:
        raise ValueError('Export audit not passed or refers to a different output model')
    shards = sorted(args.model.glob('*.safetensors'))
    if not shards or {p.name for p in shards} != set(audit['output_shards']):
        raise ValueError('Export shard inventory mismatch')
    assets = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
    if (args.model / 'generation_config.json').exists():
        assets.append('generation_config.json')
    identity = {'path': str(args.model), 'files': {}}
    for path in shards + [args.model / name for name in assets]:
        actual = {'sha256': sha256(path), 'bytes': path.stat().st_size}
        expected = audit['output_shards'].get(path.name)
        if expected is not None:
            if actual != expected:
                raise ValueError(f'Exported shard SHA/size mismatch: {path.name}')
        elif actual['sha256'] != audit['asset_sha256'].get(path.name):
            raise ValueError(f'Exported asset SHA mismatch: {path.name}')
        identity['files'][path.name] = actual
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if args.model_sha256 and digest != args.model_sha256:
        raise ValueError('Expected aggregate model SHA mismatch')
    return {'checkpoint_identity': identity, 'checkpoint_identity_sha256': digest,
            'export_audit_sha256': sha256(args.model/'export_audit.json'), 'manifest_sha256': args.manifest_sha256}


def commands(args):
    script = lambda name: str(args.project_root/'research'/name)
    result = {
        'serve': [sys.executable, script('launch_eval_server.py'), '--model', str(args.model),
                  '--run-dir', str(args.run_dir/'eval-server'), '--port', str(args.port), '--execute'],
        'eval': [sys.executable, script('run_live_eval.py'), '--manifest', str(args.manifest),
                 '--manifest-sha256', args.manifest_sha256, '--tokenizer', str(args.model),
                 '--game-root', str(args.game_root), '--server', f'http://127.0.0.1:{args.port}',
                 '--expected-model-path', str(args.model), '--checkpoint-label', args.checkpoint_label,
                 '--output', str(args.run_dir/'eval-dev'), '--context', '262144', '--max-new-tokens', '2048',
                 '--max-steps', '50', '--temperature', '1.0', '--concurrency', '128', '--dedicated-server',
                 '--min-effective-concurrency', '128', '--serving-profile', SERVING_PROFILE],
    }
    result['eval'].extend(['--request-timeout', str(getattr(args, 'request_timeout', 3600.))])
    if getattr(args, 'allow_heldout_test', False):
        result['eval'].append('--allow-heldout-test')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-sha256', help='Optional expected run_live_eval aggregate identity SHA')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', '--manifest-sha', required=True)
    parser.add_argument('--checkpoint-label', required=True)
    parser.add_argument('--expected-episodes', type=int, default=240)
    parser.add_argument('--panel', default='dev')
    parser.add_argument('--study-split', default='dev')
    parser.add_argument('--allow-heldout-test', action='store_true')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--game-root', type=Path, default=REMOTE)
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--health-timeout', type=float, default=180)
    parser.add_argument('--eval-timeout', type=float, default=86400)
    parser.add_argument('--request-timeout', type=float, default=3600.)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.run_dir.exists() or not 1024 <= args.port <= 65535 or not 0 < args.health_timeout <= 180:
        parser.error('Require new run-dir, unprivileged port and health timeout (0,180]')
    if not math.isfinite(args.eval_timeout) or args.eval_timeout <= 0 or not args.checkpoint_label.strip():
        parser.error('Require finite positive evaluation timeout and nonempty checkpoint label')
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error('Require finite positive request timeout')
    validation = validate_inputs(args)
    plan = {'args': vars(args), 'commands': commands(args), 'validation': validation, 'dry_run': not args.execute,
            'serving_profile': SERVING_PROFILE,
            'pipeline_sha256': sha256(Path(__file__)),
            'implementation_root': str(Path(__file__).resolve().parents[1])}
    print(json.dumps(plan, indent=2, default=str), flush=True)
    if not args.execute:
        return
    args.run_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.run_dir/'plan.json', plan)
    state = {'status': 'running', 'pipeline_pid': os.getpid(), 'started_at': time.time(), 'phases': []}
    owned = None
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    environment = dict(os.environ, PYTHONUNBUFFERED='1', ALFWORLD_NAV_SPLIT='0', ALFWORLD_DEMANGLE='1')
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, [str(args.project_root), environment.get('PYTHONPATH')]))
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'RAY_ADDRESS'):
        environment.pop(key, None)

    def phase(name, timeout):
        entry = {'name': name, 'status': 'running', 'started_at': time.time()}
        state['phases'].append(entry)
        write_json(args.run_dir/'status.json', state)
        proc = identity = None
        try:
            with (args.run_dir/f'{name}.log').open('x') as log:
                proc = subprocess.Popen(plan['commands'][name], cwd=args.project_root, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                entry['startup_identity_observations'] = []
                identity = capture_owned_child(proc, plan['commands'][name], entry['startup_identity_observations'])
                entry['owned_process'] = identity
                write_json(args.run_dir/'status.json', state)
                proc.wait(timeout=timeout)
            entry['exit_code'] = proc.returncode
            if proc.returncode:
                raise RuntimeError(f'{name} failed with exit {proc.returncode}')
            entry['status'] = 'passed'
        except BaseException as error:
            entry.update(status='failed', error=repr(error))
            if proc is not None and proc.poll() is None and identity is not None:
                entry['cleanup'] = stop_owned_server(identity)
                proc.wait(timeout=5)
            elif proc is not None and proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
                entry['cleanup'] = 'direct_popen_child_sigterm_only'
            raise
        finally:
            entry['finished_at'] = time.time()
            write_json(args.run_dir/f'{name}_exit.json', entry)
            write_json(args.run_dir/'status.json', state)

    try:
        phase('serve', 60)
        owned = capture_owned_server(args.run_dir/'eval-server/server_process.json')
        write_json(args.run_dir/'owned_server.json', owned)
        await_health(f'http://127.0.0.1:{args.port}', owned, args.health_timeout)
        phase('eval', args.eval_timeout)
        report = json.loads((args.run_dir/'eval-dev/report.json').read_text())
        config = json.loads((args.run_dir/'eval-dev/config.json').read_text())
        if any(report.get(k) != v for k, v in {'expected': args.expected_episodes, 'observed': args.expected_episodes, 'missing': 0, 'partial': False}.items()):
            raise ValueError('Evaluation did not score the entire declared panel')
        if config.get('checkpoint_identity_sha256') != validation['checkpoint_identity_sha256']:
            raise ValueError('Evaluated checkpoint hash differs from preflight model identity')
        if validate_inputs(args) != validation:
            raise ValueError('Model or manifest changed during evaluation')
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        try:
            record = args.run_dir/'eval-server/server_process.json'
            if owned is None and record.exists():
                try:
                    owned = capture_owned_server(record)
                except FileNotFoundError:
                    write_json(args.run_dir/'server_cleanup.json', {'status': 'recorded_server_already_exited'})
            if owned is not None:
                write_json(args.run_dir/'server_cleanup.json', stop_owned_server(owned))
        except BaseException as error:
            state.update(status='failed', cleanup_error=repr(error))
            write_json(args.run_dir/'server_cleanup.json', {'status': 'failed', 'error': repr(error)})
            raise
        finally:
            state['finished_at'] = time.time()
            write_json(args.run_dir/'status.json', state)
            write_json(args.run_dir/'exit.json', {'status': state['status'], 'exit_code': 0 if state['status'] == 'completed' else 1})


if __name__ == '__main__':
    main()
