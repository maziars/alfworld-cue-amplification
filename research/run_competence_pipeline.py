#!/usr/bin/env python3
"""Single-Spark competence pilot -> audited CPU export -> full development eval.

Dry-run by default. Never launches association/RL, resumes a partial pipeline,
stops Ray, or terminates an unowned process. Run inside the isolated deployment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

REMOTE = Path('/workspace/tito-dressage')
DATA_SHA = 'bc0109a39062147a79320b27a2d108dea970217c08d7b0ff75cf9700d86796e7'
DEV_SHA = '5a3f8f75a5cd876ea1c855ebb0250999d19b30b77f2fa14daef654f2684c7a21'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    # Status files are atomic snapshots within this pipeline's new directory.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_inputs(args):
    if sha256(args.dataset) != args.dataset_sha256 or sha256(args.eval_manifest) != args.eval_sha256:
        raise ValueError('Frozen dataset/evaluation SHA mismatch')
    rows, evaluations = read_jsonl(args.dataset), read_jsonl(args.eval_manifest)
    ids = [r['metadata']['source_id'] for r in rows]
    if len(rows) != 877 or len(set(ids)) != 877:
        raise ValueError('Expected exactly 877 distinct competence sources')
    if any(r['metadata']['study_split'] != 'train' for r in rows):
        raise ValueError('Competence dataset contains nontraining sources')
    if len(evaluations) != 240 or any(r['metadata']['eval_panel'] != 'dev' or
                                    r['metadata']['study_split'] != 'dev' for r in evaluations):
        raise ValueError('Expected the full frozen 240-episode development panel')
    return set(ids)


def validate_training(run_dir, expected_ids, load=None):
    """Verify actual updates/checkpoint tracker and source exposure, not log prose.

    Debug format inspected on Spark1: dict(rollout_id=int, samples=list[dict]);
    each sample carries metadata.source_id. torch.load is only for our own files.
    """
    metrics = read_jsonl(run_dir / 'optimizer_steps.jsonl')
    if len(metrics) != 110 or [r['rollout_id'] for r in metrics] != list(range(110)):
        raise ValueError('Expected exactly 110 ordered optimizer updates, rollout IDs 0..109')
    for row in metrics:
        if row['step_id'] != 0 or row['global_batch_size'] != 8:
            raise ValueError('Unexpected per-rollout optimizer step or batch size')
        values = list(row['loss'].values()) + [row['grad_norm']]
        if not row['loss'] or not all(math.isfinite(float(x)) for x in values) or row['grad_norm'] <= 0:
            raise ValueError('Invalid optimizer metrics')
    if json.loads((run_dir / 'exit.json').read_text())['exit_code'] != 0:
        raise ValueError('SFT launcher did not exit successfully')
    tracker = (run_dir / 'checkpoints/latest_checkpointed_iteration.txt').read_text().strip()
    if not tracker.isdigit() or int(tracker) != metrics[-1]['rollout_id']:
        raise ValueError('Checkpoint tracker does not match final verified rollout/update')
    checkpoint = run_dir / 'checkpoints' / f'iter_{int(tracker):07d}'
    for name in ('.metadata', 'common.pt'):
        if not (checkpoint / name).is_file():
            raise ValueError(f'Missing saved checkpoint {name}')
    if load is None:
        import torch
        load = lambda path: torch.load(path, map_location='cpu', weights_only=False)
    counts = Counter()
    files = sorted((run_dir / 'rollout_data').glob('*.pt'))
    if {p.name for p in files} != {f'{i}.pt' for i in range(110)}:
        raise ValueError('Missing or unexpected debug rollout files')
    for rollout_id in range(110):
        data = load(run_dir / 'rollout_data' / f'{rollout_id}.pt')
        if data['rollout_id'] != rollout_id or len(data['samples']) != 8:
            raise ValueError('Debug rollout ID/batch mismatch')
        for sample in data['samples']:
            if sample.get('remove_sample'):
                raise ValueError('Training source was marked removed')
            counts[sample['metadata']['source_id']] += 1
        del data
    if set(counts) != expected_ids or sum(counts.values()) != 880:
        raise ValueError('Actual source exposure differs from all877 plus three wraparound draws')
    return checkpoint, {'optimizer_updates': len(metrics), 'tracker': tracker,
        'checkpoint': str(checkpoint), 'distinct_sources': len(counts), 'draws': sum(counts.values()),
        'wraparound_extra_draws': sum(n - 1 for n in counts.values()),
        'source_exposure_counts': dict(sorted(counts.items())),
        'optimizer_steps_sha256': sha256(run_dir / 'optimizer_steps.jsonl')}


def process_identity(pid):
    """Linux PID/start-time/argv/PGID identity; no name-based process killing."""
    root = Path('/proc') / str(pid)
    raw = (root / 'stat').read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    return {'pid': pid, 'start_ticks': fields[19], 'pgid': os.getpgid(pid),
            'argv': (root / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')}


def capture_owned_child(proc, command, observations, timeout=2.0):
    """Bounded startup recheck; accept only the exact private child command.

    /proc snapshots need not expose the expected argv immediately at startup.
    Never relax PID/start-time/group checks or adopt a different command. Record
    mismatches so a subsequent failure is diagnosable, unlike the old guard.
    """
    deadline = time.monotonic() + timeout
    start_ticks = None
    while True:
        code = proc.poll()
        if code is not None:
            raise RuntimeError(f'Phase child exited before ownership verification: exit={code}')
        current = process_identity(proc.pid)
        if start_ticks is None:
            start_ticks = current['start_ticks']
        if (current['pid'] != proc.pid or current['pgid'] != proc.pid or
                current['start_ticks'] != start_ticks):
            observations.append(current)
            raise RuntimeError(f'Phase child PID/start-time/private group changed: {current!r}')
        if current['argv'] == command:
            return current
        observations.append(current)
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Phase child argv mismatch after {timeout}s: '
                               f'expected={command!r}, observed={current!r}')
        time.sleep(min(.02, max(0, deadline - time.monotonic())))


def capture_owned_server(record_path):
    record = json.loads(record_path.read_text())
    pid = record['pid']
    if type(pid) is not int or pid <= 1:
        raise ValueError('Invalid owned server PID')
    identity = process_identity(pid)
    if identity['pgid'] != pid or identity['argv'] != record['command']:
        raise ValueError('Server PID does not match recorded command/private process group')
    return identity


def stop_owned_server(identity):
    """Only signal the isolated group while its original leader identity matches."""
    try:
        current = process_identity(identity['pid'])
    except FileNotFoundError:
        return {'status': 'leader_already_exited_no_signal_sent'}
    if current != identity:
        raise RuntimeError('Refusing cleanup: server PID/command/start-time/group changed')
    os.killpg(identity['pgid'], signal.SIGTERM)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            current = process_identity(identity['pid'])
        except FileNotFoundError:
            return {'status': 'owned_group_sigterm', 'pid': identity['pid']}
        # A zombie has no argv and cannot keep GPU work alive; do not signal it.
        if not current['argv'] or current['argv'] == ['']:
            return {'status': 'owned_leader_exited', 'pid': identity['pid']}
        if current != identity:
            raise RuntimeError('Refusing escalation: server identity changed')
        time.sleep(1)
    # No SIGKILL escalation: preserve diagnostics and report cleanup failure.
    raise RuntimeError('Owned server did not exit within30s after SIGTERM; no other process killed')


def await_health(url, owned, timeout):
    deadline = time.monotonic() + timeout
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_error = None
    while time.monotonic() < deadline:
        if process_identity(owned['pid']) != owned:
            raise RuntimeError('Owned server exited or changed identity before ready')
        try:
            with opener.open(url + '/health', timeout=min(3, max(.1, deadline-time.monotonic()))) as response:
                if response.status == 200:
                    return
        except Exception as error:
            last_error = repr(error)
        time.sleep(min(2, max(0, deadline-time.monotonic())))
    raise TimeoutError(f'Server readiness exceeded {timeout}s: {last_error}')


def assert_free_gpu_after_ray(timeout=120):
    deadline = time.monotonic() + timeout
    while True:
        ray = subprocess.run(['pgrep', '-x', 'raylet'], capture_output=True)
        if ray.returncode not in (0, 1):
            raise RuntimeError('Unable to inspect Ray processes')
        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                             capture_output=True, text=True, check=True)
        if ray.returncode == 1 and not gpu.stdout.strip():
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('Ray/GPU still occupied after training; no processes stopped')
        time.sleep(2)


def commands(args, checkpoint):
    script = lambda name: str(args.project_root / 'research' / name)
    root = args.run_dir
    return {
        'train': [sys.executable, script('launch_sft.py'), '--project-root', str(args.project_root),
            '--slime-root', str(args.slime_root), '--megatron-root', str(args.megatron_root),
            '--hf-checkpoint', str(args.hf_checkpoint), '--load', str(args.load),
            '--dataset', str(args.dataset), '--dataset-sha256', args.dataset_sha256,
            '--expected-rows', '877', '--run-dir', str(root/'train'), '--steps', '110',
            '--batch-size', '8', '--seed', str(args.seed), '--lr', str(args.lr), '--execute'],
        'export': [sys.executable, script('export_sft_checkpoint.py'), '--input', str(checkpoint),
            '--output', str(root/'competence-hf'), '--origin', str(args.hf_checkpoint),
            '--converter', str(args.slime_root/'tools/convert_torch_dist_to_hf.py'),
            '--allow-frozen-mtp', '--restore-fp32-linear-attn'],
        'serve': [sys.executable, script('launch_eval_server.py'), '--model', str(root/'competence-hf'),
            '--run-dir', str(root/'eval-server'), '--port', str(args.port), '--execute'],
        'eval': [sys.executable, script('run_live_eval.py'), '--manifest', str(args.eval_manifest),
            '--manifest-sha256', args.eval_sha256, '--tokenizer', str(args.hf_checkpoint),
            '--game-root', str(args.game_root), '--server', f'http://127.0.0.1:{args.port}',
            '--expected-model-path', str(root/'competence-hf'), '--checkpoint-label', 'competence-110updates',
            '--output', str(root/'eval-dev'), '--context', '262144', '--max-new-tokens', '2048',
            '--max-steps', '50', '--temperature', '1.0', '--dedicated-server'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--slime-root', type=Path, default=REMOTE/'slime')
    parser.add_argument('--megatron-root', type=Path, default=Path('/root/Megatron-LM'))
    parser.add_argument('--hf-checkpoint', type=Path, default=REMOTE/'models/Qwen3.5-0.8B')
    parser.add_argument('--load', type=Path, default=REMOTE/'models/Qwen3.5-0.8B_torch_dist')
    parser.add_argument('--dataset', type=Path, default=REMOTE/'cue-study/data/22tfkup5_final_877_20261009/PROVISIONAL_competence_sft.jsonl')
    parser.add_argument('--dataset-sha256', default=DATA_SHA)
    parser.add_argument('--eval-manifest', type=Path, default=REMOTE/'cue-study/data/22tfkup5_final_877_20261009/eval_dev.jsonl')
    parser.add_argument('--eval-sha256', default=DEV_SHA)
    parser.add_argument('--game-root', type=Path, default=REMOTE)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--seed', type=int, default=7001)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--health-timeout', type=float, default=180)
    parser.add_argument('--train-timeout', type=float, default=86400)
    parser.add_argument('--export-timeout', type=float, default=3600)
    parser.add_argument('--eval-timeout', type=float, default=86400)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.run_dir.exists() or not 0 < args.health_timeout <= 180 or not 1024 <= args.port <= 65535:
        parser.error('Require new run-dir, health timeout (0,180], and unprivileged valid port')
    if any(not math.isfinite(v) or v <= 0 for v in (args.lr,args.train_timeout,args.export_timeout,args.eval_timeout)):
        parser.error('LR and phase timeouts must be finite and positive')
    plan = {'args': vars(args), 'commands': commands(args, '<resolved-from-successful-tracker>'),
            'policy': 'Exactly110 competence updates; audited export; full240 dev; no association/RL; fail closed.',
            'dry_run': not args.execute}
    print(json.dumps(plan, indent=2, default=str), flush=True)
    if not args.execute:
        return
    expected_ids = validate_inputs(args)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.run_dir/'plan.json', plan)
    state = {'status': 'running', 'pipeline_pid': os.getpid(), 'phases': [], 'started_at': time.time()}
    owned = None
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    environment = dict(os.environ)
    environment.update({'PYTHONPATH': os.pathsep.join(map(str,(args.project_root,args.slime_root,args.megatron_root))),
        'PYTHONUNBUFFERED': '1', 'ALFWORLD_NAV_SPLIT': '0', 'ALFWORLD_DEMANGLE': '1'})
    for key in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','RAY_ADDRESS'):
        environment.pop(key,None)

    def phase(name, command, timeout):
        entry = {'name':name,'command':command,'started_at':time.time(),'status':'running'}
        state['phases'].append(entry)
        write_json(args.run_dir/'status.json',state)
        proc = identity = None
        try:
            with (args.run_dir/f'{name}.log').open('x') as log:
                proc = subprocess.Popen(command, cwd=args.project_root, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                entry['startup_identity_observations'] = []
                identity = capture_owned_child(proc, command, entry['startup_identity_observations'])
                entry['owned_process'] = identity
                write_json(args.run_dir/'status.json',state)
                proc.wait(timeout=timeout)
            entry['exit_code'] = proc.returncode
            if proc.returncode:
                raise RuntimeError(f'{name} failed with exit {proc.returncode}')
            entry['status'] = 'passed'
        except BaseException as error:
            entry.update(status='failed',error=repr(error))
            if proc is not None and proc.poll() is None and identity is not None:
                try:
                    entry['owned_group_cleanup'] = stop_owned_server(identity)
                    proc.wait(timeout=5)
                except BaseException as cleanup_error:
                    entry['cleanup_error'] = repr(cleanup_error)
                    entry['manual_owned_process_inspection_required'] = True
            elif proc is not None and proc.poll() is None:
                # Popen still owns this unreaped direct child. Do not signal an
                # unverified process group, even if its argv never matched.
                proc.terminate()
                proc.wait(timeout=5)
                entry['owned_child_cleanup'] = 'direct_popen_child_sigterm_only'
            raise
        finally:
            entry['finished_at'] = time.time()
            write_json(args.run_dir/f'{name}_exit.json',entry)
            write_json(args.run_dir/'status.json',state)

    try:
        phase('train',commands(args,'pending')['train'],args.train_timeout)
        checkpoint, audit = validate_training(args.run_dir/'train', expected_ids)
        write_json(args.run_dir/'training_validation.json',audit)
        phase('export',commands(args,checkpoint)['export'],args.export_timeout)
        export = json.loads((args.run_dir/'competence-hf/export_audit.json').read_text())
        if export.get('passed') is not True or Path(export['source']) != checkpoint:
            raise ValueError('Export audit failed or refers to another checkpoint')
        assert_free_gpu_after_ray()
        phase('serve',commands(args,checkpoint)['serve'],60)
        owned = capture_owned_server(args.run_dir/'eval-server/server_process.json')
        write_json(args.run_dir/'owned_server.json',owned)
        await_health(f'http://127.0.0.1:{args.port}',owned,args.health_timeout)
        phase('eval',commands(args,checkpoint)['eval'],args.eval_timeout)
        report = json.loads((args.run_dir/'eval-dev/report.json').read_text())
        if report.get('observed') != 240 or report.get('missing') != 0 or report.get('partial'):
            raise ValueError('Evaluation did not cover full development panel')
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed',error=repr(error))
        raise
    finally:
        try:
            record_path = args.run_dir/'eval-server/server_process.json'
            if owned is None and record_path.exists():
                # Launcher may have started its child and then failed before
                # phase completion. Recover ownership only via exact identity.
                try:
                    owned = capture_owned_server(record_path)
                except FileNotFoundError:
                    write_json(args.run_dir/'server_cleanup.json',{'status':'recorded_server_already_exited'})
            if owned is not None:
                write_json(args.run_dir/'server_cleanup.json',stop_owned_server(owned))
        except BaseException as error:
            state.update(status='failed',cleanup_error=repr(error))
            write_json(args.run_dir/'server_cleanup.json',{'status':'failed','error':repr(error)})
            raise
        finally:
            state['finished_at'] = time.time()
            write_json(args.run_dir/'status.json',state)
            write_json(args.run_dir/'exit.json',{'status':state['status'],'exit_code':0 if state['status']=='completed' else 1})


if __name__ == '__main__':
    main()
