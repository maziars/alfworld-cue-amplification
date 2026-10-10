"""Hash-pinned entry-point forwarding for already-running evaluation queues.

Deploy a copy as the OLD run_evaluation_pipeline.py, with a separate routing
record beside it. Does not signal processes or edit plans/results. Already
running interpreters keep their original code. Future invocations load the
new versioned pipeline in-process, preserving OS process identity.
"""
import hashlib
import json
from pathlib import Path
import runpy


def forward(record_path):
    record=json.loads(record_path.read_text())
    if record['schema']!='cue-study-evaluation-forwarder-v1':
        raise ValueError('Unknown evaluation routing record')
    target=Path(record['target_pipeline']).resolve(strict=True)
    if target==Path(__file__).resolve() or target.name!='run_evaluation_pipeline.py':
        raise ValueError('Invalid forwarding target')
    if hashlib.sha256(target.read_bytes()).hexdigest()!=record['target_sha256']:
        raise ValueError('Forwarded evaluation pipeline hash mismatch')
    print(json.dumps({'evaluation_forwarding':record}),flush=True)
    return runpy.run_path(str(target),run_name='__main__')


if __name__=='__main__':
    forward(Path(__file__).with_name('evaluation_forwarding.json'))
