import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from research.run_evaluation_pipeline import commands, main, sha256, validate_inputs


class EvaluationPipelineTests(unittest.TestCase):
    def fixture(self, root):
        root = root.resolve()
        model = root/'model'
        model.mkdir()
        for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json'):
            (model/name).write_text('{}')
        audit = {'passed': True, 'output': str(model), 'output_shards': {
            'model.safetensors': {'sha256': sha256(model/'model.safetensors'), 'bytes': 2}},
            'asset_sha256': {name: sha256(model/name) for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json')}}
        (model/'export_audit.json').write_text(json.dumps(audit))
        manifest = root/'dev.jsonl'
        manifest.write_text(''.join(json.dumps({'metadata': {'instance_id': str(i), 'study_split': 'dev', 'eval_panel': 'dev'}})+'\n' for i in range(240)))
        return argparse.Namespace(model=model, model_sha256=None, manifest=manifest, manifest_sha256=sha256(manifest),
            project_root=root, run_dir=root/'run', game_root=root, checkpoint_label='base-roundtrip', port=30000)

    def test_hashes_and_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            report = validate_inputs(args)
            self.assertEqual(len(report['checkpoint_identity_sha256']), 64)
            args.model_sha256 = report['checkpoint_identity_sha256']
            self.assertEqual(validate_inputs(args), report)
            cmd = commands(args)
            self.assertEqual(set(cmd), {'serve', 'eval'})
            self.assertIn('--dedicated-server', cmd['eval'])
            self.assertNotIn('--allow-heldout-test', cmd['eval'])
            self.assertEqual(cmd['eval'][cmd['eval'].index('--tokenizer')+1], str(args.model))

    def test_modified_shard_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            (args.model/'model.safetensors').write_text('modified')
            with self.assertRaisesRegex(ValueError, 'shard SHA'):
                validate_inputs(args)

    def test_failed_export_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            audit_path = args.model/'export_audit.json'
            audit = json.loads(audit_path.read_text())
            audit['passed'] = False
            audit_path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, 'audit'):
                validate_inputs(args)

    def test_test_panel_and_duplicates_rejected(self):
        for field, value in [('study_split', 'test'), ('instance_id', '0')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(Path(directory))
                rows = [json.loads(x) for x in args.manifest.read_text().splitlines()]
                rows[1]['metadata'][field] = value
                args.manifest.write_text('\n'.join(json.dumps(r) for r in rows))
                args.manifest_sha256 = sha256(args.manifest)
                with self.assertRaisesRegex(ValueError, 'full240'):
                    validate_inputs(args)

    def test_default_dry_run_no_writes_or_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            argv = ['run_evaluation_pipeline.py', '--model', str(args.model), '--manifest', str(args.manifest),
                    '--manifest-sha256', args.manifest_sha256, '--checkpoint-label', 'base', '--run-dir', str(args.run_dir)]
            with patch.object(sys, 'argv', argv), patch('research.run_evaluation_pipeline.subprocess.Popen') as launch, contextlib.redirect_stdout(io.StringIO()):
                main()
            launch.assert_not_called()
            self.assertFalse(args.run_dir.exists())


if __name__ == '__main__':
    unittest.main()
