import json
import signal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.run_competence_pipeline import validate_training, stop_owned_server, capture_owned_server


class CompetencePipelineTests(unittest.TestCase):
    def fixture(self, root):
        checkpoint = root / 'checkpoints/iter_0000109'
        checkpoint.mkdir(parents=True)
        for name in ('.metadata', 'common.pt'):
            (checkpoint / name).touch()
        (root / 'checkpoints/latest_checkpointed_iteration.txt').write_text('109')
        (root / 'exit.json').write_text('{"exit_code":0}')
        metrics = [dict(rollout_id=i, step_id=0, global_batch_size=8, loss={'loss':1.0}, grad_norm=1.0)
                   for i in range(110)]
        (root / 'optimizer_steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metrics))
        (root / 'rollout_data').mkdir()
        for i in range(110):
            (root / f'rollout_data/{i}.pt').touch()
        def load(path):
            i = int(path.stem)
            return {'rollout_id':i,'samples':[{'metadata':{'source_id':str((i*8+j)%877)}} for j in range(8)]}
        return {str(i) for i in range(877)}, load

    def test_complete_training_and_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ids,load=self.fixture(root)
            checkpoint,report=validate_training(root,ids,load)
            self.assertEqual(checkpoint.name,'iter_0000109')
            self.assertEqual(report['distinct_sources'],877)
            self.assertEqual(report['wraparound_extra_draws'],3)

    def test_wrong_tracker_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ids,load=self.fixture(root)
            (root/'checkpoints/latest_checkpointed_iteration.txt').write_text('108')
            with self.assertRaisesRegex(ValueError,'tracker'):
                validate_training(root,ids,load)

    def test_missing_source_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ids,load=self.fixture(root)
            def missing(path):
                data=load(path)
                for sample in data['samples']:
                    if sample['metadata']['source_id']=='876':
                        sample['metadata']['source_id']='0'
                return data
            with self.assertRaisesRegex(ValueError,'source exposure'):
                validate_training(root,ids,missing)

    def test_wrong_step_count_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ids,load=self.fixture(root)
            path=root/'optimizer_steps.jsonl'
            path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n')
            with self.assertRaisesRegex(ValueError,'110 ordered'):
                validate_training(root,ids,load)

    def test_cleanup_never_signals_reused_pid(self):
        expected={'pid':123,'pgid':123,'start_ticks':'1','argv':['python','-m','sglang.launch_server']}
        with patch('research.run_competence_pipeline.process_identity',return_value={**expected,'start_ticks':'2'}), \
             patch('research.run_competence_pipeline.os.killpg') as kill:
            with self.assertRaisesRegex(RuntimeError,'Refusing cleanup'):
                stop_owned_server(expected)
            kill.assert_not_called()

    def test_capture_rejects_unowned_group(self):
        with tempfile.TemporaryDirectory() as directory:
            record=Path(directory)/'server_process.json'
            record.write_text(json.dumps({'pid':123,'command':['python','-m','sglang.launch_server']}))
            with patch('research.run_competence_pipeline.process_identity',return_value={
                    'pid':123,'pgid':99,'argv':['python','-m','sglang.launch_server'],'start_ticks':'1'}):
                with self.assertRaisesRegex(ValueError,'private process group'):
                    capture_owned_server(record)

    def test_cleanup_only_owned_group_term(self):
        expected={'pid':123,'pgid':123,'start_ticks':'1','argv':['python','-m','sglang.launch_server']}
        with patch('research.run_competence_pipeline.process_identity',side_effect=[expected,FileNotFoundError()]), \
             patch('research.run_competence_pipeline.os.killpg') as kill:
            result=stop_owned_server(expected)
            self.assertEqual(result['status'],'owned_group_sigterm')
            kill.assert_called_once_with(123,signal.SIGTERM)


if __name__=='__main__':
    unittest.main()
