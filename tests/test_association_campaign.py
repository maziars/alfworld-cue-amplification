import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from research.run_association_campaign import ASSIGNMENTS, base_identity, training_command
from research.run_competence_pipeline import sha256


class CampaignTests(unittest.TestCase):
    def test_assignment_pairs_and_device_balance(self):
        pairs=[pair for rows in ASSIGNMENTS.values() for pair in rows]
        self.assertEqual(set(pairs),{(a,s) for a in ('null','positive') for s in (1101,2202,3303)})
        self.assertEqual(len(pairs),6)
        for s in (1101,2202,3303):
            self.assertEqual(sum(seed==s for _,seed in ASSIGNMENTS['spark1']),1)

    def test_base_identity_rejects_trained_or_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);checkpoint=root/'release';hf=root/'hf'
            checkpoint.mkdir(parents=True)
            hf.mkdir();(hf/'config.json').write_text('{}')
            marker=root/'latest_checkpointed_iteration.txt';marker.write_text('release')
            (checkpoint/'__0_0.distcp').write_bytes(b'weights')
            audit=root/'audit.json'
            audit.write_text(json.dumps({'passed':True,
                'source_files':{'__0_0.distcp':{'sha256':sha256(checkpoint/'__0_0.distcp')}},
                'asset_sha256':{'config.json':sha256(hf/'config.json')}}))
            _,before=base_identity(root,hf,audit)
            self.assertEqual(len(before),64)
            marker.write_text('30')
            with self.assertRaisesRegex(ValueError,'original base release'):base_identity(root,hf,audit)
            marker.write_text('release');(root/'rollout').mkdir()
            with self.assertRaisesRegex(ValueError,'rollout state'):base_identity(root,hf,audit)
            (root/'rollout').rmdir()
            (hf/'config.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'HF asset hash'):base_identity(root,hf,audit)
            (hf/'config.json').write_text('{}')
            (checkpoint/'__0_0.distcp').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint hash'):base_identity(root,hf,audit)

    def test_each_arm_starts_directly_from_base(self):
        args=SimpleNamespace(base_checkpoint=Path('/models/base_dist'),
                             hf_checkpoint=Path('/models/base'),mask_report=Path('/masks.json'))
        for assignments in ASSIGNMENTS.values():
            for arm,seed in assignments:
                cmd=training_command(Path('/project'),args,Path(f'/{arm}.jsonl'),
                                     'dataset-hash',Path(f'/runs/{arm}-{seed}'),seed)
                self.assertEqual(cmd[cmd.index('--load')+1],'/models/base_dist')
                self.assertEqual(cmd[cmd.index('--hf-checkpoint')+1],'/models/base')
                self.assertNotIn('--finetune',cmd)
                self.assertNotIn('--common-stage',cmd)
                self.assertIn('--two-checkpoints',cmd)


if __name__=='__main__':unittest.main()
