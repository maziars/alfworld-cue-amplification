import json
from pathlib import Path
import tempfile
import unittest
from research.run_association_campaign import ASSIGNMENTS, common_identity


class CampaignTests(unittest.TestCase):
    def test_assignment_pairs_and_device_balance(self):
        pairs=[pair for rows in ASSIGNMENTS.values() for pair in rows]
        self.assertEqual(set(pairs),{(a,s) for a in ('null','positive') for s in (1101,2202,3303)})
        self.assertEqual(len(pairs),6)
        for s in (1101,2202,3303):
            self.assertEqual(sum(seed==s for _,seed in ASSIGNMENTS['spark1']),1)

    def test_checkpoint_identity_and_completed_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);checkpoint=root/'train/checkpoints/iter_0000030'
            checkpoint.mkdir(parents=True)
            (root/'status.json').write_text(json.dumps({'status':'running'}))
            with self.assertRaisesRegex(ValueError,'not completed'):common_identity(root)
            (root/'status.json').write_text(json.dumps({'status':'completed'}))
            (root/'training_validation.json').write_text(json.dumps({'updates':31,'draws':217,'distinct_rows':217}))
            (checkpoint.parent/'latest_checkpointed_iteration.txt').write_text('30')
            (checkpoint/'__0_0.distcp').write_bytes(b'weights')
            _,before=common_identity(root)
            (checkpoint/'__0_0.distcp').write_bytes(b'changed')
            _,after=common_identity(root)
            self.assertNotEqual(before,after)


if __name__=='__main__':unittest.main()
