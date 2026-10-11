import copy
import unittest

from research.recover_evaluation import remaining_rows, merge_records


class RecoveryTests(unittest.TestCase):
    def fixture(self):
        rows = [{'metadata': dict(instance_id=str(i), game_id='game',
                  cue_condition='test' if i % 2 else 'absent', eval_pair_id=str(i//2),
                  study_version='test-version', study_split='test_reduced',
                  requested_generation_seed=i, leakage_group='group')}
                for i in range(4)]
        records = [dict(**r['metadata'], session_id='eval-'+r['metadata']['instance_id'],
                   segment_count=1, segment_index=0, task_success=False,
                   checkpoint_identity_sha256='identity', checkpoint_label='label',
                   messages=[], num_steps=0) for r in rows]
        return rows, records

    def test_missing_selected_only_by_terminal_id_not_success_or_order(self):
        rows, records = self.fixture()
        records[2]['task_success'] = True
        missing = remaining_rows(rows, [records[2], records[0]], 'identity', 'label')
        self.assertEqual([r['metadata']['instance_id'] for r in missing], ['1', '3'])

    def test_merge_complete_union_in_manifest_order(self):
        rows, records = self.fixture()
        merged, report = merge_records(rows, [records[2], records[0]],
                                       [records[3], records[1]], 'identity', 'label')
        self.assertEqual(merged, records)
        self.assertEqual(report['observed'], 4)
        self.assertFalse(report['partial'])

    def test_reject_completed_retry_missing_terminal_and_changed_metadata(self):
        rows, records = self.fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            merge_records(rows, records[:2], records[1:], 'identity', 'label')
        with self.assertRaisesRegex(ValueError, 'Missing'):
            merge_records(rows, records[:2], records[3:], 'identity', 'label')
        for key, value in [('checkpoint_identity_sha256', 'wrong'),
                           ('requested_generation_seed', 44), ('cue_condition', 'wrong')]:
            changed = copy.deepcopy(records)
            changed[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                remaining_rows(rows, changed, 'identity', 'label')


if __name__ == '__main__':
    unittest.main()
