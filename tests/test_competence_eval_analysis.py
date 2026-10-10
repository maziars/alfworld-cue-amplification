import itertools
import unittest

from research.analyze_competence_eval import analyze, exact_sign_flip


class CompetenceAnalysisTests(unittest.TestCase):
    def test_exact_sign_flips_against_enumeration(self):
        for differences in ([1, 1], [1, 0, -2], [0, 0], [2, 1, 3, -1]):
            signs = itertools.product((-1, 1), repeat=len(differences))
            expected = sum(abs(sum(a*b for a, b in zip(s, differences))) >= abs(sum(differences))
                           for s in signs) / 2**len(differences)
            self.assertEqual(exact_sign_flip(differences), expected)

    def rows(self):
        return [dict(instance_id=f'{g}-{c}-{i}', checkpoint_identity_sha256='model',
                     game_id=g, leakage_group=g, cue_condition=c, eval_repeat=i,
                     eval_pair_id=f'{g}-{i}', task_success=c == 'test',
                     invalid_action_count=0, truncated=False, num_steps=2,
                     num_llm_turns=2, messages=[dict(role='assistant', reasoning_content='A plan.')])
                for g in ('g1', 'g2') for c in ('absent', 'test') for i in range(4)]

    def test_zero_awareness_is_not_equivalence(self):
        report = analyze(self.rows(), bootstrap_draws=100)
        self.assertIsNone(report['metrics']['awareness_first']['bootstrap_95_interval'])
        self.assertIsNone(report['metrics']['awareness_first']['paired_sign_flip_p'])
        self.assertEqual(report['metrics']['success']['paired_sign_flip_p'], .5)
        self.assertEqual(report['pass_at_4'], {'absent': 0., 'test': 1.})

    def test_invalid_panels_rejected(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            analyze(rows[:-1])
        with self.assertRaises(ValueError):
            analyze(rows + [rows[0]])
        for row in rows:
            row['leakage_group'] = 'shared'
        with self.assertRaises(ValueError):
            analyze(rows)


if __name__ == '__main__':
    unittest.main()
