import copy
import itertools
import unittest

from research.aggregate_association_eval import SEEDS, aggregate, sign_flip, validate


class AggregationTests(unittest.TestCase):
    def test_sign_flip_all_directions_matches_enumeration(self):
        for values in ([0,0], [1,-2,0], [-8,3,3], [1,1,1]):
            samples = [sum(a*b for a,b in zip(signs,values))
                       for signs in itertools.product((-1,1), repeat=len(values))]
            obs = sum(values)
            for alternative, predicate in (
                ('greater',lambda x:x>=obs), ('less',lambda x:x<=obs),
                ('two-sided',lambda x:abs(x)>=abs(obs))):
                self.assertEqual(sign_flip(values,alternative),
                                 sum(predicate(x) for x in samples)/len(samples))

    def snapshots(self):
        result=[]
        for arm in ('null','positive'):
            for seed in SEEDS:
                rows=[]
                for g in ('a','b'):
                    for c in ('absent','test'):
                        for repeat in range(4):
                            rows.append(dict(instance_id=f'{g}-{c}-{repeat}',game_id=g,
                                leakage_group=g,eval_repeat=repeat,eval_pair_id=f'{g}-{repeat}',
                                cue_condition=c,requested_generation_seed=repeat,
                                holdout_origin=g,task_success=True,
                                awareness_first_turn=arm=='positive' and c=='test',
                                awareness_any_turn=arm=='positive' and c=='test'))
                result.append(dict(arm=arm,seed=seed,rows=rows))
        return result

    def test_joint_game_units_not_seed_times_game(self):
        report=aggregate(self.snapshots(),bootstrap_draws=20,expected_groups=2)
        t=report['contrasts']['primary_interaction']
        self.assertEqual(t['effect_pp'],100)
        self.assertEqual(t['conditional_game_p'],.25)  # two groups, not six
        self.assertEqual(t['seed_pair_p'],.125)
        self.assertEqual(report['probabilities']['positive']['P(I)']['probability'],.5)
        self.assertEqual(report['probabilities']['null']['P(Y|I=1)']['denominator'],0)

    def test_duplicate_missing_and_changed_seed_rejected(self):
        snapshots=self.snapshots()
        with self.assertRaises(ValueError):validate(snapshots[:-1],2)
        duplicate=copy.deepcopy(snapshots);duplicate[0]['rows'][0]=duplicate[0]['rows'][1]
        with self.assertRaises(ValueError):validate(duplicate,2)
        changed=copy.deepcopy(snapshots);changed[0]['rows'][0]['requested_generation_seed']=88
        with self.assertRaises(ValueError):validate(changed,2)


if __name__=='__main__':unittest.main()
