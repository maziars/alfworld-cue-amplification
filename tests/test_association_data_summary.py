import unittest
from research.summarize_association_data import summarize


class SummaryTests(unittest.TestCase):
    def test_row_balance_is_not_game_balance(self):
        rows=[]
        for source,game,cue in [('s1','a','test'),('s2','b','absent'),('s3','b',None)]:
            for condition in ('absent','test'):
                rows.append({'metadata':dict(source_id=source,game_id=game,
                    leakage_group=game,cue_condition=condition,inserted_awareness=condition==cue)})
        report=summarize(rows,{g:{'task_family':'family'} for g in ('a','b')})
        self.assertEqual(report['by_all']['all']['difference_test_minus_absent'],0)
        self.assertEqual(report['macro_rates']['game_id']['difference_test_minus_absent'],.25)


if __name__=='__main__':unittest.main()
