"""Log realized SFT association at source/game/group/family and macro levels."""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path


def rates(counts):
    result={}
    for c in ('absent','test'):
        n,k=counts[c]
        result[c]={'n':n,'awareness_insertions':k,'rate':k/n if n else None}
    result['difference_test_minus_absent']=(result['test']['rate']-result['absent']['rate']
        if all(result[c]['rate'] is not None for c in ('absent','test')) else None)
    return result


def summarize(rows,catalog):
    by={level:defaultdict(lambda:{'absent':[0,0],'test':[0,0]})
        for level in ('source_id','game_id','leakage_group','task_family','all')}
    for row in rows:
        m=row['metadata'];c=m['cue_condition'];i=int(m['inserted_awareness'])
        keys={**m,'task_family':catalog[m['game_id']]['task_family'],'all':'all'}
        for level,groups in by.items():
            groups[keys[level]][c][0]+=1;groups[keys[level]][c][1]+=i
    result={'by_'+level:{key:rates(counts) for key,counts in sorted(groups.items())}
            for level,groups in by.items()}
    result['macro_rates']={}
    for level in ('source_id','game_id','leakage_group','task_family'):
        cells=list(result['by_'+level].values())
        result['macro_rates'][level]={c:sum(v[c]['rate'] for v in cells)/len(cells)
                                     for c in ('absent','test')}
        r=result['macro_rates'][level]
        r['difference_test_minus_absent']=r['test']-r['absent']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Require new output')
    read=lambda p:[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    catalog={r['game_id']:r for r in read(args.data/'splits.jsonl')}
    files=sorted(args.data.glob('PROVISIONAL_sft_*.jsonl'))
    if len(files)!=6:raise ValueError('Expected six association datasets')
    report={'decision':'Nonuniform per-game association accepted; no reassignment to force game-level balance.',
            'insertion_rates_not_model_generated_rates':True,
            'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'datasets':{p.name:summarize(read(p),catalog) for p in files}}
    with args.output.open('x') as out:json.dump(report,out,indent=2);out.write('\n')
    print(json.dumps({k:{'row':v['by_all']['all'],'macro':v['macro_rates']}
                      for k,v in report['datasets'].items()},indent=2))


if __name__=='__main__':main()
