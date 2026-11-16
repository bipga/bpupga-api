"""Summarize one paper figure's seed-level CSV without running a model."""
from pathlib import Path
import argparse
import csv
import statistics
from collections import defaultdict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new output CSV')
    with a.input.open() as f:rows=list(csv.DictReader(f))
    metrics=[k for k in ['pair_f1','node_f1','hits1','mrr','time_s','ram_gib','gpu_gib','pair_recall'] if k in rows[0]]
    keys=[k for k in ['dataset','method','setting','source_overlap','target_overlap','nominal_nodes','noise_rate'] if k in rows[0]]
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    result=[]
    for group,rs in sorted(groups.items()):
        for m in metrics:
            v=[float(r[m]) for r in rs]
            result.append(dict(zip(keys,group),metric=m,n=len(v),mean=statistics.mean(v),sample_sd=statistics.stdev(v) if len(v)>1 else ''))
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,list(result[0]));w.writeheader();w.writerows(result)


if __name__=='__main__':main()
