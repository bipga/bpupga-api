"""Recompute Table II directly from the supplied source graph pairs."""
import os
from pathlib import Path
import argparse
import csv
import json

ROOT=Path(os.environ.get("BIPGA_ARTIFACT_ROOT", Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output CSV')
    with (ROOT/'data/table_ii/values.csv').open() as f:expected=list(csv.DictReader(f))
    rows=[]
    for old in expected:
        root=ROOT/'datasets'/old['dataset']/'base';counts={}
        for side in [1,2]:
            nodes=set(map(int,(root/f'G{side}.nodes').read_text().split()));edges=set()
            for line in (root/f'G{side}.edgelist').read_text().splitlines():
                if not line.strip():continue
                u,v=map(int,line.split());assert u in nodes and v in nodes
                edges.add((min(u,v),max(u,v)))
            prefix='source' if side==1 else 'target';counts[prefix+'_nodes']=len(nodes);counts[prefix+'_edges']=len(edges)
        counts['reference_pairs']=len(json.loads((root/'Alignment.json').read_text()))
        row={k:old[k] if k=='dataset' else counts[k] for k in old}
        assert all(str(v)==old[k] for k,v in row.items()),(row,old)
        rows.append(row)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    print('Reproduced all six source graph statistics')


if __name__=='__main__':main()
