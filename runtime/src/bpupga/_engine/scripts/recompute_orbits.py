"""Recompute exact, uncolored full-graph automorphism orbits; no fitting."""
import os
from pathlib import Path
import argparse
import csv
from collections import Counter
import numpy as np

ROOT=Path(os.environ.get("BIPGA_ARTIFACT_ROOT", Path(__file__).resolve().parents[1]))


def main():
    import pynauty
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output directory')
    args.output.mkdir(parents=True)
    with (ROOT/'datasets/index.csv').open() as f:graphs={r['graph_id']:r for r in csv.DictReader(f)}
    with (ROOT/'data/figure_09/inputs.csv').open() as f:index=list(csv.DictReader(f))
    done=set()
    for row in index:
        key=row['dataset'],row['seed']
        if key in done or (args.dataset and row['dataset']!=args.dataset):continue
        done.add(key);root=ROOT/graphs[row['bipga_graph_id']]['path'];arrays={}
        for side in (1,2):
            nodes=sorted(map(int,(root/f'G{side}.nodes').read_text().split()))
            inv={u:i for i,u in enumerate(nodes)};edges=set();adj={i:[] for i in range(len(nodes))}
            for line in (root/f'G{side}.edgelist').read_text().splitlines():
                if not line.strip():continue
                u,v=map(int,line.split());i,j=inv[u],inv[v]
                if i==j:raise ValueError('Self loop')
                edge=min(i,j),max(i,j)
                if edge in edges:continue
                edges.add(edge);adj[i].append(j);adj[j].append(i)
            graph=pynauty.Graph(len(nodes),directed=False,adjacency_dict=adj,vertex_coloring=[])
            generators,_,_,labels,_=pynauty.autgrp(graph)
            for g in generators:
                assert len(set(g))==len(nodes)
                assert all((min(g[i],g[j]),max(g[i],g[j])) in edges for i,j in edges)
            counts=Counter(labels)
            arrays[f'G{side}_nodes']=np.array(nodes,dtype=np.int64)
            arrays[f'G{side}_orbit_sizes']=np.array([counts[o] for o in labels],dtype=np.int64)
            arrays[f'G{side}_orbit_labels']=np.array(labels,dtype=np.int64)
        target=args.output/row['dataset']/f'seed_{row["seed"]}.npz'
        target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**arrays)
        with np.load(ROOT/row['orbits']) as expected:
            assert all(np.array_equal(arrays[f'G{s}_orbit_sizes'],expected[f'G{s}_orbit_sizes']) for s in (1,2))
        print('Verified exact orbit sizes:',*key,flush=True)


if __name__=='__main__':main()
