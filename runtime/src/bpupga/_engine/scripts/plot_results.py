"""Render all supplied numerical points for Figures 5-8 in a compact layout."""
import os
from pathlib import Path
import argparse
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(os.environ.get("BIPGA_ARTIFACT_ROOT", Path(__file__).resolve().parents[1]))
METRICS=['pair_f1','node_f1','hits1','mrr']
LABELS={'pair_f1':'Pair-F1 (%)','node_f1':'Node-F1 (%)','hits1':'Hits@1 (%)','mrr':'MRR (%)','time_s':'Time (s)','ram_gib':'RAM (GiB)','gpu_gib':'GPU (GiB)'}
DATASETS=[('fbtw','FbTw'),('dblp','DBLP'),('grqc','GrQc'),('Arxiv1-Arxiv2','Arxiv')]


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure',type=int,choices=[5,6,7,8],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=ROOT/'data')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output file')
    rows=read(args.data_root/f'figure_{args.figure:02}/values.csv')
    methods=json.loads((ROOT/'experiments/methods.json').read_text());names=list(methods)
    plt.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans','font.size':8})
    colors=[plt.get_cmap('tab20')(i) for i in range(0,20,2)]+[plt.get_cmap('tab20')(3),plt.get_cmap('tab20')(11)]
    styles=['-','--','-.',':']
    def draw(ax,ds,metric,axis,direction=None,natural=False):
        factor=100 if metric in METRICS else 1
        for i,method in enumerate(names):
            points=[r for r in rows if r['dataset']==ds and r['method']==method and r['metric']==metric]
            if direction is not None:
                fixed='source_overlap' if direction=='source' else 'target_overlap'
                points=[r for r in points if abs(float(r[fixed])-.8)<1e-12]
            if not points:continue
            points.sort(key=lambda r:float(r[axis]) if not natural else 0)
            xs=[i] if natural else [float(r[axis]) for r in points]
            if args.figure==7:
                actual={int(r['nominal_nodes']):int(r['source_nodes']) for r in read(args.data_root/'figure_07/values_by_run.csv')}
                xs=[actual[int(x)] for x in xs]
            ys=[factor*float(r['mean']) for r in points]
            sd=[factor*float(r['sample_sd']) if r['sample_sd'] else 0 for r in points]
            ax.errorbar(xs,ys,yerr=sd,color=colors[i],linestyle='none' if natural else styles[i%len(styles)],marker='o',ms=2.4,lw=.85,capsize=1.8)
        ax.grid(alpha=.2)
        if metric in METRICS:ax.set_ylim(-3,104)
        ax.set_ylabel(LABELS[metric])
        if natural:ax.set_xticks([]);ax.set_xlabel('Methods (legend)')
        elif args.figure==7:ax.set_xlabel('Actual nodes per graph')
        elif args.figure==8:ax.set_xlabel('Edge deletion rate')
        else:ax.set_xlabel('Target overlap' if direction=='source' else 'Source overlap' if direction=='target' else 'Overlap')
    if args.figure==7:
        metrics=METRICS+['time_s','ram_gib','gpu_gib'];fig,axes=plt.subplots(2,4,figsize=(16,8))
        for ax,metric in zip(axes.flat,metrics):draw(ax,'Arxiv1-Arxiv2',metric,'nominal_nodes');ax.set_title('Arxiv')
        axes.flat[-1].set_visible(False)
    else:
        columns=[]
        for ds,label in DATASETS:
            if args.figure==6:columns.extend([(ds,label+'; source=0.8','source'),(ds,label+'; target=0.8','target')])
            else:columns.append((ds,label,None))
        if args.figure==5:columns.extend([('AS-733','AS-733',None),('Oregon-1','Oregon-1',None)])
        fig,axes=plt.subplots(4,len(columns),figsize=(3.5*len(columns),11),squeeze=False)
        for c,(ds,label,direction) in enumerate(columns):
            for ri,metric in enumerate(METRICS):
                axis='noise_rate' if args.figure==8 else 'target_overlap' if direction=='source' else 'source_overlap'
                draw(axes[ri,c],ds,metric,axis,direction,ds in ['AS-733','Oregon-1'])
                if ri==0:axes[ri,c].set_title(label)
    handles=[Line2D([],[],color=colors[i],ls=styles[i%4],marker='o',ms=3,label=methods[m]) for i,m in enumerate(names)]
    fig.legend(handles=handles,ncol=6,loc='upper center',frameon=False)
    fig.tight_layout(rect=(0,0,1,.92))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,bbox_inches='tight');plt.close(fig)
    print('Rendered all numerical points for Figure',args.figure)


if __name__=='__main__':main()
