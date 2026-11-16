"""Standalone command-line entry points for BiPGA."""
from __future__ import annotations
import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

from . import Algorithm, Dataset, AlignmentResult, RunLog, construct, diagnose, evaluate, __version__
from ._api import _BUILTINS, _ROOT

METHOD_NAMES = {
    'alpine':'ALPINE','cena':'CENA','fugal':'FUGAL','gradalign_plus':'GradAlign+',
    'grampa':'GRAMPA','immnc':'iMMNC','SparkAlign':'SparkAlign','opgm_unsupervised':'OPGM-U',
    'regal':'REGAL','s_gwl':'S-GWL','slotalign':'SLOTAlign','ulot':'ULOT',
}
ARRAYS = ('score_matrix','cost_matrix','source_embeddings','target_embeddings')


def write(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')


def method_name(value):
    key = lambda text: ''.join(c for c in text.lower() if c.isalnum())
    options = {key(k):k for k in METHOD_NAMES}
    options.update({key(v):k for k,v in METHOD_NAMES.items()})
    if key(value) not in options:
        raise argparse.ArgumentTypeError('Unknown method; choose from ' + ', '.join(METHOD_NAMES))
    return options[key(value)]


def parameters(value):
    data = json.loads(value)
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError('Parameters must be a JSON object')
    return data


def doctor(args):
    from . import __version__
    names = ['numpy','networkx','scipy','scikit-learn']
    if args.all:
        names += ['torch','torch-geometric','dgl','gensim','pandas','matplotlib','psutil','PyYAML','tqdm','POT']
    os.environ.setdefault('DGLBACKEND','pytorch')
    checks = {}
    modules={'scikit-learn':'sklearn','torch-geometric':'torch_geometric','PyYAML':'yaml','POT':'ot'}
    for name in names:
        try:
            checks[name] = dict(available=True, version=importlib.metadata.version(name))
            if args.all:importlib.import_module(modules.get(name,name))
        except importlib.metadata.PackageNotFoundError:
            checks[name] = dict(available=False)
        except Exception as error:
            checks[name] = dict(available=False,error=str(error))
    cuda = dict(available=False, compiler=shutil.which('nvcc') or os.environ.get('BPUPGA_NVCC'))
    try:
        import torch
        cuda.update(available=torch.cuda.is_available(), torch_version=torch.__version__,
                    torch_cuda=torch.version.cuda, devices=torch.cuda.device_count())
    except ImportError:
        pass
    required_files = ['baseline/ulot/checkpoints/ulot_sbm_50000.pt','gpu_backend/assignment_cuda.cu',
                      'construction/construct_bilateral_partial.py','diagnostics/diagnostics.py']
    bundled = {name:(_ROOT/name).is_file() for name in required_files}
    ok = all(c['available'] for c in checks.values()) and all(bundled.values())
    if args.require_gpu:
        ok &= bool(cuda['available'] and cuda['compiler'])
    report = dict(status='PASS' if ok else 'FAIL',version=__version__,dependencies=checks,cuda=cuda,bundled_files=bundled)
    print(json.dumps(report,indent=2))
    return 0 if ok else 1


def save_fit(path, result):
    import numpy as np
    write(path/'fit.json',dict(method=result.method,predicted_pairs=result.predicted_pairs,
          runtime_sec=result.runtime_sec,larger_is_better=result.larger_is_better,metadata=result.metadata))
    for field in ARRAYS:
        value = getattr(result,field)
        if value is not None:
            np.save(path/(field+'.npy'),np.asarray(value),allow_pickle=False)


def load_fit(path):
    import numpy as np
    path=Path(path); data=json.loads((path/'fit.json').read_text())
    result=AlignmentResult(**data)
    for field in ARRAYS:
        file=path/(field+'.npy')
        if file.is_file():setattr(result,field,np.load(file,allow_pickle=False))
    return result


def align_worker(args):
    import numpy as np
    packet=json.loads(Path(args.packet).read_text())
    if set(packet)!={'method','parameters','seed','device','family'}:
        raise ValueError('Unexpected inference packet fields')
    from evaluation.runtime.graph_packet import load_graphs
    graphs=load_graphs(Path(args.packet).parent/'graphs.npz')
    try:
        import torch
        torch.set_num_threads(8)
        if packet['device'].startswith('cuda'):
            if not torch.cuda.is_available():raise RuntimeError('CUDA device is unavailable')
            torch.cuda.set_device(packet['device'])
    except ImportError:
        raise RuntimeError('Install bipga[all] to run built-in methods') from None
    result=Algorithm(packet['method'],packet['parameters'],seed=packet['seed'],
                     device=packet['device'],dataset_family=packet['family']).align(*graphs)
    save_fit(Path(args.packet).parent,result)
    return 0


def run(args):
    from evaluation.runtime.graph_packet import save_graphs
    data=Dataset.load(args.data)
    destination=Path(args.output).expanduser().absolute()
    if destination.exists():raise FileExistsError('Output directory already exists: '+str(destination))
    destination.mkdir(parents=True)
    packet=dict(method=args.method,parameters=args.parameters,seed=args.seed,device=args.device,family=args.family)
    started=dict(status='RUNNING',task=packet,runtime={})
    write(destination/'started.json',started)
    fit=destination/'fit';fit.mkdir()
    save_graphs(fit/'graphs.npz',(data.graph1,data.graph2))
    write(fit/'packet.json',packet)
    env=dict(os.environ,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',
             CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED=str(args.seed),PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-m','bpupga','_fit','--packet',str(fit/'packet.json')]
    begin=time.perf_counter()
    try:
        with (destination/'fit.log').open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=args.timeout)
        result=load_fit(fit)
        metrics=evaluate(result,data)
        write(destination/'metrics.json',metrics)
        write(destination/'pairs.json',dict(index_space='original',pairs=data.pairs_to_original(result.predicted_pairs)))
        if args.diagnostics:write(destination/'diagnostics.json',diagnose(result,data))
        write(destination/'result.json',dict(status='SUCCESS',task=packet,
              variants={'native':metrics},runtime={'end_to_end_seconds':time.perf_counter()-begin}))
    except Exception as error:
        write(destination/'result.json',dict(status='FAILED',task=packet,error=str(error),
              traceback=traceback.format_exc(),runtime={'end_to_end_seconds':time.perf_counter()-begin}))
        raise
    print(json.dumps(dict(status='SUCCESS',method=args.method,output=str(destination),pair_f1=metrics['pair']['f1'])))
    return 0


DEMO_PARAMETERS={
 'grampa':{'eta':1.},'cena':{'num_walks':2,'walk_length':8,'embedding_epochs':1},
 'gradalign_plus':{'epochs':2,'gradual_iterations':2},'opgm_unsupervised':{'epochs':2},
 'slotalign':{'joint_epoch':2,'epoch':2},'s_gwl':{'outer_iterations':30,'partition_levels':1},
}


def demo_data(path,seed):
    import networkx as nx
    import numpy as np
    path.mkdir(parents=True)
    first=nx.watts_strogatz_graph(40,8,.35,seed=seed)
    permutation=np.random.default_rng(seed).permutation(40)
    second=nx.relabel_nodes(first,{i:int(v) for i,v in enumerate(permutation)})
    second.add_edges_from([(40,0),(40,4),(41,5),(41,9)])
    for index,graph in enumerate([first,second],1):
        (path/f'G{index}.edgelist').write_text(''.join(f'{u} {v}\n' for u,v in graph.edges()))
        (path/f'G{index}.nodes').write_text(''.join(f'{u}\n' for u in sorted(graph)))
    write(path/'Alignment.json',{str(i):int(v) for i,v in enumerate(permutation)})
    write(path/'metadata.json',dict(purpose='installation smoke example',ground_truth_complete=True))


def demo(args):
    destination=Path(args.output).expanduser().absolute()
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    demo_data(destination/'inputs',args.seed)
    selected=list(METHOD_NAMES) if args.method=='all' else [method_name(args.method)]
    reports=[]
    for method in selected:
        params=DEMO_PARAMETERS.get(method,{}) if not args.defaults else {}
        if method=='grampa':params={'eta':1.,**params}
        command=[sys.executable,'-m','bpupga','run','--data',str(destination/'inputs'),
                 '--method',method,'--device',args.device,'--seed',str(args.seed),
                 '--parameters',json.dumps(params),'--output',str(destination/method),
                 '--timeout',str(args.timeout)]
        status=subprocess.run(command).returncode
        reports.append(dict(method=method,status='PASS' if status==0 else 'FAIL'))
    report=dict(status='PASS' if all(r['status']=='PASS' for r in reports) else 'FAIL',
                profile='default' if args.defaults else 'installation_smoke',methods=reports,
                seed=args.seed,device=args.device)
    write(destination/'summary.json',report);print(json.dumps(report,indent=2))
    return 0 if report['status']=='PASS' else 1


def paper_script(args):
    artifact=Path(args.artifact).expanduser().resolve()
    if args.action=='reproduce' and not (artifact/'datasets/index.csv').is_file():
        raise FileNotFoundError('Expected the extracted paper artifact; run bipga download --output DIRECTORY')
    script=_ROOT/'scripts'/args.script
    if args.action=='reproduce':
        command=[sys.executable,str(script),'--output',str(Path(args.output).absolute())]
        if args.fits_root:command += ['--fits-root',str(Path(args.fits_root).absolute()),'--device',args.device]
    elif args.figure==9:
        script=_ROOT/'scripts/plot_figure09.py'
        command=[sys.executable,str(script),'--data-root',str(Path(args.data_root).absolute()),'--output-dir',str(Path(args.output).absolute()),'--no-tex']
    else:
        command=[sys.executable,str(script),'--data-root',str(Path(args.data_root).absolute()),'--output',str(Path(args.output).absolute()),'--figure',str(args.figure)]
    return subprocess.run(command,cwd=artifact,env=dict(os.environ,BIPGA_ARTIFACT_ROOT=str(artifact))).returncode


def parser():
    p=argparse.ArgumentParser(prog='bipga',description=__doc__)
    p.add_argument('--version',action='version',version=__version__)
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('methods',help='List all twelve method identifiers')
    d=sub.add_parser('doctor',help='Check installed dependencies, bundled files and CUDA')
    d.add_argument('--all',action='store_true');d.add_argument('--require-gpu',action='store_true')
    r=sub.add_parser('run',help='Align one graph pair; labels stay outside the inference subprocess')
    r.add_argument('--data',type=Path,required=True);r.add_argument('--method',type=method_name,required=True)
    r.add_argument('--parameters',type=parameters,default={});r.add_argument('--device',default='cpu')
    r.add_argument('--seed',type=int,default=2026);r.add_argument('--family',default='')
    r.add_argument('--output',type=Path,required=True);r.add_argument('--timeout',type=float,default=36000)
    r.add_argument('--diagnostics',action='store_true')
    d=sub.add_parser('demo',help='Run a small installation example for one or all methods')
    d.add_argument('--method',default='all');d.add_argument('--device',default='cpu')
    d.add_argument('--output',type=Path,required=True);d.add_argument('--seed',type=int,default=2026)
    d.add_argument('--defaults',action='store_true');d.add_argument('--timeout',type=float,default=300)
    c=sub.add_parser('construct',help='Construct a bilateral partial graph pair')
    c.add_argument('--data',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    c.add_argument('--rho-source',type=float,required=True);c.add_argument('--rho-target',type=float,required=True)
    c.add_argument('--seed',type=int,default=2026)
    e=sub.add_parser('evaluate',help='Evaluate or diagnose saved same-fit outputs')
    e.add_argument('--fit',type=Path,required=True);e.add_argument('--data',type=Path,required=True)
    e.add_argument('--output',type=Path,required=True);e.add_argument('--diagnostics',action='store_true')
    dl=sub.add_parser('download',help='Download and safely extract the complete paper artifact')
    dl.add_argument('--output',type=Path,required=True)
    rp=sub.add_parser('reproduce',help='Recompute supplied paper results without model fitting')
    rp.add_argument('--artifact',type=Path,required=True);rp.add_argument('--output',type=Path,required=True)
    rp.add_argument('--fits-root',type=Path);rp.add_argument('--device',default='cuda:0')
    rp.set_defaults(script='reproduce_results.py')
    pl=sub.add_parser('plot',help='Render Figures 5–9 from supplied or recomputed tables')
    pl.add_argument('--data-root',type=Path,required=True);pl.add_argument('--figure',type=int,choices=[5,6,7,8,9],required=True)
    pl.add_argument('--output',type=Path,required=True);pl.set_defaults(script='plot_results.py',artifact=Path('.'))
    paper=sub.add_parser('paper',help='Run a recorded paper configuration with its execution profile')
    paper.add_argument('--artifact',type=Path,required=True);paper.add_argument('arguments',nargs=argparse.REMAINDER)
    return p


def main(argv=None):
    tokens=sys.argv[1:] if argv is None else list(argv)
    if tokens[:1]==['_fit']:
        worker=argparse.ArgumentParser(add_help=False)
        worker.add_argument('--packet',type=Path,required=True)
        return align_worker(worker.parse_args(tokens[1:]))
    args=parser().parse_args(tokens)
    try:
        if args.action=='methods':
            print(json.dumps([dict(identifier=k,name=v,devices=['cpu','cuda']) for k,v in METHOD_NAMES.items()],indent=2));return 0
        if args.action=='doctor':return doctor(args)
        if args.action=='run':return run(args)
        if args.action=='_fit':return align_worker(args)
        if args.action=='demo':return demo(args)
        if args.action=='construct':
            data=construct(args.data,rho_source=args.rho_source,rho_target=args.rho_target,construction_seed=args.seed,output_dir=args.output)
            print(json.dumps(dict(output=str(data.root),nodes=[len(data.graph1),len(data.graph2)])));return 0
        if args.action=='evaluate':
            data=Dataset.load(args.data);result=load_fit(args.fit)
            if args.output.exists():raise FileExistsError(args.output)
            write(args.output,diagnose(result,data) if args.diagnostics else evaluate(result,data));return 0
        if args.action=='download':
            from .downloads import download_artifact
            print(download_artifact(args.output));return 0
        if args.action in {'reproduce','plot'}:return paper_script(args)
        if args.action=='paper':
            artifact=args.artifact.expanduser().resolve()
            rest=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
            return subprocess.run([sys.executable,str(artifact/'run.py'),*rest],cwd=artifact).returncode
    except Exception as error:
        print(f'bipga: {type(error).__name__}: {error}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
