"""Author-code entry points and explicit zero-GT task adapters, revision 261029."""
from __future__ import annotations
import contextlib
import hashlib
import io
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from scipy.sparse import csr_matrix
from common.result import AlignmentResult


class ProtocolIncompatible(RuntimeError):
    pass


class BudgetOnly:
    """No alignment identities: only graph-derived cardinality is accessible."""
    def __init__(self, count): self.count=count
    def __len__(self): return self.count
    def __getitem__(self, key): raise RuntimeError('GT access prohibited')
    def items(self): raise RuntimeError('GT access prohibited')
    def keys(self): raise RuntimeError('GT access prohibited')


@dataclass
class AuthorGradConfig:
    epochs: int=10
    k_hop: int=3
    hid_dim: int=100
    centrality: str='eigenvector'
    num_attr: int=15
    gradual_iterations: int=10
    device: str='auto'
    seed: int=2026
    show_progress: bool=False


class AuthorGradAlign:
    def __init__(self, config=None): self.config=config or AuthorGradConfig()
    def fit(self,g1,g2):
        from .vendor.gradalign_plus import models as author
        from .vendor.gradalign_plus.data import augment_attributes, aug_trimming
        cfg=self.config; start=time.perf_counter()
        random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
        if min(len(g1),len(g2))<2:
            raise ValueError('Author GIN BatchNorm requires at least two nodes per graph; no substitute model')
        base1=np.ones((len(g1),1),dtype=np.float32);base2=np.ones((len(g2),1),dtype=np.float32)
        aug1,aug2=augment_attributes(g1,g2,base1,base2,cfg.num_attr,version=cfg.centrality,khop=1,normalize=False)
        aug1,aug2=aug_trimming(aug1,aug2)
        trace=[]
        class ZeroGTAuthor(author.GradAlign):
            def Evaluation(self,s,t):
                trace.append(dict(round=len(trace)+1,matched_pairs=len(s),gt_evaluation=False))
            def FinalEvaluation(self,S,*args): return S,{'GT_used':False}
            def normdif_checker(self,*args): return None
        budget=BudgetOnly(min(len(g1),len(g2)))
        index1={u:u for u in g1};index2={u:u for u in g2}
        model=ZeroGTAuthor(g1.copy(),g2.copy(),base1,base2,aug1,aug2,cfg.k_hop,cfg.hid_dim,
                          budget,budget,0.,index1,index2,len(g2)/len(g1),1.)
        model.epochs=cfg.epochs;model.iter=cfg.gradual_iterations
        model.device=torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if cfg.device=='auto' else cfg.device)
        with contextlib.nullcontext() if cfg.show_progress else contextlib.redirect_stdout(io.StringIO()):
            score,_,sources,targets,_=model.run_algorithm()
        pairs=list(zip(map(int,sources),map(int,targets)))
        if not np.isfinite(score).all(): raise FloatingPointError('Non-finite author GradAlign score')
        return AlignmentResult('GradAlign+',pairs,time.perf_counter()-start,score_matrix=np.asarray(score),larger_is_better=True,
            metadata=dict(config=asdict(cfg),official_commit='95ef6366b18589fb88d01e95af3aa3d621ff71bf',
                          actual_backend='author_models_py_progressive_ACN',progressive_ACN_enabled=True,
                          attribute_mode='author_single_column_unattributed_branch',ground_truth_used_during_inference=False,
                          matching_budget_source='min_graph_order_only',matching_budget=len(budget),
                          bootstrap='all topology nodes when no initial seeds; original frontier thereafter',
                          rounds=trace,prediction_rule='author gradual ACN greedy pairs; no final Hungarian',
                          loss_mode='author_dense_layerwise_reconstruction_no_sampled_fallback'))


@dataclass
class SingleCENAConfig:
    q: float=.5
    layers: int=3
    alpha: float=5.
    embedding_dim: int=64
    embedding_epochs: int=5
    num_walks: int=20
    walk_length: int=80
    negative_samples: int=10
    seed: int=2026
    show_progress: bool=False


class SingleCENA:
    def __init__(self,config=None): self.config=config or SingleCENAConfig()
    def fit(self,g1,g2):
        from .vendor.cena.src.model import structing
        from .vendor.cena.src.CENALP import cal_degree_dict
        from .vendor.cena.src.walks import single_simulate_walks
        from .cena.core import _normalize_rows,_round_scores
        from gensim.models import Word2Vec
        from gensim.models.word2vec import LineSentence
        cfg=self.config;start=time.perf_counter();random.seed(cfg.seed);np.random.seed(cfg.seed)
        d1=cal_degree_dict(list(g1),g1,cfg.layers);d2=cal_degree_dict(list(g2),g2,cfg.layers)
        a,b,pa,pb=structing(cfg.layers,g1,g2,d1,d2,[],cfg.alpha,.5)
        if not all(np.isfinite(v).all() for values in (pa,pb) for v in values.values()):
            raise FloatingPointError('Author CENA structural transition probabilities non-finite')
        cwd=Path.cwd()
        with tempfile.TemporaryDirectory(prefix='single-cena-') as temp:
            try:
                os.chdir(temp)
                single_simulate_walks(g1,g2,cfg.q,a,b,pa,pb,[],[],num_walks=cfg.num_walks,walk_length=cfg.walk_length,workers=1)
                model=Word2Vec(LineSentence('random_walks.txt'),vector_size=cfg.embedding_dim,window=5,min_count=0,
                               hs=1,sg=1,negative=cfg.negative_samples,workers=1,epochs=cfg.embedding_epochs,seed=cfg.seed)
            finally: os.chdir(cwd)
        offset=max(len(g1),len(g2))
        x=_normalize_rows(np.asarray([model.wv[str(u)] for u in range(len(g1))]))
        y=_normalize_rows(np.asarray([model.wv[str(v+offset)] for v in range(len(g2))]))
        score=x@y.T
        return AlignmentResult('CENA-single-no-LP',_round_scores(score,'greedy'),time.perf_counter()-start,score_matrix=score,
            larger_is_better=True,source_embeddings=x,target_embeddings=y,metadata=dict(config=asdict(cfg),
            official_commit='2073ba8a9bca180913b41355fe71f60d714cd6a2',algorithm_core='author_components_CENA_one_embedding_no_LP',
            embedding_fit_count=1,link_prediction_calls=0,predicted_links_added=0,graph_modification=False,
            safety={'isolated_walk_policy':'stay_at_current_node'},prediction_rule='explicit greedy assignment of single-embedding cosine',
            ground_truth_used_during_inference=False))


@dataclass
class AuthorSGWLConfig:
    partition_levels: int=3
    cluster_count: int=2
    probability_epsilon: float=1e-12
    outer_iterations: int=2000
    inner_iterations: int=2
    beta: float=.025
    seed: int=2026
    show_progress: bool=False


class AuthorSGWL:
    def __init__(self,config=None): self.config=config or AuthorSGWLConfig()
    def fit(self,g1,g2):
        from .vendor.s_gwl.methods import GromovWassersteinGraphToolkit as toolkit
        from .vendor.s_gwl.methods import GromovWassersteinFramework as gw
        cfg=self.config;start=time.perf_counter();np.random.seed(cfg.seed)
        swapped=len(g1)<len(g2);a,b=(g2,g1) if swapped else (g1,g2)
        def packet(g):
            cost=csr_matrix(nx.to_scipy_sparse_array(g,nodelist=range(len(g)),dtype=np.float64))
            p=np.asarray(cost.sum(1)).reshape(-1,1)+cfg.probability_epsilon
            if not p.sum():p[:]=1.
            return cost,p/p.sum(),{u:u for u in range(len(g))}
        ca,pa,ia=packet(a);cb,pb,ib=packet(b)
        params=dict(loss_type='L2',ot_method='proximal',beta=cfg.beta,outer_iteration=cfg.outer_iterations,
                    iter_bound=1e-30,inner_iteration=cfg.inner_iterations,sk_bound=1e-30,node_prior=1e3,max_iter=4,
                    cost_bound=1e-26,update_p=False,lr=0,alpha=0)
        score=np.zeros((len(a),len(b)),dtype=np.float64);leaves=[];partitions=[]
        original_partition=toolkit.multi_graph_partition
        original_discrepancy=gw.gromov_wasserstein_discrepancy
        original_sinkhorn=gw.sinkhorn_knopp_iteration
        from .sgwl_numerics import log_sinkhorn_knopp
        numerics={'calls':0,'raw_kernel_underflow_entries':0,'inner_iterations':0}
        def stable_sinkhorn(*args,**kwargs):
            return log_sinkhorn_knopp(*args,**kwargs,diagnostics=numerics)
        def finite_discrepancy(*args,**kwargs):
            output=original_discrepancy(*args,**kwargs)
            if not np.isfinite(output[0]).all():
                raise FloatingPointError('Non-finite author S-GWL transport during recursive partition or matching')
            return output
        def observed_partition(*args,**kwargs):
            output=original_partition(*args,**kwargs)
            partitions.append({'input_orders':{str(k):len(v) for k,v in args[1].items()},
                               'output_clusters':[{str(k):len(v) for k,v in block.items()} for block in output[2]]})
            return output
        original_direct=toolkit.direct_graph_matching
        def observed(cs,ct,ps,pt,ids,idt,parameters):
            saved=gw.gromov_wasserstein_discrepancy;last={}
            def captured(*args,**kwargs):
                result=saved(*args,**kwargs);last['plan']=result[0];return result
            gw.gromov_wasserstein_discrepancy=captured
            try:result=original_direct(cs,ct,ps,pt,ids,idt,parameters)
            finally:gw.gromov_wasserstein_discrepancy=saved
            plan=last['plan']
            if not np.isfinite(plan).all():raise FloatingPointError('Non-finite author S-GWL leaf transport')
            rows=[ids[i] for i in range(len(ids))];cols=[idt[i] for i in range(len(idt))]
            score[np.ix_(rows,cols)]=plan
            leaves.append(dict(source_nodes=len(rows),target_nodes=len(cols),pairs=len(result[0]),outer_iterations=parameters['outer_iteration']))
            return result
        toolkit.direct_graph_matching=observed
        toolkit.multi_graph_partition=observed_partition
        gw.gromov_wasserstein_discrepancy=finite_discrepancy
        gw.sinkhorn_knopp_iteration=stable_sinkhorn
        try:
            _,pairs,_=toolkit.recursive_direct_graph_matching(ca,cb,pa,pb,ia,ib,params,weights=None,
                predefine_barycenter=False,cluster_num=cfg.cluster_count,partition_level=cfg.partition_levels,max_node_num=0)
        finally:
            toolkit.direct_graph_matching=original_direct
            toolkit.multi_graph_partition=original_partition
            gw.gromov_wasserstein_discrepancy=original_discrepancy
            gw.sinkhorn_knopp_iteration=original_sinkhorn
        if swapped: pairs=[(v,u) for u,v in pairs];score=score.T
        return AlignmentResult('S-GWL-author-recursive',[tuple(map(int,p)) for p in pairs],time.perf_counter()-start,
            score_matrix=score,larger_is_better=True,metadata=dict(config=asdict(cfg),official_commit='d682bcd1729e2c76cb65cb53e7d38ee6914e4263',
            actual_backend='author_recursive_direct_graph_matching',leaf_count=len(leaves),leaves=leaves,partition_calls=partitions,transposed_for_size_order=swapped,
            numerical_adapter={'version':'author_sinkhorn_equations_log_domain_r1','same_iteration_budgets':True,'kernel_clipping':False,**numerics},
            probability_adapter={'epsilon':cfg.probability_epsilon,'source_isolates':len(list(nx.isolates(g1))),'target_isolates':len(list(nx.isolates(g2)))},
            ranking_representation='same-fit author leaf transport lifted to global block matrix; excluded cross-leaf pairs have zero mass',
            prediction_rule='author recursive leaf node_pair_assignment',ground_truth_used_during_inference=False))


class AuthorOPGMIncompatible:
    def __init__(self): self.config={'official_commit':'39b5344c7a31c3a764f91be754932c7160a8bcfa','required_training':'supervised','allowed_training':'fully_unsupervised'}
    def fit(self,*args):
        raise ProtocolIncompatible('OPGMrs author neural model requires supervised cost/bias training; user explicitly requires fully unsupervised protocol. No xNetMF replacement, random weights, or zero score substituted.')
