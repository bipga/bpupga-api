"""Full-spectrum GRAMPA and xNetMF REGAL without CPU graph arithmetic."""
from dataclasses import asdict
import math
import time
import torch
import baseline
from common.result import AlignmentResult
from baseline.grampa import GRAMPAConfig
from baseline.regal import REGALConfig
from gpu_backend.assignment import linear_sum_assignment
from gpu_backend.graph import DeviceGraph
from gpu_backend.numerics import export,device_finite


def graph_eigh(graph,dtype,center):
    matrix=graph.dense(dtype)
    if center and matrix.shape[0]>1:
        mask=~torch.eye(matrix.shape[0],device=matrix.device,dtype=torch.bool)
        matrix[mask]-=matrix[mask].mean()
        del mask
    if matrix.is_cuda:torch.cuda.empty_cache()
    eigen,vectors=torch.linalg.eigh(matrix)
    del matrix
    if vectors.is_cuda:torch.cuda.empty_cache()
    return eigen,vectors


class DeviceSpectral:
    def __init__(self,method,config,seed,device):
        self.method,self.config,self.seed,self.device=method,config,seed,torch.device(device)

    def fit(self,g1,g2):
        started=time.perf_counter();cfg=self.config
        first,second=DeviceGraph(g1,self.device),DeviceGraph(g2,self.device)
        if self.method=='grampa':
            dtype=torch.float32 if cfg.dtype=='float32' else torch.float64
            # Eigensolver workspace is much larger than the adjacency.
            # Build and release one adjacency at a time, retaining its full spectrum.
            eigen1,u=graph_eigh(first,dtype,cfg.center_adjacency)
            eigen2,v=graph_eigh(second,dtype,cfg.center_adjacency)
            weights=1/((eigen1[:,None]-eigen2[None,:]).square()+cfg.eta**2)
            score=u@(weights*u.sum(0)[:,None]*v.sum(0)[None,:])@v.T
            del eigen1,eigen2,u,v,weights
            rows,columns=linear_sum_assignment(-score)
            pairs=torch.stack([rows,columns],1)
            device_finite(score)
            return AlignmentResult('GRAMPA',[tuple(p) for p in pairs.cpu().tolist()],
                time.perf_counter()-started,cost_matrix=export(-score),larger_is_better=False,
                metadata=dict(config=asdict(cfg),ground_truth_used_during_inference=False,
                    spectrum='complete symmetric CUDA eigendecomposition',memory_layout_revision='memory_efficient',
                    prediction_rule='exact CUDA maximum-weight assignment'))
        combined=DeviceGraph.union(first,second)
        features=combined.hop_histograms(cfg.max_layer,cfg.alpha,cfg.buckets)
        n=combined.n
        count=min(n,int(cfg.landmark_multiplier*math.log(n,2)) if n>1 else 1)
        generator=torch.Generator(device=self.device);generator.manual_seed(cfg.landmark_seed)
        landmarks=torch.randperm(n,device=self.device,generator=generator)[:count]
        similarities=torch.exp(-cfg.gamma_structure*torch.cdist(features,features[landmarks],p=2,
            compute_mode='donot_use_mm_for_euclid_dist'))
        inverse=torch.linalg.pinv(similarities[landmarks],rtol=1e-15)
        left,singular,_=torch.linalg.svd(inverse,full_matrices=False)
        embedding=similarities@(left*singular.clamp_min(0).sqrt()[None,:])
        norm=embedding.norm(dim=1,keepdim=True)
        embedding=(embedding/torch.where(norm>0,norm,torch.ones_like(norm))).float()
        x,y=embedding[:first.n],embedding[first.n:]
        score=torch.empty((first.n,second.n),device=self.device,dtype=torch.float32)
        for start in range(0,first.n,256):
            score[start:start+256]=torch.exp(-torch.cdist(x[start:start+256],y,p=2,
                compute_mode='donot_use_mm_for_euclid_dist'))
        pairs=torch.stack([torch.arange(first.n,device=self.device),score.argmax(1)],1)
        device_finite(score,x,y)
        return AlignmentResult('REGAL',[tuple(p) for p in pairs.cpu().tolist()],
            time.perf_counter()-started,score_matrix=export(score),larger_is_better=True,
            source_embeddings=export(x),target_embeddings=export(y),
            metadata=dict(config=asdict(cfg),ground_truth_used_during_inference=False,
                landmark_count=count,landmark_sampling='seeded CUDA randperm',
                author_numpy_landmark_trajectory_identical=False,
                prediction_rule='GPU maximum similarity, low-ID ties'))


def construct(method,parameters,seed=2026,device='cuda:0',dataset_family=''):
    params=dict(parameters);params.pop('device',None)
    params.update(show_progress=False,max_dense_entries=10**18)
    if method=='grampa':config=GRAMPAConfig(**params)
    else:config=REGALConfig(**params)
    config.validate()
    return DeviceSpectral(method,config,seed,device),asdict(config)
