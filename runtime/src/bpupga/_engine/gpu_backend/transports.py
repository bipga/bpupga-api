"""ALPINE/FUGAL with graph statistics, rounding and diagnostics on CUDA."""
from dataclasses import asdict
import time
import torch
import baseline
from common.result import AlignmentResult
from baseline.alpine import AlpineConfig
from baseline.fugal import FUGALConfig
from .assignment import linear_sum_assignment
from .graph import DeviceGraph
from .numerics import sinkhorn,export,device_finite,diagnostics_to_host


class DeviceTransport:
    def __init__(self,method,config,device):
        self.method,self.config,self.device=method,config,torch.device(device)

    def fit(self,graph1,graph2):
        started=time.perf_counter();cfg=self.config
        first,second=DeviceGraph(graph1,self.device),DeviceGraph(graph2,self.device)
        dtype=torch.float32 if cfg.dtype=='float32' else torch.float64
        diagnostics=[];stages={};stage_metadata={}
        if self.method=='fugal':
            n1,n2=first.n,second.n;size=max(n1,n2)
            a,b=first.dense(dtype,size),second.dense(dtype,size)
            x,y=first.local_features(dtype=dtype),second.local_features(dtype=dtype)
            x=torch.nn.functional.pad(x,(0,0,0,size-n1))
            y=torch.nn.functional.pad(y,(0,0,0,size-n2))
            feature_cost=cfg.mu*torch.cdist(x,y,p=2)
            plan=torch.ones((size,size),device=self.device,dtype=dtype)/size
            mass=torch.ones(size,device=self.device,dtype=dtype)
            for outer in range(cfg.niter):
                for inner in range(1,cfg.inner_iterations+1):
                    gradient=-a@plan@b.T-a.T@plan@b+feature_cost+outer*(1-2*plan)
                    candidate=sinkhorn(gradient,mass,mass,cfg.sinkhorn_reg,
                        cfg.sinkhorn_iters,cfg.sinkhorn_stop_thr,diagnostics=diagnostics)
                    plan=plan+(2/(2+inner))*(candidate-plan)
            rows,columns=linear_sum_assignment(-plan)
            valid=(rows<n1)&(columns<n2)
            pairs=torch.stack([rows[valid],columns[valid]],1)
            score=plan[:n1,:n2]
            device_finite(score,plan)
            state={'padded_transport':export(plan)}
            metadata=dict(prediction_rule='full padded exact CUDA LAP then dummy filtering',padded_size=size)
        else:
            swapped=first.n>second.n
            source,target=(second,first) if swapped else (first,second)
            source,target=source.prepared(),target.prepared()
            source_dummy=source.prepared(dummy=True)
            n,m=source.n,target.n
            a,b=source.dense(dtype),target.dense(dtype)
            x=source_dummy.local_features(exclude_self=True,clustering=cfg.weight==1,dtype=dtype)
            y=target.local_features(exclude_self=True,clustering=cfg.weight==1,dtype=dtype)
            if cfg.weight==2:
                x=x[:,[0,2]];y=y[:,[0,2]]
            feature_cost=cfg.mu*torch.cdist(x,y,p=2)
            plan=torch.ones((n+1,m),device=self.device,dtype=dtype)/m
            plan[-1]*=m-n
            rows_mass=torch.ones(n+1,device=self.device,dtype=dtype);rows_mass[-1]=m-n
            columns_mass=torch.ones(m,device=self.device,dtype=dtype)
            def snapshot(name,value):
                device_finite(value)
                stages[name]=export(value.T if swapped else value).copy()
            if cfg.collect_stage_trace:snapshot('feature_quality',-feature_cost[:n])
            for outer in range(cfg.niter):
                for inner in range(1,cfg.inner_iterations+1):
                    real=plan[:n]
                    residual=a-real@b@real.T
                    structural=torch.zeros_like(plan)
                    structural[:n]=-4*residual@real@b
                    gradient=structural+outer*(1-2*plan)+feature_cost
                    candidate=sinkhorn(gradient,rows_mass,columns_mass,cfg.sinkhorn_reg,
                        cfg.sinkhorn_iterations,cfg.sinkhorn_tolerance,both=n+1!=m,diagnostics=diagnostics)
                    # Author semantics: the dummy row is never updated.
                    plan[:n]+=2/(2+inner)*(candidate[:n]-plan[:n])
                if cfg.collect_stage_trace:snapshot(f'outer_{outer+1:02d}',plan[:n])
            small_score=plan[:n]
            r,c=linear_sum_assignment(-small_score)
            permutation=torch.zeros((n,m),device=self.device,dtype=torch.float64)
            permutation[r,c]=1
            frobenius=(a.double()-permutation@b.double()@permutation.T).square().sum()
            score=small_score.T if swapped else small_score
            r,c=linear_sum_assignment(-score);pairs=torch.stack([r,c],1)
            if cfg.collect_stage_trace:snapshot('final_score',small_score)
            device_finite(score)
            state={}
            metadata=dict(prediction_rule='exact rectangular CUDA LAP',swapped=swapped,
                frobenius=diagnostics_to_host(frobenius),dummy_row_update='author_unchanged')
        result=AlignmentResult(method='Alpine' if self.method=='alpine' else 'FUGAL',
            predicted_pairs=[tuple(pair) for pair in pairs.detach().cpu().tolist()],
            runtime_sec=time.perf_counter()-started,cost_matrix=export(-score),larger_is_better=False,
            partial_decoder_data=state,stage_score_matrices=stages,
            stage_diagnostics_metadata=stage_metadata,
            metadata=dict(config=asdict(cfg),ground_truth_used_during_inference=False,
                numerical_projection='same_objective_log_sinkhorn',
                numerical_revision_diagnostics=diagnostics_to_host(diagnostics),
                native_decoder_device=str(self.device),**metadata))
        return result


def construct(method,parameters,seed=2026,device='cuda:0',dataset_family=''):
    params=dict(parameters);params.pop('device',None)
    params.update(device=device,show_progress=False,max_dense_entries=10**18)
    config=(AlpineConfig if method=='alpine' else FUGALConfig)(**params)
    config.validate()
    return DeviceTransport(method,config,device),asdict(config)
