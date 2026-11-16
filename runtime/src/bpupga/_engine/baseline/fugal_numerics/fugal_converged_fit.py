"""Original full FUGAL updates with exact class arithmetic and converged OT."""
from contextlib import contextmanager
from dataclasses import asdict
import importlib
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from common.result import AlignmentResult
from .fugal_converged_solver import VERSION,SolverPolicy,solve_transport,marginal_errors
from .fugal_exact_quotient import exact_classes,quotient_gradient,expand_values


def fit_full(model,graph1,graph2,*,policy=None,progress=None,artifact_directory=None):
    policy=policy or SolverPolicy()
    cfg=model.config
    n1,n2=len(graph1),len(graph2);n=max(n1,n2)
    if not n1 or not n2 or n*n>cfg.max_dense_entries:
        raise ValueError("Invalid full FUGAL input size; no truncation permitted")
    if cfg.dtype!="float64" or set(graph1)!=set(range(n1)) or set(graph2)!=set(range(n2)):
        raise ValueError("Float64 and consecutive input nodes required")
    start=time.perf_counter()
    source,target=exact_classes(graph1,n),exact_classes(graph2,n)
    device=torch.device(cfg.device)
    a,b=(torch.as_tensor(c.counts,device=device,dtype=torch.float64) for c in (source,target))
    A,B=(torch.as_tensor(c.adjacency,device=device,dtype=torch.float64) for c in (source,target))
    f1,f2=(torch.as_tensor(c.features,device=device,dtype=torch.float64) for c in (source,target))
    feature=cfg.mu*torch.cdist(f1,f2,p=2,compute_mode="donot_use_mm_for_euclid_dist")
    q=a[:,None]*b[None,:]/n
    trace=[];warm=None
    compression=dict(version="exact_identical_adjacency_and_feature_rows_r1",padded_nodes=n,
                     original_source_nodes=n1,original_target_nodes=n2,source_classes=len(a),target_classes=len(b),
                     source_largest_class=int(a.max()),target_largest_class=int(b.max()),
                     all_nodes_retained=True,all_candidates_retained=True,approximate_partitioning=False,
                     entropy_equivalence="class entropy minus fixed marginal-dependent constants",
                     residual="sum_class (column_mass_error^2 / class_size), equal to full-node squared residual")
    print("Exact full-node quotient",compression,flush=True)
    for outer in range(cfg.niter):
        for inner in range(1,cfg.inner_iterations+1):
            gradient=quotient_gradient(q,A,B,feature,a,b,outer)
            def observed(info):
                if progress:progress(dict(info,outer=outer+1,inner=inner,outer_budget=cfg.niter,inner_budget=cfg.inner_iterations))
            candidate,warm,info=solve_transport(gradient,a,b,cfg.sinkhorn_reg,policy=policy,warm=warm,progress=observed)
            info.update(outer=outer+1,inner=inner)
            trace.append(info)
            q=q+(2./(2+inner))*(candidate-q)
    final_errors=marginal_errors(q,a,b)
    if final_errors["column_squared_residual"]>policy.column_squared_tolerance+1e-10 or final_errors["column_max_absolute_residual"]>policy.maximum_marginal_tolerance+1e-10:
        raise FloatingPointError("Final full-node FW iterate violated the accepted marginal precision")
    values=(q/a[:,None]/b[None,:]).detach().cpu().numpy()
    score=expand_values(values,source,target,n1,n2)
    if not np.isfinite(score).all():raise FloatingPointError("Non-finite expanded FUGAL score")
    if artifact_directory is not None:
        artifact_directory=Path(artifact_directory)
        np.savez(artifact_directory/"exact_quotient_state.npz",block_mass=q.detach().cpu().numpy(),
                 source_membership=source.membership,target_membership=target.membership,
                 source_counts=source.counts,target_counts=target.counts)
    padded=expand_values(values,source,target,n,n)
    rows,columns=linear_sum_assignment(-padded)
    keep=(rows<n1)&(columns<n2)
    rows,columns=rows[keep],columns[keep]
    return AlignmentResult(method="FUGAL",predicted_pairs=[(int(u),int(v)) for u,v in zip(rows,columns)],
        runtime_sec=time.perf_counter()-start,cost_matrix=-score,larger_is_better=False,partial_decoder_data={"padded_transport":padded},
        metadata=dict(config=asdict(cfg),device=str(device),solver="full FUGAL, exact repeated-row arithmetic, residual-controlled stabilized OT",
                      feature_set="original degree, clustering, mean-neighbor degree, mean-neighbor clustering",
                      euclidean_distance="float64 direct Euclidean evaluation (no dot-product cancellation)",
                      prediction_rule="author full padded Hungarian then dummy filtering",
                      padded_size=n,ground_truth_used_during_inference=False,exact_compression=compression,
                      numerical_adaptation=dict(version=VERSION,calls=trace,solver_policy=asdict(policy),
                          final_plan_marginal_errors=final_errors,final_regularization=cfg.sinkhorn_reg,
                          all_subproblems_converged=all(t["stopping_test_satisfied"] for t in trace),
                          feature_ot_fallback=False,nan_replacement=False,gradient_clipping=False,
                          global_nonconvex_optimality_claimed=False)))


@contextmanager
def adapt_full_fugal(policy=None,progress=None,artifact_directory=None):
    classes={id(c):c for c in (importlib.import_module("baselines.fugal").FUGAL,importlib.import_module("baseline_261023new.fugal").FUGAL)}
    original={key:cls.fit for key,cls in classes.items()}
    def adapted(self,g1,g2):return fit_full(self,g1,g2,policy=policy,progress=progress,artifact_directory=artifact_directory)
    try:
        for cls in classes.values():cls.fit=adapted
        yield
    finally:
        for key,cls in classes.items():cls.fit=original[key]
