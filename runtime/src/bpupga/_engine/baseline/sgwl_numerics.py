"""Log-domain evaluation of author Sinkhorn equations, with identical budgets.

Only used with author update_p=False. No clipping, added OT iterations,
GT, objective change, or replacement of recursive partitioning.
"""
import numpy as np
from scipy.special import logsumexp

class LogScaling:
    def __init__(self,values):self.values=values

def log_sinkhorn_knopp(cost,p_s=None,p_t=None,a=None,trans0=None,beta=.1,error_bound=.001,max_iter=50,diagnostics=None):
    ns,nt=cost.shape
    if p_s is None:p_s=np.ones((ns,1))/ns
    if p_t is None:p_t=np.ones((nt,1))/nt
    if not np.isfinite(cost).all() or np.any(p_s<=0) or np.any(p_t<=0):raise FloatingPointError('S-GWL log Sinkhorn requires finite cost and positive marginals')
    log_a=a.values.copy() if isinstance(a,LogScaling) else np.log(np.ones((ns,1))/ns if a is None else a)
    kernel=-cost/beta
    if trans0 is not None:
        with np.errstate(divide='ignore'):kernel=kernel+np.log(trans0)
    if diagnostics is not None:
        diagnostics['calls']+=1
        diagnostics['raw_kernel_underflow_entries']+=int((kernel<np.log(np.finfo(float).tiny)).sum())
    for _ in range(max_iter):
        log_b=np.log(p_t)-logsumexp(kernel+log_a,axis=0).reshape(-1,1)
        updated=np.log(p_s)-logsumexp(kernel+log_b.T,axis=1).reshape(-1,1)
        # Relative L1 scaling change without exponentiating huge dual values.
        shift=max(float(np.max(updated)),float(np.max(log_a)))
        denominator=np.exp(log_a-shift).sum()
        relative=np.abs(np.exp(updated-shift)-np.exp(log_a-shift)).sum()/denominator if denominator else np.inf
        log_a=updated
        if diagnostics is not None:diagnostics['inner_iterations']+=1
        if relative<error_bound:break
    plan=np.exp(kernel+log_a+log_b.T)
    if not np.isfinite(plan).all():raise FloatingPointError('Non-finite log-domain author S-GWL transport')
    return plan,LogScaling(log_a)
