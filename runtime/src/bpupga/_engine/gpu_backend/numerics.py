"""Device transport arithmetic; host control only launches GPU-decided work."""
import torch
from numerical_backend.numerics import _row_lse,_column_lse


def continue_device(flag):
    """Dynamic-shape launch gate; the stopping predicate is computed on GPU."""
    return flag.reshape(-1).nonzero().shape[0] != 0


def sinkhorn(cost,rows,columns,regularization,iterations,tolerance,*,both=False,diagnostics=None):
    log_a=torch.where(rows>0,rows.log(),-torch.inf)
    log_b=torch.where(columns>0,columns.log(),-torch.inf)
    left=torch.where(rows>0,-torch.ones_like(rows)*torch.log(rows.new_tensor(rows.numel())),-torch.inf)
    right=torch.where(columns>0,-torch.ones_like(columns)*torch.log(columns.new_tensor(columns.numel())),-torch.inf)
    kernel=-cost/regularization
    for step in range(iterations):
        right=log_b-_column_lse(kernel,left)
        left=log_a-_row_lse(kernel,right)
        finite=torch.isfinite(left)
        gauge=torch.where(finite,left,0.).sum()/finite.sum()
        left=left-gauge;right=right+gauge
        if tolerance>0 and (step+1)%10==0:
            residual=(torch.exp(right+_column_lse(kernel,left))-columns).square().sum()
            if both:
                residual=(residual+(torch.exp(left+_row_lse(kernel,right))-rows).square().sum())*.5
            if not continue_device(residual>tolerance):break
    width=max(1,2_000_000//kernel.shape[1])
    for start in range(0,kernel.shape[0],width):
        kernel[start:start+width].add_(left[start:start+width,None]).add_(right[None,:]).exp_()
    if diagnostics is not None:
        diagnostics.append(dict(iterations=step+1,iteration_budget=iterations,
            row_squared_residual=(kernel.sum(1)-rows).square().sum(),
            column_squared_residual=(kernel.sum(0)-columns).square().sum(),
            zero_marginal_transport='exactly_zero'))
    return kernel


def export(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def diagnostics_to_host(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:diagnostics_to_host(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [diagnostics_to_host(v) for v in value]
    return value


def device_finite(*values):
    checked=[torch.isfinite(v).all() for v in values if v is not None]
    if checked and continue_device(~torch.stack(checked).all()):
        raise FloatingPointError('Non-finite GPU result')
