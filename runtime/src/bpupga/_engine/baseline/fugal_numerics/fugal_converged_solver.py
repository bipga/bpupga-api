"""Residual-controlled, stabilized balanced OT for the original FUGAL objective.

Epsilon continuation is only a numerical warm start. Every accepted answer is
polished against the requested final regularization and both original-node
marginals. Exact repeated-node classes carry multiplicities, not sampled mass.
"""
from dataclasses import asdict, dataclass
import math
import torch
from .fugal_newton_ot import newton_transport

VERSION = "fugal_full_converged_exact_author_rounding_r3"


@dataclass(frozen=True)
class SolverPolicy:
    max_iterations: int = 30000
    check_every: int = 10
    absorb_every: int = 100
    stage_iterations: int = 200
    column_squared_tolerance: float = 1e-3
    maximum_marginal_tolerance: float = 1e-3
    row_tolerance: float = 1e-10
    continuation: bool = True
    newton_after: int = 200
    newton_iterations: int = 300
    newton_cg_iterations: int = 100
    newton_cleanup_iterations: int = 10


class TransportNotConverged(RuntimeError):
    pass


def marginal_errors(plan, row_mass, column_mass):
    dr, dc = plan.sum(1)-row_mass, plan.sum(0)-column_mass
    return dict(row_max_absolute_residual=float((dr/row_mass).abs().max()),
                column_max_absolute_residual=float((dc/column_mass).abs().max()),
                column_squared_residual=float((dc.square()/column_mass).sum()))


def solve_transport(cost, row_mass, column_mass, regularization=1., *, policy=None, warm=None, progress=None):
    policy = policy or SolverPolicy()
    if cost.dtype != torch.float64:
        raise ValueError("Convergence-controlled FUGAL requires float64")
    if cost.shape != (row_mass.numel(), column_mass.numel()):
        raise ValueError("Invalid transport shape")
    if not all(bool(torch.isfinite(x).all()) for x in (cost,row_mass,column_mass)):
        raise FloatingPointError("Non-finite transport input; no silent repair")
    if not bool((row_mass>0).all()) or not bool((column_mass>0).all()):
        raise ValueError("Strictly positive class multiplicities required")
    if abs(float(row_mass.sum()-column_mass.sum())) > 1e-9 or regularization <= 0:
        raise ValueError("Invalid balanced transport masses or regularization")
    if min(policy.max_iterations,policy.stage_iterations,policy.check_every,policy.absorb_every) < 1:
        raise ValueError("Invalid solver budget")
    logs_a, logs_b = row_mass.log(), column_mass.log()
    if warm is None:
        alpha, beta = torch.zeros_like(row_mass), torch.zeros_like(column_mass)
    else:
        alpha,beta = warm[0].clone(),warm[1].clone()
        if alpha.shape!=row_mass.shape or beta.shape!=column_mass.shape:
            raise ValueError("Incompatible dual warm start")
    minimum,maximum = float(cost.min()),float(cost.max())
    schedule = [regularization]
    if warm is None and policy.continuation:
        factor = max(0,math.ceil(math.log2(max(1.,(maximum-minimum)/(32*regularization)))))
        schedule = [regularization*2**power for power in range(factor,-1,-1)]
    stages=[]
    total_steps=0

    def report(phase,**values):
        if progress:
            observed=values.pop("total_iterations_observed",total_steps)
            progress(dict(phase=phase,version=VERSION,cost_min=minimum,cost_max=maximum,
                          regularization=regularization,budget=policy.max_iterations,
                          iterations=observed,**values))

    report("sinkhorn_start",running_regularization=schedule[0])
    for running_reg in schedule:
        final = running_reg == regularization
        budget = policy.max_iterations if final else policy.stage_iterations
        alpha,beta,plan,info,steps = _stage(cost,row_mass,column_mass,logs_a,logs_b,
            alpha,beta,running_reg,budget,policy,final,report,total_steps)
        total_steps += steps
        stages.append(dict(regularization=running_reg,iterations=steps,**info))
    info.update(version=VERSION,regularization=regularization,iterations=total_steps,
                final_stage_iterations=stages[-1]["iterations"],budget=policy.max_iterations,
                continuation_stages=stages,policy=asdict(policy),cost_min=minimum,cost_max=maximum,
                original_exp_overflow_possible=(-minimum/regularization)>math.log(torch.finfo(cost.dtype).max),
                finite=bool(torch.isfinite(plan).all()),stopping_test_satisfied=True,
                exact_multiplicity_weighted_residual=True,final_log_polish=True,warm_start_used=warm is not None)
    if not info["finite"]:
        raise FloatingPointError("Non-finite converged plan")
    report("sinkhorn_done",**{k:v for k,v in info.items() if k not in {"version","regularization","iterations","budget","cost_min","cost_max"}})
    return plan,(alpha,beta),info


def _stage(cost,a,b,loga,logb,alpha,beta,reg,budget,policy,final,report,previous_steps):
    def center(alpha,beta):
        shift=(alpha*a).sum()/a.sum()
        return alpha-shift,beta+shift

    def log_sweep(alpha,beta):
        beta=reg*(logb-torch.logsumexp((alpha[:,None]-cost)/reg,dim=0))
        alpha=reg*(loga-torch.logsumexp((beta[None,:]-cost)/reg,dim=1))
        return center(alpha,beta)

    def kernel(alpha,beta):
        matrix=torch.exp((alpha[:,None]+beta[None,:]-cost)/reg)
        if not bool(torch.isfinite(matrix).all()):
            raise FloatingPointError("Stabilized kernel overflow")
        return matrix

    alpha,beta=log_sweep(alpha,beta)
    matrix=kernel(alpha,beta)
    left,right=torch.ones_like(a),torch.ones_like(b)
    steps=1
    info={}
    newton_records=[]
    newton_tried=False
    while True:
        if final and not newton_tried and steps>=policy.newton_after and steps<budget and policy.newton_iterations>0:
            alpha=alpha+reg*left.log();beta=beta+reg*right.log()
            def observe_newton(iteration,squared,largest):
                report("newton",running_regularization=reg,newton_iteration=iteration,
                       total_iterations_observed=previous_steps+steps+iteration,
                       column_squared_residual=squared,column_max_absolute_residual=largest)
            alpha,beta,matrix,newton_records=newton_transport(cost,a,b,reg,beta,
                max_iterations=min(policy.newton_iterations,max(1,(budget-steps)//(1+policy.newton_cleanup_iterations))),cg_iterations=policy.newton_cg_iterations,
                squared_tolerance=policy.column_squared_tolerance*.25,maximum_tolerance=policy.maximum_marginal_tolerance*.5,
                cleanup_iterations=policy.newton_cleanup_iterations,
                report=observe_newton)
            steps+=len(newton_records)+sum(t.get('sinkhorn_cleanup_iterations',0) for t in newton_records);newton_tried=True
            report("newton_completed",running_regularization=reg,newton_records=newton_records,
                   total_iterations_observed=previous_steps+steps)
            left.fill_(1.);right.fill_(1.)
        if steps%policy.check_every==0 or steps>=budget:
            columns=right*(matrix.T@left)
            squared=float(((columns-b).square()/b).sum())
            largest=float(((columns-b)/b).abs().max())
            if final:
                passed=squared<=policy.column_squared_tolerance and largest<=policy.maximum_marginal_tolerance
            else:
                passed=squared<=max(.1,policy.column_squared_tolerance) and largest<=max(.01,policy.maximum_marginal_tolerance)
            report("sinkhorn",running_regularization=reg,stage_iterations=steps,
                   column_squared_residual=squared,column_max_absolute_residual=largest,
                   total_iterations_observed=previous_steps+steps)
            if passed or steps>=budget:
                # Rebuild from the ORIGINAL cost, not the underflowed cached
                # kernel. This also verifies the final answer after absorption.
                alpha=alpha+reg*left.log();beta=beta+reg*right.log()
                if steps<budget:
                    alpha,beta=log_sweep(alpha,beta)
                    steps+=1
                else:
                    alpha,beta=center(alpha,beta)
                matrix=kernel(alpha,beta);left.fill_(1.);right.fill_(1.)
                info=marginal_errors(matrix,a,b)
                accepted=(info["column_squared_residual"]<=policy.column_squared_tolerance and
                          info["column_max_absolute_residual"]<=policy.maximum_marginal_tolerance and
                          info["row_max_absolute_residual"]<=policy.row_tolerance)
                if (final and accepted) or (not final and (passed or steps>=budget)):
                    return alpha,beta,matrix,dict(info,converged_at_final_tolerance=accepted,
                                                  newton_iterations=len(newton_records),newton_records=newton_records),steps
                if steps>=budget:
                    raise TransportNotConverged(f"Final-reg={reg} OT did not converge in {steps} steps: {info}")
        denominator=matrix.T@left
        if not bool((denominator>0).all()) or not bool(torch.isfinite(denominator).all()):
            alpha=alpha+reg*left.log();beta=beta+reg*right.log()
            alpha,beta=log_sweep(alpha,beta)
            matrix=kernel(alpha,beta);left.fill_(1.);right.fill_(1.);steps+=1
            continue
        right=b/denominator
        denominator=matrix@right
        if not bool((denominator>0).all()) or not bool(torch.isfinite(denominator).all()):
            # Absorb the valid left scaling, but do not log an infinite right.
            alpha=alpha+reg*left.log()
            alpha,beta=log_sweep(alpha,beta)
            matrix=kernel(alpha,beta);left.fill_(1.);right.fill_(1.);steps+=1
            continue
        left=a/denominator
        steps+=1
        if steps%policy.absorb_every==0 or max(float(left.abs().max()),float(right.abs().max()))>math.exp(30):
            alpha=alpha+reg*left.log();beta=beta+reg*right.log()
            alpha,beta=center(alpha,beta)
            matrix=kernel(alpha,beta);left.fill_(1.);right.fill_(1.)
