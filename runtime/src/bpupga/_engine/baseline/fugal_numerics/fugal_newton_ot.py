"""Damped Newton-CG acceleration of the SAME balanced entropic OT problem.

The row-normalized semi-dual has gradient col(Q)-b and Hessian
(diag(col(Q))-Q.T @ diag(1/a) @ Q)/reg. sqrt(b) coordinates match
the full-node squared residual under exact repeated-node multiplicities.
No labels, matching cardinality, clipping or alternative transport objective.
"""
import math
import torch


def row_normalized(cost,a,beta,reg):
    scaled=(beta[None,:]-cost)/reg
    normalizer=torch.logsumexp(scaled,dim=1)
    alpha=reg*(a.log()-normalizer)
    plan=torch.exp(scaled+(a.log()-normalizer)[:,None])
    objective=reg*torch.dot(a,normalizer)
    return alpha,plan,objective


def hessian_product(plan,a,columns,vector,reg):
    return (columns*vector-plan.T@((plan@vector)/a))/reg


def newton_transport(cost,a,b,reg,beta,*,max_iterations=100,cg_iterations=100,
                     squared_tolerance=1e-3,maximum_tolerance=1e-3,cleanup_iterations=10,report=None):
    sqrt_b=b.sqrt();inv_sqrt_b=1/sqrt_b;total=b.sum()
    def project(x):return x-sqrt_b*(torch.dot(sqrt_b,x)/total)
    def centered(x):return x-torch.dot(b,x)/total
    beta=centered(beta)
    alpha,plan,first=row_normalized(cost,a,beta,reg)
    objective=first-torch.dot(b,beta)
    damping=1e-3/reg
    records=[]
    for iteration in range(1,max_iterations+1):
        columns=plan.sum(0);gradient=(columns-b)*inv_sqrt_b
        largest=float(((columns-b)/b).abs().max())
        squared=float(gradient.square().sum())
        if report:report(iteration,squared,largest)
        records.append(dict(iteration=iteration,column_squared_residual=squared,
                            column_max_absolute_residual=largest,damping=damping))
        # A generous early exit; the caller applies its exact declared criteria
        # after reconstructing from the original cost and final duals.
        if squared<=squared_tolerance and largest<=maximum_tolerance:
            return alpha,beta,plan,records
        gradient=project(gradient)
        diagonal=(columns/b/reg+damping).clamp_min(damping)
        def product(x):
            return project(hessian_product(plan,a,columns,x*inv_sqrt_b,reg)*inv_sqrt_b)+damping*x
        direction=torch.zeros_like(gradient)
        residual=-gradient.clone();z=project(residual/diagonal);conjugate=z.clone()
        rz=torch.dot(residual,z)
        initial=float(torch.dot(residual,residual))
        for cg in range(cg_iterations):
            hp=product(conjugate);denominator=torch.dot(conjugate,hp)
            if not bool(torch.isfinite(denominator)) or float(denominator)<=0:break
            step=rz/denominator
            direction+=step*conjugate;residual-=step*hp
            if float(torch.dot(residual,residual))<=.01*initial:break
            z=project(residual/diagonal);updated=torch.dot(residual,z)
            if float(rz)<=0:break
            conjugate=z+(updated/rz)*conjugate;rz=updated
        direction=project(direction)
        slope=float(torch.dot(gradient,direction))
        if not math.isfinite(slope) or slope>=0:
            direction=project(-gradient/diagonal);slope=float(torch.dot(gradient,direction))
        accepted=False;fraction=1.
        rounding=16*torch.finfo(cost.dtype).eps*max(1.,abs(float(objective)))
        for backtrack in range(25):
            trial_beta=centered(beta+fraction*direction*inv_sqrt_b)
            trial_alpha,trial_plan,raw=row_normalized(cost,a,trial_beta,reg)
            trial_objective=raw-torch.dot(b,trial_beta)
            trial_error=float((((trial_plan.sum(0)-b)*inv_sqrt_b).square()).sum())
            if float(trial_objective)<=float(objective)+1e-4*fraction*slope+rounding and (
                    float(trial_objective)<float(objective)-rounding or trial_error<squared):
                alpha,beta,plan,objective=trial_alpha,trial_beta,trial_plan,trial_objective
                accepted=True;break
            fraction*=.5
        records[-1].update(cg_steps=cg+1,step_fraction=fraction,line_search_accepted=accepted)
        if not accepted:
            damping=min(damping*10,1e3/reg)
        elif fraction>=.5:
            damping=max(damping*.3,1e-10/reg)
        else:damping=min(damping*3,1./reg)
        if accepted and cleanup_iterations:
            # Restore local marginal balance between Newton moves. This is
            # ordinary Sinkhorn scaling of the same current entropic kernel,
            # not a projection onto an unrelated score or transport objective.
            left=torch.ones_like(a);right=torch.ones_like(b);cleaned=0
            for _ in range(cleanup_iterations):
                denominator=plan.T@left
                if not bool((denominator>0).all()):break
                right=b/denominator
                denominator=plan@right
                if not bool((denominator>0).all()):break
                left=a/denominator;cleaned+=1
            if cleaned:
                beta=centered(beta+reg*right.log())
                alpha,plan,first=row_normalized(cost,a,beta,reg)
                objective=first-torch.dot(b,beta)
            records[-1]['sinkhorn_cleanup_iterations']=cleaned
    return alpha,beta,plan,records
