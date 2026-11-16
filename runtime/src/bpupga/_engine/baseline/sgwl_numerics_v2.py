"""Gauge-stable log Sinkhorn with unrounded proximal log-transport state.

The GW objective, marginals, beta, iteration budgets and stopping rule are
unchanged. Float64 underflow in the exported plan must not become a structural
zero in the next proximal iteration. No positive floor or clipping is used.
"""
from contextlib import contextmanager
import weakref

import numpy as np
from scipy.special import logsumexp
from .sgwl_numerics import LogScaling


class LogSinkhorn:
    def __init__(self):
        self.states = {}
        self.statistics = dict(proximal_log_state_reuses=0, exported_underflow_entries=0,
                               scaling_gauge_normalizations=0)

    def __call__(self, cost, p_s=None, p_t=None, a=None, trans0=None, beta=.1,
                 error_bound=.001, max_iter=50, diagnostics=None):
        ns, nt = cost.shape
        p_s = np.full((ns, 1), 1 / ns) if p_s is None else p_s
        p_t = np.full((nt, 1), 1 / nt) if p_t is None else p_t
        if (not np.isfinite(cost).all() or not np.isfinite(p_s).all()
                or not np.isfinite(p_t).all() or np.any(p_s <= 0)
                or np.any(p_t <= 0) or beta <= 0 or max_iter < 1):
            raise FloatingPointError('Invalid log-domain S-GWL inputs')
        log_a = (a.values.copy() if isinstance(a, LogScaling) else
                 np.log(np.full((ns, 1), 1 / ns) if a is None else a))
        log_a -= np.max(log_a)
        kernel = -cost / beta
        if trans0 is not None:
            previous = self.states.get(id(trans0))
            if previous is not None and previous[0]() is trans0:
                kernel = kernel + previous[1]
                self.statistics['proximal_log_state_reuses'] += 1
            else:
                with np.errstate(divide='ignore'):
                    kernel = kernel + np.log(trans0)
        if diagnostics is not None:
            diagnostics['calls'] = diagnostics.get('calls', 0) + 1
            diagnostics['raw_kernel_underflow_entries'] = diagnostics.get('raw_kernel_underflow_entries', 0) + int(np.count_nonzero(kernel < np.log(np.nextafter(0., 1.))))
        for _ in range(max_iter):
            log_b = np.log(p_t) - logsumexp(kernel + log_a, axis=0).reshape(-1, 1)
            updated = np.log(p_s) - logsumexp(kernel + log_b.T, axis=1).reshape(-1, 1)
            shift = max(float(np.max(updated)), float(np.max(log_a)))
            denominator = np.exp(log_a - shift).sum()
            relative = (np.abs(np.exp(updated - shift) - np.exp(log_a - shift)).sum()
                        / denominator if denominator else np.inf)
            # a*c and b/c represent the identical transport. Normalize both
            # factors together, after evaluating the original stopping rule.
            center = float(np.max(updated))
            log_a = updated - center
            log_b = log_b + center
            self.statistics['scaling_gauge_normalizations'] += 1
            if diagnostics is not None:
                diagnostics['inner_iterations'] = diagnostics.get('inner_iterations', 0) + 1
            if relative < error_bound:
                break
        log_plan = kernel + log_a + log_b.T
        if not np.isfinite(log_plan).all():
            raise FloatingPointError('Non-finite S-GWL log transport, revision v2')
        plan = np.exp(log_plan)
        if not np.isfinite(plan).all():
            raise FloatingPointError('Non-finite S-GWL transport, revision v2')
        self.statistics['exported_underflow_entries'] += int(np.count_nonzero(plan == 0))
        key = id(plan)
        def release(reference):
            if self.states.get(key, (None,))[0] is reference:
                self.states.pop(key, None)
        self.states[key] = (weakref.ref(plan, release), log_plan)
        return plan, LogScaling(log_a)


@contextmanager
def stable_proximal_state():
    """Replace only the numerical hook inside an isolated fit process."""
    from . import sgwl_numerics
    previous = sgwl_numerics.log_sinkhorn_knopp
    solver = LogSinkhorn()
    sgwl_numerics.log_sinkhorn_knopp = solver
    try:
        yield solver.statistics
    finally:
        sgwl_numerics.log_sinkhorn_knopp = previous
