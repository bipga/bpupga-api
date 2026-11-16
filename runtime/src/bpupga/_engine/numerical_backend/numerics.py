"""Isolated finite-arithmetic ALPINE/FUGAL adaptation for revision 261112.

Frozen baseline and author source files remain untouched.  A private module is
loaded for each model and only its entropic projection is replaced.  The graph
features, objectives, Frank--Wolfe iterations, dummy handling and Hungarian
rounding therefore use the same implementation as the archived baseline.
"""
from __future__ import annotations

from dataclasses import asdict, fields
import importlib.util
import math
from pathlib import Path
import sys
import uuid

import torch

VERSION = "same_objective_log_sinkhorn_gpu_261112_r1"
# Bounds temporary broadcast matrices, not the problem/candidate population.
REDUCTION_ENTRIES = 2_000_000


def _row_lse(kernel, right):
    step = max(1, REDUCTION_ENTRIES // kernel.shape[1])
    return torch.cat([
        torch.logsumexp(kernel[start:start + step] + right[None, :], dim=1)
        for start in range(0, kernel.shape[0], step)
    ])


def _column_lse(kernel, left):
    step = max(1, REDUCTION_ENTRIES // kernel.shape[0])
    return torch.cat([
        torch.logsumexp(kernel[:, start:start + step] + left[:, None], dim=0)
        for start in range(0, kernel.shape[1], step)
    ])


def stable_sinkhorn(cost, row_mass, column_mass, regularization, iterations,
                    tolerance, *, residual="columns", diagnostics=None):
    """Evaluate the unchanged entropic projection using logarithmic scaling.

    No cost clipping, NaN replacement, extra iterations or feature-only fallback
    is used.  Zero dummy marginals have exactly zero transport.  All operations
    execute on the cost tensor's device, including bounded-memory reductions.
    """
    if cost.ndim != 2 or cost.shape != (row_mass.numel(), column_mass.numel()):
        raise ValueError("Sinkhorn cost/marginal shape mismatch")
    if not (math.isfinite(regularization) and regularization > 0 and iterations > 0):
        raise ValueError("Invalid Sinkhorn regularization/iteration budget")
    if not bool(torch.isfinite(cost).all()):
        raise FloatingPointError("Non-finite ALPINE/FUGAL gradient before projection")
    if not bool(torch.isfinite(row_mass).all() and torch.isfinite(column_mass).all()):
        raise FloatingPointError("Non-finite transport marginal")
    if bool((row_mass < 0).any() or (column_mass < 0).any()):
        raise ValueError("Negative transport marginal")
    total = float(row_mass.sum())
    if total <= 0 or not math.isclose(total, float(column_mass.sum()), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("Sinkhorn requires equal positive total marginal mass")
    log_a = torch.where(row_mass > 0, row_mass.log(), -torch.inf)
    log_b = torch.where(column_mass > 0, column_mass.log(), -torch.inf)
    left = torch.where(row_mass > 0, torch.full_like(row_mass, -math.log(row_mass.numel())), -torch.inf)
    right = torch.where(column_mass > 0, torch.full_like(column_mass, -math.log(column_mass.numel())), -torch.inf)
    kernel = -cost / regularization
    cost_min, cost_max = float(cost.min()), float(cost.max())
    threshold = math.log(torch.finfo(cost.dtype).max)
    direct_exp_overflow = -cost_min / regularization > threshold
    direct_exp_underflow = -cost_max / regularization < math.log(torch.finfo(cost.dtype).tiny)
    row_error = column_error = None
    converged = False
    used = 0
    for used in range(1, iterations + 1):
        right = log_b - _column_lse(kernel, left)
        left = log_a - _row_lse(kernel, right)
        # Gauge freedom is exact: subtracting c from u and adding it to v
        # leaves every plan entry unchanged and prevents dual drift.
        finite_left = left[torch.isfinite(left)]
        gauge = finite_left.mean()
        left = left - gauge
        right = right + gauge
        if tolerance > 0 and used % 10 == 0:
            columns = torch.exp(right + _column_lse(kernel, left))
            column_error = (columns - column_mass).square().sum()
            if residual == "both":
                rows = torch.exp(left + _row_lse(kernel, right))
                row_error = (rows - row_mass).square().sum()
                error = .5 * (row_error + column_error)
            else:
                error = column_error
            if float(error) <= tolerance:
                converged = True
                break
    # Reuse this dense allocation instead of constructing another dense plan.
    step = max(1, REDUCTION_ENTRIES // cost.shape[1])
    for start in range(0, cost.shape[0], step):
        block = kernel[start:start + step]
        block.add_(left[start:start + step, None]).add_(right[None, :]).exp_()
    if not bool(torch.isfinite(kernel).all()):
        raise FloatingPointError("Log-domain ALPINE/FUGAL projection became non-finite")
    final_row = kernel.sum(1) - row_mass
    final_column = kernel.sum(0) - column_mass
    detail = dict(iterations=used, iteration_budget=iterations,
                  stopping_test_satisfied=converged,
                  row_squared_residual=float(final_row.square().sum()),
                  column_squared_residual=float(final_column.square().sum()),
                  row_max_absolute_residual=float(final_row.abs().max()),
                  column_max_absolute_residual=float(final_column.abs().max()),
                  gradient_min=cost_min, gradient_max=cost_max,
                  direct_exp_would_overflow=direct_exp_overflow,
                  direct_exp_would_underflow=direct_exp_underflow,
                  device=str(cost.device))
    if diagnostics is not None:
        diagnostics.append(detail)
    return kernel


class StableTransport:
    """Per-model private globals prevent changes to the frozen module registry."""
    def __init__(self, method, parameters, device="cuda:0"):
        if method not in {"alpine", "fugal"}:
            raise ValueError("Only ALPINE and FUGAL use this projection")
        import baseline  # establishes the portable common/result import root
        source = Path(baseline.__file__).parent / (method + ".py")
        name = "_gpu261112_" + method + "_" + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses resolves annotations here
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        self._module = module
        self.method = method
        self.projections = []
        cfg_class = module.AlpineConfig if method == "alpine" else module.FUGALConfig
        allowed = {f.name for f in fields(cfg_class)}
        params = dict(parameters)
        unknown = set(params) - allowed
        if unknown:
            raise ValueError("Unexpected transport parameters: " + repr(sorted(unknown)))
        params.update(device=device, show_progress=False)
        # Real memory is monitored by the campaign; keep the exact dense solve.
        params.setdefault("max_dense_entries", 10**18)
        self.config = cfg_class(**params)
        self.config.validate()
        def projection(cost, rows, columns, regularization, iterations, tolerance):
            mode = "both" if method == "alpine" and rows.numel() != columns.numel() else "columns"
            return stable_sinkhorn(cost, rows, columns, regularization, iterations,
                                   tolerance, residual=mode, diagnostics=self.projections)
        if method == "alpine":
            module._sinkhorn_log = projection
            self.model = module.Alpine(self.config)
        else:
            module._sinkhorn = projection
            self.model = module.FUGAL(self.config)

    def fit(self, graph1, graph2):
        self.projections.clear()
        result = self.model.fit(graph1, graph2)
        result.metadata.update(
            actual_backend="torch_graph_objective_and_log_sinkhorn",
            numerical_revision=VERSION,
            numerical_revision_diagnostics=dict(
                calls=list(self.projections),
                same_iteration_budgets=True, same_objective=True,
                nan_replacement=False, cost_clipping=False,
                padded_Hungarian_preserved=self.method == "fugal",
                dummy_row_update_preserved=self.method == "alpine",
                zero_marginal_transport="exactly_zero",
                frozen_source_modified=False),
            efficiency_scope=dict(
                gpu_operations="feature distance, graph objective/gradient, entropic projection",
                cpu_operations="NetworkX feature extraction and SciPy Hungarian rounding"))
        return result


def construct_transport(method, parameters, device="cuda:0"):
    model = StableTransport(method, parameters, device)
    return model, dict(asdict(model.config), numerical_revision=VERSION)
