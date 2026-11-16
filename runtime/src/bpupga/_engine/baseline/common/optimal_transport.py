"""Numerically stable optimal-transport primitives shared by baselines."""

from __future__ import annotations

import math

import torch


def log_sinkhorn(
    cost: torch.Tensor,
    row_mass: torch.Tensor,
    column_mass: torch.Tensor,
    regularization: float,
    iterations: int,
    tolerance: float,
) -> torch.Tensor:
    """Project ``exp(-cost / regularization)`` onto prescribed marginals.

    All scaling updates stay in log space. This is algebraically equivalent
    to exponential-kernel Sinkhorn, but remains finite when a structural
    gradient has a range too large for a floating-point exponential.
    """

    if cost.ndim != 2:
        raise ValueError("Sinkhorn cost must be a matrix")
    if cost.shape != (row_mass.numel(), column_mass.numel()):
        raise ValueError("Sinkhorn cost and marginal shapes do not agree")
    if regularization <= 0.0 or iterations < 1 or tolerance < 0.0:
        raise ValueError("Invalid Sinkhorn regularization, iterations, or tolerance")
    if not torch.isfinite(cost).all():
        raise FloatingPointError("Sinkhorn cost contains NaN or infinite values")
    if not torch.isfinite(row_mass).all() or not torch.isfinite(column_mass).all():
        raise FloatingPointError("Sinkhorn marginals contain NaN or infinite values")
    if torch.any(row_mass < 0.0) or torch.any(column_mass < 0.0):
        raise ValueError("Sinkhorn marginals must be non-negative")
    row_total = float(row_mass.sum())
    column_total = float(column_mass.sum())
    if not math.isclose(row_total, column_total, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("Sinkhorn marginals must have equal total mass")

    negative_infinity = torch.tensor(
        float("-inf"), device=cost.device, dtype=cost.dtype
    )
    log_row = torch.where(row_mass > 0.0, torch.log(row_mass), negative_infinity)
    log_column = torch.where(
        column_mass > 0.0, torch.log(column_mass), negative_infinity
    )
    log_kernel = -cost / regularization
    log_left = torch.where(
        row_mass > 0.0,
        torch.full_like(row_mass, -math.log(max(1, row_mass.numel()))),
        negative_infinity,
    )
    log_right = torch.where(
        column_mass > 0.0,
        torch.full_like(column_mass, -math.log(max(1, column_mass.numel()))),
        negative_infinity,
    )

    for iteration in range(1, iterations + 1):
        log_right = log_column - torch.logsumexp(
            log_kernel + log_left[:, None], dim=0
        )
        log_left = log_row - torch.logsumexp(
            log_kernel + log_right[None, :], dim=1
        )
        active_left = log_left[row_mass > 0.0]
        active_right = log_right[column_mass > 0.0]
        if not torch.isfinite(active_left).all() or not torch.isfinite(active_right).all():
            raise FloatingPointError("Sinkhorn scaling became non-finite")
        if tolerance > 0.0 and iteration % 10 == 0:
            plan = torch.exp(log_left[:, None] + log_kernel + log_right[None, :])
            row_error = torch.max(torch.abs(plan.sum(dim=1) - row_mass))
            column_error = torch.max(torch.abs(plan.sum(dim=0) - column_mass))
            if float(torch.maximum(row_error, column_error)) <= tolerance:
                break

    plan = torch.exp(log_left[:, None] + log_kernel + log_right[None, :])
    if not torch.isfinite(plan).all():
        raise FloatingPointError("Sinkhorn transport plan became non-finite")
    return plan


def stable_sinkhorn(
    cost: torch.Tensor,
    row_mass: torch.Tensor,
    column_mass: torch.Tensor,
    regularization: float,
    iterations: int,
    tolerance: float,
) -> torch.Tensor:
    """Fast stabilized-kernel Sinkhorn with an exact log-domain fallback.

    Subtracting one constant from each row or column of ``-cost / reg`` only
    rescales the Gibbs kernel along that axis; the corresponding Sinkhorn factor
    absorbs the rescaling, so the projected transport plan is unchanged. The
    two-axis centering keeps at least one represented entry on every row and
    column while preventing overflow. If scaling nevertheless becomes
    non-finite, the computation restarts with :func:`log_sinkhorn` rather than
    modifying or clipping the resulting plan.
    """

    if cost.ndim != 2:
        raise ValueError("Sinkhorn cost must be a matrix")
    if cost.shape != (row_mass.numel(), column_mass.numel()):
        raise ValueError("Sinkhorn cost and marginal shapes do not agree")
    if regularization <= 0.0 or iterations < 1 or tolerance < 0.0:
        raise ValueError("Invalid Sinkhorn regularization, iterations, or tolerance")
    if not torch.isfinite(cost).all():
        raise FloatingPointError("Sinkhorn cost contains NaN or infinite values")
    if not torch.isfinite(row_mass).all() or not torch.isfinite(column_mass).all():
        raise FloatingPointError("Sinkhorn marginals contain NaN or infinite values")
    if torch.any(row_mass < 0.0) or torch.any(column_mass < 0.0):
        raise ValueError("Sinkhorn marginals must be non-negative")
    row_total = float(row_mass.sum())
    column_total = float(column_mass.sum())
    if not math.isclose(row_total, column_total, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("Sinkhorn marginals must have equal total mass")

    log_kernel = -cost / regularization
    # A row shift alone prevents overflow and guarantees one represented entry
    # per row, but an entire column can still underflow when every row strongly
    # prefers the same column. A subsequent column shift is also absorbed by
    # the right Sinkhorn factor and guarantees represented support on both axes
    # without changing the projected transport plan.
    log_kernel = log_kernel - torch.amax(log_kernel, dim=1, keepdim=True)
    log_kernel = log_kernel - torch.amax(log_kernel, dim=0, keepdim=True)
    kernel = torch.exp(log_kernel)
    if not torch.isfinite(kernel).all():
        return log_sinkhorn(
            cost, row_mass, column_mass, regularization, iterations, tolerance
        )

    denominator_floor = torch.finfo(cost.dtype).eps
    left = torch.where(
        row_mass > 0.0,
        torch.full_like(row_mass, 1.0 / max(1, row_mass.numel())),
        torch.zeros_like(row_mass),
    )
    right = torch.where(
        column_mass > 0.0,
        torch.full_like(column_mass, 1.0 / max(1, column_mass.numel())),
        torch.zeros_like(column_mass),
    )
    for iteration in range(1, iterations + 1):
        column_denominator = kernel.T @ left
        right = torch.where(
            column_mass > 0.0,
            column_mass / torch.clamp(column_denominator, min=denominator_floor),
            torch.zeros_like(column_mass),
        )
        row_denominator = kernel @ right
        left = torch.where(
            row_mass > 0.0,
            row_mass / torch.clamp(row_denominator, min=denominator_floor),
            torch.zeros_like(row_mass),
        )
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            return log_sinkhorn(
                cost, row_mass, column_mass, regularization, iterations, tolerance
            )
        if tolerance > 0.0 and iteration % 10 == 0:
            plan = left[:, None] * kernel * right[None, :]
            row_error = (plan.sum(dim=1) - row_mass).square().sum()
            column_error = (plan.sum(dim=0) - column_mass).square().sum()
            if float(torch.maximum(row_error, column_error)) <= tolerance:
                return plan

    plan = left[:, None] * kernel * right[None, :]
    if not torch.isfinite(plan).all():
        return log_sinkhorn(
            cost, row_mass, column_mass, regularization, iterations, tolerance
        )
    return plan
