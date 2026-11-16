from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sps

from common.result import AlignmentResult

try:
    import torch
except ImportError as error:  # pragma: no cover - depends on the runtime
    raise ImportError("SLOTAlign requires PyTorch; install the 'torch' package") from error

try:
    from tqdm import trange
except ImportError:  # progress output is optional
    def trange(*args, **kwargs):
        kwargs.pop("desc", None)
        kwargs.pop("disable", None)
        return range(*args, **kwargs)


@dataclass(frozen=True)
class SLOTAlignConfig:
    """Official optimizer defaults plus a plain-graph feature adapter.

    The reference implementation consumes node attributes. BPUPGA currently
    exposes only topology, so ``feature_rounds`` builds deterministic structural
    features without using cross-graph correspondences.
    """

    bases: int = 2
    joint_epoch: int = 100
    epoch: int = 1000
    gw_beta: float = 0.01
    step_size: float = 0.1
    joint_sinkhorn_iters: int = 50
    final_sinkhorn_iters: int = 20
    feature_rounds: int = 3
    feature_norm: str = "center"
    dtype: str = "float32"
    device: str = "auto"
    seed: int = 2026
    show_progress: bool = True
    max_dense_entries: int = 25_000_000

    def validate(self) -> None:
        if self.bases < 2:
            raise ValueError("bases must be at least 2")
        if self.joint_epoch < 0 or self.epoch < self.joint_epoch:
            raise ValueError("require 0 <= joint_epoch <= epoch")
        if self.gw_beta <= 0.0 or self.step_size <= 0.0:
            raise ValueError("gw_beta and step_size must be positive")
        if self.joint_sinkhorn_iters < 1 or self.final_sinkhorn_iters < 1:
            raise ValueError("Sinkhorn iteration counts must be positive")
        if self.feature_rounds < 0:
            raise ValueError("feature_rounds must be non-negative")
        if self.feature_norm not in {"none", "center", "separate", "joint"}:
            raise ValueError("feature_norm must be none, center, separate, or joint")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if self.max_dense_entries < 1:
            raise ValueError("max_dense_entries must be positive")


def _require_consecutive_nodes(graph: nx.Graph, label: str) -> None:
    if set(graph.nodes()) != set(range(graph.number_of_nodes())):
        raise ValueError(f"{label} must use consecutive integer node ids")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        value = "cuda:0"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} was requested but CUDA is unavailable")
    return torch.device(value)


def _torch_dtype(value: str) -> torch.dtype:
    return torch.float32 if value == "float32" else torch.float64


def _adjacency(graph: nx.Graph) -> sps.csr_matrix:
    return sps.csr_matrix(
        nx.to_scipy_sparse_array(
            graph,
            nodelist=range(graph.number_of_nodes()),
            dtype=np.float32,
            format="csr",
        )
    )


def _structural_features(adjacency: sps.csr_matrix, rounds: int) -> np.ndarray:
    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float32)
    current = np.stack(
        [degree, np.log1p(degree), np.sqrt(degree + 1e-6), degree * degree],
        axis=1,
    )
    blocks = [current]
    denominator = np.maximum(degree, 1.0)[:, None]
    for _ in range(rounds):
        neighbor_sum = adjacency @ current
        neighbor_mean = neighbor_sum / denominator
        neighbor_second = (adjacency @ (current * current)) / denominator
        neighbor_std = np.sqrt(
            np.maximum(neighbor_second - neighbor_mean * neighbor_mean, 0.0) + 1e-6
        )
        current = np.concatenate([neighbor_mean, neighbor_std, neighbor_sum], axis=1)
        mean = current.mean(axis=0, keepdims=True)
        std = np.maximum(current.std(axis=0, keepdims=True), 1e-6)
        current = ((current - mean) / std).astype(np.float32)
        blocks.append(current)
    return np.nan_to_num(np.concatenate(blocks, axis=1)).astype(np.float32)


def _normalize_feature_pair(
    source: np.ndarray,
    target: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "none":
        return source, target
    if mode == "center":
        return (
            (source - source.mean(axis=0, keepdims=True)).astype(np.float32),
            (target - target.mean(axis=0, keepdims=True)).astype(np.float32),
        )
    if mode == "separate":
        result = []
        for features in (source, target):
            mean = features.mean(axis=0, keepdims=True)
            std = np.maximum(features.std(axis=0, keepdims=True), 1e-6)
            result.append(((features - mean) / std).astype(np.float32))
        return result[0], result[1]
    combined = np.vstack([source, target])
    mean = combined.mean(axis=0, keepdims=True)
    std = np.maximum(combined.std(axis=0, keepdims=True), 1e-6)
    return (
        ((source - mean) / std).astype(np.float32),
        ((target - mean) / std).astype(np.float32),
    )


def _row_normalize(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _similarity_stack(
    adjacency: sps.csr_matrix,
    features: np.ndarray,
    bases: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    adjacency_tensor = torch.as_tensor(adjacency.toarray(), device=device, dtype=dtype)
    views = [adjacency_tensor]

    identity = torch.eye(adjacency.shape[0], device=device, dtype=dtype)
    adjacency_with_loops = adjacency_tensor + identity
    degree = adjacency_with_loops.sum(dim=1).clamp_min(1.0)
    scale = torch.rsqrt(degree)
    propagation = scale[:, None] * adjacency_with_loops * scale[None, :]

    propagated = torch.as_tensor(features, device=device, dtype=dtype)
    feature_views = [propagated]
    for _ in range(bases - 2):
        propagated = propagation @ propagated
        feature_views.append(propagated)
    views.extend(_row_normalize(view) @ _row_normalize(view).T for view in feature_views)
    return torch.stack(views, dim=2)


def _project_simplex(vector: np.ndarray) -> np.ndarray:
    if np.isclose(vector.sum(), 1.0) and np.all(vector >= 0.0):
        return vector
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered)
    valid = np.nonzero(
        ordered * np.arange(1, vector.size + 1) > cumulative - 1.0
    )[0]
    threshold = (cumulative[valid[-1]] - 1.0) / float(valid[-1] + 1)
    return np.maximum(vector - threshold, 0.0)


@torch.no_grad()
def _gw_step(
    source_cost: torch.Tensor,
    target_cost: torch.Tensor,
    transport: torch.Tensor,
    source_mass: torch.Tensor,
    target_mass: torch.Tensor,
    beta: float,
    sinkhorn_iters: int,
    diagnostics: dict | None = None,
) -> torch.Tensor:
    cost = -2.0 * source_cost @ transport @ target_cost.T
    raw_exponent = -(cost - cost.min()) / beta
    if diagnostics is not None:
        diagnostics["updates"] += 1
        diagnostics["clip_entries"] += int(((raw_exponent < -60) | (raw_exponent > 60)).sum())
        diagnostics["raw_nonfinite_exponent"] += int((~torch.isfinite(raw_exponent)).sum())
    exponent = torch.clamp(raw_exponent, -60.0, 60.0)
    if diagnostics is not None:
        diagnostics['floored_transport_entries'] = diagnostics.get('floored_transport_entries',0)+int((transport < 1e-30).sum())
    kernel = torch.exp(exponent) * transport.clamp_min(1e-30)
    left = torch.full_like(source_mass, 1.0/source_mass.numel())
    right = torch.ones_like(target_mass)
    for _ in range(sinkhorn_iters):
        column_denominator = kernel.T @ left
        if diagnostics is not None:
            diagnostics["zero_column_denominator"] += int((column_denominator == 0).sum())
            diagnostics["clamped_column_denominator"] += int((column_denominator < 1e-30).sum())
        right = target_mass / column_denominator.clamp_min(1e-30)
        row_denominator = kernel @ right
        if diagnostics is not None:
            diagnostics["zero_row_denominator"] += int((row_denominator == 0).sum())
            diagnostics["clamped_row_denominator"] += int((row_denominator < 1e-30).sum())
        updated_left = source_mass / row_denominator.clamp_min(1e-30)
        relative_error = (updated_left-left).abs().sum()/left.abs().sum()
        left = updated_left
        if float(relative_error) < 1e-10:
            if diagnostics is not None: diagnostics["inner_early_stops"] += 1
            break
    candidate = left[:, None] * kernel * right[None, :]
    if not torch.isfinite(candidate).all():
        if diagnostics is not None: diagnostics["nonfinite_return_previous"] += 1
        return transport
    return candidate


class SLOTAlign:
    """Leakage-safe SLOTAlign for unattributed BPUPGA graphs."""

    def __init__(self, config: SLOTAlignConfig | None = None) -> None:
        self.config = config or SLOTAlignConfig()
        self.config.validate()

    def fit(self, graph1: nx.Graph, graph2: nx.Graph) -> AlignmentResult:
        if graph1.number_of_nodes() == 0 or graph2.number_of_nodes() == 0:
            raise ValueError("SLOTAlign cannot align empty graphs")
        _require_consecutive_nodes(graph1, "G1")
        _require_consecutive_nodes(graph2, "G2")

        n1, n2 = graph1.number_of_nodes(), graph2.number_of_nodes()
        dense_entries = max(n1 * n1, n2 * n2, n1 * n2)
        if dense_entries > self.config.max_dense_entries:
            raise ValueError(
                "SLOTAlign's exact dense solver requires a matrix with "
                f"{dense_entries:,} matrix entries, exceeding max_dense_entries="
                f"{self.config.max_dense_entries:,}. Increase the explicit limit; "
                "no non-SLOTAlign fallback is used."
            )

        diagnostics = {key:0 for key in ("updates","clip_entries","raw_nonfinite_exponent","zero_column_denominator","zero_row_denominator","clamped_column_denominator","clamped_row_denominator","inner_early_stops","nonfinite_return_previous")}
        start = time.perf_counter()
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        device = _resolve_device(self.config.device)
        dtype = _torch_dtype(self.config.dtype)

        adjacency1, adjacency2 = _adjacency(graph1), _adjacency(graph2)
        features1 = _structural_features(adjacency1, self.config.feature_rounds)
        features2 = _structural_features(adjacency2, self.config.feature_rounds)
        features1, features2 = _normalize_feature_pair(
            features1, features2, self.config.feature_norm
        )
        similarities1 = _similarity_stack(
            adjacency1, features1, self.config.bases, device, dtype
        )
        similarities2 = _similarity_stack(
            adjacency2, features2, self.config.bases, device, dtype
        )

        source_mass = torch.full((n1,), 1.0 / n1, device=device, dtype=dtype)
        target_mass = torch.full((n2,), 1.0 / n2, device=device, dtype=dtype)
        # Reference SLOTAlign initialization: independent uniform coupling.
        transport = source_mass[:, None] * target_mass[None, :]
        layers = similarities1.shape[2]
        alpha_np = np.full(layers, 1.0 / layers, dtype=np.float64)
        beta_np = np.full(layers, 1.0 / layers, dtype=np.float64)
        source_cost = (
            similarities1
            * torch.as_tensor(alpha_np, device=device, dtype=dtype)
        ).sum(dim=2)
        target_cost = (
            similarities2
            * torch.as_tensor(beta_np, device=device, dtype=dtype)
        ).sum(dim=2)

        for _ in trange(
            self.config.joint_epoch,
            desc="SLOTAlign joint optimization",
            disable=not self.config.show_progress,
        ):
            alpha = torch.tensor(alpha_np, device=device, dtype=dtype, requires_grad=True)
            beta = torch.tensor(beta_np, device=device, dtype=dtype, requires_grad=True)
            source_cost = (similarities1 * alpha).sum(dim=2)
            target_cost = (similarities2 * beta).sum(dim=2)
            objective = (
                source_cost.square().mean()
                + target_cost.square().mean()
                - torch.trace(source_cost @ transport @ target_cost @ transport.T)
            )
            alpha_gradient, beta_gradient = torch.autograd.grad(
                objective, (alpha, beta)
            )
            alpha_np = _project_simplex(
                (alpha - self.config.step_size * alpha_gradient).detach().cpu().numpy()
            )
            beta_np = _project_simplex(
                (beta - self.config.step_size * beta_gradient).detach().cpu().numpy()
            )
            transport = _gw_step(
                source_cost.detach(),
                target_cost.detach(),
                transport.detach(),
                source_mass,
                target_mass,
                self.config.gw_beta,
                self.config.joint_sinkhorn_iters,
                diagnostics=diagnostics,
            ).detach()

        source_cost = source_cost.detach()
        target_cost = target_cost.detach()
        transport = transport.detach()

        if self.config.joint_epoch > 0:
            del alpha, beta, alpha_gradient, beta_gradient, objective

        with torch.no_grad():
            for _ in trange(
                self.config.epoch - self.config.joint_epoch,
                desc="SLOTAlign transport optimization",
                disable=not self.config.show_progress,
            ):
                previous_transport = transport
                transport = _gw_step(
                    source_cost,
                    target_cost,
                    transport,
                    source_mass,
                    target_mass,
                    self.config.gw_beta,
                    self.config.final_sinkhorn_iters,
                    diagnostics=diagnostics,
                )
                relative_error = (transport-previous_transport).abs().sum()/previous_transport.abs().sum()
                if float(relative_error) < 1e-10:
                    diagnostics['outer_early_stops'] = diagnostics.get('outer_early_stops',0)+1
                    break

        score_matrix = transport.detach().cpu().numpy()
        predicted_pairs = [
            (source, int(target))
            for source, target in enumerate(np.argmax(score_matrix, axis=1))
        ]
        return AlignmentResult(
            method="SLOTAlign",
            predicted_pairs=predicted_pairs,
            runtime_sec=float(time.perf_counter() - start),
            cost_matrix=-score_matrix,
            larger_is_better=False,
            metadata={
                "config": asdict(self.config),
                "device": str(device),
                "alpha": alpha_np.tolist(),
                "beta": beta_np.tolist(),
                "feature_adapter": "deterministic topology-only structural features",
                "prediction_rule": "row-wise maximum transport mass",
                "numerical_guard_diagnostics": diagnostics,
                "nonfinite_guard_triggered": diagnostics["nonfinite_return_previous"] > 0,
                "author_update_coefficient": -1,
                "author_code_priority": True,
                "cost_matrix": "negative transport mass (ranking-equivalent)",
                "solver": "exact dense SLOTAlign; no feature-OT fallback",
                "ground_truth_used_during_inference": False,
            },
        )
