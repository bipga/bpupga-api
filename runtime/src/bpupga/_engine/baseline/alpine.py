from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from common.result import AlignmentResult

try:
    import torch
except ImportError as error:  # pragma: no cover - depends on the runtime
    raise ImportError("Alpine requires PyTorch; install the 'torch' package") from error

try:
    from tqdm import trange
except ImportError:  # progress output is optional
    def trange(*args, **kwargs):
        kwargs.pop("desc", None)
        kwargs.pop("disable", None)
        return range(*args, **kwargs)


@dataclass(frozen=True)
class AlpineConfig:
    """Parameters of the authors' unsupervised Alpine implementation.

    The official code uses ten outer iterations and ten Frank--Wolfe steps per
    outer iteration.  ``niter`` was present in the public API but ignored by a
    hard-coded ``range(10)``; this implementation makes it effective while
    retaining the official default.
    """

    mu: float = 1.0
    niter: int = 10
    inner_iterations: int = 10
    weight: int = 2
    sinkhorn_reg: float = 1.0
    sinkhorn_iterations: int = 500
    sinkhorn_tolerance: float = 1e-5
    dtype: str = "float64"
    device: str = "auto"
    show_progress: bool = True
    max_dense_entries: int = 25_000_000
    collect_stage_trace: bool = False

    def validate(self) -> None:
        if self.mu < 0.0:
            raise ValueError("mu must be non-negative")
        if self.niter < 1 or self.inner_iterations < 1:
            raise ValueError("iteration counts must be positive")
        if self.weight not in {1, 2}:
            raise ValueError("weight must be 1 or 2")
        if self.sinkhorn_reg <= 0.0 or self.sinkhorn_iterations < 1:
            raise ValueError("Sinkhorn regularization and iterations must be positive")
        if self.sinkhorn_tolerance < 0.0:
            raise ValueError("sinkhorn_tolerance must be non-negative")
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


def _prepare_graph(graph: nx.Graph) -> nx.Graph:
    prepared = nx.Graph()
    prepared.add_nodes_from(range(graph.number_of_nodes()))
    prepared.add_edges_from(graph.edges())
    for node in list(nx.isolates(prepared)):
        prepared.add_edge(node, node)
    return prepared


def _adjacency(
    graph: nx.Graph,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    matrix = nx.to_numpy_array(
        graph,
        nodelist=range(graph.number_of_nodes()),
        dtype=np.float64 if dtype == torch.float64 else np.float32,
    )
    return torch.as_tensor(matrix, device=device, dtype=dtype)


def _structural_features(graph: nx.Graph, weight: int) -> np.ndarray:
    """Reproduce ``feature_extraction1``/``feature_extraction`` from Alpine."""

    nodes = range(graph.number_of_nodes())
    degree = dict(graph.degree())
    feature_count = 2 if weight == 2 else 4
    features = np.zeros((graph.number_of_nodes(), feature_count), dtype=np.float64)
    clustering = nx.clustering(graph) if weight == 1 else None

    for node in nodes:
        neighbors = [neighbor for neighbor in graph.neighbors(node) if neighbor != node]
        features[node, 0] = float(degree.get(node, 0))
        if weight == 1:
            features[node, 1] = float(clustering.get(node, 0.0))
            if neighbors:
                features[node, 2] = float(
                    np.mean([degree.get(neighbor, 0) for neighbor in neighbors])
                )
                features[node, 3] = float(
                    np.mean([clustering.get(neighbor, 0.0) for neighbor in neighbors])
                )
        elif neighbors:
            features[node, 1] = float(
                np.mean([degree.get(neighbor, 0) for neighbor in neighbors])
            )
    return np.nan_to_num(features)


def _pairwise_euclidean(
    source: np.ndarray,
    target: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    source_tensor = torch.as_tensor(source, device=device, dtype=dtype)
    target_tensor = torch.as_tensor(target, device=device, dtype=dtype)
    return torch.cdist(source_tensor, target_tensor, p=2)


def _sinkhorn_log(
    cost: torch.Tensor,
    row_mass: torch.Tensor,
    column_mass: torch.Tensor,
    regularization: float,
    iterations: int,
    tolerance: float,
) -> torch.Tensor:
    """Log-domain form of Alpine's entropic transport projection.

    It solves the same projection as the authors' exponential-kernel routine,
    while avoiding the overflow that their Facebook example explicitly warns
    about.  Zero dummy mass is represented by ``-inf`` in log space.
    """

    negative_infinity = torch.tensor(float("-inf"), device=cost.device, dtype=cost.dtype)
    log_row = torch.where(row_mass > 0.0, torch.log(row_mass), negative_infinity)
    log_column = torch.where(
        column_mass > 0.0,
        torch.log(column_mass),
        negative_infinity,
    )
    log_kernel = -cost / regularization
    log_left = torch.where(
        row_mass > 0.0,
        torch.full_like(row_mass, -float(np.log(max(1, row_mass.numel())))),
        negative_infinity,
    )
    log_right = torch.where(
        column_mass > 0.0,
        torch.full_like(
            column_mass,
            -float(np.log(max(1, column_mass.numel()))),
        ),
        negative_infinity,
    )

    for iteration in range(1, iterations + 1):
        log_right = log_column - torch.logsumexp(
            log_kernel + log_left[:, None], dim=0
        )
        log_left = log_row - torch.logsumexp(
            log_kernel + log_right[None, :], dim=1
        )
        if tolerance > 0.0 and iteration % 10 == 0:
            log_plan = log_left[:, None] + log_kernel + log_right[None, :]
            plan = torch.exp(log_plan)
            row_error = (plan.sum(dim=1) - row_mass).square().sum()
            column_error = (plan.sum(dim=0) - column_mass).square().sum()
            if float(0.5 * (row_error + column_error)) <= tolerance:
                return plan

    return torch.exp(log_left[:, None] + log_kernel + log_right[None, :])

def _snapshot(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().copy()


def _stage_metadata(
    plan: torch.Tensor,
    adjacency_source: torch.Tensor,
    adjacency_target: torch.Tensor,
    feature_cost: torch.Tensor,
    continuation_weight: float,
) -> dict[str, float]:
    real_plan = plan[: adjacency_source.shape[0], :]

    residual = (
        adjacency_source
        - real_plan @ adjacency_target @ real_plan.T
    )
    structural_loss = torch.sum(residual.square())
    feature_loss = torch.sum(
        real_plan * feature_cost[: adjacency_source.shape[0], :]
    )
    binary_penalty = torch.sum(real_plan * (1.0 - real_plan))

    safe_plan = torch.clamp(real_plan, min=1e-300)
    entropy = -torch.sum(safe_plan * torch.log(safe_plan))

    return {
        "continuation_weight": float(continuation_weight),
        "structural_loss": float(structural_loss.detach().cpu()),
        "feature_loss": float(feature_loss.detach().cpu()),
        "binary_penalty": float(binary_penalty.detach().cpu()),
        "surrogate_objective": float(
            (
                structural_loss
                + feature_loss
                + continuation_weight * binary_penalty
            ).detach().cpu()
        ),
        "entropy": float(entropy.detach().cpu()),
        "row_sum_error_l1": float(
            torch.sum(torch.abs(real_plan.sum(dim=1) - 1.0))
            .detach()
            .cpu()
        ),
        "column_sum_error_l1": float(
            torch.sum(torch.abs(plan.sum(dim=0) - 1.0))
            .detach()
            .cpu()
        ),
        "score_min": float(real_plan.min().detach().cpu()),
        "score_max": float(real_plan.max().detach().cpu()),
        "row_max_mean": float(
            real_plan.max(dim=1).values.mean().detach().cpu()
        ),
        "column_max_mean": float(
            real_plan.max(dim=0).values.mean().detach().cpu()
        ),
    }


def _run_alpine(
    source_graph: nx.Graph,
    target_graph: nx.Graph,
    config: AlpineConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    np.ndarray,
    float,
    dict[str, np.ndarray],
    dict[str, dict[str, object]],
]:
    """Run Alpine with ``|source| <= |target|`` and return its soft plan."""

    source_size = source_graph.number_of_nodes()
    target_size = target_graph.number_of_nodes()
    if source_size > target_size:
        raise ValueError("Alpine requires the first graph to be no larger than the second")

    source = _prepare_graph(source_graph)
    target = _prepare_graph(target_graph)
    source_with_dummy = source.copy()
    source_with_dummy.add_node(source_size)
    source_with_dummy.add_edge(source_size, source_size)

    adjacency_source = _adjacency(source, device, dtype)
    adjacency_target = _adjacency(target, device, dtype)
    source_features = _structural_features(source_with_dummy, config.weight)
    target_features = _structural_features(target, config.weight)
    feature_cost = config.mu * _pairwise_euclidean(
        source_features,
        target_features,
        device,
        dtype,
    )

    stage_scores: dict[str, np.ndarray] = {}
    stage_metadata: dict[str, dict[str, object]] = {}

    if config.collect_stage_trace:
        stage_scores["feature_quality"] = _snapshot(
            -feature_cost[:source_size, :]
        )
        stage_metadata["feature_quality"] = {
            "kind": "negative_pairwise_feature_distance",
            "quality_convention": "larger_is_better",
        }

    plan = torch.ones(
        (source_size + 1, target_size),
        device=device,
        dtype=dtype,
    )
    plan[:-1, :] *= 1.0 / target_size
    plan[-1, :] *= float(target_size - source_size) / target_size

    row_mass = torch.ones(source_size + 1, device=device, dtype=dtype)
    row_mass[-1] = float(target_size - source_size)
    column_mass = torch.ones(target_size, device=device, dtype=dtype)

    for outer in trange(
        config.niter,
        desc="Alpine quasi-permutation",
        disable=not config.show_progress,
    ):
        for inner in range(1, config.inner_iterations + 1):
            real_plan = plan[:source_size, :]
            residual = (
                adjacency_source
                - real_plan @ adjacency_target @ real_plan.T
            )
            structural_gradient = torch.zeros_like(plan)
            structural_gradient[:source_size, :] = (
                -4.0 * residual @ real_plan @ adjacency_target
            )
            gradient = (
                structural_gradient
                + float(outer) * (1.0 - 2.0 * plan)
                + feature_cost
            )
            candidate = _sinkhorn_log(
                gradient,
                row_mass,
                column_mass,
                config.sinkhorn_reg,
                config.sinkhorn_iterations,
                config.sinkhorn_tolerance,
            )
            if not torch.isfinite(candidate).all():
                raise FloatingPointError("Alpine Sinkhorn projection became non-finite")
            step = 2.0 / float(2 + inner)
            # The public implementation updates only the real-node rows.  The
            # dummy row remains at its initial unmatched-mass allocation.
            plan[:source_size, :] += step * (
                candidate[:source_size, :] - plan[:source_size, :]
            )
        if config.collect_stage_trace:
            stage_name = f"outer_{outer + 1:02d}"
            stage_scores[stage_name] = _snapshot(
                plan[:source_size, :]
            )
            stage_metadata[stage_name] = {
                "kind": "transport_plan",
                **_stage_metadata(
                    plan,
                    adjacency_source,
                    adjacency_target,
                    feature_cost,
                    continuation_weight=float(outer),
                ),
            }


    score = _snapshot(plan[:source_size, :])

    if config.collect_stage_trace:
        stage_scores["final_score"] = score.copy()
        stage_metadata["final_score"] = {
            "kind": "final_transport_plan",
            **_stage_metadata(
                plan,
                adjacency_source,
                adjacency_target,
                feature_cost,
                continuation_weight=float(config.niter - 1),
            ),
        }

    row_indices, column_indices = linear_sum_assignment(-score)
    permutation = np.zeros((source_size, target_size), dtype=np.float64)
    permutation[row_indices, column_indices] = 1.0
    reconstructed = permutation @ adjacency_target.detach().cpu().numpy() @ permutation.T
    frobenius = float(np.linalg.norm(
        adjacency_source.detach().cpu().numpy() - reconstructed,
        ord="fro",
    ) ** 2)
    return score, frobenius, stage_scores, stage_metadata


class Alpine:
    """Leakage-safe unsupervised Alpine for partial unlabeled graph alignment."""

    def __init__(self, config: AlpineConfig | None = None) -> None:
        self.config = config or AlpineConfig()
        self.config.validate()

    def fit(self, graph1: nx.Graph, graph2: nx.Graph) -> AlignmentResult:
        if graph1.number_of_nodes() == 0 or graph2.number_of_nodes() == 0:
            raise ValueError("Alpine cannot align empty graphs")
        _require_consecutive_nodes(graph1, "G1")
        _require_consecutive_nodes(graph2, "G2")

        n1 = graph1.number_of_nodes()
        n2 = graph2.number_of_nodes()
        smaller = min(n1, n2)
        larger = max(n1, n2)
        working_entries = max(
            smaller * smaller,
            larger * larger,
            (smaller + 1) * larger,
        )
        if working_entries > self.config.max_dense_entries:
            raise ValueError(
                "Alpine's exact dense solver requires a working matrix with "
                f"{working_entries:,} entries, exceeding max_dense_entries="
                f"{self.config.max_dense_entries:,}."
            )

        device = _resolve_device(self.config.device)
        dtype = _torch_dtype(self.config.dtype)
        start = time.perf_counter()

        swapped = n1 > n2
        if swapped:
            (
                score_small_to_large,
                frobenius,
                stage_scores,
                stage_metadata,
            ) = _run_alpine(
                graph2,
                graph1,
                self.config,
                device,
                dtype,
            )
            score_matrix = score_small_to_large.T
            stage_scores = {
                name: matrix.T.copy()
                for name, matrix in stage_scores.items()
            }
        else:
            (
                score_matrix,
                frobenius,
                stage_scores,
                stage_metadata,
            ) = _run_alpine(
                graph1,
                graph2,
                self.config,
                device,
                dtype,
            )

        source_indices, target_indices = linear_sum_assignment(-score_matrix)
        predicted_pairs = [
            (int(source), int(target))
            for source, target in zip(source_indices, target_indices)
        ]
        return AlignmentResult(
            method="Alpine",
            predicted_pairs=predicted_pairs,
            runtime_sec=float(time.perf_counter() - start),
            cost_matrix=-score_matrix,
            larger_is_better=False,
            metadata={
                "config": asdict(self.config),
                "device": str(device),
                "frobenius_objective": frobenius,
                "feature_set": (
                    "degree and mean-neighbor degree"
                    if self.config.weight == 2
                    else "degree, clustering, mean-neighbor degree, and "
                    "mean-neighbor clustering"
                ),
                "prediction_rule": "maximum-weight linear assignment",
                "cost_matrix": "negative Alpine quasi-permutation score",
                "solver": "official unsupervised Alpine objective with stable Sinkhorn",
                "swapped_for_smaller_source": swapped,
                "matched_cardinality": smaller,
                "ground_truth_used_during_inference": False,
            },
            stage_score_matrices=stage_scores,
            stage_diagnostics_metadata=stage_metadata,
        )
