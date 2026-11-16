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
    raise ImportError("FUGAL requires PyTorch; install the 'torch' package") from error

try:
    from tqdm import trange
except ImportError:  # progress output is optional
    def trange(*args, **kwargs):
        kwargs.pop("desc", None)
        kwargs.pop("disable", None)
        return range(*args, **kwargs)


@dataclass(frozen=True)
class FUGALConfig:
    """Defaults used by the authors' ``predict_alignment`` implementation."""

    mu: float = 1.0
    niter: int = 15
    inner_iterations: int = 10
    sinkhorn_reg: float = 1.0
    sinkhorn_iters: int = 500
    sinkhorn_stop_thr: float = 1e-3
    dtype: str = "float64"
    device: str = "cpu"
    show_progress: bool = True
    max_dense_entries: int = 25_000_000

    def validate(self) -> None:
        if self.mu < 0.0:
            raise ValueError("mu must be non-negative")
        if self.niter < 1 or self.inner_iterations < 1:
            raise ValueError("iteration counts must be positive")
        if self.sinkhorn_reg <= 0.0 or self.sinkhorn_iters < 1:
            raise ValueError("Sinkhorn regularization and iterations must be positive")
        if self.sinkhorn_stop_thr < 0.0:
            raise ValueError("sinkhorn_stop_thr must be non-negative")
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


def _adjacency(
    graph: nx.Graph,
    padded_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    adjacency = torch.zeros(
        (padded_size, padded_size), device=device, dtype=dtype
    )
    if graph.number_of_edges():
        edges = np.asarray(list(graph.edges()), dtype=np.int64)
        rows = torch.as_tensor(edges[:, 0], device=device, dtype=torch.long)
        columns = torch.as_tensor(edges[:, 1], device=device, dtype=torch.long)
        adjacency[rows, columns] = 1.0
        adjacency[columns, rows] = 1.0
    return adjacency


def _original_features(graph: nx.Graph, dtype: np.dtype) -> np.ndarray:
    """The four effective local features in the reference FUGAL extractor."""

    n = graph.number_of_nodes()
    degree = dict(graph.degree())
    clustering = nx.clustering(graph)
    features = np.zeros((n, 4), dtype=dtype)
    for node in range(n):
        neighbors = list(graph.neighbors(node))
        features[node, 0] = float(degree.get(node, 0))
        features[node, 1] = float(clustering.get(node, 0.0))
        if neighbors:
            features[node, 2] = float(
                np.mean([degree.get(neighbor, 0) for neighbor in neighbors])
            )
            features[node, 3] = float(
                np.mean([clustering.get(neighbor, 0.0) for neighbor in neighbors])
            )
    return np.nan_to_num(features)


def _sinkhorn(
    cost: torch.Tensor,
    row_mass: torch.Tensor,
    column_mass: torch.Tensor,
    regularization: float,
    iterations: int,
    stop_threshold: float,
) -> torch.Tensor:
    kernel = torch.exp(-cost / regularization)
    left = torch.ones_like(row_mass) / max(1, row_mass.numel())
    right = torch.ones_like(column_mass) / max(1, column_mass.numel())
    for iteration in range(1, iterations + 1):
        right = column_mass / (kernel.T @ left + 1e-16)
        left = row_mass / (kernel @ right + 1e-16)
        if stop_threshold > 0.0 and iteration % 10 == 0:
            estimated_columns = (kernel.T @ left) * right
            residual = (column_mass - estimated_columns).square().sum()
            if float(residual) <= stop_threshold:
                break
    return left[:, None] * kernel * right[None, :]


class FUGAL:
    """Leakage-safe Feature-fortified Unrestricted Graph Alignment."""

    def __init__(self, config: FUGALConfig | None = None) -> None:
        self.config = config or FUGALConfig()
        self.config.validate()

    def fit(self, graph1: nx.Graph, graph2: nx.Graph) -> AlignmentResult:
        if graph1.number_of_nodes() == 0 or graph2.number_of_nodes() == 0:
            raise ValueError("FUGAL cannot align empty graphs")
        _require_consecutive_nodes(graph1, "G1")
        _require_consecutive_nodes(graph2, "G2")

        n1, n2 = graph1.number_of_nodes(), graph2.number_of_nodes()
        padded_size = max(n1, n2)
        dense_entries = padded_size * padded_size
        if dense_entries > self.config.max_dense_entries:
            raise ValueError(
                "FUGAL's exact dense solver requires matrices with "
                f"{dense_entries:,} working matrix entries, exceeding "
                f"max_dense_entries={self.config.max_dense_entries:,}. Increase "
                "the explicit limit; no feature-only fallback is used."
            )

        start = time.perf_counter()
        device = _resolve_device(self.config.device)
        dtype = _torch_dtype(self.config.dtype)
        numpy_dtype = np.float32 if self.config.dtype == "float32" else np.float64
        adjacency1 = _adjacency(graph1, padded_size, device, dtype)
        adjacency2 = _adjacency(graph2, padded_size, device, dtype)

        features1 = _original_features(graph1, numpy_dtype)
        features2 = _original_features(graph2, numpy_dtype)
        if n1 < padded_size:
            features1 = np.vstack(
                [features1, np.zeros((padded_size - n1, 4), dtype=numpy_dtype)]
            )
        if n2 < padded_size:
            features2 = np.vstack(
                [features2, np.zeros((padded_size - n2, 4), dtype=numpy_dtype)]
            )
        feature_cost = self.config.mu * torch.cdist(
            torch.as_tensor(features1, device=device, dtype=dtype),
            torch.as_tensor(features2, device=device, dtype=dtype),
            p=2,
        )

        plan = torch.full(
            (padded_size, padded_size),
            1.0 / padded_size,
            device=device,
            dtype=dtype,
        )
        marginal = torch.ones(padded_size, device=device, dtype=dtype)
        for outer in trange(
            self.config.niter,
            desc="FUGAL quasi-permutation",
            disable=not self.config.show_progress,
        ):
            for inner in range(1, self.config.inner_iterations + 1):
                gradient = (
                    -(adjacency1.T @ plan @ adjacency2)
                    - (adjacency1 @ plan @ adjacency2.T)
                    + feature_cost
                    + float(outer) * (1.0 - 2.0 * plan)
                )
                candidate = _sinkhorn(
                    gradient,
                    marginal,
                    marginal,
                    self.config.sinkhorn_reg,
                    self.config.sinkhorn_iters,
                    self.config.sinkhorn_stop_thr,
                )
                step = 2.0 / float(2 + inner)
                plan = plan + step * (candidate - plan)

        padded_transport = plan.detach().cpu().numpy()
        source_indices, target_indices = linear_sum_assignment(-padded_transport)
        keep = (source_indices < n1) & (target_indices < n2)
        source_indices, target_indices = source_indices[keep], target_indices[keep]
        score_matrix = padded_transport[:n1, :n2]
        predicted_pairs = [
            (int(source), int(target))
            for source, target in zip(source_indices, target_indices)
        ]
        return AlignmentResult(
            method="FUGAL",
            predicted_pairs=predicted_pairs,
            runtime_sec=float(time.perf_counter() - start),
            cost_matrix=-score_matrix,
            partial_decoder_data={"padded_transport": padded_transport},
            larger_is_better=False,
            metadata={
                "config": asdict(self.config),
                "device": str(device),
                "feature_set": (
                    "degree, clustering coefficient, mean-neighbor degree, "
                    "mean-neighbor clustering"
                ),
                "prediction_rule": "author full padded Hungarian, then dummy filtering",
                "cost_matrix": "negative quasi-permutation score (ranking-equivalent)",
                "solver": "exact dense FUGAL quasi-permutation solver",
                "padded_size": padded_size,
                "ground_truth_used_during_inference": False,
            },
        )
