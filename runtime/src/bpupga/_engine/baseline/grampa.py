from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from common.result import AlignmentResult


@dataclass(frozen=True)
class GRAMPAConfig:
    """Configuration for exact full-spectrum GRAMPA."""

    eta: float = 1.0
    center_adjacency: bool = False
    dtype: str = "float64"
    show_progress: bool = True
    max_dense_entries: int = 25_000_000

    def validate(self) -> None:
        if self.eta <= 0.0:
            raise ValueError("eta must be positive")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if self.max_dense_entries < 1:
            raise ValueError("max_dense_entries must be positive")


def _require_consecutive_nodes(graph: nx.Graph, label: str) -> None:
    if set(graph.nodes()) != set(range(graph.number_of_nodes())):
        raise ValueError(f"{label} must use consecutive integer node ids")


def _adjacency(
    graph: nx.Graph,
    dtype: np.dtype,
    center: bool,
) -> np.ndarray:
    matrix = nx.to_numpy_array(
        graph,
        nodelist=range(graph.number_of_nodes()),
        dtype=dtype,
        weight=None,
    )
    if center and matrix.shape[0] > 1:
        off_diagonal = ~np.eye(matrix.shape[0], dtype=bool)
        matrix[off_diagonal] -= matrix[off_diagonal].mean()
    return matrix


def _similarity_matrix(
    adjacency1: np.ndarray,
    adjacency2: np.ndarray,
    eta: float,
) -> np.ndarray:
    eigenvalues1, eigenvectors1 = np.linalg.eigh(adjacency1)
    eigenvalues2, eigenvectors2 = np.linalg.eigh(adjacency2)

    # Algebraically identical to U' * ones(n1, n2) * V, but avoids
    # materializing a second rectangular dense matrix.
    left_projection = eigenvectors1.T @ np.ones(adjacency1.shape[0])
    right_projection = eigenvectors2.T @ np.ones(adjacency2.shape[0])
    cauchy = 1.0 / (
        (eigenvalues1[:, None] - eigenvalues2[None, :]) ** 2 + eta**2
    )
    coefficients = cauchy * (
        left_projection[:, None] * right_projection[None, :]
    )
    return eigenvectors1 @ coefficients @ eigenvectors2.T


class GRAMPA:
    """Leakage-safe GRAph Matching by Pairwise eigen-Alignments."""

    def __init__(self, config: GRAMPAConfig | None = None) -> None:
        self.config = config or GRAMPAConfig()
        self.config.validate()

    def fit(self, graph1: nx.Graph, graph2: nx.Graph) -> AlignmentResult:
        if graph1.number_of_nodes() == 0 or graph2.number_of_nodes() == 0:
            raise ValueError("GRAMPA cannot align empty graphs")
        _require_consecutive_nodes(graph1, "G1")
        _require_consecutive_nodes(graph2, "G2")

        n1, n2 = graph1.number_of_nodes(), graph2.number_of_nodes()
        dense_entries = max(n1 * n1, n2 * n2, n1 * n2)
        if dense_entries > self.config.max_dense_entries:
            raise ValueError(
                "Exact GRAMPA requires "
                f"{dense_entries:,} dense matrix entries, exceeding "
                f"max_dense_entries={self.config.max_dense_entries:,}. Increase "
                "the explicit limit; no low-rank approximation is substituted."
            )

        start = time.perf_counter()
        dtype = np.float32 if self.config.dtype == "float32" else np.float64
        adjacency1 = _adjacency(
            graph1, dtype=dtype, center=self.config.center_adjacency
        )
        adjacency2 = _adjacency(
            graph2, dtype=dtype, center=self.config.center_adjacency
        )
        score_matrix = np.asarray(
            _similarity_matrix(adjacency1, adjacency2, self.config.eta),
            dtype=dtype,
        )

        source_indices, target_indices = linear_sum_assignment(-score_matrix)
        predicted_pairs = [
            (int(source), int(target))
            for source, target in zip(source_indices, target_indices)
        ]
        return AlignmentResult(
            method="GRAMPA",
            predicted_pairs=predicted_pairs,
            runtime_sec=float(time.perf_counter() - start),
            cost_matrix=-score_matrix,
            larger_is_better=False,
            metadata={
                "config": asdict(self.config),
                "spectrum": "complete symmetric eigendecomposition",
                "kernel": "1 / ((lambda_i - mu_j)^2 + eta^2)",
                "prediction_rule": "maximum-weight linear assignment",
                "cost_matrix": "negative GRAMPA similarity (ranking-equivalent)",
                "solver": "exact full-spectrum GRAMPA; no low-rank fallback",
                "ground_truth_used_during_inference": False,
            },
        )
