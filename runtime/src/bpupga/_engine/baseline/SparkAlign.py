from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Iterable

import networkx as nx
import numpy as np
import scipy.sparse as sps
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

try:
    from tqdm import trange
except ImportError:  # progress output is optional
    def trange(*args, **kwargs):
        kwargs.pop("desc", None)
        kwargs.pop("disable", None)
        return range(*args, **kwargs)

from common.result import AlignmentResult


@dataclass(frozen=True)
class SparkAlignConfig:
    dp_min_degree: int = 6
    num_dp_select: int = 10
    num_diffusion_select: int = 20
    diffusion_step: int = 5
    matching_budget: int | None = None
    nearest_candidates: int = 50
    distance_batch_size: int = 5000
    seed: int = 2026
    show_progress: bool = True
    compute_cost_matrix: bool = False
    max_dense_entries: int = 25_000_000

    def validate(self) -> None:
        if self.dp_min_degree < 0:
            raise ValueError("dp_min_degree must be non-negative")
        if self.num_dp_select < 1 or self.num_diffusion_select < 1:
            raise ValueError("pair selection counts must be positive")
        if self.diffusion_step < 0:
            raise ValueError("diffusion_step must be non-negative")
        if self.matching_budget is not None and self.matching_budget < 1:
            raise ValueError("matching_budget must be positive when provided")
        if self.nearest_candidates < 1:
            raise ValueError("nearest_candidates must be positive")
        if self.distance_batch_size < 1:
            raise ValueError("distance_batch_size must be positive")
        if self.max_dense_entries < 1:
            raise ValueError("max_dense_entries must be positive")


def _require_consecutive_nodes(graph: nx.Graph, label: str) -> None:
    expected = set(range(graph.number_of_nodes()))
    if set(graph.nodes()) != expected:
        raise ValueError(f"{label} must use consecutive integer node ids")


def _adjacency(graph: nx.Graph) -> sps.csr_matrix:
    return sps.csr_matrix(
        nx.to_scipy_sparse_array(
            graph,
            nodelist=range(graph.number_of_nodes()),
            dtype=np.float32,
            format="csr",
        )
    )


def _symmetric_normalize(adjacency: sps.csr_matrix) -> sps.csr_matrix:
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_sqrt = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0.0
    inverse_sqrt[nonzero] = np.power(degree[nonzero], -0.5)
    scale = sps.diags(inverse_sqrt, format="csr")
    return sps.csr_matrix(scale @ adjacency @ scale, dtype=np.float32)


def _plain_diffusion_features(
    normalized_adjacency: sps.csr_matrix,
    diffusion_step: int,
    output_dim: int,
) -> np.ndarray:
    n = normalized_adjacency.shape[0]
    hot = np.eye(n, dtype=np.float32)
    # Matches the official range(T + 1): H <- A_norm H + H.
    for _ in range(diffusion_step + 1):
        hot = normalized_adjacency @ hot + hot
    hot.sort(axis=1)
    return np.asarray(hot[:, -output_dim:], dtype=np.float32)


def _sorted_distances(
    source: np.ndarray,
    target: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n1, n2 = source.shape[0], target.shape[0]
    distances = np.empty((n1, n2), dtype=np.float32)
    indices = np.empty((n1, n2), dtype=np.int64)
    for start in range(0, n1, batch_size):
        stop = min(start + batch_size, n1)
        batch_distances = pairwise_distances(
            source[start:stop],
            target,
            metric="euclidean",
        ).astype(np.float32, copy=False)
        order = np.argsort(batch_distances, axis=1, kind="stable")
        distances[start:stop] = np.take_along_axis(batch_distances, order, axis=1)
        indices[start:stop] = order
    return distances, indices


def _select_pairs(
    features1: np.ndarray,
    features2: np.ndarray,
    degree1: np.ndarray,
    degree2: np.ndarray,
    num_select: int,
    config: SparkAlignConfig,
    used_sources: set[int],
    used_targets: set[int],
    initial: bool,
) -> tuple[list[tuple[int, int]], list[list[int]]]:
    if initial:
        source_nodes = np.flatnonzero(degree1 >= config.dp_min_degree)
        target_nodes = np.flatnonzero(degree2 >= config.dp_min_degree)
    else:
        source_nodes = np.asarray(
            [node for node in range(features1.shape[0]) if node not in used_sources],
            dtype=np.int64,
        )
        target_nodes = np.asarray(
            [node for node in range(features2.shape[0]) if node not in used_targets],
            dtype=np.int64,
        )

    if source_nodes.size == 0 or target_nodes.size == 0 or num_select <= 0:
        return [], []

    sorted_distance, sorted_target_local = _sorted_distances(
        features1[source_nodes],
        features2[target_nodes],
        batch_size=config.distance_batch_size,
    )
    candidate_count = min(config.nearest_candidates, target_nodes.size)
    candidate_local = sorted_target_local[:, :candidate_count]
    sparse_cost = np.full(
        (source_nodes.size, target_nodes.size),
        1e9,
        dtype=np.float32,
    )
    row_numbers = np.arange(source_nodes.size)[:, None]
    sparse_cost[row_numbers, candidate_local] = sorted_distance[:, :candidate_count]

    row_index, column_index = linear_sum_assignment(sparse_cost)
    matched_cost = sparse_cost[row_index, column_index]
    order = np.argsort(matched_cost, kind="stable")[:num_select]
    selected_rows = row_index[order]
    selected_columns = column_index[order]

    pairs = [
        (int(source_nodes[row]), int(target_nodes[column]))
        for row, column in zip(selected_rows, selected_columns)
    ]
    rankings = [
        target_nodes[sorted_target_local[row]].astype(int).tolist()
        for row in selected_rows
    ]
    return pairs, rankings


def _diffuse_anchors(
    kernel: sps.csr_matrix,
    nodes: Iterable[int],
    diffusion_step: int,
) -> np.ndarray:
    selected = list(nodes)
    hot = np.zeros((kernel.shape[0], len(selected)), dtype=np.float32)
    if selected:
        hot[np.asarray(selected, dtype=np.int64), np.arange(len(selected))] = 1.0
    for _ in range(diffusion_step):
        hot = kernel @ hot
    return np.asarray(hot, dtype=np.float32)


class SparkAlign:
    """Leakage-safe optimal SparkAlign.

    The official code derives the epoch count from ``len(alignment)``. Here the
    matching budget is graph-derived by default, so ``fit`` never receives or
    observes ground truth.
    """

    def __init__(self, config: SparkAlignConfig | None = None) -> None:
        self.config = config or SparkAlignConfig()
        self.config.validate()

    def fit(self, graph1: nx.Graph, graph2: nx.Graph) -> AlignmentResult:
        if graph1.number_of_nodes() == 0 or graph2.number_of_nodes() == 0:
            raise ValueError("SparkAlign cannot align empty graphs")
        _require_consecutive_nodes(graph1, "G1")
        _require_consecutive_nodes(graph2, "G2")

        start = time.perf_counter()
        n1, n2 = graph1.number_of_nodes(), graph2.number_of_nodes()
        maximum_budget = min(n1, n2)
        budget = self.config.matching_budget or maximum_budget
        if budget > maximum_budget:
            raise ValueError(
                f"matching_budget={budget} exceeds min(|V1|, |V2|)={maximum_budget}"
            )

        adjacency1 = _adjacency(graph1)
        adjacency2 = _adjacency(graph2)
        degree1 = np.asarray(adjacency1.sum(axis=1)).ravel()
        degree2 = np.asarray(adjacency2.sum(axis=1)).ravel()
        normalized1 = _symmetric_normalize(adjacency1)
        normalized2 = _symmetric_normalize(adjacency2)

        output_dim = maximum_budget
        initial1 = _plain_diffusion_features(
            normalized1,
            self.config.diffusion_step,
            output_dim,
        )
        initial2 = _plain_diffusion_features(
            normalized2,
            self.config.diffusion_step,
            output_dim,
        )
        kernel1 = sps.identity(n1, dtype=np.float32, format="csr") + normalized1
        kernel2 = sps.identity(n2, dtype=np.float32, format="csr") + normalized2

        predicted_pairs: list[tuple[int, int]] = []
        used_sources: set[int] = set()
        used_targets: set[int] = set()
        feature_blocks1: list[np.ndarray] = []
        feature_blocks2: list[np.ndarray] = []
        selection_history: list[dict[str, int | bool]] = []

        initial_capacity = min(
            budget,
            self.config.num_dp_select,
            int(np.count_nonzero(degree1 >= self.config.dp_min_degree)),
            int(np.count_nonzero(degree2 >= self.config.dp_min_degree)),
        )
        expected_epochs = 1
        if budget > initial_capacity:
            remaining = budget - initial_capacity
            expected_epochs += int(
                np.ceil(remaining / float(self.config.num_diffusion_select))
            )
        iterator = trange(
            expected_epochs,
            desc="SparkAlign",
            disable=not self.config.show_progress,
        )
        for epoch in iterator:
            remaining_budget = budget - len(predicted_pairs)
            if remaining_budget <= 0:
                break
            initial = epoch == 0
            requested = min(
                remaining_budget,
                self.config.num_dp_select if initial else self.config.num_diffusion_select,
            )
            if initial:
                selection_features1, selection_features2 = initial1, initial2
            else:
                selection_features1 = np.concatenate(feature_blocks1, axis=1)
                selection_features2 = np.concatenate(feature_blocks2, axis=1)

            new_pairs, _ = _select_pairs(
                selection_features1,
                selection_features2,
                degree1,
                degree2,
                requested,
                self.config,
                used_sources,
                used_targets,
                initial=initial,
            )
            if not new_pairs:
                if initial:
                    raise ValueError(
                        "No initial SparkAlign anchors satisfy dp_min_degree; "
                        "lower --dp-min-degree for this graph"
                    )
                break

            source_nodes = [source for source, _ in new_pairs]
            target_nodes = [target for _, target in new_pairs]
            feature_blocks1.append(
                _diffuse_anchors(kernel1, source_nodes, self.config.diffusion_step)
            )
            feature_blocks2.append(
                _diffuse_anchors(kernel2, target_nodes, self.config.diffusion_step)
            )
            predicted_pairs.extend(new_pairs)
            used_sources.update(source_nodes)
            used_targets.update(target_nodes)
            selection_history.append(
                {
                    "epoch": epoch + 1,
                    "initial": initial,
                    "requested_pairs": requested,
                    "selected_pairs": len(new_pairs),
                    "cumulative_pairs": len(predicted_pairs),
                }
            )

        source_embedding = np.concatenate(feature_blocks1, axis=1)
        target_embedding = np.concatenate(feature_blocks2, axis=1)
        dense_entries = n1 * n2
        cost_matrix = None
        dense_skipped = False
        if self.config.compute_cost_matrix:
            if dense_entries <= self.config.max_dense_entries:
                cost_matrix = pairwise_distances(
                    source_embedding,
                    target_embedding,
                    metric="euclidean",
                ).astype(np.float32, copy=False)
            else:
                dense_skipped = True

        runtime_sec = time.perf_counter() - start
        return AlignmentResult(
            method="SparkAlign",
            predicted_pairs=predicted_pairs,
            runtime_sec=float(runtime_sec),
            cost_matrix=cost_matrix,
            source_embeddings=source_embedding,
            target_embeddings=target_embedding,
            larger_is_better=False,
            metadata={
                "config": asdict(self.config),
                "matching_budget": budget,
                "matching_budget_source": (
                    "explicit_config" if self.config.matching_budget is not None
                    else "min_graph_order"
                ),
                "selection_history": selection_history,
                "selected_pairs": len(predicted_pairs),
                "dense_matrix_entries": dense_entries,
                "dense_cost_matrix_skipped": dense_skipped,
                "initial_feature_l2_normalization": False,
                "isolated_node_safety": "zero degree uses zero inverse sqrt",
                "prediction_rule": "iterative sparse optimal one-to-one assignment",
                "ground_truth_used_during_inference": False,
            },
        )