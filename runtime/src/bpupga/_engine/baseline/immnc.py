from __future__ import annotations

import heapq
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import svds
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree
try:
    from tqdm import trange
except ImportError:  # tqdm is only needed for cosmetic progress output.
    def trange(*args, **kwargs):
        kwargs.pop("desc", None)
        return range(*args, **kwargs)

from common.result import AlignmentResult


@dataclass(frozen=True)
class IMMNCConfig:
    k_de: int = 3
    k_nei: int = 3
    initial_anchor_ratio: float = 0.04
    anchor_growth_rate: float = 0.10
    iterations: int = 10
    degree_threshold: int = 6
    fast_select: bool = True
    embed_dim: int = 128
    netmf_window: int = 5
    negative_sampling: float = 1.0
    seed: int = 2026
    show_progress: bool = True
    compute_cost_matrix: bool = False
    max_dense_entries: int = 25_000_000

    def validate(self) -> None:
        if self.k_de < 0 or self.k_nei < 0:
            raise ValueError("k_de and k_nei must be non-negative")
        if not 0.0 < self.initial_anchor_ratio <= 1.0:
            raise ValueError("initial_anchor_ratio must be in (0, 1]")
        if self.anchor_growth_rate < 0.0:
            raise ValueError("anchor_growth_rate must be non-negative")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.degree_threshold < 0:
            raise ValueError("degree_threshold must be non-negative")
        if self.embed_dim < 1:
            raise ValueError("embed_dim must be positive")
        if self.netmf_window < 1:
            raise ValueError("netmf_window must be positive")
        if self.negative_sampling <= 0:
            raise ValueError("negative_sampling must be positive")


def _row_normalized_adjacency(graph: nx.Graph) -> sps.csr_matrix:
    adjacency = nx.to_scipy_sparse_array(
        graph,
        nodelist=range(graph.number_of_nodes()),
        dtype=np.float64,
        format="csr",
    )
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree, dtype=np.float64)
    nonzero = degree > 0
    inverse[nonzero] = 1.0 / degree[nonzero]
    return sps.diags(inverse) @ adjacency


def _netmf_matrix(
    adjacency: sps.spmatrix,
    window: int,
    negative_sampling: float,
) -> sps.csr_matrix:
    adjacency = sps.csr_matrix(adjacency, dtype=np.float64)
    n = adjacency.shape[0]
    volume = float(adjacency.sum())
    if n == 0:
        raise ValueError("Cannot embed an empty graph")
    if volume <= 0:
        return sps.csr_matrix((n, n), dtype=np.float64)

    laplacian, degree_sqrt = sps.csgraph.laplacian(
        adjacency,
        normed=True,
        return_diag=True,
    )
    inverse_degree_sqrt = np.zeros_like(degree_sqrt, dtype=np.float64)
    nonzero = degree_sqrt > 0
    inverse_degree_sqrt[nonzero] = 1.0 / degree_sqrt[nonzero]
    d_inverse = sps.diags(inverse_degree_sqrt)

    normalized = sps.identity(n, dtype=np.float64, format="csr") - laplacian
    accumulated = sps.csr_matrix((n, n), dtype=np.float64)
    power = sps.identity(n, dtype=np.float64, format="csr")
    for _ in range(window):
        power = power @ normalized
        accumulated = accumulated + power

    accumulated *= volume / float(window) / float(negative_sampling)
    matrix = d_inverse @ (d_inverse @ accumulated).T
    dense = np.asarray(matrix.toarray(), dtype=np.float64)
    dense = np.log(np.maximum(dense, 1.0))
    return sps.csr_matrix(dense)


def _safe_svd_embedding(
    proximity: sps.spmatrix,
    dimension: int,
    seed: int,
) -> np.ndarray:
    n = proximity.shape[0]
    if n == 1:
        return np.ones((1, dimension), dtype=np.float32)

    effective_dim = min(dimension, n - 1)
    if proximity.nnz == 0:
        embedding = np.eye(n, effective_dim, dtype=np.float64)
    elif n <= 256:
        dense = proximity.toarray()
        u, singular_values, _ = np.linalg.svd(dense, full_matrices=False)
        embedding = u[:, :effective_dim] * np.sqrt(
            np.maximum(singular_values[:effective_dim], 0.0)
        )[None, :]
    else:
        u, singular_values, _ = svds(
            proximity,
            k=effective_dim,
            return_singular_vectors=True,
            random_state=seed,
        )
        order = np.argsort(singular_values)[::-1]
        singular_values = singular_values[order]
        u = u[:, order]
        embedding = u * np.sqrt(np.maximum(singular_values, 0.0))[None, :]

    if effective_dim < dimension:
        padding = np.zeros((n, dimension - effective_dim), dtype=np.float64)
        embedding = np.concatenate([embedding, padding], axis=1)

    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return (embedding / norms).astype(np.float32, copy=False)


def _netmf_embedding(graph: nx.Graph, config: IMMNCConfig) -> np.ndarray:
    adjacency = nx.to_scipy_sparse_array(
        graph,
        nodelist=range(graph.number_of_nodes()),
        dtype=np.float64,
        format="csr",
    )
    proximity = _netmf_matrix(
        adjacency,
        window=config.netmf_window,
        negative_sampling=config.negative_sampling,
    )
    return _safe_svd_embedding(proximity, config.embed_dim, config.seed)


def _degree_signature_features(graph: nx.Graph, max_hop: int) -> np.ndarray:
    """CENA-style exact-hop degree distribution signatures."""
    n = graph.number_of_nodes()
    degree = np.asarray([graph.degree[node] for node in range(n)], dtype=np.float64)
    features = np.zeros((n, (max_hop + 1) * 5), dtype=np.float32)

    for source in range(n):
        distances = nx.single_source_shortest_path_length(graph, source, cutoff=max_hop)
        for hop in range(max_hop + 1):
            hop_nodes = [node for node, distance in distances.items() if distance == hop]
            if hop_nodes:
                values = degree[hop_nodes]
                summary = np.percentile(values, [100, 0, 50, 75, 25])
                summary = np.log1p(summary)
            else:
                summary = np.zeros(5, dtype=np.float64)
            start = hop * 5
            features[source, start : start + 5] = summary.astype(np.float32)
    return features


def _select_pseudo_anchors(
    graph1: nx.Graph,
    graph2: nx.Graph,
    features1: np.ndarray,
    features2: np.ndarray,
    ratio: float,
    fast_select: bool,
    degree_threshold: int,
) -> Dict[int, int]:
    n = min(graph1.number_of_nodes(), graph2.number_of_nodes())
    if n == 0:
        raise ValueError("Cannot select pseudo anchors from an empty graph")

    if fast_select:
        source_candidates = [
            node for node in graph1.nodes() if graph1.degree[node] >= degree_threshold
        ]
        target_candidates = [
            node for node in graph2.nodes() if graph2.degree[node] >= degree_threshold
        ]
        if not source_candidates:
            source_candidates = list(graph1.nodes())
        if not target_candidates:
            target_candidates = list(graph2.nodes())
    else:
        source_candidates = list(graph1.nodes())
        target_candidates = list(graph2.nodes())

    source_matrix = features1[source_candidates]
    target_matrix = features2[target_candidates]
    tree = KDTree(target_matrix, metric="euclidean")
    distances, indices = tree.query(source_matrix, k=1)

    requested = max(1, int(float(ratio) * n))
    requested = min(
        requested, 
        len(source_candidates), 
        len(target_candidates)
    )
    scores = -distances[:, 0]
    selected_local = heapq.nlargest(
        requested,
        range(len(scores)),
        scores.__getitem__,
    )
    return {
        int(source_candidates[local_source]): int(target_candidates[indices[local_source, 0]])
        for local_source in selected_local
    }


def _multiscale_aligned_embeddings(
    graph1: nx.Graph,
    graph2: nx.Graph,
    base1: np.ndarray,
    base2: np.ndarray,
    anchors: Dict[int, int],
    propagation_orders: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not anchors:
        raise ValueError("iMMNC requires at least one pseudo anchor")

    source_anchor = np.asarray(list(anchors.keys()), dtype=np.int64)
    target_anchor = np.asarray(list(anchors.values()), dtype=np.int64)
    transition1 = _row_normalized_adjacency(graph1)
    transition2 = _row_normalized_adjacency(graph2)

    scales1 = [base1]
    scales2 = [base2]
    current1, current2 = base1, base2
    for _ in range(propagation_orders):
        current1 = transition1 @ current1
        current2 = transition2 @ current2
        scales1.append(np.asarray(current1, dtype=np.float32))
        scales2.append(np.asarray(current2, dtype=np.float32))

    cross_covariance = np.zeros((base1.shape[1], base2.shape[1]), dtype=np.float64)
    for scale1, scale2 in zip(scales1, scales2):
        cross_covariance += scale1[source_anchor].T @ scale2[target_anchor]
    cross_covariance /= len(scales1)

    u, _, vh = np.linalg.svd(cross_covariance, full_matrices=False)
    rotation = u @ vh

    aligned_base1 = np.asarray(base1 @ rotation, dtype=np.float32)
    combined1 = np.concatenate(
        [np.asarray(scale @ rotation, dtype=np.float32) for scale in scales1],
        axis=1,
    )
    combined2 = np.concatenate(scales2, axis=1).astype(np.float32, copy=False)
    return aligned_base1, base2, combined1, combined2


def _nearest_neighbor_pairs(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
) -> list[tuple[int, int]]:
    tree = KDTree(target_embeddings, metric="euclidean")
    _, indices = tree.query(source_embeddings, k=1)
    return [(int(source), int(target)) for source, target in enumerate(indices[:, 0])]


class IMMNC:
    """Leakage-safe iMMNC implementation.

    ``fit`` deliberately accepts only two graphs. Ground truth is neither an
    argument nor stored on the model.
    """

    def __init__(self, config: IMMNCConfig | None = None) -> None:
        self.config = config or IMMNCConfig()
        self.config.validate()

    def fit(self, graph1: nx.Graph, graph2: nx.Graph) -> AlignmentResult:
        if graph1.number_of_nodes() == 0 or graph2.number_of_nodes() == 0:
            raise ValueError("iMMNC cannot align empty graphs")
        expected1 = set(range(graph1.number_of_nodes()))
        expected2 = set(range(graph2.number_of_nodes()))
        if set(graph1.nodes()) != expected1 or set(graph2.nodes()) != expected2:
            raise ValueError("Graphs must be reindexed to consecutive integer nodes")

        start = time.perf_counter()
        degree_features1 = _degree_signature_features(graph1, self.config.k_de)
        degree_features2 = _degree_signature_features(graph2, self.config.k_de)
        base1 = _netmf_embedding(graph1, self.config)
        base2 = _netmf_embedding(graph2, self.config)

        anchors = _select_pseudo_anchors(
            graph1,
            graph2,
            degree_features1,
            degree_features2,
            ratio=self.config.initial_anchor_ratio,
            fast_select=self.config.fast_select,
            degree_threshold=self.config.degree_threshold,
        )
        anchor_history: list[dict[str, object]] = []
        active_ratio = self.config.initial_anchor_ratio

        iterator: Iterable[int]
        if self.config.show_progress:
            iterator = trange(self.config.iterations, desc="iMMNC")
        else:
            iterator = range(self.config.iterations)

        aligned1 = base1
        aligned2 = base2
        for iteration in iterator:
            aligned1, aligned2, combined1, combined2 = _multiscale_aligned_embeddings(
                graph1,
                graph2,
                base1,
                base2,
                anchors,
                propagation_orders=self.config.k_nei,
            )
            anchor_history.append(
                {
                    "iteration": iteration + 1,
                    "requested_ratio": active_ratio,
                    "num_source_anchors": len(anchors),
                    "num_unique_target_anchors": len(set(anchors.values())),
                }
            )

            next_ratio = max(
                self.config.initial_anchor_ratio
                + self.config.anchor_growth_rate * (iteration + 1),
                1.0 if self.config.fast_select else 0.5,
            )
            active_ratio = next_ratio
            anchors = _select_pseudo_anchors(
                graph1,
                graph2,
                combined1,
                combined2,
                ratio=next_ratio,
                fast_select=self.config.fast_select,
                degree_threshold=self.config.degree_threshold,
            )

        predicted_pairs = _nearest_neighbor_pairs(aligned1, aligned2)
        runtime_sec = time.perf_counter() - start

        cost_matrix = None
        dense_entries = graph1.number_of_nodes() * graph2.number_of_nodes()
        dense_skipped = False
        if self.config.compute_cost_matrix:
            if dense_entries <= self.config.max_dense_entries:
                cost_matrix = pairwise_distances(
                    aligned1,
                    aligned2,
                    metric="euclidean",
                ).astype(np.float32, copy=False)
            else:
                dense_skipped = True

        return AlignmentResult(
            method="iMMNC",
            predicted_pairs=predicted_pairs,
            runtime_sec=float(runtime_sec),
            cost_matrix=cost_matrix,
            source_embeddings=aligned1,
            target_embeddings=aligned2,
            larger_is_better=False,
            metadata={
                "config": asdict(self.config),
                "pseudo_anchor_history": anchor_history,
                "anchor_schedule": "author_max",
                "netmf_operator": "I_minus_normalized_Laplacian",
                "boundary_guard_triggers": {
                    "source_empty_high_degree_fallback": self.config.fast_select and not any(d >= self.config.degree_threshold for _,d in graph1.degree()),
                    "target_empty_high_degree_fallback": self.config.fast_select and not any(d >= self.config.degree_threshold for _,d in graph2.degree()),
                    "source_zero_volume": graph1.number_of_edges() == 0,
                    "target_zero_volume": graph2.number_of_edges() == 0,
                    "source_small_graph_exact_svd": 1 < len(graph1) <= 256,
                    "target_small_graph_exact_svd": 1 < len(graph2) <= 256,
                    "source_single_node_constant": len(graph1) == 1,
                    "target_single_node_constant": len(graph2) == 1,
                    "source_svd_dimension_reduced": self.config.embed_dim > len(graph1)-1,
                    "target_svd_dimension_reduced": self.config.embed_dim > len(graph2)-1,
                },
                "safety": {"zero_degree_source": len(list(nx.isolates(graph1))), "zero_degree_target": len(list(nx.isolates(graph2))), "empty_high_degree_policy": "all_nodes", "small_graph_svd": "k=min(dim,n-1), exact SVD if n<=256, n=1 constant", "zero_volume_policy": "zero_proximity_then_identity_embedding", "svd_random_seed": self.config.seed},
                "dense_matrix_entries": dense_entries,
                "dense_cost_matrix_skipped": dense_skipped,
                "prediction_rule": "row-wise nearest target in Euclidean embedding space",
                "ground_truth_used_during_inference": False,
            },
        )
