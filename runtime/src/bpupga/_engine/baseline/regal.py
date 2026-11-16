from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree

from common.result import AlignmentResult


@dataclass(frozen=True)
class REGALConfig:
    """Configuration matching the official unattributed REGAL CLI defaults."""

    max_layer: int = 2
    alpha: float = 0.01
    buckets: float = 2.0
    landmark_multiplier: int = 10
    gamma_structure: float = 1.0
    gamma_attribute: float = 1.0
    landmark_seed: int = 42
    max_dense_entries: int = 25_000_000
    show_progress: bool = True

    def validate(self) -> None:
        if self.max_layer < 0:
            raise ValueError("max_layer must be non-negative; use 0 for all layers")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.buckets < 1.0:
            raise ValueError("buckets must be at least 1; use 1 to disable binning")
        if self.landmark_multiplier < 1:
            raise ValueError("landmark_multiplier must be positive")
        if self.gamma_structure < 0.0 or self.gamma_attribute < 0.0:
            raise ValueError("gamma weights must be non-negative")
        if self.max_dense_entries < 1:
            raise ValueError("max_dense_entries must be positive")


def _require_consecutive_nodes(graph: nx.Graph, label: str) -> None:
    expected = set(range(graph.number_of_nodes()))
    if set(graph.nodes()) != expected:
        raise ValueError(f"{label} must use consecutive integer node ids")


def _degree_bucket_count(max_degree: int, buckets: float) -> int:
    if buckets == 1.0:
        return max_degree + 1
    if max_degree <= 0:
        return 1
    return int(math.log(max_degree, buckets)) + 1


def _degree_bucket(degree: int, buckets: float) -> int | None:
    if buckets == 1.0:
        return degree
    if degree <= 0:
        # This is how the reference implementation treats degree-zero nodes:
        # log(0) cannot be assigned to a logarithmic bucket.
        return None
    return int(math.log(degree, buckets))


def _structural_features(
    graph: nx.Graph,
    max_layer: int,
    alpha: float,
    buckets: float,
) -> np.ndarray:
    """Compute xNetMF degree-distribution identity features.

    The reference code explicitly materializes exact-hop neighborhoods. A BFS
    produces the same layer membership while avoiding a large nested set cache.
    """

    n = graph.number_of_nodes()
    degrees = np.fromiter((graph.degree[node] for node in range(n)), dtype=np.int64)
    max_degree = int(degrees.max(initial=0))
    features = np.zeros(
        (n, _degree_bucket_count(max_degree, buckets)),
        dtype=np.float64,
    )
    cutoff = None if max_layer == 0 else max_layer

    for source in range(n):
        distances = nx.single_source_shortest_path_length(
            graph,
            source,
            cutoff=cutoff,
        )
        for node, layer in distances.items():
            bucket = _degree_bucket(int(degrees[node]), buckets)
            if bucket is not None:
                features[source, bucket] += alpha ** int(layer)
    return features


def _xnetmf_embedding(graph: nx.Graph, config: REGALConfig) -> tuple[np.ndarray, int]:
    features = _structural_features(
        graph,
        max_layer=config.max_layer,
        alpha=config.alpha,
        buckets=config.buckets,
    )
    n = graph.number_of_nodes()
    landmark_count = min(
        n,
        int(config.landmark_multiplier * math.log(n, 2)) if n > 1 else 1,
    )
    rng = np.random.RandomState(config.landmark_seed)
    landmarks = rng.permutation(np.arange(n))[:landmark_count]

    distances = pairwise_distances(features, features[landmarks], metric="euclidean")
    similarities = np.exp(-config.gamma_structure * distances)

    landmark_pseudoinverse = np.linalg.pinv(similarities[landmarks])
    left, singular_values, _ = np.linalg.svd(
        landmark_pseudoinverse,
        full_matrices=False,
    )
    factor = left * np.sqrt(np.maximum(singular_values, 0.0))[None, :]
    embedding = similarities @ factor

    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    # The official code divides directly. Keeping zero rows at zero prevents
    # isolated nodes from poisoning every downstream distance with NaN.
    np.divide(embedding, norms, out=embedding, where=norms > 0.0)
    return embedding.astype(np.float32, copy=False), landmark_count


def _rowwise_nearest_pairs(
    source_embedding: np.ndarray,
    target_embedding: np.ndarray,
) -> list[tuple[int, int]]:
    tree = KDTree(target_embedding, metric="euclidean")
    _, indices = tree.query(source_embedding, k=1)
    return [
        (int(source), int(target))
        for source, target in enumerate(indices[:, 0])
    ]


class REGAL:
    """Leakage-safe REGAL/xNetMF for unattributed graph alignment."""

    def __init__(self, config: REGALConfig | None = None) -> None:
        self.config = config or REGALConfig()
        self.config.validate()

    def fit(self, graph1: nx.Graph, graph2: nx.Graph) -> AlignmentResult:
        if graph1.number_of_nodes() == 0 or graph2.number_of_nodes() == 0:
            raise ValueError("REGAL cannot align empty graphs")
        _require_consecutive_nodes(graph1, "G1")
        _require_consecutive_nodes(graph2, "G2")

        start = time.perf_counter()
        n1 = graph1.number_of_nodes()
        n2 = graph2.number_of_nodes()
        combined = nx.disjoint_union(graph1, graph2)
        embedding, landmark_count = _xnetmf_embedding(combined, self.config)
        source_embedding = embedding[:n1]
        target_embedding = embedding[n1 : n1 + n2]

        dense_entries = n1 * n2
        score_matrix = None
        if dense_entries <= self.config.max_dense_entries:
            distances = pairwise_distances(
                source_embedding,
                target_embedding,
                metric="euclidean",
            )
            score_matrix = np.exp(-distances).astype(np.float32, copy=False)
            predicted_pairs = [
                (int(source), int(target))
                for source, target in enumerate(np.argmax(score_matrix, axis=1))
            ]
        else:
            predicted_pairs = _rowwise_nearest_pairs(
                source_embedding,
                target_embedding,
            )

        runtime_sec = time.perf_counter() - start
        return AlignmentResult(
            method="REGAL",
            predicted_pairs=predicted_pairs,
            runtime_sec=float(runtime_sec),
            score_matrix=score_matrix,
            source_embeddings=source_embedding,
            target_embeddings=target_embedding,
            larger_is_better=True,
            metadata={
                "config": asdict(self.config),
                "landmark_count": landmark_count,
                "embedding_dimension": int(embedding.shape[1]),
                "dense_matrix_entries": dense_entries,
                "dense_score_matrix_skipped": score_matrix is None,
                "prediction_rule": "row-wise maximum exp(-Euclidean distance)",
                "isolated_node_safety": "zero xNetMF rows remain zero instead of NaN",
                "ground_truth_used_during_inference": False,
            },
        )