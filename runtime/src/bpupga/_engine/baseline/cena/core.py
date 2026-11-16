from __future__ import annotations

import random
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sps
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression

from common.result import AlignmentResult

from .structure import structural_neighbors
from .walks import write_walk_corpus


@dataclass(frozen=True)
class CENAConfig:
    """Unsupervised CENALP configuration for BPUPGA.

    The algorithmic defaults follow the authors' implementation. Word2Vec uses
    one worker by default because multi-worker Hogwild training is nondeterministic.
    """

    q: float = 0.5
    alpha: float = 5.0
    layers: int = 3
    matching_budget: int | None = None
    max_iterations: int = 20
    minimum_new_anchors: int = 5
    anchor_growth: float = 1.5
    structural_topk: int = 10
    structure_device: str = "cpu"
    structure_block_size: int = 1024
    structure_dtype: str = "float64"
    num_walks: int = 20
    walk_length: int = 80
    multi_process_walks: bool = False
    walk_workers: int = 20
    embedding_dim: int = 64
    embedding_window: int = 5
    embedding_epochs: int = 5
    negative_samples: int = 5
    word2vec_workers: int = 1
    word2vec_batch_words: int = 50_000
    link_prediction_batch_size: int = 100_000
    rounding: str = "greedy"
    seed: int = 2026
    show_progress: bool = True
    max_dense_entries: int = 25_000_000

    def validate(self) -> None:
        if not 0.0 <= self.q <= 1.0:
            raise ValueError("q must be in [0, 1]")
        if self.alpha <= 0.0 or self.layers < 0:
            raise ValueError("alpha must be positive and layers non-negative")
        if self.matching_budget is not None and self.matching_budget < 1:
            raise ValueError("matching_budget must be positive when provided")
        if self.max_iterations < 1 or self.minimum_new_anchors < 1:
            raise ValueError("iteration and anchor counts must be positive")
        if self.anchor_growth < 1.0:
            raise ValueError("anchor_growth must be at least 1")
        if self.structural_topk < 1 or self.structure_block_size < 1:
            raise ValueError("structural_topk and structure_block_size must be positive")
        if self.structure_dtype not in {"float32", "float64"}:
            raise ValueError("structure_dtype must be float32 or float64")
        if self.num_walks < 1 or self.walk_length < 1 or self.walk_workers < 1:
            raise ValueError("random-walk parameters must be positive")
        if (
            self.embedding_dim < 1
            or self.embedding_window < 1
            or self.embedding_epochs < 1
            or self.negative_samples < 0
            or self.word2vec_workers < 1
            or self.word2vec_batch_words < 1
        ):
            raise ValueError("Word2Vec parameters are invalid")
        if self.link_prediction_batch_size < 1:
            raise ValueError("link_prediction_batch_size must be positive")
        if self.rounding not in {"greedy", "hungarian", "row_argmax"}:
            raise ValueError("rounding must be greedy, hungarian, or row_argmax")
        if self.max_dense_entries < 1:
            raise ValueError("max_dense_entries must be positive")


def _require_consecutive_nodes(graph: nx.Graph, label: str) -> None:
    if set(graph.nodes()) != set(range(graph.number_of_nodes())):
        raise ValueError(f"{label} must use consecutive integer node ids")


def _normalize_rows(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    return np.divide(
        embedding,
        norms,
        out=np.zeros_like(embedding),
        where=norms > 0.0,
    )


def _select_seed_pairs(
    scores: np.ndarray,
    source_nodes: list[int],
    target_nodes: list[int],
    count: int,
) -> list[tuple[int, int]]:
    order = np.argsort(-scores.reshape(-1), kind="stable")
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    selected: list[tuple[int, int]] = []
    target_count = len(target_nodes)
    for position in order:
        source_position, target_position = divmod(int(position), target_count)
        source = source_nodes[source_position]
        target = target_nodes[target_position]
        if source in used_sources or target in used_targets:
            continue
        selected.append((source, target))
        used_sources.add(source)
        used_targets.add(target)
        if len(selected) >= count:
            break
    return selected


def _anchor_jaccard_scores(
    graph1: nx.Graph,
    graph2: nx.Graph,
    anchors1: list[int],
    anchors2: list[int],
    source_nodes: list[int],
    target_nodes: list[int],
) -> np.ndarray:
    offset = max(graph1.number_of_nodes(), graph2.number_of_nodes())
    shared_offset = 2 * offset
    source_anchor_token = {
        source: shared_offset + index for index, source in enumerate(anchors1)
    }
    target_anchor_token = {
        target: shared_offset + index for index, target in enumerate(anchors2)
    }
    merged = nx.Graph()
    merged.add_edges_from(
        (
            source_anchor_token.get(source, source),
            source_anchor_token.get(target, target),
        )
        for source, target in graph1.edges()
    )
    merged.add_edges_from(
        (
            target_anchor_token.get(source, source + offset),
            target_anchor_token.get(target, target + offset),
        )
        for source, target in graph2.edges()
    )
    shifted_targets = [target + offset for target in target_nodes]
    merged.add_nodes_from(source_nodes)
    merged.add_nodes_from(shifted_targets)
    node_order = list(merged.nodes())
    node_position = {node: position for position, node in enumerate(node_order)}
    adjacency = sps.csr_matrix(
        nx.to_scipy_sparse_array(
            merged, nodelist=node_order, dtype=np.float32, format="csr"
        )
    )
    source_rows = adjacency[[node_position[node] for node in source_nodes]]
    target_rows = adjacency[[node_position[node] for node in shifted_targets]]
    intersections = np.asarray((source_rows @ target_rows.T).toarray(), dtype=np.float32)
    degree1 = np.asarray(source_rows.sum(axis=1), dtype=np.float32).reshape(-1, 1)
    degree2 = np.asarray(target_rows.sum(axis=1), dtype=np.float32).reshape(1, -1)
    unions = degree1 + degree2 - intersections
    return (intersections + 0.5) / (unions + 0.5)


def _sample_edge_examples(
    graph: nx.Graph,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    positive = [(int(u), int(v)) for u, v in graph.edges()]
    if not positive:
        return [], np.empty(0, dtype=np.int64)
    nodes = graph.number_of_nodes()
    maximum_non_edges = nodes * (nodes - 1) // 2 - graph.number_of_edges()
    requested = min(len(positive), maximum_non_edges)
    negative: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    maximum_attempts = max(10_000, requested * 20)
    while len(negative) < requested and attempts < maximum_attempts:
        attempts += 1
        left, right = rng.integers(nodes, size=2).tolist()
        if left == right:
            continue
        edge = (min(left, right), max(left, right))
        if edge in seen or graph.has_edge(*edge):
            continue
        seen.add(edge)
        negative.append(edge)
    if len(negative) < requested:
        for left, right in nx.non_edges(graph):
            edge = (min(int(left), int(right)), max(int(left), int(right)))
            if edge in seen:
                continue
            negative.append(edge)
            if len(negative) >= requested:
                break
    edges = positive + negative
    labels = np.concatenate(
        (np.ones(len(positive), dtype=np.int64), np.zeros(len(negative), dtype=np.int64))
    )
    return edges, labels


def _edge_features(model, edges: list[tuple[int, int]], offset: int) -> np.ndarray:
    if not edges:
        return np.empty((0, model.vector_size * 3), dtype=np.float32)
    left = np.asarray([model.wv[str(u + offset)] for u, _ in edges])
    right = np.asarray([model.wv[str(v + offset)] for _, v in edges])
    return np.concatenate((left, right, left * right), axis=1)


def _candidate_missing_links(
    graph1: nx.Graph,
    graph2: nx.Graph,
    anchors1: list[int],
    anchors2: list[int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    missing1: list[tuple[int, int]] = []
    missing2: list[tuple[int, int]] = []
    for first, second in combinations(range(len(anchors1)), 2):
        source_edge = (anchors1[first], anchors1[second])
        target_edge = (anchors2[first], anchors2[second])
        source_present = graph1.has_edge(*source_edge)
        target_present = graph2.has_edge(*target_edge)
        if target_present and not source_present:
            missing1.append((min(source_edge), max(source_edge)))
        if source_present and not target_present:
            missing2.append((min(target_edge), max(target_edge)))
    return missing1, missing2


def _predict_missing_links(
    model,
    graph1: nx.Graph,
    graph2: nx.Graph,
    anchors1: list[int],
    anchors2: list[int],
    offset: int,
    rng: np.random.Generator,
    seed: int,
    batch_size: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    training1, labels1 = _sample_edge_examples(graph1, rng)
    training2, labels2 = _sample_edge_examples(graph2, rng)
    if labels1.size == 0 or labels2.size == 0:
        return [], []
    features = np.vstack(
        (_edge_features(model, training1, 0), _edge_features(model, training2, offset))
    )
    labels = np.concatenate((labels1, labels2))
    if np.unique(labels).size < 2:
        return [], []
    classifier = LogisticRegression(solver="liblinear", random_state=seed)
    classifier.fit(features, labels)
    candidates1, candidates2 = _candidate_missing_links(
        graph1, graph2, anchors1, anchors2
    )

    def accepted(candidates: list[tuple[int, int]], token_offset: int) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            probability = classifier.predict_proba(
                _edge_features(model, batch, token_offset)
            )[:, 1]
            result.extend(edge for edge, value in zip(batch, probability) if value > 0.5)
        return result

    return accepted(candidates1, 0), accepted(candidates2, offset)


def _round_scores(scores: np.ndarray, mode: str) -> list[tuple[int, int]]:
    if mode == "row_argmax":
        return [
            (source, int(target))
            for source, target in enumerate(np.argmax(scores, axis=1))
        ]
    if mode == "hungarian":
        sources, targets = linear_sum_assignment(-scores)
        return [(int(source), int(target)) for source, target in zip(sources, targets)]

    order = np.argsort(-scores.reshape(-1), kind="stable")
    used_sources = np.zeros(scores.shape[0], dtype=bool)
    used_targets = np.zeros(scores.shape[1], dtype=bool)
    pairs: list[tuple[int, int]] = []
    for position in order:
        source, target = divmod(int(position), scores.shape[1])
        if used_sources[source] or used_targets[target]:
            continue
        pairs.append((source, target))
        used_sources[source] = True
        used_targets[target] = True
        if len(pairs) == min(scores.shape):
            break
    return pairs


def _train_word2vec(corpus_path: Path, config: CENAConfig):
    try:
        from gensim.models import Word2Vec
        from gensim.models.word2vec import LineSentence
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("CENA requires gensim for cross-graph Word2Vec") from error

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="gensim")
        return Word2Vec(
            LineSentence(str(corpus_path)),
            vector_size=config.embedding_dim,
            window=config.embedding_window,
            min_count=0,
            hs=1,
            sg=1,
            negative=config.negative_samples,
            workers=config.word2vec_workers,
            epochs=config.embedding_epochs,
            batch_words=config.word2vec_batch_words,
            # The reference implementation reuses the same Word2Vec seed in
            # every CENALP refinement round.
            seed=config.seed,
        )


class CENA:
    """Leakage-safe unsupervised CENALP exposed through the BPUPGA contract."""

    def __init__(self, config: CENAConfig | None = None) -> None:
        self.config = config or CENAConfig()
        self.config.validate()

    def fit(self, graph1: nx.Graph, graph2: nx.Graph) -> AlignmentResult:
        if graph1.number_of_nodes() == 0 or graph2.number_of_nodes() == 0:
            raise ValueError("CENA cannot align empty graphs")
        _require_consecutive_nodes(graph1, "G1")
        _require_consecutive_nodes(graph2, "G2")
        n1, n2 = graph1.number_of_nodes(), graph2.number_of_nodes()
        if n1 * n2 > self.config.max_dense_entries:
            raise ValueError(
                f"CENA requires an {n1} x {n2} similarity matrix, exceeding "
                f"max_dense_entries={self.config.max_dense_entries:,}. Increase "
                "the explicit limit; no non-CENA rounding fallback is used."
            )
        maximum_budget = min(n1, n2)
        matching_budget = self.config.matching_budget or maximum_budget
        if matching_budget > maximum_budget:
            raise ValueError(
                f"matching_budget={matching_budget} exceeds min(|V1|, |V2|)={maximum_budget}"
            )

        start = time.perf_counter()
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        rng = np.random.default_rng(self.config.seed)
        working1, working2 = graph1.copy(), graph2.copy()
        anchors1: list[int] = []
        anchors2: list[int] = []
        last_added_links = 1
        total_added_links = 0
        structure_cache = None
        model = None
        completed_iterations = 0
        offset = max(n1, n2)

        with tempfile.TemporaryDirectory(prefix="bpupga-cena-") as temporary:
            corpus_path = Path(temporary) / "random_walks.txt"
            for iteration in range(1, self.config.max_iterations + 1):
                completed_iterations = iteration
                if structure_cache is None or last_added_links > 0:
                    structure_cache = structural_neighbors(
                        working1,
                        working2,
                        layers=self.config.layers,
                        alpha=self.config.alpha,
                        topk=self.config.structural_topk,
                        block_size=self.config.structure_block_size,
                        dtype_name=self.config.structure_dtype,
                        device_name=self.config.structure_device,
                    )
                neighbors1, probabilities1, neighbors2, probabilities2 = structure_cache
                write_walk_corpus(
                    corpus_path,
                    working1,
                    working2,
                    within_graph_probability=self.config.q,
                    neighbors1=neighbors1,
                    probabilities1=probabilities1,
                    neighbors2=neighbors2,
                    probabilities2=probabilities2,
                    anchors1=anchors1,
                    anchors2=anchors2,
                    num_walks=self.config.num_walks,
                    walk_length=self.config.walk_length,
                    seed=self.config.seed + iteration,
                    multi_process=self.config.multi_process_walks,
                    workers=self.config.walk_workers,
                    show_progress=self.config.show_progress,
                )
                model = _train_word2vec(corpus_path, self.config)

                source_remaining = sorted(set(range(n1)) - set(anchors1))
                target_remaining = sorted(set(range(n2)) - set(anchors2))
                if (
                    len(anchors1) >= matching_budget
                    or not source_remaining
                    or not target_remaining
                    or iteration == self.config.max_iterations
                ):
                    break

                embedding1 = _normalize_rows(
                    np.asarray([model.wv[str(node)] for node in source_remaining])
                )
                embedding2 = _normalize_rows(
                    np.asarray([model.wv[str(node + offset)] for node in target_remaining])
                )
                pair_scores = np.maximum(embedding1 @ embedding2.T, 0.0)
                if anchors1:
                    pair_scores *= _anchor_jaccard_scores(
                        working1,
                        working2,
                        anchors1,
                        anchors2,
                        source_remaining,
                        target_remaining,
                    )
                requested = max(
                    self.config.minimum_new_anchors,
                    int(
                        matching_budget
                        / 100.0
                        * self.config.anchor_growth**iteration
                    ),
                )
                requested = min(
                    requested,
                    matching_budget - len(anchors1),
                    len(source_remaining),
                    len(target_remaining),
                )
                new_pairs = _select_seed_pairs(
                    pair_scores, source_remaining, target_remaining, requested
                )
                if not new_pairs:
                    break
                anchors1.extend(source for source, _ in new_pairs)
                anchors2.extend(target for _, target in new_pairs)
                predicted1, predicted2 = _predict_missing_links(
                    model,
                    working1,
                    working2,
                    anchors1,
                    anchors2,
                    offset,
                    rng,
                    self.config.seed,
                    self.config.link_prediction_batch_size,
                )
                working1.add_edges_from(predicted1)
                working2.add_edges_from(predicted2)
                last_added_links = len(predicted1) + len(predicted2)
                total_added_links += last_added_links

        if model is None:  # defensive; the loop always trains at least once
            raise RuntimeError("CENA did not produce an embedding model")
        source_embedding = _normalize_rows(
            np.asarray([model.wv[str(node)] for node in range(n1)])
        )
        target_embedding = _normalize_rows(
            np.asarray([model.wv[str(node + offset)] for node in range(n2)])
        )
        score_matrix = source_embedding @ target_embedding.T
        predicted_pairs = _round_scores(score_matrix, self.config.rounding)
        return AlignmentResult(
            method="CENA",
            predicted_pairs=predicted_pairs,
            runtime_sec=float(time.perf_counter() - start),
            cost_matrix=-score_matrix.astype(np.float32, copy=False),
            source_embeddings=source_embedding,
            target_embeddings=target_embedding,
            larger_is_better=False,
            metadata={
                "config": asdict(self.config),
                "algorithm_core": "CENALP",
                "iterations": completed_iterations,
                "matching_budget": matching_budget,
                "pseudo_anchors_selected": len(anchors1),
                "predicted_links_added": total_added_links,
                "prediction_rule": self.config.rounding,
                "cost_matrix": "negative cosine similarity (ranking-equivalent)",
                "word2vec_reproducible": self.config.word2vec_workers == 1,
                "ground_truth_used_during_inference": False,
            },
        )
