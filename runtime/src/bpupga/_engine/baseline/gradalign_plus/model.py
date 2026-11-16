"""Full GradAlign+ reproduction with an unknown-overlap plain-graph adapter.

Implements CNFA (including the paper's automatic centrality selection),
layer-wise GIN reconstruction, multi-layer embedding similarity, and gradual
alignment with ACN^p reinforcement. Ground truth and true overlap cardinality
are never accepted by this module.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sps
import torch
import torch.nn as nn
import torch.nn.functional as functional

from common.result import AlignmentResult


PAPER_CENTRALITIES = (
    "degree",
    "betweenness",
    "closeness",
    "eigenvector",
    "katz",
    "pagerank",
)


@dataclass(frozen=True)
class GradAlignPlusConfig:
    epochs: int = 20
    layers: int = 2
    hidden_dim: int = 100
    learning_rate: float = 0.005
    centrality: str = "auto"
    centrality_candidates: tuple[str, ...] = PAPER_CENTRALITIES
    centrality_bins: int = 15
    centrality_gamma: float = 1.0
    augmentation_weight: float = 0.3
    acn_exponent: float = 1.5
    gradual_iterations: int = 10
    use_frontier_candidates: bool = True
    score_block_size: int = 1024
    dtype: str = "float32"
    device: str = "auto"
    seed: int = 2026
    show_progress: bool = True
    max_dense_entries: int = 25_000_000

    def validate(self) -> None:
        if self.epochs < 0:
            raise ValueError("epochs must be non-negative")
        if self.layers < 1 or self.hidden_dim < 1:
            raise ValueError("layers and hidden_dim must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.centrality != "auto" and self.centrality not in PAPER_CENTRALITIES:
            raise ValueError("unsupported centrality")
        if not self.centrality_candidates:
            raise ValueError("centrality_candidates cannot be empty")
        if any(name not in PAPER_CENTRALITIES for name in self.centrality_candidates):
            raise ValueError("centrality_candidates contains an unsupported value")
        if self.centrality_bins < 2:
            raise ValueError("centrality_bins must be at least two")
        if self.centrality_gamma < 0:
            raise ValueError("centrality_gamma must be non-negative")
        if self.augmentation_weight < 0:
            raise ValueError("augmentation_weight must be non-negative")
        if self.acn_exponent <= 0:
            raise ValueError("acn_exponent must be positive")
        if self.gradual_iterations < 1:
            raise ValueError("gradual_iterations must be positive")
        if self.score_block_size < 1 or self.max_dense_entries < 1:
            raise ValueError("matrix limits must be positive")
        if self.dtype != "float32":
            raise ValueError("GradAlign+ reproduction currently uses float32")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        value = "cuda:0"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} was requested but CUDA is unavailable")
    return torch.device(value)


def _adjacency(graph: nx.Graph) -> sps.csr_matrix:
    return sps.csr_matrix(
        nx.to_scipy_sparse_array(
            graph,
            nodelist=range(graph.number_of_nodes()),
            dtype=np.float32,
            format="csr",
        )
    )


def _centrality(graph: nx.Graph, name: str) -> tuple[np.ndarray, str | None]:
    """Calculate one of the six centralities used in GradAlign+."""

    nodes = range(graph.number_of_nodes())
    fallback = None
    if name == "degree":
        values = dict(graph.degree())
    elif name == "betweenness":
        values = nx.betweenness_centrality(graph, normalized=True)
    elif name == "closeness":
        values = nx.closeness_centrality(graph)
    elif name == "eigenvector":
        try:
            values = nx.eigenvector_centrality(graph, max_iter=500, tol=1e-8)
        except nx.PowerIterationFailedConvergence:
            values = nx.eigenvector_centrality_numpy(graph)
            fallback = "eigenvector_numpy"
    elif name == "katz":
        try:
            values = nx.katz_centrality(
                graph,
                alpha=0.01,
                beta=1.0,
                max_iter=1000,
                tol=1e-8,
                normalized=False,
            )
        except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
            maximum_degree = max((degree for _, degree in graph.degree()), default=0)
            safe_alpha = min(0.01, 0.9 / max(maximum_degree, 1))
            values = nx.katz_centrality(
                graph,
                alpha=safe_alpha,
                beta=1.0,
                max_iter=2000,
                tol=1e-8,
                normalized=False,
            )
            fallback = f"katz_safe_alpha={safe_alpha:.8g}"
    elif name == "pagerank":
        values = nx.pagerank(graph, alpha=0.85, max_iter=100, tol=1e-8)
    else:
        raise ValueError(f"unknown centrality {name!r}")
    array = np.asarray([values[node] for node in nodes], dtype=np.float64)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0), fallback


def _distribution_score(
    source: np.ndarray,
    target: np.ndarray,
    bins: int,
    gamma: float,
) -> tuple[float, float, float]:
    """Equation (7): exp(var(ps)+var(pt)-gamma*KL(ps||pt)-1)."""

    low = float(min(np.min(source), np.min(target)))
    high = float(max(np.max(source), np.max(target)))
    if high <= low:
        return math.exp(-1.0), 0.0, 0.0
    source_hist, _ = np.histogram(source, bins=bins, range=(low, high))
    target_hist, _ = np.histogram(target, bins=bins, range=(low, high))
    epsilon = 1e-12
    source_pdf = (source_hist.astype(np.float64) + epsilon)
    target_pdf = (target_hist.astype(np.float64) + epsilon)
    source_pdf /= source_pdf.sum()
    target_pdf /= target_pdf.sum()
    divergence = float(np.sum(source_pdf * np.log(source_pdf / target_pdf)))
    variance = float(np.var(source_pdf) + np.var(target_pdf))
    log_score = variance - gamma * divergence - 1.0
    return float(np.exp(np.clip(log_score, -700, 700))), divergence, variance


def _equal_width_features(
    source: np.ndarray,
    target: np.ndarray,
    dimensions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Equation (1): joint equal-width one-hot encoding."""

    low = float(min(np.min(source), np.min(target)))
    high = float(max(np.max(source), np.max(target)))
    if high <= low:
        source_bins = np.zeros(source.shape[0], dtype=np.int64)
        target_bins = np.zeros(target.shape[0], dtype=np.int64)
    else:
        width = (high - low) / dimensions
        source_bins = np.ceil((source - low) / width).astype(np.int64) - 1
        target_bins = np.ceil((target - low) / width).astype(np.int64) - 1
        source_bins = np.clip(source_bins, 0, dimensions - 1)
        target_bins = np.clip(target_bins, 0, dimensions - 1)
    identity = np.eye(dimensions, dtype=np.float32)
    return identity[source_bins], identity[target_bins]


def _cnfa(
    graph1: nx.Graph,
    graph2: nx.Graph,
    config: GradAlignPlusConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    names = (
        config.centrality_candidates
        if config.centrality == "auto"
        else (config.centrality,)
    )
    records: dict[str, dict[str, object]] = {}
    values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in names:
        source, source_fallback = _centrality(graph1, name)
        target, target_fallback = _centrality(graph2, name)
        score, divergence, variance = _distribution_score(
            source, target, config.centrality_bins, config.centrality_gamma
        )
        values[name] = (source, target)
        records[name] = {
            "selection_score": score,
            "kl_source_to_target": divergence,
            "distribution_variance_sum": variance,
            "source_fallback": source_fallback,
            "target_fallback": target_fallback,
        }
    selected = max(names, key=lambda name: float(records[name]["selection_score"]))
    features1, features2 = _equal_width_features(
        *values[selected], config.centrality_bins
    )
    return features1, features2, {
        "selected_centrality": selected,
        "candidate_statistics": records,
        "selection_equation": "exp(var(ps)+var(pt)-gamma*KL(ps||pt)-1)",
        "binning": "joint equal-width one-hot",
    }


def _directed_edges(graph: nx.Graph, device: torch.device) -> torch.Tensor:
    edges: list[tuple[int, int]] = []
    for source, target in graph.edges():
        edges.append((int(source), int(target)))
        if source != target:
            edges.append((int(target), int(source)))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.as_tensor(edges, dtype=torch.long, device=device).T.contiguous()


class _GINEncoder(nn.Module):
    """GIN backbone matching the authors' released research implementation."""

    def __init__(self, input_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.mlps = nn.ModuleList()
        for layer in range(layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                )
            )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, features: torch.Tensor, edges: torch.Tensor) -> list[torch.Tensor]:
        hidden = features
        outputs: list[torch.Tensor] = []
        for mlp in self.mlps:
            aggregate = torch.zeros_like(hidden)
            if edges.numel():
                aggregate.index_add_(0, edges[1], hidden[edges[0]])
            hidden = torch.tanh(mlp(hidden + aggregate))
            outputs.append(hidden)
        return outputs


def _normalized_power_targets(
    adjacency: sps.csr_matrix,
    layers: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Build D_l^-1/2 sum_{k=1}^l (A+I)^k D_l^-1/2 from Eq. (8)."""

    n = adjacency.shape[0]
    base = (adjacency + sps.eye(n, dtype=np.float32, format="csr")).tocsr()
    power = base.copy()
    cumulative = sps.csr_matrix(base.shape, dtype=np.float32)
    targets: list[torch.Tensor] = []
    for layer in range(layers):
        cumulative = (cumulative + power).tocsr()
        degree = np.asarray(cumulative.sum(axis=1)).reshape(-1)
        inverse = np.zeros_like(degree, dtype=np.float32)
        positive = degree > 0
        inverse[positive] = np.power(degree[positive], -0.5)
        normalized = sps.diags(inverse) @ cumulative @ sps.diags(inverse)
        coo = normalized.tocoo()
        indices = torch.as_tensor(
            np.vstack([coo.row, coo.col]), dtype=torch.long, device=device
        )
        values = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
        targets.append(
            torch.sparse_coo_tensor(
                indices, values, size=coo.shape, device=device
            ).coalesce()
        )
        if layer + 1 < layers:
            power = (power @ base).tocsr()
            power.eliminate_zeros()
    return targets


def _frobenius_reconstruction(
    hidden: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Exact ||target - H H^T||_F without materializing an n-by-n prediction."""

    gram = hidden.T @ hidden
    predicted_squared = torch.sum(gram.square())
    target_squared = torch.sum(target.values().square())
    if hidden.is_cuda and torch.are_deterministic_algorithms_enabled():
        # CUDA sparse.mm was observed to vary in both forward and backward
        # despite the global deterministic flag (PyTorch 2.10 / CUDA 12.8).
        # Sparse-dense bmm has a documented deterministic implementation.  A
        # singleton batch computes the identical T @ H without densifying T.
        target_hidden = torch.bmm(
            target.unsqueeze(0), hidden.unsqueeze(0)
        ).squeeze(0)
    else:
        target_hidden = torch.sparse.mm(target, hidden)
    cross = torch.sum(hidden * target_hidden)
    squared = torch.clamp(target_squared + predicted_squared - 2 * cross, min=0)
    return torch.sqrt(squared + torch.finfo(hidden.dtype).eps)


def _train_encoder(
    graph1: nx.Graph,
    graph2: nx.Graph,
    features1: np.ndarray,
    features2: np.ndarray,
    config: GradAlignPlusConfig,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float]]:
    encoder = _GINEncoder(
        features1.shape[1], config.hidden_dim, config.layers
    ).to(device)
    source = torch.as_tensor(features1, dtype=torch.float32, device=device)
    target = torch.as_tensor(features2, dtype=torch.float32, device=device)
    edges1 = _directed_edges(graph1, device)
    edges2 = _directed_edges(graph2, device)
    targets1 = _normalized_power_targets(_adjacency(graph1), config.layers, device)
    targets2 = _normalized_power_targets(_adjacency(graph2), config.layers, device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    encoder.train()
    for _ in range(config.epochs):
        embeddings1 = encoder(source, edges1)
        embeddings2 = encoder(target, edges2)
        loss = sum(
            _frobenius_reconstruction(hidden, expected)
            for hidden, expected in zip(embeddings1, targets1)
        ) + sum(
            _frobenius_reconstruction(hidden, expected)
            for hidden, expected in zip(embeddings2, targets2)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    encoder.eval()
    with torch.no_grad():
        return encoder(source, edges1), encoder(target, edges2), losses


def _embedding_similarity(
    source: Sequence[torch.Tensor],
    target: Sequence[torch.Tensor],
    block_size: int,
) -> np.ndarray:
    """Multi-layer cosine similarity used by the released GradAlign+ code."""

    normalized1 = [functional.normalize(layer, p=2, dim=1) for layer in source]
    normalized2 = [functional.normalize(layer, p=2, dim=1) for layer in target]
    matrix = np.zeros(
        (source[0].shape[0], target[0].shape[0]), dtype=np.float32
    )
    for start in range(0, matrix.shape[0], block_size):
        stop = min(matrix.shape[0], start + block_size)
        block = sum(
            layer1[start:stop] @ layer2.T
            for layer1, layer2 in zip(normalized1, normalized2)
        )
        matrix[start:stop] = block.detach().cpu().numpy()
    return matrix


def _greedy_top_pairs(
    scores: np.ndarray,
    source_nodes: np.ndarray,
    target_nodes: np.ndarray,
    limit: int,
) -> list[tuple[int, int]]:
    """Exact score-ordered one-to-one greedy selection using adaptive top-k."""

    if not limit or scores.size == 0:
        return []
    limit = min(limit, len(source_nodes), len(target_nodes))
    flat = scores.reshape(-1)
    pool = min(flat.size, max(1024, 8 * limit))
    while True:
        if pool == flat.size:
            candidates = np.argsort(-flat, kind="stable")
        else:
            candidates = np.argpartition(-flat, pool - 1)[:pool]
            candidates = candidates[np.argsort(-flat[candidates], kind="stable")]
        used_source: set[int] = set()
        used_target: set[int] = set()
        selected: list[tuple[int, int]] = []
        width = scores.shape[1]
        for index in candidates:
            row, column = divmod(int(index), width)
            source, target = int(source_nodes[row]), int(target_nodes[column])
            if source in used_source or target in used_target:
                continue
            selected.append((source, target))
            used_source.add(source)
            used_target.add(target)
            if len(selected) == limit:
                return selected
        if pool == flat.size:
            return selected
        pool = min(flat.size, pool * 2)


def _acn_counts(
    adjacency1: sps.csr_matrix,
    adjacency2: sps.csr_matrix,
    matched1: Sequence[int],
    matched2: Sequence[int],
    source_nodes: np.ndarray,
    target_nodes: np.ndarray,
) -> np.ndarray:
    """Count aligned cross-network neighbor pairs exactly."""

    if not matched1:
        return np.zeros((len(source_nodes), len(target_nodes)), dtype=np.float32)
    left = adjacency1[source_nodes][:, np.asarray(matched1, dtype=np.int64)]
    right = adjacency2[target_nodes][:, np.asarray(matched2, dtype=np.int64)]
    return np.asarray((left @ right.T).toarray(), dtype=np.float32)


def _frontier(
    graph: nx.Graph,
    matched: Sequence[int],
    unavailable: set[int],
) -> np.ndarray:
    candidates: set[int] = set()
    for node in matched:
        candidates.update(int(neighbor) for neighbor in graph.neighbors(node))
    candidates.difference_update(unavailable)
    return np.asarray(sorted(candidates), dtype=np.int64)


def _gradual_alignment(
    graph1: nx.Graph,
    graph2: nx.Graph,
    embedding_score: np.ndarray,
    config: GradAlignPlusConfig,
) -> tuple[list[tuple[int, int]], np.ndarray, list[dict[str, object]]]:
    """Equations (5)-(6) and Algorithm 1, with a GT-free match budget."""

    n1, n2 = embedding_score.shape
    adjacency1, adjacency2 = _adjacency(graph1), _adjacency(graph2)
    budget = min(n1, n2)
    batch_size = max(1, math.ceil(budget / config.gradual_iterations))
    matched1: list[int] = []
    matched2: list[int] = []
    unavailable1: set[int] = set()
    unavailable2: set[int] = set()
    history: list[dict[str, object]] = []
    final_score = embedding_score.copy()

    for iteration in range(config.gradual_iterations):
        if len(matched1) >= budget:
            break
        remaining1 = np.asarray(
            [node for node in range(n1) if node not in unavailable1],
            dtype=np.int64,
        )
        remaining2 = np.asarray(
            [node for node in range(n2) if node not in unavailable2],
            dtype=np.int64,
        )
        if not len(remaining1) or not len(remaining2):
            break
        if matched1 and config.use_frontier_candidates:
            candidates1 = _frontier(graph1, matched1, unavailable1)
            candidates2 = _frontier(graph2, matched2, unavailable2)
            if not len(candidates1) or not len(candidates2):
                candidates1, candidates2 = remaining1, remaining2
        else:
            candidates1, candidates2 = remaining1, remaining2

        base = embedding_score[np.ix_(candidates1, candidates2)]
        if matched1:
            counts = _acn_counts(
                adjacency1,
                adjacency2,
                matched1,
                matched2,
                candidates1,
                candidates2,
            )
            reinforced = base * np.power(counts, config.acn_exponent)
            # Disconnected components have no ACN evidence. Continuing with
            # S_emb is the GT-free completion of Algorithm 1 for that case.
            local_score = reinforced if np.any(counts > 0) else base
        else:
            counts = np.zeros_like(base, dtype=np.float32)
            local_score = base

        take = min(batch_size, budget - len(matched1))
        selected = _greedy_top_pairs(
            local_score, candidates1, candidates2, take
        )
        if not selected and (
            len(candidates1) != len(remaining1)
            or len(candidates2) != len(remaining2)
        ):
            base = embedding_score[np.ix_(remaining1, remaining2)]
            counts = _acn_counts(
                adjacency1,
                adjacency2,
                matched1,
                matched2,
                remaining1,
                remaining2,
            )
            local_score = (
                base * np.power(counts, config.acn_exponent)
                if np.any(counts > 0)
                else base
            )
            selected = _greedy_top_pairs(
                local_score, remaining1, remaining2, take
            )
        if not selected:
            break
        for source, target in selected:
            matched1.append(source)
            matched2.append(target)
            unavailable1.add(source)
            unavailable2.add(target)
        history.append(
            {
                "iteration": iteration + 1,
                "selected_pairs": len(selected),
                "cumulative_pairs": len(matched1),
                "candidate_sources": len(candidates1),
                "candidate_targets": len(candidates2),
                "maximum_acn_count": float(counts.max(initial=0.0)),
            }
        )

    if matched1:
        all_source = np.arange(n1, dtype=np.int64)
        all_target = np.arange(n2, dtype=np.int64)
        counts = _acn_counts(
            adjacency1,
            adjacency2,
            matched1,
            matched2,
            all_source,
            all_target,
        )
        final_score = embedding_score * np.power(counts, config.acn_exponent)
        if not np.any(counts > 0):
            final_score = embedding_score.copy()
    return list(zip(matched1, matched2)), final_score.astype(np.float32), history


class GradAlignPlus:
    """GradAlign+ with paper-faithful core and partial-graph output adaptation."""

    def __init__(self, config: GradAlignPlusConfig | None = None) -> None:
        self.config = config or GradAlignPlusConfig()
        self.config.validate()

    def fit(self, graph1: nx.Graph, graph2: nx.Graph) -> AlignmentResult:
        return self.fit_with_attributes(graph1, graph2, None, None)

    def fit_with_attributes(
        self,
        graph1: nx.Graph,
        graph2: nx.Graph,
        attributes1: np.ndarray | None,
        attributes2: np.ndarray | None,
    ) -> AlignmentResult:
        if graph1.number_of_nodes() == 0 or graph2.number_of_nodes() == 0:
            raise ValueError("GradAlign+ cannot align empty graphs")
        for graph, label in ((graph1, "G1"), (graph2, "G2")):
            if set(graph.nodes()) != set(range(graph.number_of_nodes())):
                raise ValueError(f"{label} must use consecutive integer node ids")
        n1, n2 = graph1.number_of_nodes(), graph2.number_of_nodes()
        if n1 * n2 > self.config.max_dense_entries:
            raise ValueError(
                f"GradAlign+ needs {n1*n2:,} scores, exceeding "
                f"max_dense_entries={self.config.max_dense_entries:,}"
            )
        if (attributes1 is None) != (attributes2 is None):
            raise ValueError("both original attribute matrices must be supplied together")

        start = time.perf_counter()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        device = _resolve_device(self.config.device)

        augmented1, augmented2, cnfa_metadata = _cnfa(
            graph1, graph2, self.config
        )
        embeddings1, embeddings2, augmented_losses = _train_encoder(
            graph1,
            graph2,
            augmented1,
            augmented2,
            self.config,
            device,
        )
        augmented_score = _embedding_similarity(
            embeddings1, embeddings2, self.config.score_block_size
        )
        original_losses: list[float] | None = None
        if attributes1 is None:
            embedding_score = augmented_score
            input_mode = "unattributed: only CNFA term in Equation (4)"
        else:
            attributes1 = np.asarray(attributes1, dtype=np.float32)
            attributes2 = np.asarray(attributes2, dtype=np.float32)
            if (
                attributes1.shape[0] != n1
                or attributes2.shape[0] != n2
                or attributes1.ndim != 2
                or attributes2.ndim != 2
                or attributes1.shape[1] != attributes2.shape[1]
            ):
                raise ValueError("original attributes have incompatible shapes")
            base1, base2, original_losses = _train_encoder(
                graph1,
                graph2,
                attributes1,
                attributes2,
                self.config,
                device,
            )
            embedding_score = _embedding_similarity(
                base1, base2, self.config.score_block_size
            ) + self.config.augmentation_weight * augmented_score
            input_mode = "attributed: original + lambda*CNFA terms in Equation (4)"

        pairs, final_score, history = _gradual_alignment(
            graph1, graph2, embedding_score, self.config
        )
        return AlignmentResult(
            method="GradAlign+",
            predicted_pairs=pairs,
            runtime_sec=float(time.perf_counter() - start),
            score_matrix=final_score,
            larger_is_better=True,
            metadata={
                "config": asdict(self.config),
                "device": str(device),
                "reconstruction_sparse_backend": (
                    "deterministic_sparse_bmm"
                    if device.type == "cuda"
                    and torch.are_deterministic_algorithms_enabled()
                    else "sparse_mm"
                ),
                "paper": "IEEE TNSE 2024 extended GradAlign+, arXiv:2304.12751",
                "core_stages": [
                    "CNFA automatic centrality selection, Equation (7)",
                    "joint equal-width one-hot augmentation, Equation (1)",
                    "shared GIN layer-wise reconstruction, Equation (8)",
                    "multi-layer embedding similarity, Equation (4)",
                    "ten-step ACN^p gradual alignment, Equations (5)-(6)",
                ],
                "cnfa": cnfa_metadata,
                "input_mode": input_mode,
                "augmented_training_loss_first": (
                    augmented_losses[0] if augmented_losses else None
                ),
                "augmented_training_loss_last": (
                    augmented_losses[-1] if augmented_losses else None
                ),
                "original_training_loss_first": (
                    original_losses[0] if original_losses else None
                ),
                "original_training_loss_last": (
                    original_losses[-1] if original_losses else None
                ),
                "gradual_history": history,
                "paper_M_adaptation": (
                    "unknown-overlap protocol replaces unavailable true M with "
                    "min(|Vs|,|Vt|); no GT pair or GT cardinality is read"
                ),
                "partial_adaptation": (
                    "method-aware native assignment-margin rejection after "
                    "paper gradual matching"
                ),
                "ground_truth_used_during_inference": False,
                "ground_truth_cardinality_used_during_inference": False,
            },
        )
