"""S-GWL graph alignment model for the unified BPUPGA contract.

The original NeurIPS 2019 GW and recursive partition routines are vendored in
``methods/``.  This adapter converts NetworkX topology into their cost and
probability representation, never exposes ground truth, and returns a global
source-by-target score matrix for ranking and unknown-overlap decoding.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

from common.result import AlignmentResult

from .methods import GromovWassersteinFramework as gw_framework
from .methods import GromovWassersteinGraphToolkit as gw_toolkit


@dataclass(frozen=True)
class SGWLConfig:
    beta: float = 0.1
    outer_iterations: int = 200
    inner_iterations: int = 5
    iteration_tolerance: float = 1e-10
    sinkhorn_tolerance: float = 1e-10
    node_prior: float = 1_000.0
    cost_norm: str = "range"
    kernel_clip: float = 50.0
    max_barycenter_iterations: int = 4
    barycenter_tolerance: float = 1e-12
    cluster_count: int = 2
    partition_levels: int = 3
    max_leaf_nodes: int = 400
    direct_node_threshold: int = 800
    probability_epsilon: float = 1e-12
    native_rounding: str = "hungarian"
    seed: int = 2026
    show_progress: bool = True
    max_dense_entries: int = 25_000_000

    def validate(self) -> None:
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")
        if self.outer_iterations < 1 or self.inner_iterations < 1:
            raise ValueError("iteration counts must be positive")
        if self.iteration_tolerance < 0.0 or self.sinkhorn_tolerance < 0.0:
            raise ValueError("tolerances must be non-negative")
        if self.node_prior < 0.0:
            raise ValueError("node_prior must be non-negative")
        if self.cost_norm not in {"none", "max", "range"}:
            raise ValueError("cost_norm must be none, max, or range")
        if self.kernel_clip < 0.0:
            raise ValueError("kernel_clip must be non-negative")
        if self.cluster_count < 2 or self.partition_levels < 0:
            raise ValueError("cluster_count must be >=2 and partition_levels non-negative")
        if self.max_leaf_nodes < 1 or self.direct_node_threshold < 1:
            raise ValueError("leaf and direct thresholds must be positive")
        if self.probability_epsilon < 0.0:
            raise ValueError("probability_epsilon must be non-negative")
        if self.native_rounding not in {"hungarian", "historical_transport_greedy"}:
            raise ValueError("Unsupported native_rounding")
        if self.native_rounding == "historical_transport_greedy" and self.partition_levels:
            raise ValueError("Historical transport rounding requires direct GWL")
        if self.max_dense_entries < 1:
            raise ValueError("max_dense_entries must be positive")


def _require_consecutive_nodes(graph: nx.Graph, label: str) -> None:
    if set(graph.nodes()) != set(range(graph.number_of_nodes())):
        raise ValueError(f"{label} must use consecutive integer node ids")


def _cost_probability(graph: nx.Graph, epsilon: float) -> tuple[csr_matrix, np.ndarray]:
    n = graph.number_of_nodes()
    cost = csr_matrix(
        nx.to_scipy_sparse_array(
            graph,
            nodelist=range(n),
            dtype=np.float64,
            format="csr",
        )
    )
    degree = np.asarray(cost.sum(axis=1), dtype=np.float64).reshape(n, 1)
    probability = degree + float(epsilon)
    total = float(probability.sum())
    if total <= np.finfo(np.float64).tiny:
        probability = np.full((n, 1), 1.0 / n, dtype=np.float64)
    else:
        probability /= total
    return cost, probability


def _ot_parameters(config: SGWLConfig) -> dict[str, object]:
    return {
        "loss_type": "L2",
        "ot_method": "proximal",
        "beta": config.beta,
        "outer_iteration": config.outer_iterations,
        "iter_bound": config.iteration_tolerance,
        "inner_iteration": config.inner_iterations,
        "sk_bound": config.sinkhorn_tolerance,
        "node_prior": config.node_prior,
        "max_iter": config.max_barycenter_iterations,
        "cost_bound": config.barycenter_tolerance,
        "update_p": False,
        "lr": 0.0,
        "alpha": 0.0,
        "cost_norm": config.cost_norm,
        "kernel_clip": config.kernel_clip,
        "paper_exact": False,
    }


def _direct_score(
    source_cost: csr_matrix,
    target_cost: csr_matrix,
    source_probability: np.ndarray,
    target_probability: np.ndarray,
    parameters: dict[str, object],
) -> np.ndarray:
    transport, updated_source = _direct_transport(
        source_cost, target_cost, source_probability, target_probability, parameters)
    return transport / np.maximum(updated_source, np.finfo(np.float64).tiny)


def _direct_transport(source_cost, target_cost, source_probability, target_probability, parameters):
    transport, _, updated_source = gw_framework.gromov_wasserstein_discrepancy(
        source_cost,
        target_cost,
        source_probability,
        target_probability,
        parameters,
    )
    if not np.all(np.isfinite(transport)) or np.any(transport < 0.0):
        raise FloatingPointError("S-GWL produced an invalid transport matrix")
    return transport, updated_source


class SGWL:
    """Scalable Gromov-Wasserstein learning graph alignment."""

    def __init__(self, config: SGWLConfig | None = None) -> None:
        self.config = config or SGWLConfig()
        self.config.validate()

    def fit(self, graph1: nx.Graph, graph2: nx.Graph) -> AlignmentResult:
        if graph1.number_of_nodes() == 0 or graph2.number_of_nodes() == 0:
            raise ValueError("S-GWL cannot align empty graphs")
        _require_consecutive_nodes(graph1, "G1")
        _require_consecutive_nodes(graph2, "G2")
        n1, n2 = graph1.number_of_nodes(), graph2.number_of_nodes()
        largest_matrix = max(n1 * n1, n2 * n2, n1 * n2)
        if largest_matrix > self.config.max_dense_entries:
            raise ValueError(
                "S-GWL's current unified score export requires at most "
                f"{self.config.max_dense_entries:,} entries; requested "
                f"{largest_matrix:,}"
            )

        start = time.perf_counter()
        np.random.seed(self.config.seed)
        source_cost, source_probability = _cost_probability(
            graph1, self.config.probability_epsilon
        )
        target_cost, target_probability = _cost_probability(
            graph2, self.config.probability_epsilon
        )
        parameters = _ot_parameters(self.config)
        use_recursive = (
            self.config.partition_levels > 0
            and max(n1, n2) > self.config.direct_node_threshold
        )
        predicted_pairs: list[tuple[int, int]] = []

        if not use_recursive:
            transport, updated_source = _direct_transport(
                source_cost,
                target_cost,
                source_probability,
                target_probability,
                dict(parameters),
            )
            score_matrix = transport / np.maximum(updated_source, np.finfo(np.float64).tiny)
            if self.config.native_rounding == "historical_transport_greedy":
                # Use the actual transport and marginals, not a reconstructed
                # transport or a Hungarian substitute. No GT enters rounding.
                pairs, _, _ = gw_toolkit.node_pair_assignment(
                    transport, updated_source, target_probability,
                    dict(enumerate(range(n1))), dict(enumerate(range(n2))))
                predicted_pairs.extend((int(s), int(t)) for s, t in pairs)
            else:
                sources, targets = linear_sum_assignment(-score_matrix)
                predicted_pairs.extend(
                    (int(source), int(target)) for source, target in zip(sources, targets))
            leaf_count = 1
            solver_mode = "direct_gwl"
        else:
            identity1 = {index: index for index in range(n1)}
            identity2 = {index: index for index in range(n2)}
            costs, probabilities, index_maps = gw_toolkit.recursive_multi_graph_partition(
                {0: source_cost, 1: target_cost},
                {0: source_probability, 1: target_probability},
                {0: identity1, 1: identity2},
                dict(parameters),
                weights=None,
                predefine_barycenter=False,
                cluster_num=self.config.cluster_count,
                partition_level=self.config.partition_levels,
                max_node_num=self.config.max_leaf_nodes,
            )
            score_matrix = np.zeros((n1, n2), dtype=np.float64)
            leaf_count = 0
            for leaf_costs, leaf_probabilities, leaf_maps in zip(
                costs, probabilities, index_maps
            ):
                if 0 not in leaf_costs or 1 not in leaf_costs:
                    continue
                leaf_score = _direct_score(
                    leaf_costs[0],
                    leaf_costs[1],
                    leaf_probabilities[0],
                    leaf_probabilities[1],
                    dict(parameters),
                )
                source_nodes = np.asarray(
                    [leaf_maps[0][index] for index in range(len(leaf_maps[0]))],
                    dtype=np.int64,
                )
                target_nodes = np.asarray(
                    [leaf_maps[1][index] for index in range(len(leaf_maps[1]))],
                    dtype=np.int64,
                )
                score_matrix[np.ix_(source_nodes, target_nodes)] = leaf_score
                local_sources, local_targets = linear_sum_assignment(-leaf_score)
                predicted_pairs.extend(
                    (int(source_nodes[source]), int(target_nodes[target]))
                    for source, target in zip(local_sources, local_targets)
                )
                leaf_count += 1
            solver_mode = "recursive_s_gwl"

        if not np.all(np.isfinite(score_matrix)) or np.any(score_matrix < 0.0):
            raise FloatingPointError("S-GWL global score matrix is invalid")
        if len({source for source, _ in predicted_pairs}) != len(predicted_pairs):
            raise RuntimeError("S-GWL native assignment contains duplicate sources")
        if len({target for _, target in predicted_pairs}) != len(predicted_pairs):
            raise RuntimeError("S-GWL native assignment contains duplicate targets")
        return AlignmentResult(
            method="S-GWL",
            predicted_pairs=predicted_pairs,
            runtime_sec=float(time.perf_counter() - start),
            score_matrix=np.asarray(score_matrix, dtype=np.float64),
            larger_is_better=True,
            metadata={
                "config": asdict(self.config),
                "solver_mode": solver_mode,
                "leaf_count": leaf_count,
                "prediction_rule": self.config.native_rounding,
                "implementation_revision": ("upga_direct_gwl_v1" if self.config.native_rounding ==
                                            "historical_transport_greedy" else "stable_s_gwl_v1"),
                "score": "row-conditional Gromov-Wasserstein transport",
                "ground_truth_used_during_inference": False,
            },
        )
