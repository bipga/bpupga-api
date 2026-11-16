from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import pairwise_distances

from common.dataset import AlignmentDataset
from common.result import AlignmentResult

from .metrics import (
    evaluate_matchability,
    evaluate_pair_set,
    evaluate_ranking_from_embeddings,
    evaluate_ranking_from_matrix,
    validate_predicted_pairs,
)
from .postprocess import full_hungarian_pairs, oracle_cardinality_pairs


def _evaluation_matrix(
    result: AlignmentResult,
    max_dense_entries: int,
) -> tuple[np.ndarray | None, bool | None, str | None]:
    if result.score_matrix is not None:
        return np.asarray(result.score_matrix), True, "score_matrix"
    if result.cost_matrix is not None:
        return np.asarray(result.cost_matrix), False, "cost_matrix"
    if result.source_embeddings is None or result.target_embeddings is None:
        return None, None, None

    entries = result.source_embeddings.shape[0] * result.target_embeddings.shape[0]
    if entries > max_dense_entries:
        return None, None, None
    cost = pairwise_distances(
        result.source_embeddings,
        result.target_embeddings,
        metric="euclidean",
    ).astype(np.float32, copy=False)
    result.cost_matrix = cost
    return cost, False, "embedding_distance"


def evaluate_alignment_result(
    result: AlignmentResult,
    dataset: AlignmentDataset,
    ks: Sequence[int] = (1, 5, 10),
    evaluate_full_hungarian: bool = False,
    evaluate_oracle_k: bool = False,
    max_dense_entries: int = 25_000_000,
    ranking_batch_size: int = 512,
) -> Dict[str, object]:
    n1 = dataset.graph1.number_of_nodes()
    n2 = dataset.graph2.number_of_nodes()
    result.validate_shapes(n1, n2)

    diagnostics = validate_predicted_pairs(result.predicted_pairs, n1, n2)
    native_pair = evaluate_pair_set(result.predicted_pairs, dataset.ground_truth, n1, n2)
    native_matchability = evaluate_matchability(
        result.predicted_pairs,
        dataset.ground_truth,
        n1,
        n2,
    )
    reverse_ground_truth = {
        int(target): int(source) for source, target in dataset.ground_truth.items()
    }

    if result.score_matrix is not None:
        ranking_forward = evaluate_ranking_from_matrix(
            result.score_matrix,
            dataset.ground_truth,
            ks=ks,
            larger_is_better=True,
        )
        ranking_reverse = evaluate_ranking_from_matrix(
            result.score_matrix.T,
            reverse_ground_truth,
            ks=ks,
            larger_is_better=True,
        )
        ranking_source = "score_matrix"
    elif result.cost_matrix is not None:
        ranking_forward = evaluate_ranking_from_matrix(
            result.cost_matrix,
            dataset.ground_truth,
            ks=ks,
            larger_is_better=False,
        )
        ranking_reverse = evaluate_ranking_from_matrix(
            result.cost_matrix.T,
            reverse_ground_truth,
            ks=ks,
            larger_is_better=False,
        )
        ranking_source = "cost_matrix"
    elif result.source_embeddings is not None and result.target_embeddings is not None:
        ranking_forward = evaluate_ranking_from_embeddings(
            result.source_embeddings,
            result.target_embeddings,
            dataset.ground_truth,
            ks=ks,
            batch_size=ranking_batch_size,
        )
        ranking_reverse = evaluate_ranking_from_embeddings(
            result.target_embeddings,
            result.source_embeddings,
            reverse_ground_truth,
            ks=ks,
            batch_size=ranking_batch_size,
        )
        ranking_source = "embedding_distance"
    else:
        ranking_forward = None
        ranking_reverse = None
        ranking_source = None

    report: Dict[str, object] = {
        "prediction_diagnostics": diagnostics,
        "conditional_ranking": {
            "source_to_target": ranking_forward,
            "target_to_source": ranking_reverse,
        },
        "conditional_ranking_source": ranking_source,
        "native_pair": native_pair,
        "source_matchability": native_matchability["source"],
        "target_matchability": native_matchability["target"],
    }

    if evaluate_full_hungarian or evaluate_oracle_k:
        matrix, larger_is_better, matrix_source = _evaluation_matrix(result, max_dense_entries)
        report["dense_postprocess_source"] = matrix_source
        if matrix is None or larger_is_better is None:
            entries = n1 * n2
            report["dense_postprocess_skipped"] = (
                "No score representation available or dense matrix would exceed "
                f"max_dense_entries={max_dense_entries}; requested entries={entries}."
            )
        else:
            if evaluate_full_hungarian:
                full_pairs = full_hungarian_pairs(matrix, larger_is_better)
                report["full_hungarian"] = {
                    "pair": evaluate_pair_set(full_pairs, dataset.ground_truth, n1, n2),
                    "matchability": evaluate_matchability(
                        full_pairs, dataset.ground_truth, n1, n2
                    ),
                    "diagnostics": validate_predicted_pairs(full_pairs, n1, n2),
                }
            if evaluate_oracle_k:
                oracle_pairs = oracle_cardinality_pairs(
                    matrix,
                    cardinality=len(dataset.ground_truth),
                    larger_is_better=larger_is_better,
                )
                report["oracle_k"] = {
                    "cardinality": len(dataset.ground_truth),
                    "pair": evaluate_pair_set(oracle_pairs, dataset.ground_truth, n1, n2),
                    "diagnostics": validate_predicted_pairs(oracle_pairs, n1, n2),
                    "warning": "Uses ground-truth cardinality after inference; diagnostic only.",
                }

    return report


