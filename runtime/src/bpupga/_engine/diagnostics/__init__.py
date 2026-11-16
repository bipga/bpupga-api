"""Shared BPUPGA evaluation API."""

from .evaluator import evaluate_alignment_result
from .metrics import (
    evaluate_matchability,
    evaluate_pair_set,
    evaluate_ranking_from_embeddings,
    evaluate_ranking_from_matrix,
    validate_predicted_pairs,
)

__all__ = [
    "evaluate_alignment_result",
    "evaluate_matchability",
    "evaluate_pair_set",
    "evaluate_ranking_from_embeddings",
    "evaluate_ranking_from_matrix",
    "validate_predicted_pairs",
]

