"""Standalone graph alignment, construction, evaluation and diagnostics."""

__version__ = "0.1.0"

from ._api import (
    Algorithm,
    AlignmentDataset,
    AlignmentResult,
    Dataset,
    RunLog,
    construct,
    diagnose,
    evaluate,
    register_algorithm,
)

__all__ = [
    "Algorithm", "AlignmentDataset", "AlignmentResult", "Dataset", "RunLog",
    "construct", "diagnose", "evaluate", "register_algorithm",
]
