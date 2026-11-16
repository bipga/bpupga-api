from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from common.dataset import AlignmentDataset
from common.result import AlignmentResult
from evaluation.metrics import evaluate_pair_set


def _ratio_token(value: float) -> str:
    rendered = f"{float(value):.12f}".rstrip("0").rstrip(".")
    return (rendered or "0").replace(".", "p")


def load_input_anchors(
    dataset: AlignmentDataset,
    ratio: float,
    anchor_seed: int,
) -> tuple[list[tuple[int, int]], dict[str, object], Path]:
    """Load the declared public anchors without exposing any other truth pairs."""

    anchor_dir = (
        dataset.root
        / "anchors"
        / f"eta_{_ratio_token(ratio)}"
        / f"anchor_seed_{int(anchor_seed)}"
    )
    anchor_path = anchor_dir / "Anchors.json"
    metadata_path = anchor_dir / "anchor_metadata.json"
    if not anchor_path.is_file():
        raise FileNotFoundError(
            f"Seed anchor set is missing: {anchor_path}. Generate it with "
            "data/construct_seed_anchors.py."
        )
    with anchor_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Anchors.json must be a JSON object: {anchor_path}")

    original_pairs = [(int(source), int(target)) for source, target in raw.items()]
    pairs = dataset.pairs_to_reindexed(original_pairs)
    if len({source for source, _ in pairs}) != len(pairs):
        raise ValueError(f"Anchor sources are not unique: {anchor_path}")
    if len({target for _, target in pairs}) != len(pairs):
        raise ValueError(f"Anchor targets are not unique: {anchor_path}")

    metadata: dict[str, object] = {}
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"anchor_metadata.json must be an object: {metadata_path}")
        metadata = value
        expected = metadata.get("anchor_count")
        if expected is not None and int(expected) != len(pairs):
            raise ValueError(
                f"Anchor metadata count {expected} disagrees with {len(pairs)} pairs"
            )
    return pairs, metadata, anchor_path.resolve()


def evaluate_unseeded_pairs(
    result: AlignmentResult,
    dataset: AlignmentDataset,
    anchors: Sequence[tuple[int, int]],
) -> dict[str, object]:
    """Evaluate predictions after excluding correspondences exposed as anchors."""

    anchor_set = {(int(source), int(target)) for source, target in anchors}
    remaining_truth = {
        int(source): int(target)
        for source, target in dataset.ground_truth.items()
        if (int(source), int(target)) not in anchor_set
    }
    remaining_predictions = [
        (int(source), int(target))
        for source, target in result.predicted_pairs
        if (int(source), int(target)) not in anchor_set
    ]
    return {
        "anchor_count": len(anchor_set),
        "unseeded_ground_truth_pairs": len(remaining_truth),
        "pair": evaluate_pair_set(
            remaining_predictions,
            remaining_truth,
            dataset.graph1.number_of_nodes(),
            dataset.graph2.number_of_nodes(),
        ),
        "warning": "Input anchor pairs are excluded from this diagnostic.",
    }


__all__ = ["evaluate_unseeded_pairs", "load_input_anchors"]
