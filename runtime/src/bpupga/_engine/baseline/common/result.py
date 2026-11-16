from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dataset import AlignmentDataset


@dataclass
class AlignmentResult:
    method: str
    predicted_pairs: List[Tuple[int, int]]
    runtime_sec: float
    peak_memory_mb: Optional[float] = None
    score_matrix: Optional[np.ndarray] = None
    cost_matrix: Optional[np.ndarray] = None
    source_embeddings: Optional[np.ndarray] = None
    target_embeddings: Optional[np.ndarray] = None
    # Sparse algorithm-native candidate scores.  Unlike ``score_matrix``, an
    # omitted pair means that the method never retrieved that candidate.
    ranking_candidates: Optional[List[Tuple[int, int, float]]] = None
    larger_is_better: Optional[bool] = None
    # Ephemeral model-native signals consumed by a method-aware partial decoder.
    # They are intentionally not serialized as JSON because they may be dense.
    partial_decoder_data: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    # Optional read-only traces returned by the baseline Alpine model.
    stage_score_matrices: Dict[str, np.ndarray] = field(default_factory=dict)
    stage_diagnostics_metadata: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def validate_shapes(self, num_source_nodes: int, num_target_nodes: int) -> None:
        expected_matrix_shape = (num_source_nodes, num_target_nodes)
        if self.score_matrix is not None and self.score_matrix.shape != expected_matrix_shape:
            raise ValueError(
                f"score_matrix shape {self.score_matrix.shape} != {expected_matrix_shape}"
            )
        if self.cost_matrix is not None and self.cost_matrix.shape != expected_matrix_shape:
            raise ValueError(
                f"cost_matrix shape {self.cost_matrix.shape} != {expected_matrix_shape}"
            )
        if self.source_embeddings is not None and self.source_embeddings.shape[0] != num_source_nodes:
            raise ValueError("source_embeddings row count does not match G1")
        if self.target_embeddings is not None and self.target_embeddings.shape[0] != num_target_nodes:
            raise ValueError("target_embeddings row count does not match G2")
        for stage_name, matrix in self.stage_score_matrices.items():
            if matrix.shape != expected_matrix_shape:
                raise ValueError(
                    f"stage score matrix {stage_name!r} has shape "
                    f"{matrix.shape} != {expected_matrix_shape}"
                )
        if self.ranking_candidates is not None:
            seen = set()
            for source, target, score in self.ranking_candidates:
                pair = (int(source), int(target))
                if not (0 <= pair[0] < num_source_nodes and 0 <= pair[1] < num_target_nodes):
                    raise ValueError("ranking candidate contains an out-of-range node")
                if pair in seen:
                    raise ValueError("ranking_candidates contains a duplicate pair")
                if not np.isfinite(float(score)):
                    raise ValueError("ranking candidate score must be finite")
                seen.add(pair)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def _write_pair_artifact(
    path: Path,
    dataset: AlignmentDataset,
    result: AlignmentResult,
    pairs: List[Tuple[int, int]],
    output_variant: str,
    decoder_id: str | None,
) -> None:
    original_pairs = dataset.pairs_to_original(pairs)
    _write_json(
        path,
        {
            "schema_name": "bpupga.predicted_pairs",
            "schema_version": 2,
            "method": result.method,
            "index_space": "original",
            "output_variant": output_variant,
            "decoder_id": decoder_id,
            "pairs": [[u, v] for u, v in original_pairs],
        },
    )


def save_run_artifacts(
    output_dir: str | Path,
    dataset: AlignmentDataset,
    result: AlignmentResult,
    report: Dict[str, object],
    config: Dict[str, object],
    save_embeddings: bool = False,
    save_cost_matrix: bool = False,
    save_score_matrix: bool = False,
    run_metadata: Dict[str, object] | None = None,
    runtime_metadata: Dict[str, object] | None = None,
    native_predicted_pairs: List[Tuple[int, int]] | None = None,
    common_decoder_predicted_pairs: List[Tuple[int, int]] | None = None,
    method_aware_predicted_pairs: List[Tuple[int, int]] | None = None,
    primary_predicted_pairs: List[Tuple[int, int]] | None = None,
    primary_output_variant: str = "method_aware_decoder",
    common_decoder_id: str | None = None,
    method_aware_decoder_id: str | None = None,
) -> Dict[str, str]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    selected_primary_pairs = (
        list(primary_predicted_pairs)
        if primary_predicted_pairs is not None
        else list(result.predicted_pairs)
    )
    prediction_path = output_path / "predicted_pairs.json"
    primary_decoder_id = (
        common_decoder_id
        if primary_output_variant == "common_decoder"
        else method_aware_decoder_id
        if primary_output_variant == "method_aware_decoder"
        else None
    )
    _write_pair_artifact(
        prediction_path,
        dataset,
        result,
        selected_primary_pairs,
        primary_output_variant,
        primary_decoder_id,
    )

    native_prediction_path = None
    if native_predicted_pairs is not None:
        native_prediction_path = output_path / "native_predicted_pairs.json"
        _write_pair_artifact(
            native_prediction_path,
            dataset,
            result,
            list(native_predicted_pairs),
            "native",
            None,
        )

    common_prediction_path = None
    if common_decoder_predicted_pairs is not None:
        common_prediction_path = output_path / "common_decoder_predicted_pairs.json"
        _write_pair_artifact(
            common_prediction_path,
            dataset,
            result,
            list(common_decoder_predicted_pairs),
            "common_decoder",
            common_decoder_id,
        )

    method_aware_prediction_path = None
    if method_aware_predicted_pairs is not None:
        method_aware_prediction_path = (
            output_path / "method_aware_decoder_predicted_pairs.json"
        )
        _write_pair_artifact(
            method_aware_prediction_path,
            dataset,
            result,
            list(method_aware_predicted_pairs),
            "method_aware_decoder",
            method_aware_decoder_id,
        )

    index_maps_path = output_path / "index_maps.json"
    _write_json(
        index_maps_path,
        {
            "g1_new_to_old": {str(k): int(v) for k, v in dataset.g1_new_to_old.items()},
            "g2_new_to_old": {str(k): int(v) for k, v in dataset.g2_new_to_old.items()},
        },
    )

    artifacts: Dict[str, str] = {
        "predicted_pairs": str(prediction_path),
        "index_maps": str(index_maps_path),
    }
    if native_prediction_path is not None:
        artifacts["native_predicted_pairs"] = str(native_prediction_path)
    if common_prediction_path is not None:
        artifacts["common_decoder_predicted_pairs"] = str(common_prediction_path)
    if method_aware_prediction_path is not None:
        artifacts["method_aware_decoder_predicted_pairs"] = str(
            method_aware_prediction_path
        )

    if save_embeddings and result.source_embeddings is not None and result.target_embeddings is not None:
        source_path = output_path / "source_embeddings.npy"
        target_path = output_path / "target_embeddings.npy"
        np.save(source_path, result.source_embeddings)
        np.save(target_path, result.target_embeddings)
        artifacts["source_embeddings"] = str(source_path)
        artifacts["target_embeddings"] = str(target_path)

    if save_cost_matrix and result.cost_matrix is not None:
        cost_path = output_path / "cost_matrix.npy"
        np.save(cost_path, result.cost_matrix)
        artifacts["cost_matrix"] = str(cost_path)

    # ``score_matrix`` is the last method-native dense representation before
    # the common and method-aware decoders run.  It is deliberately opt-in:
    # matrices can be large, and routine benchmark artifacts should not change.
    if save_score_matrix and result.score_matrix is not None:
        score_path = output_path / "score_matrix.npy"
        np.save(score_path, result.score_matrix)
        artifacts["score_matrix"] = str(score_path)

    metrics_path = output_path / "metrics.json"
    runtime_payload: Dict[str, object] = {
        "alignment_sec": result.runtime_sec,
        "peak_process_rss_mb": result.peak_memory_mb,
    }
    if runtime_metadata:
        runtime_payload.update(runtime_metadata)
    payload = {
        "schema_name": "bpupga.partial_alignment.metrics",
        "schema_version": 2,
        "method": result.method,
        "run": run_metadata or {},
        "dataset": dataset.summary(),
        "config": config,
        "runtime": runtime_payload,
        "method_metadata": result.metadata,
        "evaluation": report,
        "artifacts": artifacts,
    }
    payload["dataset"]["input_sha256"] = {
        name: hashlib.sha256((dataset.root / name).read_bytes()).hexdigest()
        for name in (
            "G1.edgelist",
            "G2.edgelist",
            "G1.nodes",
            "G2.nodes",
            "Alignment.json",
            "metadata.json",
        )
        if (dataset.root / name).is_file()
    }
    _write_json(metrics_path, payload)
    artifacts["metrics"] = str(metrics_path)
    return artifacts
