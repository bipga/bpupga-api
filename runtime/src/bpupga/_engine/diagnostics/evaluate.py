#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.dataset import load_dataset
from common.result import AlignmentResult
from evaluation.evaluator import evaluate_alignment_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved BPUPGA predictions")
    parser.add_argument("--root", required=True, help="Dataset directory")
    parser.add_argument("--g1-file", default="G1.edgelist")
    parser.add_argument("--g2-file", default="G2.edgelist")
    parser.add_argument("--g1-nodes-file", default="G1.nodes")
    parser.add_argument("--g2-nodes-file", default="G2.nodes")
    parser.add_argument("--alignment-file", default="Alignment.json")
    parser.add_argument("--metadata-file", default="metadata.json")
    parser.add_argument("--prediction", required=True, help="predicted_pairs.json")
    parser.add_argument("--score-matrix", default=None, help="Optional .npy score/cost matrix")
    parser.add_argument(
        "--matrix-kind",
        choices=("score", "cost"),
        default="score",
        help="Whether larger or smaller matrix values are better",
    )
    parser.add_argument("--output", default=None, help="Optional report JSON path")
    parser.add_argument("--evaluate-full-hungarian", action="store_true")
    parser.add_argument("--evaluate-oracle-k", action="store_true")
    parser.add_argument("--max-dense-entries", type=int, default=25_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(
        args.root,
        g1_file=args.g1_file,
        g2_file=args.g2_file,
        g1_nodes_file=args.g1_nodes_file,
        g2_nodes_file=args.g2_nodes_file,
        alignment_file=args.alignment_file,
        metadata_file=args.metadata_file,
    )

    prediction_path = Path(args.prediction)
    with prediction_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "pairs" not in payload:
        raise ValueError("Prediction JSON must contain a 'pairs' list")
    pairs = [(int(u), int(v)) for u, v in payload["pairs"]]
    if payload.get("index_space", "original") == "original":
        pairs = dataset.pairs_to_reindexed(pairs)

    score_matrix = None
    cost_matrix = None
    if args.score_matrix:
        matrix = np.load(args.score_matrix)
        if args.matrix_kind == "score":
            score_matrix = matrix
        else:
            cost_matrix = matrix

    result = AlignmentResult(
        method=str(payload.get("method", "external")),
        predicted_pairs=pairs,
        runtime_sec=0.0,
        score_matrix=score_matrix,
        cost_matrix=cost_matrix,
    )
    report = evaluate_alignment_result(
        result,
        dataset,
        evaluate_full_hungarian=args.evaluate_full_hungarian,
        evaluate_oracle_k=args.evaluate_oracle_k,
        max_dense_entries=args.max_dense_entries,
    )

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
