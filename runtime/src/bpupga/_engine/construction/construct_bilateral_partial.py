#!/usr/bin/env python3
"""Construct leakage-safe bilateral partial-overlap graph pairs.

The constructor partitions the original one-to-one correspondences into three
disjoint groups:

* shared: keep both endpoints and retain the correspondence;
* source-only: keep only the G1 endpoint;
* target-only: keep only the G2 endpoint.

Original unmatched nodes are intentionally excluded from the controlled main
setting. This makes requested source/target overlap ratios comparable across
datasets that originally have different one-sided private-node counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_DATASETS = ("fbtw", "dblp", "grqc")
DEFAULT_OVERLAPS = (0.9, 0.8, 0.7, 0.6)
SCHEMA_VERSION = 1

Pair = tuple[int, int]


@dataclass(frozen=True)
class GraphInput:
    nodes: frozenset[int]
    edges: tuple[Pair, ...]
    raw_edge_lines: int
    self_loops_removed: int
    duplicate_edges_removed: int
    node_file_present: bool


@dataclass(frozen=True)
class RawDataset:
    name: str
    raw_dir: Path
    graph1: GraphInput
    graph2: GraphInput
    alignment: dict[int, int]
    input_hashes: dict[str, str | None]


@dataclass(frozen=True)
class PartitionCounts:
    shared: int
    source_only: int
    target_only: int


@dataclass(frozen=True)
class ConstructedDataset:
    source_nodes: frozenset[int]
    target_nodes: frozenset[int]
    source_edges: tuple[Pair, ...]
    target_edges: tuple[Pair, ...]
    alignment: dict[int, int]
    counts: PartitionCounts
    metadata: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct bilateral partial-overlap datasets from data/<dataset>/raw"
        )
    )
    parser.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing dataset folders; defaults to this script's directory",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset folder names (Arxiv is intentionally not in the default list)",
    )
    parser.add_argument(
        "--overlaps",
        nargs="+",
        type=float,
        default=None,
        help="Symmetric overlap ratios; default: 0.9 0.8 0.7 0.6",
    )
    parser.add_argument(
        "--rho-source",
        type=float,
        default=None,
        help="Source overlap ratio for one asymmetric configuration",
    )
    parser.add_argument(
        "--rho-target",
        type=float,
        default=None,
        help="Target overlap ratio for one asymmetric configuration",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[2026],
        help="Construction seeds",
    )
    parser.add_argument("--raw-dir-name", default="raw")
    parser.add_argument("--output-dir-name", default="bpupga")
    parser.add_argument("--g1-file", default="G1.edgelist")
    parser.add_argument("--g2-file", default="G2.edgelist")
    parser.add_argument("--g1-nodes-file", default="G1.nodes")
    parser.add_argument("--g2-nodes-file", default="G2.nodes")
    parser.add_argument("--alignment-file", default="Alignment.json")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing generated dataset only after a new copy is ready",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned constructions without writing files",
    )
    return parser.parse_args()


def _validate_ratio(value: float, label: str) -> float:
    ratio = float(value)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError(f"{label} must be finite and in (0, 1], got {value}")
    return ratio


def _requested_configurations(args: argparse.Namespace) -> list[tuple[float, float]]:
    has_source = args.rho_source is not None
    has_target = args.rho_target is not None
    if has_source != has_target:
        raise ValueError("--rho-source and --rho-target must be provided together")
    if (has_source or has_target) and args.overlaps is not None:
        raise ValueError("Use either --overlaps or --rho-source/--rho-target, not both")

    if has_source and has_target:
        configurations = [
            (
                _validate_ratio(args.rho_source, "rho_source"),
                _validate_ratio(args.rho_target, "rho_target"),
            )
        ]
    else:
        overlaps = args.overlaps if args.overlaps is not None else DEFAULT_OVERLAPS
        configurations = [
            (_validate_ratio(value, "overlap"), _validate_ratio(value, "overlap"))
            for value in overlaps
        ]

    deduplicated: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for rho_source, rho_target in configurations:
        key = (round(rho_source, 12), round(rho_target, 12))
        if key not in seen:
            seen.add(key)
            deduplicated.append((rho_source, rho_target))
    return deduplicated


def _clean_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line and not line.startswith("#"):
                yield line


def _read_nodes(path: Path) -> set[int]:
    nodes: set[int] = set()
    for line in _clean_lines(path):
        fields = line.split()
        if len(fields) != 1:
            raise ValueError(f"Expected one node id per line in {path}: {line!r}")
        nodes.add(int(fields[0]))
    return nodes


def _read_graph(edge_path: Path, node_path: Path) -> GraphInput:
    if not edge_path.is_file():
        raise FileNotFoundError(f"Missing graph file: {edge_path}")

    nodes: set[int] = set()
    unique_edges: set[Pair] = set()
    raw_edge_lines = 0
    self_loops = 0
    duplicate_edges = 0

    for line in _clean_lines(edge_path):
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"Expected at least two node ids in {edge_path}: {line!r}")
        source, target = int(fields[0]), int(fields[1])
        raw_edge_lines += 1
        nodes.update((source, target))
        if source == target:
            self_loops += 1
            continue
        edge = (source, target) if source < target else (target, source)
        if edge in unique_edges:
            duplicate_edges += 1
        else:
            unique_edges.add(edge)

    node_file_present = node_path.is_file()
    if node_file_present:
        nodes.update(_read_nodes(node_path))
    if not nodes:
        raise ValueError(f"Graph contains no nodes: {edge_path}")

    return GraphInput(
        nodes=frozenset(nodes),
        edges=tuple(sorted(unique_edges)),
        raw_edge_lines=raw_edge_lines,
        self_loops_removed=self_loops,
        duplicate_edges_removed=duplicate_edges,
        node_file_present=node_file_present,
    )


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_alignment(path: Path) -> dict[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing alignment file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Alignment must be a JSON object: {path}")

    alignment = {int(source): int(target) for source, target in payload.items()}
    if not alignment:
        raise ValueError(f"Alignment is empty: {path}")
    if len(set(alignment.values())) != len(alignment):
        raise ValueError(f"Alignment must be one-to-one: {path}")
    return alignment


def load_raw_dataset(
    data_root: Path,
    dataset_name: str,
    args: argparse.Namespace,
) -> RawDataset:
    raw_dir = data_root / dataset_name / args.raw_dir_name
    if not raw_dir.is_dir():
        raise NotADirectoryError(
            f"Missing raw directory: {raw_dir}. Expected files under "
            f"data/<dataset>/{args.raw_dir_name}/."
        )

    g1_edge_path = raw_dir / args.g1_file
    g2_edge_path = raw_dir / args.g2_file
    g1_node_path = raw_dir / args.g1_nodes_file
    g2_node_path = raw_dir / args.g2_nodes_file
    alignment_path = raw_dir / args.alignment_file

    graph1 = _read_graph(g1_edge_path, g1_node_path)
    graph2 = _read_graph(g2_edge_path, g2_node_path)
    alignment = _read_alignment(alignment_path)

    missing_source = sorted(set(alignment) - set(graph1.nodes))
    missing_target = sorted(set(alignment.values()) - set(graph2.nodes))
    if missing_source or missing_target:
        raise ValueError(
            "Alignment references nodes absent from raw graphs: "
            f"G1 examples={missing_source[:10]}, G2 examples={missing_target[:10]}"
        )

    return RawDataset(
        name=dataset_name,
        raw_dir=raw_dir,
        graph1=graph1,
        graph2=graph2,
        alignment=alignment,
        input_hashes={
            args.g1_file: _sha256(g1_edge_path),
            args.g2_file: _sha256(g2_edge_path),
            args.g1_nodes_file: _sha256(g1_node_path),
            args.g2_nodes_file: _sha256(g2_node_path),
            args.alignment_file: _sha256(alignment_path),
        },
    )


def allocate_partition_counts(
    num_pairs: int,
    rho_source: float,
    rho_target: float,
) -> PartitionCounts:
    """Allocate all matched identities using largest-remainder rounding."""
    if num_pairs < 1:
        raise ValueError("Cannot partition an empty alignment")
    rho_source = _validate_ratio(rho_source, "rho_source")
    rho_target = _validate_ratio(rho_target, "rho_target")

    denominator = 1.0 / rho_source + 1.0 / rho_target - 1.0
    shared_ideal = num_pairs / denominator
    ideals = (
        shared_ideal,
        shared_ideal * (1.0 / rho_source - 1.0),
        shared_ideal * (1.0 / rho_target - 1.0),
    )
    counts = [math.floor(value) for value in ideals]
    remainder = num_pairs - sum(counts)
    order = sorted(
        range(3),
        key=lambda index: (-(ideals[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1

    partition = PartitionCounts(*counts)
    if sum(asdict(partition).values()) != num_pairs:
        raise RuntimeError("Internal partition allocation error")
    if partition.shared < 1:
        raise ValueError("Requested ratios leave no shared correspondence")
    if rho_source < 1.0 and partition.source_only < 1:
        raise ValueError("Dataset is too small to realize source-private nodes")
    if rho_target < 1.0 and partition.target_only < 1:
        raise ValueError("Dataset is too small to realize target-private nodes")
    return partition


def _induced_edges(edges: Sequence[Pair], nodes: frozenset[int]) -> tuple[Pair, ...]:
    return tuple((source, target) for source, target in edges if source in nodes and target in nodes)


def _connected_component_stats(nodes: frozenset[int], edges: Sequence[Pair]) -> dict[str, int | float]:
    adjacency = {node: [] for node in nodes}
    for source, target in edges:
        adjacency[source].append(target)
        adjacency[target].append(source)

    visited: set[int] = set()
    component_sizes: list[int] = []
    for start in sorted(nodes):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        component_sizes.append(size)

    isolates = sum(not adjacency[node] for node in nodes)
    largest = max(component_sizes, default=0)
    return {
        "num_components": len(component_sizes),
        "largest_component_nodes": largest,
        "largest_component_ratio": largest / len(nodes) if nodes else 0.0,
        "num_isolates": isolates,
    }


def construct_dataset(
    raw: RawDataset,
    rho_source: float,
    rho_target: float,
    construction_seed: int,
    raw_dir_relative: str,
) -> ConstructedDataset:
    counts = allocate_partition_counts(
        len(raw.alignment),
        rho_source,
        rho_target,
    )
    pairs = sorted(raw.alignment.items())
    random.Random(construction_seed).shuffle(pairs)

    shared_end = counts.shared
    source_only_end = shared_end + counts.source_only
    shared_pairs = pairs[:shared_end]
    source_only_pairs = pairs[shared_end:source_only_end]
    target_only_pairs = pairs[source_only_end:]

    source_nodes = frozenset(
        [source for source, _ in shared_pairs]
        + [source for source, _ in source_only_pairs]
    )
    target_nodes = frozenset(
        [target for _, target in shared_pairs]
        + [target for _, target in target_only_pairs]
    )
    alignment = dict(sorted(shared_pairs))
    source_edges = _induced_edges(raw.graph1.edges, source_nodes)
    target_edges = _induced_edges(raw.graph2.edges, target_nodes)

    realized_source = len(alignment) / len(source_nodes)
    realized_target = len(alignment) / len(target_nodes)
    excluded_source_private = len(set(raw.graph1.nodes) - set(raw.alignment))
    excluded_target_private = len(
        set(raw.graph2.nodes) - set(raw.alignment.values())
    )

    metadata: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "name": (
            f"{raw.name}__bpupga__rho_s{_ratio_token(rho_source)}"
            f"__rho_t{_ratio_token(rho_target)}__cseed{construction_seed}"
        ),
        "base_dataset": raw.name,
        "task": "bilateral_partial_unsupervised_plain_graph_alignment",
        "construction": "disjoint_identity_partition",
        "construction_seed": construction_seed,
        "original_private_policy": "discard",
        "requested_source_overlap_ratio": rho_source,
        "requested_target_overlap_ratio": rho_target,
        "source_overlap_ratio": realized_source,
        "target_overlap_ratio": realized_target,
        "partition": asdict(counts),
        "raw": {
            "directory": raw_dir_relative,
            "alignment_pairs": len(raw.alignment),
            "g1_nodes": len(raw.graph1.nodes),
            "g1_edges_after_normalization": len(raw.graph1.edges),
            "g2_nodes": len(raw.graph2.nodes),
            "g2_edges_after_normalization": len(raw.graph2.edges),
            "original_source_private_nodes_excluded": excluded_source_private,
            "original_target_private_nodes_excluded": excluded_target_private,
            "g1_self_loops_removed": raw.graph1.self_loops_removed,
            "g2_self_loops_removed": raw.graph2.self_loops_removed,
            "g1_duplicate_edges_removed": raw.graph1.duplicate_edges_removed,
            "g2_duplicate_edges_removed": raw.graph2.duplicate_edges_removed,
            "input_sha256": raw.input_hashes,
        },
        "output": {
            "g1_nodes": len(source_nodes),
            "g1_edges": len(source_edges),
            "g2_nodes": len(target_nodes),
            "g2_edges": len(target_edges),
            "ground_truth_pairs": len(alignment),
            "source_only_nodes": counts.source_only,
            "target_only_nodes": counts.target_only,
            "g1_connectivity": _connected_component_stats(source_nodes, source_edges),
            "g2_connectivity": _connected_component_stats(target_nodes, target_edges),
        },
        "invariants": {
            "ground_truth_is_one_to_one": True,
            "all_original_matched_identities_assigned_once": True,
            "discarded_pair_never_keeps_both_endpoints": True,
            "isolated_nodes_preserved_in_node_files": True,
        },
    }

    constructed = ConstructedDataset(
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        source_edges=source_edges,
        target_edges=target_edges,
        alignment=alignment,
        counts=counts,
        metadata=metadata,
    )
    validate_constructed_dataset(raw, constructed)
    return constructed


def validate_constructed_dataset(
    raw: RawDataset,
    constructed: ConstructedDataset,
) -> None:
    if len(constructed.alignment) != constructed.counts.shared:
        raise RuntimeError("Constructed alignment cardinality is inconsistent")
    if len(set(constructed.alignment.values())) != len(constructed.alignment):
        raise RuntimeError("Constructed alignment is not one-to-one")
    if not set(constructed.alignment).issubset(constructed.source_nodes):
        raise RuntimeError("A shared source endpoint is missing from G1")
    if not set(constructed.alignment.values()).issubset(constructed.target_nodes):
        raise RuntimeError("A shared target endpoint is missing from G2")

    source_only = constructed.source_nodes - set(constructed.alignment)
    target_only = constructed.target_nodes - set(constructed.alignment.values())
    if len(source_only) != constructed.counts.source_only:
        raise RuntimeError("Source-private node count is inconsistent")
    if len(target_only) != constructed.counts.target_only:
        raise RuntimeError("Target-private node count is inconsistent")

    for source, target in raw.alignment.items():
        source_present = source in constructed.source_nodes
        target_present = target in constructed.target_nodes
        is_shared = constructed.alignment.get(source) == target
        if is_shared != (source_present and target_present):
            raise RuntimeError(
                "A discarded correspondence retained both endpoints, or a shared "
                f"correspondence is incomplete: ({source}, {target})"
            )
        if not source_present and not target_present:
            raise RuntimeError(
                f"Original matched identity was assigned to neither graph: ({source}, {target})"
            )

    if any(
        source not in constructed.source_nodes or target not in constructed.source_nodes
        for source, target in constructed.source_edges
    ):
        raise RuntimeError("G1 contains an edge outside its declared node set")
    if any(
        source not in constructed.target_nodes or target not in constructed.target_nodes
        for source, target in constructed.target_edges
    ):
        raise RuntimeError("G2 contains an edge outside its declared node set")


def _ratio_token(value: float) -> str:
    rendered = f"{value:.6f}".rstrip("0").rstrip(".")
    return rendered.replace(".", "p")


def output_directory(
    data_root: Path,
    dataset_name: str,
    output_dir_name: str,
    rho_source: float,
    rho_target: float,
    construction_seed: int,
) -> Path:
    configuration = (
        f"rho_s{_ratio_token(rho_source)}_rho_t{_ratio_token(rho_target)}"
    )
    return (
        data_root
        / dataset_name
        / output_dir_name
        / configuration
        / f"construction_seed_{construction_seed}"
    )


def _write_nodes(path: Path, nodes: Iterable[int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for node in sorted(nodes):
            handle.write(f"{node}\n")


def _write_edges(path: Path, edges: Iterable[Pair]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for source, target in sorted(edges):
            handle.write(f"{source} {target}\n")


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_constructed_dataset(
    output_dir: Path,
    constructed: ConstructedDataset,
    force: bool,
) -> None:
    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.tmp-",
        dir=output_parent,
    ) as temporary_parent:
        staging = Path(temporary_parent) / output_dir.name
        staging.mkdir()
        _write_nodes(staging / "G1.nodes", constructed.source_nodes)
        _write_nodes(staging / "G2.nodes", constructed.target_nodes)
        _write_edges(staging / "G1.edgelist", constructed.source_edges)
        _write_edges(staging / "G2.edgelist", constructed.target_edges)
        _write_json(
            staging / "Alignment.json",
            {str(source): target for source, target in constructed.alignment.items()},
        )
        _write_json(staging / "metadata.json", constructed.metadata)

        if output_dir.exists():
            if not force:
                raise FileExistsError(f"Output already exists: {output_dir}")
            if output_dir.is_symlink() or not output_dir.is_dir():
                raise RuntimeError(f"Refusing to replace non-directory output: {output_dir}")
            shutil.rmtree(output_dir)
        os.replace(staging, output_dir)


def _relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root does not exist: {data_root}")
    if not args.datasets:
        raise ValueError("At least one dataset must be specified")
    if not args.seeds:
        raise ValueError("At least one construction seed must be specified")

    configurations = _requested_configurations(args)
    jobs: list[tuple[RawDataset, float, float, int, Path]] = []
    for dataset_name in args.datasets:
        raw = load_raw_dataset(data_root, dataset_name, args)
        for rho_source, rho_target in configurations:
            for construction_seed in dict.fromkeys(args.seeds):
                destination = output_directory(
                    data_root,
                    dataset_name,
                    args.output_dir_name,
                    rho_source,
                    rho_target,
                    construction_seed,
                )
                jobs.append(
                    (raw, rho_source, rho_target, construction_seed, destination)
                )

    destinations = [job[-1] for job in jobs]
    if len(destinations) != len(set(destinations)):
        raise RuntimeError("Duplicate output destinations were requested")
    existing = [path for path in destinations if path.exists()]
    if existing and not args.force and not args.dry_run:
        preview = "\n".join(f"  {path}" for path in existing[:10])
        raise FileExistsError(
            "One or more outputs already exist; no dataset was written. "
            f"Use --force to replace them:\n{preview}"
        )

    completed = []
    for raw, rho_source, rho_target, construction_seed, destination in jobs:
        constructed = construct_dataset(
            raw,
            rho_source,
            rho_target,
            construction_seed,
            raw_dir_relative=_relative_path(raw.raw_dir, data_root),
        )
        if not args.dry_run:
            write_constructed_dataset(destination, constructed, force=args.force)

        output_stats = constructed.metadata["output"]
        record = {
            "dataset": raw.name,
            "construction_seed": construction_seed,
            "requested_rho_source": rho_source,
            "requested_rho_target": rho_target,
            "realized_rho_source": constructed.metadata["source_overlap_ratio"],
            "realized_rho_target": constructed.metadata["target_overlap_ratio"],
            "g1_nodes": output_stats["g1_nodes"],
            "g1_edges": output_stats["g1_edges"],
            "g2_nodes": output_stats["g2_nodes"],
            "g2_edges": output_stats["g2_edges"],
            "ground_truth_pairs": output_stats["ground_truth_pairs"],
            "output": str(destination),
            "status": "validated" if args.dry_run else "written",
        }
        completed.append(record)
        print(json.dumps(record, ensure_ascii=False))

    print(
        json.dumps(
            {
                "status": "dry-run complete" if args.dry_run else "complete",
                "datasets": len(set(record["dataset"] for record in completed)),
                "outputs": len(completed),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()