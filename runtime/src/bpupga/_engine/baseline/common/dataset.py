from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import networkx as nx


@dataclass(frozen=True)
class GraphIndex:
    graph: nx.Graph
    old_to_new: Dict[int, int]
    new_to_old: Dict[int, int]


@dataclass(frozen=True)
class AlignmentDataset:
    root: Path
    name: str
    graph1: nx.Graph
    graph2: nx.Graph
    ground_truth: Dict[int, int]
    g1_old_to_new: Dict[int, int]
    g1_new_to_old: Dict[int, int]
    g2_old_to_new: Dict[int, int]
    g2_new_to_old: Dict[int, int]
    metadata: Dict[str, object]

    @property
    def source_only_nodes(self) -> set[int]:
        return set(self.graph1.nodes()) - set(self.ground_truth.keys())

    @property
    def target_only_nodes(self) -> set[int]:
        return set(self.graph2.nodes()) - set(self.ground_truth.values())

    @property
    def source_overlap_ratio(self) -> float:
        n = self.graph1.number_of_nodes()
        return len(self.ground_truth) / n if n else 0.0

    @property
    def target_overlap_ratio(self) -> float:
        n = self.graph2.number_of_nodes()
        return len(self.ground_truth) / n if n else 0.0

    @property
    def node_jaccard_overlap(self) -> float:
        n1 = self.graph1.number_of_nodes()
        n2 = self.graph2.number_of_nodes()
        shared = len(self.ground_truth)
        union = n1 + n2 - shared
        return shared / union if union else 0.0

    def pairs_to_original(
        self, pairs: Iterable[Tuple[int, int]]
    ) -> list[tuple[int, int]]:
        return [
            (self.g1_new_to_old[int(u)], self.g2_new_to_old[int(v)])
            for u, v in pairs
        ]

    def pairs_to_reindexed(
        self, pairs: Iterable[Tuple[int, int]]
    ) -> list[tuple[int, int]]:
        converted = []
        for u, v in pairs:
            if int(u) not in self.g1_old_to_new:
                raise ValueError(f"Unknown original G1 node id in prediction: {u}")
            if int(v) not in self.g2_old_to_new:
                raise ValueError(f"Unknown original G2 node id in prediction: {v}")
            converted.append((self.g1_old_to_new[int(u)], self.g2_old_to_new[int(v)]))
        return converted

    def summary(self) -> Dict[str, object]:
        def graph_statistics(graph: nx.Graph) -> Dict[str, int | float]:
            node_count = graph.number_of_nodes()
            component_sizes = [
                len(component) for component in nx.connected_components(graph)
            ]
            largest = max(component_sizes, default=0)
            return {
                "components": len(component_sizes),
                "isolates": nx.number_of_isolates(graph),
                "largest_component_ratio": largest / node_count if node_count else 0.0,
                "average_degree": (
                    2.0 * graph.number_of_edges() / node_count if node_count else 0.0
                ),
                "density": nx.density(graph),
            }

        mapped_source_edges = {
            tuple(sorted((self.ground_truth[u], self.ground_truth[v])))
            for u, v in self.graph1.edges()
            if u in self.ground_truth and v in self.ground_truth
        }
        matched_targets = set(self.ground_truth.values())
        target_core_edges = {
            tuple(sorted((u, v)))
            for u, v in self.graph2.edges()
            if u in matched_targets and v in matched_targets
        }
        common_core_edges = mapped_source_edges & target_core_edges
        union_core_edges = mapped_source_edges | target_core_edges
        n1 = self.graph1.number_of_nodes()
        n2 = self.graph2.number_of_nodes()
        shared = len(self.ground_truth)
        return {
            "name": self.name,
            "root": str(self.root),
            "construction_protocol": self.metadata.get("protocol", self.metadata.get("construction")),
            "g1_nodes": n1,
            "g1_edges": self.graph1.number_of_edges(),
            "g2_nodes": n2,
            "g2_edges": self.graph2.number_of_edges(),
            "ground_truth_pairs": shared,
            "source_only_nodes": len(self.source_only_nodes),
            "target_only_nodes": len(self.target_only_nodes),
            "source_overlap_ratio": self.source_overlap_ratio,
            "target_overlap_ratio": self.target_overlap_ratio,
            "node_jaccard_overlap": self.node_jaccard_overlap,
            "source_private_ratio": len(self.source_only_nodes) / n1 if n1 else 0.0,
            "target_private_ratio": len(self.target_only_nodes) / n2 if n2 else 0.0,
            "size_ratio": min(n1, n2) / max(n1, n2) if max(n1, n2) else 0.0,
            "g1_structure": graph_statistics(self.graph1),
            "g2_structure": graph_statistics(self.graph2),
            "ground_truth_core": {
                "g1_edges": len(mapped_source_edges),
                "g2_edges": len(target_core_edges),
                "common_edges": len(common_core_edges),
                "edge_jaccard": (
                    len(common_core_edges) / len(union_core_edges)
                    if union_core_edges
                    else 0.0
                ),
            },
        }


def _iter_clean_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line and not line.startswith("#"):
                yield line


def _read_node_file(path: Path) -> set[int]:
    nodes: set[int] = set()
    for line in _iter_clean_lines(path):
        fields = line.split()
        if len(fields) != 1:
            raise ValueError(f"Expected one node id per line in {path}, got: {line!r}")
        nodes.add(int(fields[0]))
    return nodes


def _read_edge_file(path: Path) -> tuple[list[tuple[int, int]], set[int]]:
    edges: list[tuple[int, int]] = []
    nodes: set[int] = set()
    for line in _iter_clean_lines(path):
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"Expected at least two node ids in {path}, got: {line!r}")
        u, v = int(fields[0]), int(fields[1])
        edges.append((u, v))
        nodes.update((u, v))
    return edges, nodes


def _load_graph(edge_path: Path, node_path: Path | None) -> GraphIndex:
    if not edge_path.exists():
        raise FileNotFoundError(f"Missing edge file: {edge_path}")

    edges, edge_nodes = _read_edge_file(edge_path)
    nodes = set(edge_nodes)
    if node_path is not None and node_path.exists():
        nodes.update(_read_node_file(node_path))

    if not nodes:
        raise ValueError(f"Graph has no nodes: {edge_path}")

    old_to_new = {old: new for new, old in enumerate(sorted(nodes))}
    new_to_old = {new: old for old, new in old_to_new.items()}

    graph = nx.Graph()
    graph.add_nodes_from(range(len(old_to_new)))
    graph.add_edges_from((old_to_new[u], old_to_new[v]) for u, v in edges)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    return GraphIndex(graph=graph, old_to_new=old_to_new, new_to_old=new_to_old)


def _load_json_object(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _validate_ground_truth(
    ground_truth: Mapping[int, int],
    num_source_nodes: int,
    num_target_nodes: int,
) -> None:
    invalid_sources = [u for u in ground_truth if not 0 <= u < num_source_nodes]
    invalid_targets = [v for v in ground_truth.values() if not 0 <= v < num_target_nodes]
    if invalid_sources or invalid_targets:
        raise ValueError(
            "Ground truth contains out-of-range nodes: "
            f"sources={invalid_sources[:5]}, targets={invalid_targets[:5]}"
        )
    targets = list(ground_truth.values())
    if len(targets) != len(set(targets)):
        raise ValueError("Alignment.json must define a one-to-one correspondence")


def load_dataset(
    root: str | Path,
    g1_file: str = "G1.edgelist",
    g2_file: str = "G2.edgelist",
    alignment_file: str = "Alignment.json",
    g1_nodes_file: str = "G1.nodes",
    g2_nodes_file: str = "G2.nodes",
    metadata_file: str = "metadata.json",
) -> AlignmentDataset:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist: {root_path}")

    g1 = _load_graph(root_path / g1_file, root_path / g1_nodes_file)
    g2 = _load_graph(root_path / g2_file, root_path / g2_nodes_file)

    alignment_path = root_path / alignment_file
    if not alignment_path.exists():
        raise FileNotFoundError(f"Missing alignment file: {alignment_path}")
    raw_alignment = _load_json_object(alignment_path)

    ground_truth: Dict[int, int] = {}
    missing_pairs: list[tuple[int, int]] = []
    for raw_u, raw_v in raw_alignment.items():
        old_u, old_v = int(raw_u), int(raw_v)
        if old_u not in g1.old_to_new or old_v not in g2.old_to_new:
            missing_pairs.append((old_u, old_v))
            continue
        ground_truth[g1.old_to_new[old_u]] = g2.old_to_new[old_v]

    if missing_pairs:
        preview = missing_pairs[:5]
        raise ValueError(
            "Alignment.json references nodes absent from the graph node sets; "
            f"examples: {preview}"
        )

    _validate_ground_truth(
        ground_truth,
        g1.graph.number_of_nodes(),
        g2.graph.number_of_nodes(),
    )

    metadata = _load_json_object(root_path / metadata_file)
    dataset = AlignmentDataset(
        root=root_path,
        name=str(metadata.get("name", root_path.name)),
        graph1=g1.graph,
        graph2=g2.graph,
        ground_truth=ground_truth,
        g1_old_to_new=g1.old_to_new,
        g1_new_to_old=g1.new_to_old,
        g2_old_to_new=g2.old_to_new,
        g2_new_to_old=g2.new_to_old,
        metadata=metadata,
    )

    expected_rho_s = metadata.get("source_overlap_ratio")
    expected_rho_t = metadata.get("target_overlap_ratio")
    # Synthetic-data metadata is often rounded (for example, 0.6667), so use
    # a small reporting tolerance while still catching inconsistent files.
    tolerance = 1e-4
    if expected_rho_s is not None and abs(float(expected_rho_s) - dataset.source_overlap_ratio) > tolerance:
        raise ValueError(
            "metadata source_overlap_ratio disagrees with realized dataset: "
            f"{expected_rho_s} vs {dataset.source_overlap_ratio}"
        )
    if expected_rho_t is not None and abs(float(expected_rho_t) - dataset.target_overlap_ratio) > tolerance:
        raise ValueError(
            "metadata target_overlap_ratio disagrees with realized dataset: "
            f"{expected_rho_t} vs {dataset.target_overlap_ratio}"
        )

    return dataset
