from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

try:
    from tqdm import trange
except ImportError:  # progress output is optional
    def trange(*args, **kwargs):
        kwargs.pop("desc", None)
        kwargs.pop("disable", None)
        return range(*args, **kwargs)


@dataclass(frozen=True)
class WalkContext:
    graph1: nx.Graph
    graph2: nx.Graph
    within_graph_probability: float
    neighbors1: dict[int, list[int]]
    probabilities1: dict[int, np.ndarray]
    neighbors2: dict[int, list[int]]
    probabilities2: dict[int, np.ndarray]
    anchors1_to_2: dict[int, int]
    anchors2_to_1: dict[int, int]
    walk_length: int
    offset: int


_WORKER_CONTEXT: WalkContext | None = None


def _initialize_worker(context: WalkContext) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _cross_graph_node(
    node: int,
    current_graph: int,
    context: WalkContext,
    python_rng: random.Random,
) -> tuple[int, int]:
    if current_graph == 1:
        if node in context.anchors1_to_2:
            return context.anchors1_to_2[node], 2
        candidates = context.neighbors1[node]
        weights = context.probabilities1[node]
        return int(python_rng.choices(candidates, weights=weights, k=1)[0]), 2
    if node in context.anchors2_to_1:
        return context.anchors2_to_1[node], 1
    candidates = context.neighbors2[node]
    weights = context.probabilities2[node]
    return int(python_rng.choices(candidates, weights=weights, k=1)[0]), 1


def _one_walk(
    start_node: int,
    start_graph: int,
    context: WalkContext,
    numpy_rng: np.random.Generator,
    python_rng: random.Random,
) -> list[int]:
    node = start_node
    current_graph = start_graph
    tokens = [node if current_graph == 1 else node + context.offset]
    for _ in range(context.walk_length - 1):
        if numpy_rng.random() < context.within_graph_probability:
            graph = context.graph1 if current_graph == 1 else context.graph2
            graph_neighbors = tuple(graph.neighbors(node))
            # Isolated nodes are treated as implicit self-loops. This does not
            # alter the reference behavior on graphs without isolated nodes.
            if graph_neighbors:
                node = int(graph_neighbors[int(numpy_rng.integers(len(graph_neighbors)))])
        else:
            node, current_graph = _cross_graph_node(
                node, current_graph, context, python_rng
            )
        tokens.append(node if current_graph == 1 else node + context.offset)
    return tokens


def _walk_pass(
    context: WalkContext,
    source_nodes: list[int],
    target_nodes: list[int],
    seed: int,
) -> list[list[int]]:
    numpy_rng = np.random.default_rng(seed)
    python_rng = random.Random(seed)
    walks = [
        _one_walk(node, 1, context, numpy_rng, python_rng)
        for node in source_nodes
    ]
    walks.extend(
        _one_walk(node, 2, context, numpy_rng, python_rng)
        for node in target_nodes
    )
    return walks


def _worker_walk_pass(
    pass_index: int,
    source_nodes: list[int],
    target_nodes: list[int],
    seed: int,
    part_path: str,
) -> tuple[int, str]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("CENA walk worker was not initialized")
    walks = _walk_pass(
        _WORKER_CONTEXT, source_nodes, target_nodes, seed
    )
    path = Path(part_path)
    with path.open("w", encoding="utf-8") as handle:
        _write_walks(handle, walks)
    return pass_index, str(path)


def _write_walks(handle, walks: list[list[int]]) -> None:
    for walk in walks:
        handle.write(" ".join(map(str, walk)))
        handle.write("\n")


def write_walk_corpus(
    output_path: Path,
    graph1: nx.Graph,
    graph2: nx.Graph,
    within_graph_probability: float,
    neighbors1: dict[int, list[int]],
    probabilities1: dict[int, np.ndarray],
    neighbors2: dict[int, list[int]],
    probabilities2: dict[int, np.ndarray],
    anchors1: list[int],
    anchors2: list[int],
    num_walks: int,
    walk_length: int,
    seed: int,
    multi_process: bool,
    workers: int,
    show_progress: bool,
) -> None:
    """Write the CENALP cross-graph walk corpus in deterministic pass order."""

    offset = max(graph1.number_of_nodes(), graph2.number_of_nodes())
    context = WalkContext(
        graph1=graph1,
        graph2=graph2,
        within_graph_probability=within_graph_probability,
        neighbors1=neighbors1,
        probabilities1=probabilities1,
        neighbors2=neighbors2,
        probabilities2=probabilities2,
        anchors1_to_2=dict(zip(anchors1, anchors2)),
        anchors2_to_1=dict(zip(anchors2, anchors1)),
        walk_length=walk_length,
        offset=offset,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_nodes = list(range(graph1.number_of_nodes()))
    target_nodes = list(range(graph2.number_of_nodes()))
    seed_rng = random.Random(seed)

    if not multi_process or workers == 1:
        with output_path.open("w", encoding="utf-8") as handle:
            for _ in trange(
                num_walks,
                desc="CENA random walks",
                disable=not show_progress,
            ):
                pass_seed = seed_rng.getrandbits(63)
                _write_walks(
                    handle,
                    _walk_pass(context, source_nodes, target_nodes, pass_seed),
                )
        return

    part_paths: dict[int, Path] = {}
    try:
        with ProcessPoolExecutor(
            max_workers=min(workers, num_walks),
            initializer=_initialize_worker,
            initargs=(context,),
        ) as executor:
            futures = {}
            for pass_index in range(num_walks):
                seed_rng.shuffle(source_nodes)
                seed_rng.shuffle(target_nodes)
                part_path = output_path.with_name(
                    f"{output_path.name}.part{pass_index:03d}"
                )
                future = executor.submit(
                    _worker_walk_pass,
                    pass_index,
                    source_nodes.copy(),
                    target_nodes.copy(),
                    seed_rng.getrandbits(63),
                    str(part_path),
                )
                futures[future] = pass_index
            for future in as_completed(futures):
                pass_index, raw_path = future.result()
                part_paths[pass_index] = Path(raw_path)

        with output_path.open("wb") as output:
            for pass_index in range(num_walks):
                with part_paths[pass_index].open("rb") as part:
                    while chunk := part.read(1024 * 1024):
                        output.write(chunk)
    finally:
        for part_path in part_paths.values():
            part_path.unlink(missing_ok=True)
