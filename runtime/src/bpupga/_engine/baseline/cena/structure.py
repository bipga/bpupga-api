from __future__ import annotations

from collections import deque

import networkx as nx
import numpy as np


def degree_extrema(graph: nx.Graph, layers: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the official exact-hop minimum/maximum degree signatures."""

    nodes = list(range(graph.number_of_nodes()))
    degree = np.fromiter((graph.degree[node] for node in nodes), dtype=np.float64)
    maximum = np.zeros((layers + 1, len(nodes)), dtype=np.float64)
    minimum = np.zeros((layers + 1, len(nodes)), dtype=np.float64)

    adjacency = [tuple(graph.neighbors(node)) for node in nodes]
    for source in nodes:
        rings: list[list[int]] = [[] for _ in range(layers + 1)]
        rings[0].append(source)
        distance = {source: 0}
        queue: deque[int] = deque([source])
        while queue:
            node = queue.popleft()
            next_distance = distance[node] + 1
            if next_distance > layers:
                continue
            for neighbor in adjacency[node]:
                if neighbor in distance:
                    continue
                distance[neighbor] = next_distance
                rings[next_distance].append(neighbor)
                queue.append(neighbor)

        for layer, ring in enumerate(rings):
            if ring:
                ring_degree = degree[np.asarray(ring, dtype=np.int64)]
                maximum[layer, source] = float(ring_degree.max())
                minimum[layer, source] = float(ring_degree.min())
            # The reference implementation substitutes [0] for an empty ring.
    return maximum, minimum


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(
        values,
        totals,
        out=np.zeros_like(values),
        where=totals > 0.0,
    )


def _sorted_topk(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    # The reference code calls NumPy argsort through Pandas without specifying
    # a stable algorithm, whose default is quicksort.
    indices = np.argsort(-values, axis=1, kind="quicksort")[:, :k]
    return np.take_along_axis(values, indices, axis=1), indices


def _numpy_structural_neighbors(
    max1: np.ndarray,
    min1: np.ndarray,
    max2: np.ndarray,
    min2: np.ndarray,
    alpha: float,
    topk: int,
    block_size: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n1, n2 = max1.shape[1], max2.shape[1]
    row_k, column_k = min(topk, n2), min(topk, n1)
    row_values = np.empty((n1, row_k), dtype=np.float64)
    row_indices = np.empty((n1, row_k), dtype=np.int64)
    column_values = np.full((n2, column_k), -np.inf, dtype=np.float64)
    column_indices = np.full((n2, column_k), -1, dtype=np.int64)

    max1 = max1.astype(dtype, copy=False)
    min1 = min1.astype(dtype, copy=False)
    max2 = max2.astype(dtype, copy=False)
    min2 = min2.astype(dtype, copy=False)
    for start in range(0, n1, block_size):
        stop = min(start + block_size, n1)
        distance = np.zeros((stop - start, n2), dtype=dtype)
        for layer in range(max1.shape[0]):
            distance += np.abs(max1[layer, start:stop, None] - max2[layer, None, :])
            distance += np.abs(min1[layer, start:stop, None] - min2[layer, None, :])
        scores = np.exp(-0.5 * alpha * distance)

        block_values, block_indices = _sorted_topk(scores, row_k)
        row_values[start:stop] = block_values
        row_indices[start:stop] = block_indices

        local_k = min(column_k, stop - start)
        transposed_values, transposed_indices = _sorted_topk(scores.T, local_k)
        transposed_indices += start
        if start == 0 and local_k == column_k:
            column_values = transposed_values
            column_indices = transposed_indices
        else:
            candidates = np.concatenate((column_values, transposed_values), axis=1)
            candidate_indices = np.concatenate((column_indices, transposed_indices), axis=1)
            column_values, selected = _sorted_topk(candidates, column_k)
            column_indices = np.take_along_axis(candidate_indices, selected, axis=1)

    return (
        row_indices,
        _normalize_rows(row_values),
        column_indices,
        _normalize_rows(column_values),
    )


def _torch_structural_neighbors(
    max1: np.ndarray,
    min1: np.ndarray,
    max2: np.ndarray,
    min2: np.ndarray,
    alpha: float,
    topk: int,
    block_size: int,
    dtype_name: str,
    device_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("CUDA structural matching requires PyTorch") from error

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device_name!r} was requested but CUDA is unavailable")
    dtype = torch.float32 if dtype_name == "float32" else torch.float64
    max1_tensor = torch.as_tensor(max1, device=device, dtype=dtype)
    min1_tensor = torch.as_tensor(min1, device=device, dtype=dtype)
    max2_tensor = torch.as_tensor(max2, device=device, dtype=dtype)
    min2_tensor = torch.as_tensor(min2, device=device, dtype=dtype)
    n1, n2 = max1.shape[1], max2.shape[1]
    row_k, column_k = min(topk, n2), min(topk, n1)
    row_values = np.empty((n1, row_k), dtype=np.float64)
    row_indices = np.empty((n1, row_k), dtype=np.int64)
    column_values = torch.full((n2, column_k), -torch.inf, device=device, dtype=dtype)
    column_indices = torch.full((n2, column_k), -1, device=device, dtype=torch.long)

    try:
        with torch.inference_mode():
            for start in range(0, n1, block_size):
                stop = min(start + block_size, n1)
                distance = torch.zeros((stop - start, n2), device=device, dtype=dtype)
                for layer in range(max1.shape[0]):
                    distance.add_(
                        torch.abs(max1_tensor[layer, start:stop, None] - max2_tensor[layer, None, :])
                    )
                    distance.add_(
                        torch.abs(min1_tensor[layer, start:stop, None] - min2_tensor[layer, None, :])
                    )
                scores = torch.exp(distance.mul_(-0.5 * alpha))
                values, indices = torch.topk(scores, k=row_k, dim=1, sorted=True)
                row_values[start:stop] = values.cpu().numpy()
                row_indices[start:stop] = indices.cpu().numpy()

                local_k = min(column_k, stop - start)
                local_values, local_indices = torch.topk(
                    scores, k=local_k, dim=0, sorted=True
                )
                local_values = local_values.T
                local_indices = local_indices.T + start
                candidate_values = torch.cat((column_values, local_values), dim=1)
                candidate_indices = torch.cat((column_indices, local_indices), dim=1)
                column_values, selected = torch.topk(
                    candidate_values, k=column_k, dim=1, sorted=True
                )
                column_indices = torch.gather(candidate_indices, 1, selected)
            column_values_array = column_values.cpu().numpy().astype(np.float64)
            column_indices_array = column_indices.cpu().numpy()
    except torch.OutOfMemoryError as error:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError(
            "CENA structural matching ran out of GPU memory; reduce structure_block_size"
        ) from error

    return (
        row_indices,
        _normalize_rows(row_values),
        column_indices_array,
        _normalize_rows(column_values_array),
    )


def structural_neighbors(
    graph1: nx.Graph,
    graph2: nx.Graph,
    layers: int,
    alpha: float,
    topk: int,
    block_size: int,
    dtype_name: str,
    device_name: str,
) -> tuple[dict[int, list[int]], dict[int, np.ndarray], dict[int, list[int]], dict[int, np.ndarray]]:
    """Compute the bidirectional structural Top-k used by CENALP."""

    max1, min1 = degree_extrema(graph1, layers)
    max2, min2 = degree_extrema(graph2, layers)
    # Exact transforms from the reference ``structing`` implementation.
    max1 = np.log(max1 + np.e)
    max2 = np.log(max2 + np.e)
    min1 = np.log(min1 + 1.0)
    min2 = np.log(min2 + 1.0)
    if device_name == "auto":
        try:
            import torch

            device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device_name = "cpu"

    if device_name == "cpu":
        dtype = np.float32 if dtype_name == "float32" else np.float64
        indices1, weights1, indices2, weights2 = _numpy_structural_neighbors(
            max1, min1, max2, min2, alpha, topk, block_size, dtype
        )
    else:
        indices1, weights1, indices2, weights2 = _torch_structural_neighbors(
            max1,
            min1,
            max2,
            min2,
            alpha,
            topk,
            block_size,
            dtype_name,
            device_name,
        )

    neighbors1 = {node: indices1[node].astype(int).tolist() for node in range(indices1.shape[0])}
    probabilities1 = {node: weights1[node] for node in range(weights1.shape[0])}
    neighbors2 = {node: indices2[node].astype(int).tolist() for node in range(indices2.shape[0])}
    probabilities2 = {node: weights2[node] for node in range(weights2.shape[0])}
    return neighbors1, probabilities1, neighbors2, probabilities2
