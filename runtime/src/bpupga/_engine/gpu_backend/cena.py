"""CENA single-embedding all-algorithm CUDA implementation, revision 261103.

This is an explicitly versioned implementation of the author's walk and
skip-gram objectives, not a numerical-equivalence claim for gensim's online SGD.
This revision scales sparse gradients by their embedding-row occurrence count
to prevent shared Huffman nodes from taking batch-size-amplified SGD steps.
Huffman construction, native greedy and graph statistics execute on GPU.
Only graph edge import, RAM corpus storage, launch and artifact export use CPU. Exact BFS shell extrema, structural comparisons, walks, subsampling,
context/negative sampling, *both* training losses and cosine products execute on
the selected torch device. CPU is accepted only when explicitly requested, to
make the numerical and protocol tests possible; CUDA never falls back to CPU.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import time

import torch
from torch import nn
from torch.nn import functional as F


IMPLEMENTATION = "cena_all_algorithm_device_frequency_scaled_261103_v3"
WINDOW = 5
SUBSAMPLE = 1e-3
INITIAL_ALPHA = 0.025
MIN_ALPHA = 0.0001


def _check_graph(graph):
    if not len(graph):
        raise ValueError("CENA requires a nonempty source and target graph")
    if set(graph) != set(range(len(graph))):
        raise ValueError("CENA requires contiguous graph-local integer node IDs")


def _shell_features_device(graph, layers, device, block_size=128):
    """Exact BFS shell min/max reductions on device, with bounded frontiers."""
    if graph.is_directed():
        # The benchmark and author adapter use undirected graphs; silently
        # symmetrizing an unexpected directed graph would alter the objective.
        raise ValueError("CENA GPU exact-hop features require undirected graphs")
    packet = graph if isinstance(graph, _DeviceGraph) else _DeviceGraph(graph, device)
    adjacency, degree = packet.adjacency, packet.degree
    maxima = torch.zeros((len(graph), layers + 1), dtype=torch.float64, device=device)
    minima = torch.zeros_like(maxima)
    for start in range(0, len(graph), block_size):
        stop = min(start + block_size, len(graph))
        frontier = torch.zeros((len(graph), stop-start), dtype=torch.float64, device=device)
        frontier[torch.arange(start, stop, device=device), torch.arange(stop-start, device=device)] = 1
        visited = frontier.bool()
        for hop in range(layers + 1):
            mask = frontier.T.bool()
            maxima[start:stop, hop] = torch.where(mask, degree[None, :], 0.).amax(1)
            lower = torch.where(mask, degree[None, :], torch.inf).amin(1)
            minima[start:stop, hop] = torch.where(mask.any(1), lower, 0.)
            if hop < layers:
                reached = (torch.sparse.mm(adjacency, frontier) > 0) & ~visited
                visited |= reached
                frontier = reached.to(torch.float64)
    return torch.cat((torch.log(maxima + math.e), torch.log(minima + 1.0)), dim=1)


def _stable_nearest(distances, count):
    """Small exact top-k with node-index tie breaks and no full argsort matrix."""
    indices = []
    values = []
    for _ in range(count):
        value, index = distances.min(dim=1)
        indices.append(index)
        values.append(value)
        distances.scatter_(1, index[:, None], torch.inf)
    return torch.stack(indices, 1), torch.stack(values, 1)


def _structural_transitions(features1, features2, alpha, block_size=256,
                            max_block_entries=2_000_000):
    """Exact all-candidate structural distance on device; retain author top ten.

    Log-space normalization prevents exp underflow; there is no similarity
    approximation. Equal structural similarities select the lower graph-local
    node index, rather than relying on NumPy quicksort's unspecified tie order.
    """
    device = features1.device
    n1, n2 = len(features1), len(features2)
    width = min(10, max(n1, n2))
    neighbors = torch.empty((n1 + n2, width), dtype=torch.int64, device=device)
    probabilities = torch.zeros((n1 + n2, width), dtype=torch.float64, device=device)
    for source, target, row_offset, target_offset in (
        (features1, features2, 0, n1), (features2, features1, n1, 0)
    ):
        k = min(10, len(target))
        step = max(1, min(block_size, max_block_entries // len(target)))
        for start in range(0, len(source), step):
            distances = torch.cdist(source[start:start + step], target, p=1) * 0.5
            index, value = _stable_nearest(distances, k)
            rows = slice(row_offset + start, row_offset + start + len(index))
            neighbors[rows, :k] = index + target_offset
            if k < width:
                neighbors[rows, k:] = index[:, -1:] + target_offset
            probabilities[rows, :k] = torch.softmax(-alpha * value, dim=1)
    torch._assert_async(torch.isfinite(probabilities).all(), "Non-finite CENA transition probabilities")
    # CDF's final entry must be exactly one, including float64 rounding.
    cdf = probabilities.cumsum(1)
    cdf[:, -1] = 1.0
    return neighbors, probabilities, cdf


class _DeviceGraph:
    """One NetworkX edge import; every statistic and CSR operation is on device."""
    def __init__(self, graph, device):
        _check_graph(graph)
        if graph.is_directed() or graph.is_multigraph():
            raise ValueError("CENA requires simple undirected graphs")
        self.n = len(graph)
        self.edges = torch.tensor(list(graph.to_directed().edges()), dtype=torch.int64,
                                  device=device).reshape(-1, 2).T.contiguous()
        row, column = self.edges
        values = torch.ones(row.numel(), device=device, dtype=torch.float64)
        self.adjacency = torch.sparse_coo_tensor(self.edges, values, (self.n, self.n)).coalesce()
        self.degree = torch.bincount(row, minlength=self.n).to(torch.float64)
        self.degree.scatter_add_(0, row[row == column], torch.ones_like(row[row == column], dtype=torch.float64))
        self.device = torch.device(device)

    def __len__(self):
        return self.n

    def is_directed(self):
        return False


def _csr_neighbors(g1, g2, device):
    n1 = len(g1)
    edges = torch.cat((g1.edges, g2.edges + n1), 1)
    order = torch.argsort(edges[0], stable=True)
    row, column = edges[:, order]
    degree = torch.bincount(row, minlength=n1 + len(g2))
    pointers = torch.cat((torch.zeros(1, dtype=torch.int64, device=device), degree.cumsum(0)))
    return pointers, torch.cat((column, torch.zeros(1, dtype=torch.int64, device=device)))


def _walk_batch(starts, length, q, pointers, adjacency, cross_neighbors,
                cross_cdf, generator):
    device = starts.device
    walk = torch.empty((len(starts), length), dtype=torch.int64, device=device)
    node = starts.clone()
    walk[:, 0] = node
    for position in range(1, length):
        local = torch.rand(len(node), device=device, generator=generator) < q
        count = pointers[node + 1] - pointers[node]
        choice = (torch.rand(len(node), device=device, generator=generator)
                  * count.clamp_min(1)).to(torch.int64)
        location = pointers[node] + choice
        adjacent = torch.where(count > 0, adjacency[location], node)
        threshold = torch.rand((len(node), 1), dtype=torch.float64,
                               device=device, generator=generator)
        slot = (threshold > cross_cdf[node]).sum(1).clamp_max(cross_cdf.shape[1] - 1)
        cross = cross_neighbors[node, slot]
        node = torch.where(local, adjacent, cross)
        walk[:, position] = node
    return walk


def _generate_corpus(g1, g2, config, transitions, device, generator, batch_size):
    """Generate the declared walk budget once on device; replay RAM corpus epochs.

    A CPU int32 corpus bounds GPU memory without an author random_walks.txt
    round trip. Every original walk/token is retained, including isolates.
    """
    vocab_size = len(g1) + len(g2)
    if vocab_size >= 2**31:
        raise ValueError("CENA RAM corpus int32 token limit exceeded")
    rows = config.num_walks * vocab_size
    corpus = torch.empty((rows, config.walk_length), dtype=torch.int32, device="cpu")
    counts = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    pointers, adjacency = _csr_neighbors(g1, g2, device)
    neighbors, _, cdf = transitions
    for repeat in range(config.num_walks):
        for start in range(0, vocab_size, batch_size):
            starts = torch.arange(start, min(start + batch_size, vocab_size), device=device)
            walks = _walk_batch(starts, config.walk_length, config.q, pointers,
                                adjacency, neighbors, cdf, generator)
            row = repeat * vocab_size + start
            corpus[row:row + len(walks)].copy_(walks.to("cpu", dtype=torch.int32))
            counts += torch.bincount(walks.reshape(-1), minlength=vocab_size)
    return corpus, counts


import triton
import triton.language as tl


@triton.jit
def _huffman_merge_kernel(Counts, Order, Weights, Parents, Codes, N: tl.constexpr):
    # Sorted leaves and nondecreasing internal weights form two priority queues.
    # (weight,node_id) ordering exactly matches heapq, including internal ties.
    leaf = 0
    internal = N
    next_node = N
    while next_node < 2 * N - 1:
        first_weight = tl.full((), 0, tl.int64)
        for position in range(2):
            leaf_node = tl.load(Order + tl.minimum(leaf, N - 1))
            leaf_weight = tl.load(Counts + leaf_node)
            inner_weight = tl.load(Weights + tl.minimum(internal, 2 * N - 2))
            use_leaf = (leaf < N) & ((internal >= next_node) | (leaf_weight <= inner_weight))
            node = tl.where(use_leaf, leaf_node, internal)
            weight = tl.where(use_leaf, leaf_weight, inner_weight)
            leaf += use_leaf.to(tl.int32)
            internal += (~use_leaf).to(tl.int32)
            tl.store(Parents + node, next_node)
            tl.store(Codes + node, position)
            if position == 0:
                first_weight = weight
            else:
                tl.store(Weights + next_node, first_weight + weight)
        next_node += 1


def _huffman_paths(counts):
    """Exact integer Huffman tree and token-ID ties; construction stays on GPU.

    A positive int64 total bounds any depth by 94 (Fibonacci lower bound).
    Padding is excluded from every HS loss/gradient occurrence count. Columns
    are compacted by a device mask, retaining the prior root-to-leaf paths.
    """
    if counts.device.type != "cuda":
        raise RuntimeError("GPU Huffman requires CUDA; no CPU fallback")
    counts = counts.to(torch.int64)
    n = len(counts)
    torch._assert_async((counts > 0).all(), "CENA vocabulary contains an unobserved node")
    order = torch.argsort(counts, stable=True)
    weights = torch.zeros(2 * n - 1, dtype=torch.int64, device=counts.device)
    parents = torch.full_like(weights, -1)
    codes = torch.zeros_like(weights, dtype=torch.float32)
    _huffman_merge_kernel[(1,)](counts, order, weights, parents, codes, n, num_warps=1)
    width = min(max(n - 1, 0), 94)
    node = torch.arange(n, device=counts.device)
    depth = torch.zeros(n, dtype=torch.int64, device=counts.device)
    reverse_paths = torch.zeros((n, width), dtype=torch.int64, device=counts.device)
    reverse_labels = torch.zeros((n, width), dtype=torch.float32, device=counts.device)
    for step in range(width):
        parent = parents[node]
        active = parent >= 0
        reverse_paths[:, step] = torch.where(active, parent - n, 0)
        reverse_labels[:, step] = torch.where(active, 1. - codes[node], 0.)
        depth += active
        node = torch.where(active, parent, node)
    position = torch.arange(width, device=counts.device)[None, :]
    mask = position < depth[:, None]
    index = (depth[:, None] - 1 - position).clamp_min(0)
    paths = torch.where(mask, reverse_paths.gather(1, index), 0)
    labels = torch.where(mask, reverse_labels.gather(1, index), 0.)
    used = mask.any(0).nonzero(as_tuple=True)[0]
    return paths[:, used], labels[:, used], mask[:, used].float()


def _context_pairs(walks, keep_probability, generator):
    """Gensim-style subsampling *before* dynamic window selection, on device.

    As in gensim's skip-gram implementation, context is the input embedding and
    the central token is the HS/negative target. Pairs never cross walk borders.
    """
    rows, length = walks.shape
    positions = torch.arange(length, device=walks.device).expand(rows, -1)
    keep = (torch.rand(walks.shape, device=walks.device, generator=generator)
            < keep_probability[walks])
    counts = keep.sum(1)
    # Compact each subsampled sentence, maintaining retained-token order.
    order = torch.argsort(torch.where(keep, positions, positions + length), dim=1)
    compact = torch.gather(walks, 1, order)
    radius = torch.randint(1, WINDOW + 1, walks.shape, device=walks.device,
                           generator=generator)
    inputs = []
    targets = []
    for delta in range(-WINDOW, WINDOW + 1):
        if delta == 0:
            continue
        other = positions + delta
        valid = ((positions < counts[:, None]) & (other >= 0)
                 & (other < counts[:, None]) & (radius >= abs(delta)))
        context = torch.gather(compact, 1, other.clamp(0, length - 1))
        inputs.append(context[valid])
        targets.append(compact[valid])
    return torch.cat(inputs), torch.cat(targets), counts.sum()


class _JointSkipGram(nn.Module):
    """Single joint input embedding with independent HS and SGNS output tables."""
    def __init__(self, vocab_size, dimension, device, generator):
        super().__init__()
        self.input = nn.Embedding(vocab_size, dimension, sparse=True, device=device)
        self.hs = nn.Embedding(max(vocab_size - 1, 1), dimension, sparse=True, device=device)
        self.negative = nn.Embedding(vocab_size, dimension, sparse=True, device=device)
        with torch.no_grad():
            self.input.weight.uniform_(-0.5 / dimension, 0.5 / dimension,
                                       generator=generator)
            self.hs.weight.zero_()
            self.negative.weight.zero_()

    def objective(self, inputs, targets, negatives, paths, labels, mask):
        vector = self.input(inputs)
        nodes = paths[targets]
        hs_logits = (vector[:, None, :] * self.hs(nodes)).sum(2)
        hs_loss = (F.binary_cross_entropy_with_logits(
            hs_logits, labels[targets], reduction="none") * mask[targets]).sum(1)
        positive = (vector * self.negative(targets)).sum(1)
        negative_logits = (vector[:, None, :] * self.negative(negatives)).sum(2)
        # Author gensim skips a negative draw equal to the positive target; it
        # does not redraw. Duplicated negative samples remain permitted.
        negative_mask = negatives != targets[:, None]
        negative_loss = (F.softplus(negative_logits) * negative_mask).sum(1)
        sgns_loss = F.softplus(-positive) + negative_loss
        return hs_loss + sgns_loss, hs_loss, sgns_loss, negative_mask.sum()



def _normalize_sparse_row_gradient(parameter, occurrences):
    """Scale each sparse embedding row by its actual use count in this batch.

    This is an explicit positive diagonal preconditioner of the summed
    skip-gram gradient. It avoids multiplying an online learning rate by up to
    thousands at shared Huffman nodes. No contribution, sample, loss term or
    epoch is discarded. Padding and skipped negative collisions are excluded
    from the occurrence counts rather than diluting valid row gradients.
    """
    gradient = parameter.grad
    if gradient is None or not gradient.is_sparse:
        raise RuntimeError("CENA frequency-scaled SGD requires sparse embedding gradients")
    gradient = gradient.coalesce()
    counts = torch.bincount(occurrences.reshape(-1), minlength=parameter.shape[0])
    scale = counts[gradient.indices()[0]].clamp_min(1).to(gradient.dtype)
    gradient.values().div_(scale[:, None])
    parameter.grad = gradient
    return counts.max()


def _frequency_scaled_step(model, optimizer, inputs, targets, negatives, paths, mask):
    active_hs = paths[targets][mask[targets] > 0]
    active_negative = negatives[negatives != targets[:, None]]
    frequencies = torch.stack((
        _normalize_sparse_row_gradient(model.input.weight, inputs),
        _normalize_sparse_row_gradient(model.hs.weight, active_hs),
        _normalize_sparse_row_gradient(model.negative.weight, torch.cat((targets, active_negative))),
    ))
    optimizer.step()
    return frequencies


class _DeviceSGD:
    """Sparse SGD with device tensor learning rate and device row updates."""
    def __init__(self, parameters, lr):
        self.parameters = tuple(parameters)
        self.lr = lr

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self):
        for parameter in self.parameters:
            gradient = parameter.grad.coalesce()
            parameter.index_add_(0, gradient.indices()[0], -self.lr * gradient.values())

def _train_embedding(corpus, counts, config, device, generator, pair_batch_size,
                     walk_batch_size=256):
    vocab_size = len(counts)
    points, targets, mask = _huffman_paths(counts)
    paths = torch.as_tensor(points, device=device)
    labels = torch.as_tensor(targets, device=device)
    masks = torch.as_tensor(mask, device=device)
    count_float = counts.to(torch.float64)
    threshold = SUBSAMPLE * count_float.sum()
    keep = ((count_float / threshold).sqrt() + 1) * (threshold / count_float)
    keep = keep.clamp_max(1).to(torch.float32)
    noise = count_float.pow(0.75)
    noise_cdf = (noise / noise.sum()).cumsum(0)
    noise_cdf[-1] = 1.0
    model = _JointSkipGram(vocab_size, config.embedding_dim, device, generator)
    optimizer = _DeviceSGD(model.parameters(), torch.tensor(INITIAL_ALPHA, device=device))
    # Keep the declared minibatch policy and all training pairs. Scale each
    # embedding-row gradient by actual use count before the SGD update.
    effective_batch_size = min(pair_batch_size, max(1, vocab_size // 2))
    raw_words = corpus.numel()
    epoch_trace = []
    total_steps = 0
    maximum_row_frequency = torch.zeros(3, dtype=torch.int64, device=device)
    for epoch in range(config.embedding_epochs):
        sums = torch.zeros(6, dtype=torch.float64, device=device)
        pairs_seen = 0
        epoch_steps = 0
        for start in range(0, len(corpus), walk_batch_size):
            walks = corpus[start:start + walk_batch_size].to(device=device, dtype=torch.int64)
            inputs, central, retained = _context_pairs(walks, keep, generator)
            sums[3] += retained
            progress = torch.tensor(epoch * raw_words + start * corpus.shape[1],
                                    dtype=torch.float64, device=device) / (config.embedding_epochs * raw_words)
            lr = (INITIAL_ALPHA - (INITIAL_ALPHA - MIN_ALPHA) * progress).clamp_min(MIN_ALPHA)
            optimizer.lr = lr
            for position in range(0, len(inputs), effective_batch_size):
                a = inputs[position:position + effective_batch_size]
                b = central[position:position + effective_batch_size]
                random = torch.rand((len(a), config.negative_samples),
                                    dtype=torch.float64, device=device, generator=generator)
                negative = torch.searchsorted(noise_cdf, random).clamp_max(vocab_size - 1)
                optimizer.zero_grad(set_to_none=True)
                loss, hs_loss, sgns_loss, valid_negative = model.objective(
                    a, b, negative, paths, labels, masks)
                batch_loss = loss.sum()
                batch_loss.backward()
                frequencies = _frequency_scaled_step(model, optimizer, a, b, negative, paths, masks)
                maximum_row_frequency = torch.maximum(maximum_row_frequency, frequencies)
                sums[0] += batch_loss.detach()
                sums[1] += hs_loss.detach().sum()
                sums[2] += sgns_loss.detach().sum()
                sums[4] += masks[b].sum()
                sums[5] += valid_negative
                pairs_seen += len(a)
                total_steps += 1
                epoch_steps += 1
                if total_steps % 256 == 0:
                    torch._assert_async(torch.isfinite(sums[:3]).all(), "Non-finite CENA CUDA training loss")
        values = sums.cpu().tolist()
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("Non-finite CENA CUDA HS+SGNS epoch loss")
        for parameter in model.parameters():
            torch._assert_async(torch.isfinite(parameter).all(), "Non-finite CENA embedding parameters")
        epoch_trace.append(dict(epoch=epoch + 1, raw_walks=len(corpus), raw_tokens=raw_words,
            retained_tokens=int(values[3]), context_pairs=pairs_seen, optimizer_steps=epoch_steps,
            hs_decisions=int(values[4]), negative_draws=pairs_seen * config.negative_samples,
            accepted_negative_draws=int(values[5]), loss_sum=values[0], hs_loss_sum=values[1],
            sgns_loss_sum=values[2], mean_loss_per_pair=(values[0] / pairs_seen if pairs_seen else None),
            learning_rate_last=float(lr.cpu())))
    if not any(item["context_pairs"] for item in epoch_trace):
        raise RuntimeError("CENA subsampled corpus has no context pairs; no untrained fallback allowed")
    return model.input.weight.detach(), dict(
        objective="skipgram_hierarchical_softmax_plus_negative_sampling",
        hs=1, sg=1, negative=config.negative_samples, window=WINDOW, shrink_windows=True,
        subsample=SUBSAMPLE, negative_sampling_exponent=0.75,
        negative_collision_policy="skip_draw_equal_to_positive_without_resampling",
        optimizer="device_sparse_index_add_SGD_with_embedding_row_occurrence_preconditioner",
        gradient_reduction="sum_of_per_pair_objectives_then_per_row_occurrence_normalization",
        gradient_preconditioner="inverse_actual_occurrences_per_embedding_row_in_current_minibatch",
        padded_HS_nodes_counted=False, skipped_negative_collisions_counted=False,
        maximum_row_frequency=dict(zip(("input", "hierarchical_softmax", "negative_output"),
                                       maximum_row_frequency.cpu().tolist())),
        gradient_clipping=False, nonfinite_replacement=False,
        initial_learning_rate=INITIAL_ALPHA, minimum_learning_rate=MIN_ALPHA,
        learning_rate_schedule="linear_by_raw_corpus_tokens_across_declared_epochs",
        pair_batch_size_requested=pair_batch_size, pair_batch_size_effective=effective_batch_size,
        training_walk_batch_size=walk_batch_size, optimizer_steps=total_steps,
        epochs_requested=config.embedding_epochs, epochs_completed=len(epoch_trace),
        vocabulary_size=vocab_size, embedding_dimension=config.embedding_dim,
        huffman_max_depth=points.shape[1], huffman_tie_break="token_id",
        input_initialization="uniform(-0.5/dim,0.5/dim)", output_initialization="zeros",
        stochastic_backend="torch.Generator on selected device",
        reproducibility="seeded device RNG; no cross-device bitwise or CUDA atomic-update determinism claim",
        pair_order="walk-chunk then context-offset then sentence-position; GPU minibatches",
        numerical_loss="exact stable logistic BCE/softplus, no gensim sigmoid lookup table",
        gensim_online_sgd_equivalent=False,
        implementation_difference="same declared walk/epoch/objective budget; GPU RNG and frequency-scaled minibatch SGD change trajectory",
        epoch_trace=epoch_trace)


class CUDAComputeCENA:
    """Fit exactly one CENA joint embedding; no LP, seeds, GT or overlap inputs."""
    def __init__(self, config=None, kernels=None, *, device=None, pair_batch_size=4096,
                 walk_batch_size=4096, structural_block_size=256):
        if config is None:
            from baseline.author_adapters import SingleCENAConfig
            config = SingleCENAConfig()
        selected = device if device is not None else (kernels.device if kernels is not None else "cuda:0")
        self.device = torch.device(selected)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CENA CUDA was requested but is unavailable; CPU fallback is disabled")
        if self.device.type != "cuda":
            raise ValueError("All-device CENA requires CUDA; CPU fallback disabled")
        if kernels is not None and torch.device(kernels.device) != self.device:
            raise ValueError("CENA kernels and selected device disagree")
        if min(pair_batch_size, walk_batch_size, structural_block_size) < 1:
            raise ValueError("CENA batch sizes must be positive")
        if not 0 <= config.q <= 1 or config.layers < 0 or config.alpha <= 0:
            raise ValueError("Invalid CENA structural/walk configuration")
        if min(config.embedding_dim, config.embedding_epochs, config.num_walks, config.walk_length) < 1:
            raise ValueError("CENA embedding and walk budgets must be positive")
        if config.negative_samples < 0:
            raise ValueError("CENA negative_samples must be nonnegative")
        self.config = config
        self.kernels = kernels
        self.pair_batch_size = pair_batch_size
        self.walk_batch_size = walk_batch_size
        self.structural_block_size = structural_block_size

    def _count(self, stage):
        if self.kernels is not None:
            self.kernels.count(stage)

    def fit(self, g1, g2):
        import baseline  # Install the repository's bundled common.* alias.
        from gpu_backend.assignment import flat_greedy
        from common.result import AlignmentResult

        cfg = self.config
        started = time.perf_counter()
        _check_graph(g1)
        _check_graph(g2)
        g1, g2 = _DeviceGraph(g1, self.device), _DeviceGraph(g2, self.device)
        timings = {}
        generator = torch.Generator(device=self.device)
        generator.manual_seed(cfg.seed)

        def mark(name, since):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timings[name] = time.perf_counter() - since

        stage = time.perf_counter()
        f1 = _shell_features_device(g1, cfg.layers, self.device)
        f2 = _shell_features_device(g2, cfg.layers, self.device)
        self._count("cena_exact_hop_degree_extrema")
        mark("exact_bfs_shell_features_device", stage)
        stage = time.perf_counter()
        transitions = _structural_transitions(f1, f2, cfg.alpha, self.structural_block_size)
        self._count("cena_structural_all_candidate_top10")
        mark("structural_probabilities_device", stage)
        del f1, f2
        stage = time.perf_counter()
        corpus, counts = _generate_corpus(g1, g2, cfg, transitions, self.device,
                                          generator, self.walk_batch_size)
        self._count("cena_random_walks")
        mark("random_walks_device", stage)
        del transitions
        stage = time.perf_counter()
        embedding, training = _train_embedding(corpus, counts, cfg, self.device,
                                                generator, self.pair_batch_size)
        self._count("cena_skipgram_hs_negative_training")
        mark("joint_hs_sgns_training_device", stage)
        corpus_shape = list(corpus.shape)
        corpus_bytes = corpus.numel() * corpus.element_size()
        del corpus
        stage = time.perf_counter()
        embedding = F.normalize(embedding, p=2, dim=1)
        n1, n2 = len(g1), len(g2)
        x, y = embedding[:n1], embedding[n1:]
        score_device = x @ y.T
        self._count("cena_final_cosine")
        torch._assert_async(torch.isfinite(score_device).all(), "Non-finite CENA cosine score")
        mark("same_fit_cosine_device", stage)
        stage = time.perf_counter()
        rows, columns = flat_greedy(score_device, maximize=True, device=self.device)
        mark("greedy_decoder_device", stage)
        # The complete fit is finished before artifact/diagnostic export.
        score = score_device.cpu().numpy()
        pairs = list(map(tuple, torch.stack((rows, columns), 1).cpu().tolist()))
        x_numpy, y_numpy = x.cpu().numpy(), y.cpu().numpy()
        metadata = dict(config=asdict(cfg), implementation_revision=IMPLEMENTATION,
            official_commit="2073ba8a9bca180913b41355fe71f60d714cd6a2",
            algorithm_core="CENA_one_joint_embedding_no_LP_hs_plus_negative",
            author_components_numerically_equivalent=False,
            embedding_fit_count=1, link_prediction_calls=0, predicted_links_added=0,
            graph_modification=False, ground_truth_used_during_inference=False,
            safety={"isolated_walk_policy": "stay_at_current_node_for_local_step"},
            prediction_rule="explicit greedy assignment of single-embedding cosine",
            structural_distance="exact half_L1_of_log_shell_degree_maxima_and_minima",
            structural_topk=10, structural_tie_break="ascending_opposite_graph_node_index",
            structural_numeric_policy="float64 exact distances and log_space_probability_normalization",
            walk_budget=dict(num_walks=cfg.num_walks, walk_length=cfg.walk_length,
                             walk_count=cfg.num_walks * (n1 + n2),
                             token_count=cfg.num_walks * (n1 + n2) * cfg.walk_length),
            corpus=dict(shape=corpus_shape, bytes=corpus_bytes, storage="CPU_RAM_int32",
                        regenerated_per_epoch=False, disk_random_walk_file=False),
            token_identity=dict(author_target_offset=max(n1, n2),
                source_author_token="u", target_author_token="v+max(n1,n2)",
                compact_internal_bijection="source u -> u; target v+offset -> n1+v"),
            embedding_training=training, stage_runtime_sec=timings,
            compute_device=str(self.device), cuda_training=self.device.type == "cuda",
            gpu_stages=["exact_hop_degree_extrema", "all_candidate_structural_distance_and_top10", "random_walks",
                "subsampling_context_and_negative_sampling", "joint_skipgram_HS_and_negative_training",
                "same_fit_embedding_cosine", "Huffman_tree_and_paths", "stable_flatten_greedy_decoder"],
            cpu_stages=["one_graph_nodes_edges_import", "RAM_walk_corpus_storage_and_transfer",
                        "host_budget_kernel_launch", "artifact_and_diagnostic_export"],
            all_algorithm_decisions_on_gpu=True,
            execution_mode="all_algorithm_gpu", cpu_fallback=False,
            host_dispatch="fixed budgets and GPU-computed dynamic tensor shapes", dynamic_shape_sync=True,
            greedy_tie_break="stable_score_descending_then_flatten_index_ascending",
            greedy_tie_equivalent_to_native_stable_numpy=True)
        return AlignmentResult("CENA-single-no-LP", pairs, time.perf_counter() - started,
            score_matrix=score, larger_is_better=True, source_embeddings=x_numpy,
            target_embeddings=y_numpy, metadata=metadata)


# A readable alias for the factory while retaining a versioned implementation ID.
TensorCENA = CUDAComputeCENA
