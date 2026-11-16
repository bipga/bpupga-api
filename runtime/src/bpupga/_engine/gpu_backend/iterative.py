"""Device-resident iMMNC and SparkAlign, including graph and anchor operations.

The host parses the input edge records once, launches kernels, reads dynamic
tensor shapes for allocation, and exports the final artifacts.  There is no
NumPy/SciPy numerical path and no device scalar controls a host algorithm loop.
CPU devices are supported only for small numerical regression tests.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import time
import torch
import baseline  # Install the frozen common/baselines import aliases.
from common.result import AlignmentResult


REVISION = 'paper_experiments'


def export(tensor):
    """Final artifact/diagnostic boundary; never called inside an algorithm."""
    return tensor.detach().cpu().numpy()


def pairs_export(pairs):
    return [tuple(pair) for pair in export(pairs).tolist()]


class Graph:
    """Parse topology once, then derive every graph statistic on the device."""
    def __init__(self, graph, device, dtype=torch.float64):
        self.n = len(graph)
        if self.n == 0 or set(graph) != set(range(self.n)):
            raise ValueError('Graphs must be nonempty with consecutive integer node IDs')
        self.device = torch.device(device)
        # Reading edge attributes and node IDs is the input serialization boundary.
        records = list(graph.edges(data=True))
        edge_ids = [(u, v) for u, v, _ in records]
        weights = [data.get('weight', 1.) for _, _, data in records]
        edges = torch.tensor(edge_ids, dtype=torch.int64, device=device).reshape(-1, 2).T
        values = torch.tensor(weights, dtype=dtype, device=device)
        loops = edges[0] == edges[1]
        if not graph.is_directed():
            edges = torch.cat((edges, edges.flip(0)[:, ~loops]), 1)
            values = torch.cat((values, values[~loops]))
        self.adjacency = torch.sparse_coo_tensor(edges, values, (self.n, self.n),
                                                 device=device).coalesce()
        self.indices = self.adjacency.indices()
        self.values = self.adjacency.values()
        self.degree = torch.zeros(self.n, dtype=dtype, device=device)
        self.degree.scatter_add_(0, self.indices[0], self.values)
        self.unweighted_degree = torch.zeros(self.n, dtype=dtype, device=device)
        self.unweighted_degree.scatter_add_(0, self.indices[0], torch.ones_like(self.values))
        diag = self.indices[0] == self.indices[1]
        if not graph.is_directed():
            self.unweighted_degree.scatter_add_(0, self.indices[0, diag],
                                                torch.ones_like(self.values[diag]))

    def sparse(self, values, dtype=None):
        return torch.sparse_coo_tensor(self.indices, values.to(dtype or values.dtype),
                                      (self.n, self.n), device=self.device).coalesce()

    def row_normalized(self):
        inv = torch.where(self.degree > 0, self.degree.reciprocal(), 0.)
        return self.sparse(self.values * inv[self.indices[0]])

    def symmetric_normalized(self, dtype=torch.float32):
        degree = self.degree.to(dtype)
        inv = torch.where(degree > 0, degree.rsqrt(), 0.)
        values = self.values.to(dtype) * inv[self.indices[0]] * inv[self.indices[1]]
        return self.sparse(values, dtype)

    def laplacian_complement(self):
        """Match scipy I - csgraph.laplacian(A, normed=True), including isolates.

        SciPy excludes self loops from Laplacian degree, and an isolate has a
        zero Laplacian row, hence an identity row in its complement.
        """
        i, j = self.indices
        nonloop = i != j
        degree = torch.zeros_like(self.degree)
        degree.scatter_add_(0, i[nonloop], self.values[nonloop])
        inv = torch.where(degree > 0, degree.rsqrt(), 1.)
        values = self.values[nonloop] * inv[i[nonloop]] * inv[j[nonloop]]
        nodes = torch.arange(self.n, device=self.device)
        indices = torch.cat((self.indices[:, nonloop], torch.stack((nodes, nodes))), 1)
        values = torch.cat((values, (degree == 0).to(values.dtype)))
        return (torch.sparse_coo_tensor(indices, values, (self.n, self.n),
                                       device=self.device).coalesce(), inv)


def mm(matrix, value):
    return torch.sparse.mm(matrix, value) if matrix.layout != torch.strided else matrix @ value


def cdist(a, b, block_size=256):
    dtype = torch.promote_types(a.dtype, b.dtype)
    a, b = a.to(dtype), b.to(dtype)
    output = torch.empty((len(a), len(b)), device=a.device, dtype=dtype)
    for start in range(0, len(a), block_size):
        output[start:start + block_size] = torch.cdist(a[start:start + block_size], b,
            p=2, compute_mode='donot_use_mm_for_euclid_dist')
    return output


def nearest(a, b, block_size=256):
    dtype = torch.promote_types(a.dtype, b.dtype)
    a, b = a.to(dtype), b.to(dtype)
    distances = torch.empty(len(a), device=a.device, dtype=dtype)
    indices = torch.empty(len(a), device=a.device, dtype=torch.int64)
    for start in range(0, len(a), block_size):
        batch = torch.cdist(a[start:start + block_size], b, p=2,
                           compute_mode='donot_use_mm_for_euclid_dist')
        distances[start:start + block_size], indices[start:start + block_size] = batch.min(1)
    return distances, indices


def degree_signatures(graph, max_hop, block_size=128):
    n = graph.n
    output = torch.zeros((n, (max_hop + 1) * 5), device=graph.device, dtype=torch.float32)
    levels = torch.tensor([1., 0., .5, .75, .25], device=graph.device, dtype=torch.float64)
    block_size = max(1, min(block_size, 2_000_000 // n))
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        frontier = torch.zeros((n, stop - start), device=graph.device, dtype=torch.float64)
        frontier[torch.arange(start, stop, device=graph.device),
                 torch.arange(stop - start, device=graph.device)] = 1
        visited = frontier.bool()
        # Empty frontiers remain empty; no host-side BFS termination decision.
        for hop in range(min(max_hop, n - 1) + 1):
            mask = frontier.T.bool()
            count = mask.sum(1)
            ordered = torch.sort(torch.where(mask, graph.unweighted_degree[None], torch.inf), 1).values
            ranks = (count.clamp_min(1) - 1)[:, None] * levels
            lower, upper = ranks.floor().long(), ranks.ceil().long()
            lo = torch.where(count[:, None] > 0, ordered.gather(1, lower), 0.)
            hi = torch.where(count[:, None] > 0, ordered.gather(1, upper), 0.)
            output[start:stop, hop * 5:hop * 5 + 5] = torch.log1p(lo + (hi - lo) * (ranks - lower)).float()
            if hop != max_hop:
                reached = (mm(graph.adjacency, frontier) > 0) & ~visited
                visited |= reached
                frontier = reached.double()
    return output


def netmf_embedding(graph, cfg):
    n, dim = graph.n, min(cfg.embed_dim, graph.n - 1)
    if n == 1:
        return torch.ones((1, cfg.embed_dim), device=graph.device, dtype=torch.float32)
    matrix, inv = graph.laplacian_complement()
    power = torch.eye(n, dtype=torch.float64, device=graph.device)
    accumulated = torch.zeros_like(power)
    for _ in range(cfg.netmf_window):
        power = mm(matrix, power)
        accumulated += power
    # Volume includes diagonal weights exactly as the native adjacency does.
    proximity = (accumulated * (graph.values.sum() / cfg.netmf_window / cfg.negative_sampling)).T
    proximity = (inv[:, None] * proximity * inv[None, :]).clamp_min(1).log()
    u, s, _ = torch.linalg.svd(proximity, full_matrices=False)
    embedding = u[:, :dim] * s[:dim].clamp_min(0).sqrt()[None]
    # The native zero-proximity fallback is identity, chosen on the device.
    embedding = torch.where(torch.count_nonzero(proximity) > 0, embedding,
                            torch.eye(n, dim, dtype=torch.float64, device=graph.device))
    if dim < cfg.embed_dim:
        embedding = torch.nn.functional.pad(embedding, (0, cfg.embed_dim - dim))
    norms = torch.linalg.vector_norm(embedding, dim=1, keepdim=True)
    return (embedding / torch.where(norms < 1e-12, 1., norms)).float()


def select_anchors(g1, g2, f1, f2, ratio, fast_select, threshold):
    masks = []
    for graph in (g1, g2):
        mask = graph.unweighted_degree >= threshold if fast_select else torch.ones(graph.n, dtype=torch.bool, device=graph.device)
        masks.append(torch.where(mask.any(), mask, torch.ones_like(mask)))
    source, target = [torch.nonzero(mask, as_tuple=True)[0] for mask in masks]
    # Tensor shapes only determine buffer/launch size; eligibility and ordering
    # were computed on the device.  Stable sorting retains lower source IDs.
    requested = min(max(1, int(ratio * min(g1.n, g2.n))), source.numel(), target.numel())
    distances, indices = nearest(f1[source], f2[target])
    selected = torch.argsort(distances, stable=True)[:requested]
    return source[selected], target[indices[selected]]


def multiscale(g1, g2, base1, base2, anchors, propagation_orders):
    scales1, scales2 = [base1], [base2]
    current1, current2 = base1.double(), base2.double()
    transition1, transition2 = g1.row_normalized(), g2.row_normalized()
    for _ in range(propagation_orders):
        current1, current2 = mm(transition1, current1), mm(transition2, current2)
        scales1.append(current1.float())
        scales2.append(current2.float())
    source, target = anchors
    covariance = torch.zeros((base1.shape[1], base2.shape[1]), device=base1.device, dtype=torch.float64)
    for x, y in zip(scales1, scales2):
        covariance += (x[source].T @ y[target]).double()
    covariance /= len(scales1)
    u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
    rotation = u @ vh
    aligned = (base1.double() @ rotation).float()
    combined1 = torch.cat([(x.double() @ rotation).float() for x in scales1], 1)
    return aligned, base2, combined1, torch.cat(scales2, 1)


def execution_metadata(device, stages):
    return dict(implementation_revision=REVISION, requested_device=str(device),
        actual_device=str(device), cuda_active=device.type == 'cuda',
        execution_mode='device_resident_algorithm' if device.type == 'cuda' else 'cpu_regression_only',
        fully_device_resident=True, cpu_fallback=False,
        gpu_stages=stages if device.type == 'cuda' else [],
        cpu_stages=['input graph edge/node parsing', 'CUDA kernel launch and dynamic-shape buffer scheduling',
                    'final artifact and diagnostic export'],
        algorithm_numerical_cpu_stages=[], host_algorithm_scalar_decisions=0,
        dynamic_shape_synchronization=True, all_stages_included_in_runtime=True)


class TensorIMMNC:
    def __init__(self, config, device):
        self.config, self.device = config, torch.device(device)

    def fit(self, graph1, graph2):
        cfg, device = self.config, self.device
        started = time.perf_counter()
        g1, g2 = Graph(graph1, device), Graph(graph2, device)
        f1, f2 = degree_signatures(g1, cfg.k_de), degree_signatures(g2, cfg.k_de)
        base1, base2 = netmf_embedding(g1, cfg), netmf_embedding(g2, cfg)
        anchors = select_anchors(g1, g2, f1, f2, cfg.initial_anchor_ratio,
                                 cfg.fast_select, cfg.degree_threshold)
        histories, target_history = [], []
        active_ratio = cfg.initial_anchor_ratio
        aligned1, aligned2 = base1, base2
        for iteration in range(cfg.iterations):
            aligned1, aligned2, combined1, combined2 = multiscale(g1, g2, base1, base2, anchors, cfg.k_nei)
            _, targets = nearest(aligned1, aligned2)
            target_history.append(targets)
            # Counts are GPU reductions; their export is deferred until fit ends.
            unique = torch.unique(anchors[1]).numel()
            histories.append(dict(iteration=iteration + 1, requested_ratio=active_ratio,
                                  num_source_anchors=anchors[0].numel(), num_unique_target_anchors=unique))
            active_ratio = max(cfg.initial_anchor_ratio + cfg.anchor_growth_rate * (iteration + 1),
                               1.0 if cfg.fast_select else .5)
            anchors = select_anchors(g1, g2, combined1, combined2, active_ratio,
                                     cfg.fast_select, cfg.degree_threshold)
        _, targets = nearest(aligned1, aligned2)
        pairs = torch.stack((torch.arange(g1.n, device=device), targets), 1)
        cost = cdist(aligned1, aligned2).float() if cfg.compute_cost_matrix else None
        guards = torch.stack((
            ~(g1.unweighted_degree >= cfg.degree_threshold).any(),
            ~(g2.unweighted_degree >= cfg.degree_threshold).any(),
            g1.values.sum() <= 0, g2.values.sum() <= 0,
            (g1.unweighted_degree == 0).sum(), (g2.unweighted_degree == 0).sum()))
        execution = execution_metadata(device, ['GPU graph normalization and exact-hop percentile signatures',
            'GPU NetMF and complete SVD', 'GPU pseudo-anchor eligibility, nearest retrieval and stable ranking',
            'GPU propagation, Procrustes and final nearest decoder'])
        # All device-to-host numerical transfers below are final outputs.
        guard = export(guards).tolist()
        result = AlignmentResult('iMMNC', pairs_export(pairs), 0.,
            cost_matrix=export(cost) if cost is not None else None,
            source_embeddings=export(aligned1), target_embeddings=export(aligned2), larger_is_better=False,
            partial_decoder_data={'iteration_history': export(torch.stack(target_history))},
            metadata=dict(config=asdict(cfg), pseudo_anchor_history=histories, anchor_schedule='author_max',
                netmf_operator='I_minus_normalized_Laplacian',
                boundary_guard_triggers=dict(source_empty_high_degree_fallback=bool(cfg.fast_select and guard[0]),
                    target_empty_high_degree_fallback=bool(cfg.fast_select and guard[1]),
                    source_zero_volume=bool(guard[2]), target_zero_volume=bool(guard[3])),
                safety=dict(zero_degree_source=int(guard[4]), zero_degree_target=int(guard[5]),
                    empty_high_degree_policy='all_nodes'),
                dense_matrix_entries=g1.n * g2.n, dense_cost_matrix_skipped=False,
                prediction_rule='row-wise nearest target in Euclidean embedding space',
                numerical_adapter=dict(svd='complete device SVD; retain native top dimensions',
                    tie_policy='stable lower source ID; nearest ties lower target ID',
                    seeded_arpack_trajectory_identical=False),
                implementation_revision=REVISION, gpu_revision=REVISION, device_execution=execution,
                ground_truth_used_during_inference=False))
        result.runtime_sec = time.perf_counter() - started
        execution['full_algorithm_end_to_end_seconds'] = result.runtime_sec
        return result


def plain_diffusion(adjacency, steps, output_dim):
    hot = torch.eye(adjacency.shape[0], device=adjacency.device, dtype=torch.float32)
    for _ in range(steps + 1):
        hot = mm(adjacency, hot) + hot
    return torch.sort(hot, dim=1).values[:, -output_dim:]


def diffuse_anchors(adjacency, nodes, steps):
    hot = torch.zeros((adjacency.shape[0], nodes.numel()), device=adjacency.device, dtype=torch.float32)
    hot[nodes, torch.arange(nodes.numel(), device=nodes.device)] = 1
    for _ in range(steps):
        hot = mm(adjacency, hot) + hot
    return hot


def select_pairs(f1, f2, degree1, degree2, count, cfg, used1, used2, initial):
    from .assignment import linear_sum_assignment
    s = torch.nonzero(degree1 >= cfg.dp_min_degree if initial else ~used1, as_tuple=True)[0]
    t = torch.nonzero(degree2 >= cfg.dp_min_degree if initial else ~used2, as_tuple=True)[0]
    if s.numel() == 0 or t.numel() == 0 or count <= 0:
        return torch.empty((0, 2), dtype=torch.int64, device=f1.device)
    costs = torch.full((s.numel(), t.numel()), 1e9, device=f1.device, dtype=torch.float32)
    candidate_count = min(cfg.nearest_candidates, t.numel())
    batch_size = min(256, cfg.distance_batch_size)
    for start in range(0, s.numel(), batch_size):
        ds = cdist(f1[s[start:start + batch_size]], f2[t], batch_size)
        order = torch.argsort(ds, dim=1, stable=True)[:, :candidate_count]
        costs[start:start + batch_size].scatter_(1, order, ds.gather(1, order))
    row, col = linear_sum_assignment(costs, device=str(f1.device))
    selected = torch.argsort(costs[row, col], stable=True)[:count]
    return torch.stack((s[row[selected]], t[col[selected]]), 1)


class TensorSparkAlign:
    def __init__(self, config, device):
        self.config, self.device = config, torch.device(device)

    def fit(self, graph1, graph2):
        cfg, device = self.config, self.device
        started = time.perf_counter()
        g1, g2 = Graph(graph1, device), Graph(graph2, device)
        maximum = min(g1.n, g2.n)
        budget = cfg.matching_budget or maximum
        if budget > maximum:
            raise ValueError('matching_budget exceeds minimum graph order')
        norm1, norm2 = g1.symmetric_normalized(), g2.symmetric_normalized()
        # float32 adjacency-sum degree is the native SparkAlign convention.
        degree1, degree2 = g1.degree.float(), g2.degree.float()
        initial1, initial2 = plain_diffusion(norm1, cfg.diffusion_step, maximum), plain_diffusion(norm2, cfg.diffusion_step, maximum)
        used1 = torch.zeros(g1.n, dtype=torch.bool, device=device)
        used2 = torch.zeros(g2.n, dtype=torch.bool, device=device)
        blocks1, blocks2, all_pairs, history = [], [], [], []
        # Worst-case graph-derived epoch count also covers a small initial set.
        # Host uses selected tensor shapes only for launch/buffer sizes.
        epochs = 1 + (budget + cfg.num_diffusion_select - 1) // cfg.num_diffusion_select
        selected_total = 0
        for epoch in range(epochs):
            remaining = budget - selected_total
            if remaining <= 0:
                break
            initial = epoch == 0
            requested = min(remaining, cfg.num_dp_select if initial else cfg.num_diffusion_select)
            x, y = (initial1, initial2) if initial else (torch.cat(blocks1, 1), torch.cat(blocks2, 1))
            pairs = select_pairs(x, y, degree1, degree2, requested, cfg, used1, used2, initial)
            if pairs.shape[0] == 0:
                if initial:
                    raise ValueError('No initial SparkAlign anchors satisfy dp_min_degree')
                break
            s, t = pairs.unbind(1)
            blocks1.append(diffuse_anchors(norm1, s, cfg.diffusion_step))
            blocks2.append(diffuse_anchors(norm2, t, cfg.diffusion_step))
            used1[s], used2[t] = True, True
            all_pairs.append(pairs)
            selected_total += pairs.shape[0]
            history.append(dict(epoch=epoch + 1, initial=initial, requested_pairs=requested,
                                selected_pairs=pairs.shape[0], cumulative_pairs=selected_total))
        source, target = torch.cat(blocks1, 1), torch.cat(blocks2, 1)
        pairs = torch.cat(all_pairs)
        cost = cdist(source, target).float() if cfg.compute_cost_matrix else None
        execution = execution_metadata(device, ['GPU adjacency degree and symmetric normalization',
            'GPU plain and anchor diffusion', 'GPU unused-anchor masks and stable nearest candidate sorting',
            'GPU candidate scatter and exact rectangular assignment', 'GPU iterative anchor bookkeeping'])
        result = AlignmentResult('SparkAlign', pairs_export(pairs), 0.,
            cost_matrix=export(cost) if cost is not None else None,
            source_embeddings=export(source), target_embeddings=export(target), larger_is_better=False,
            metadata=dict(config=asdict(cfg), matching_budget=budget,
                matching_budget_source='explicit_config' if cfg.matching_budget is not None else 'min_graph_order',
                selection_history=history, selected_pairs=pairs.shape[0], dense_matrix_entries=g1.n * g2.n,
                dense_cost_matrix_skipped=False, initial_feature_l2_normalization=False,
                isolated_node_safety='zero degree uses zero inverse sqrt',
                prediction_rule='iterative sparse optimal one-to-one assignment',
                exact_assignment_backend='device shortest augmenting path',
                implementation_revision=REVISION, gpu_revision=REVISION, device_execution=execution,
                ground_truth_used_during_inference=False))
        result.runtime_sec = time.perf_counter() - started
        execution['full_algorithm_end_to_end_seconds'] = result.runtime_sec
        return result


def construct(method, parameters, seed=2026, device='cuda:0', dataset_family=''):
    device = 'cuda:0' if device == 'auto' else str(device)
    if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU fallback is disabled')
    params = dict(parameters)
    params.pop('device', None)
    params.update(seed=seed, show_progress=False)
    if method == 'immnc':
        from baseline.immnc import IMMNCConfig
        params['compute_cost_matrix'] = True
        cfg = replace(IMMNCConfig(**params), max_dense_entries=10 ** 18)
        cfg.validate()
        model = TensorIMMNC(cfg, device)
    elif method == 'SparkAlign':
        from baseline.SparkAlign import SparkAlignConfig
        params['compute_cost_matrix'] = True
        cfg = replace(SparkAlignConfig(**params), max_dense_entries=10 ** 18)
        cfg.validate()
        model = TensorSparkAlign(cfg, device)
    elif method == 's_gwl':
        from baseline.author_adapters import AuthorSGWLConfig
        if 'probability_shift' in params:
            from .sgwl_prior import TensorSGWL
            from .sgwl_config import DegreePriorConfig
            AuthorSGWLConfig=DegreePriorConfig
        else:
            from .sgwl import TensorSGWL
        cfg = AuthorSGWLConfig(**params)
        model = TensorSGWL(cfg, device)
    else:
        raise ValueError('Unknown device iterative method: ' + method)
    return model, dict(asdict(cfg), implementation_revision=REVISION, gpu_revision=REVISION,
                       cpu_fallback=False, execution_mode='device_resident_algorithm')
