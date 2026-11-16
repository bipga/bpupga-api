"""GradAlign+ with GPU graph statistics, reconstruction, frontiers and decoding.

The author GIN/Adam/objective/epoch and match budgets are retained. Initialization
uses device RNG; equal scores use stable flattened-index ordering instead of two
unspecified NumPy quicksort permutations. These are explicit version changes.
A custom exact reconstruction autograd saves embeddings and fixed targets only;
its backward recomputes bounded row blocks instead of retaining dense loss graphs.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import time

import torch
import torch.nn.functional as F

import baseline
from baseline.author_adapters import AuthorGradConfig
from common.result import AlignmentResult

CPU_STAGES = ['one_graph_nodes_edges_import', 'host_budget_and_dynamic_shape_kernel_launch',
              'artifact_and_diagnostic_export']
GPU_STAGES = ['device_rng_GIN_initialization', 'centrality_and_attribute_bins',
              'exact_dense_multihop_targets', 'author_GIN_and_exact_block_reconstruction_training',
              'layerwise_cosine_float64_accumulation', 'progressive_frontier_and_ACN',
              'stable_score_sort_and_sequential_greedy', 'final_union_ACN']
IMPLEMENTATION = 'gradalign_all_algorithm_device_exact_block_reconstruction_261103_v3'


class DeviceGraph:
    def __init__(self, graph, device):
        self.n = len(graph)
        if graph.is_directed() or graph.is_multigraph():
            raise ValueError('GradAlign requires simple undirected graphs')
        if list(graph) != list(range(self.n)):
            raise ValueError('GradAlign requires ascending consecutive graph-local IDs')
        self.device = torch.device(device)
        # Once-only edge parsing preserves from_networkx scatter order.
        self.edges = torch.tensor(list(graph.to_directed().edges()), device=self.device,
                                  dtype=torch.int64).reshape(-1, 2).T.contiguous()
        self.adjacency = torch.sparse_coo_tensor(self.edges,
            torch.ones(self.edges.shape[1], device=self.device, dtype=torch.float64),
            (self.n, self.n)).coalesce()
        self.neighbors = torch.sparse.sum(self.adjacency, dim=1).to_dense()
        self.diagonal = torch.zeros(self.n, device=self.device, dtype=torch.float64)
        row, col = self.edges
        self.diagonal.scatter_add_(0, row[row == col], torch.ones_like(row[row == col], dtype=torch.float64))

    def frontier(self, matched, first=False):
        if first:
            return torch.arange(self.n, device=self.device)
        reached = torch.sparse.mm(self.adjacency, matched.double()[:, None])[:, 0] > 0
        return (reached & ~matched).nonzero(as_tuple=True)[0]


def centrality(graph, version):
    if version == 'khop':
        return graph.neighbors + 1. - graph.diagonal
    if version != 'eigenvector':
        raise ValueError('Campaign GradAlign supports eigenvector/khop only')
    x = torch.full((graph.n,), 1. / graph.n, dtype=torch.float64, device=graph.device)
    active = torch.ones((), device=graph.device, dtype=torch.bool)
    for _ in range(500):
        candidate = x + torch.sparse.mm(graph.adjacency, x[:, None])[:, 0]
        candidate = candidate / torch.linalg.vector_norm(candidate).clamp_min(torch.finfo(torch.float64).tiny)
        converged = (candidate - x).abs().sum() < graph.n * 1e-8
        x = torch.where(active, candidate, x)
        active = active & ~converged
    torch._assert_async(~active, 'GradAlign eigenvector centrality did not converge in 500 steps')
    return x


def augmented_attributes(left, right, num_attr, version):
    values = (centrality(left, version), centrality(right, version))
    interval = torch.maximum(values[0].max(), values[1].max()) / num_attr
    features = []
    for value in values:
        bins = torch.trunc(value / interval).long() - 1
        bins = torch.where(bins < 0, bins + num_attr, bins)
        hot = torch.zeros((len(value), num_attr), device=value.device, dtype=torch.float64)
        hot[torch.arange(len(value), device=value.device), bins] = 1
        features.append(hot)
    active = torch.cat(features).ne(0).any(0)
    return features[0][:, active].float(), features[1][:, active].float()


def reconstruction_targets(graph, layers):
    a = graph.adjacency.float().to_dense()
    a.fill_diagonal_(1.)
    candidate = a
    result = [None]
    for layer in range(layers):
        degree = a.sum(0).pow(-.5)
        result.append(degree[:, None] * a * degree[None, :])
        # Last multiplication is unused by the author objective.
        if layer + 1 < layers:
            candidate = a @ candidate
            a = a + candidate
    return result


def reconstruction_reference(embedding, target):
    unit = F.normalize(embedding)
    predicted = torch.minimum(unit @ unit.T, embedding.new_tensor([1.]))
    predicted = F.normalize(predicted, dim=1)
    return (predicted - target).square().sum() / target.shape[1]


class _ExactBlockReconstruction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, embedding, target, block_rows):
        ctx.save_for_backward(embedding, target)
        ctx.block_rows = block_rows
        unit = F.normalize(embedding)
        loss = embedding.new_zeros(())
        for start in range(0, len(unit), block_rows):
            stop = min(start + block_rows, len(unit))
            predicted = torch.minimum(unit[start:stop] @ unit.T, embedding.new_tensor([1.]))
            predicted = F.normalize(predicted, dim=1)
            loss += (predicted - target[start:stop]).square().sum() / target.shape[1]
        return loss

    @staticmethod
    def backward(ctx, scale):
        embedding, target = ctx.saved_tensors
        # Differentiate each complete normalized row and then sum its two
        # appearances in U_block @ U.T. No sampled pairs or changed loss terms.
        with torch.enable_grad():
            source = embedding.detach().requires_grad_(True)
            unit = F.normalize(source)
            unit_gradient = torch.zeros_like(unit)
            for start in range(0, len(unit), ctx.block_rows):
                stop = min(start + ctx.block_rows, len(unit))
                predicted = torch.minimum(unit[start:stop] @ unit.T, source.new_tensor([1.]))
                predicted = F.normalize(predicted, dim=1)
                loss = (predicted - target[start:stop]).square().sum() / target.shape[1]
                gradient, = torch.autograd.grad(loss, unit, retain_graph=False)
                unit_gradient += gradient
            gradient, = torch.autograd.grad(unit, source, grad_outputs=unit_gradient)
        return gradient * scale, None, None


def reconstruction_loss(embedding, target, block_rows=128):
    return _ExactBlockReconstruction.apply(embedding, target, int(block_rows))


def progressive_signatures(graph, seeds, index):
    column_map = torch.full((graph.n,), -1, dtype=torch.int64, device=graph.device)
    column_map[seeds] = torch.arange(len(seeds), device=graph.device)
    row_map = torch.full_like(column_map, -1)
    row_map[index] = torch.arange(len(index), device=graph.device)
    row, column = graph.edges
    mapped_row, mapped_column = row_map[row], column_map[column]
    active = (mapped_row >= 0) & (mapped_column >= 0)
    edges = torch.stack((mapped_row[active], mapped_column[active]))
    return torch.sparse_coo_tensor(edges,
        torch.ones(edges.shape[1], device=graph.device, dtype=torch.float64),
        (len(index), len(seeds))).coalesce()


def weighted_candidates(left, right, seeds1, seeds2, index, columns, score, block_rows):
    weighted = score[index[:, None], columns[None, :]]
    if seeds1.numel():  # tensor shape controls launch, never a copied GPU scalar
        source = progressive_signatures(left, seeds1, index)
        target = progressive_signatures(right, seeds2, columns)
        target_t = target.T.coalesce()
        row, col = source.indices()
        for start in range(0, len(index), block_rows):
            stop = min(start + block_rows, len(index))
            mask = (row >= start) & (row < stop)
            block = torch.sparse_coo_tensor(torch.stack((row[mask] - start, col[mask])),
                source.values()[mask], (stop-start, len(seeds1)), device=left.device).coalesce()
            common = torch.sparse.mm(block, target_t).to_dense()
            weighted[start:stop] *= (common + .01).square()
            weighted[start:stop] *= ((left.neighbors[index[start:stop]] > 0)[:, None]
                                     & (right.neighbors[columns] > 0)[None, :])
    return weighted


def apply_final_acn_device(left, right, pairs, score, block_rows=128):
    n1, n2 = left.n, right.n
    count = pairs.shape[0]
    universe = n1 + n2 + count
    ids1 = torch.arange(n1, device=left.device)
    ids2 = torch.arange(n2, device=left.device) + n1
    ids1[pairs[:, 0]] = torch.arange(count, device=left.device) + n1 + n2
    ids2[pairs[:, 1]] = torch.arange(count, device=left.device) + n1 + n2
    edges = torch.cat((ids1[left.edges], ids2[right.edges]), 1)
    union = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1], device=left.device,
        dtype=torch.float64), (universe, universe)).coalesce()
    union = torch.sparse_coo_tensor(union.indices(), torch.ones_like(union.values()), union.shape).coalesce()
    # Sparse remapping uses GPU inverse lookup for merged vertices. Reindexing
    # rows preserves every merged union edge and all isolate empty neighborhoods.
    signatures = []
    row, col = union.indices()
    for ids in (ids1, ids2):
        inverse = torch.full((universe,), -1, device=left.device, dtype=torch.int64)
        inverse[ids] = torch.arange(len(ids), device=left.device)
        mapped = inverse[row]
        active = mapped >= 0
        ij = torch.stack((mapped[active], col[active]))
        signatures.append(torch.sparse_coo_tensor(ij, torch.ones(ij.shape[1], device=left.device,
            dtype=torch.float64), (len(ids), universe)).coalesce())
    source, target = signatures
    row, col = source.indices()
    output = torch.empty_like(score)
    for start in range(0, n1, block_rows):
        stop = min(start + block_rows, n1)
        active = (row >= start) & (row < stop)
        block = torch.sparse_coo_tensor(torch.stack((row[active] - start, col[active])),
            source.values()[active], (stop-start, universe), device=left.device).coalesce()
        common = torch.sparse.mm(block, target.T).to_dense()
        output[start:stop] = score[start:stop] * (common + .01).square()
    return output


class TensorGradAlign:
    def __init__(self, config, device='cuda:0', final_acn=False, block_rows=128):
        self.config = config
        self.device = torch.device(device)
        self.final_acn = final_acn
        self.block_rows = int(block_rows)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('All-device GradAlign requires CUDA; CPU fallback disabled')

    def fit(self, graph1, graph2):
        from .gin import myGIN
        from gpu_backend.assignment import flat_greedy
        cfg = self.config
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        left, right = DeviceGraph(graph1, self.device), DeviceGraph(graph2, self.device)
        n1, n2 = left.n, right.n
        if min(n1, n2) < 2:
            raise ValueError('Author GIN BatchNorm requires at least two nodes per graph')
        torch.cuda.manual_seed_all(cfg.seed)
        timings = {}

        def mark(name, since):
            torch.cuda.synchronize(self.device)
            timings[name] = time.perf_counter() - since

        stage = time.perf_counter()
        augmented1, augmented2 = augmented_attributes(left, right, cfg.num_attr, cfg.centrality)
        base1 = torch.ones((n1, 1), device=self.device)
        base2 = torch.ones((n2, 1), device=self.device)
        # Device construction moves all parameter initialization to CUDA RNG.
        with torch.device(self.device):
            base_model = myGIN(1, hidden_channels=cfg.hid_dim, num_layers=cfg.k_hop)
            augmented_model = myGIN(augmented1.shape[1], hidden_channels=cfg.hid_dim, num_layers=cfg.k_hop)
        mark('centrality_attributes_and_device_initialization', stage)
        stage = time.perf_counter()
        targets1 = reconstruction_targets(left, cfg.k_hop)
        targets2 = reconstruction_targets(right, cfg.k_hop)
        mark('exact_reconstruction_targets_device', stage)
        stage = time.perf_counter()
        embeddings = []
        losses = []
        for model, x1, x2 in ((base_model, base1, base2), (augmented_model, augmented1, augmented2)):
            optimizer = torch.optim.Adam(model.parameters(), lr=.005, capturable=True)
            model.train()
            epoch_losses = []
            for epoch in range(cfg.epochs):
                e1 = model.full_forward(x1, left.edges)
                e2 = model.full_forward(x2, right.edges)
                optimizer.zero_grad()
                loss = torch.zeros((), device=self.device)
                for layer in range(1, cfg.k_hop + 1):
                    loss = loss + reconstruction_loss(e1[layer], targets1[layer], self.block_rows)
                    loss = loss + reconstruction_loss(e2[layer], targets2[layer], self.block_rows)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.detach())
            # Author retains training-mode BatchNorm at final full_forward.
            with torch.no_grad():
                e1 = model.full_forward(x1, left.edges)
                e2 = model.full_forward(x2, right.edges)
            embeddings.append((e1, e2))
            losses.append(torch.stack(epoch_losses))
        del targets1, targets2
        mark('author_GIN_exact_block_reconstruction_training_device', stage)
        stage = time.perf_counter()
        score = torch.zeros((n1, n2), device=self.device, dtype=torch.float64)
        # Campaign uses one-column base attributes, so author's branch returns
        # augmented similarities. Base-model training budget is still executed.
        for e1, e2 in zip(*embeddings[1]):
            u, v = F.normalize(e1), F.normalize(e2)
            for start in range(0, n1, self.block_rows):
                stop = min(start + self.block_rows, n1)
                score[start:stop] += ((u[start:stop] @ v.T) * (1 / (cfg.k_hop + 1))).double()
        del embeddings
        mark('neural_cosine_device', stage)
        stage = time.perf_counter()
        seeds1 = torch.empty(0, dtype=torch.int64, device=self.device)
        seeds2 = torch.empty_like(seeds1)
        matched1 = torch.zeros(n1, device=self.device, dtype=torch.bool)
        matched2 = torch.zeros(n2, device=self.device, dtype=torch.bool)
        budget = max(1, math.ceil(min(n1, n2) / cfg.gradual_iterations))
        trace = []
        for iteration in range(min(n1, n2)):
            index = left.frontier(matched1, first=iteration == 0)
            columns = right.frontier(matched2, first=iteration == 0)
            # Shapes are produced by GPU nonzero. CPU schedules work on their
            # sizes; no scores, counts, graph statistics or comparisons copied.
            if not index.numel() or not columns.numel() or seeds1.numel() == min(n1, n2):
                break
            weighted = weighted_candidates(left, right, seeds1, seeds2, index, columns,
                                             score, self.block_rows)
            rows, cols = flat_greedy(weighted, maximize=True, device=self.device, budget=budget)
            chosen1, chosen2 = index[rows], columns[cols]
            seeds1 = torch.cat((seeds1, chosen1))
            seeds2 = torch.cat((seeds2, chosen2))
            matched1[chosen1] = True
            matched2[chosen2] = True
            trace.append(dict(round=iteration + 1, matched_pairs=seeds1.numel(), gt_evaluation=False))
        pairs_device = torch.stack((seeds1, seeds2), 1)
        mark('progressive_ACN_sort_and_greedy_device', stage)
        if self.final_acn:
            stage = time.perf_counter()
            score = apply_final_acn_device(left, right, pairs_device, score, self.block_rows)
            mark('final_union_ACN_device', stage)
        torch._assert_async(torch.isfinite(score).all(), 'Non-finite GradAlign scores')
        # Model decisions are complete before CPU artifact/diagnostic transfer.
        loss_trace = [value.cpu().tolist() for value in losses]
        pairs = list(map(tuple, pairs_device.cpu().tolist()))
        score_numpy = score.cpu().numpy()
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        metadata = dict(config=asdict(cfg), implementation_revision=IMPLEMENTATION,
            official_commit='95ef6366b18589fb88d01e95af3aa3d621ff71bf',
            actual_backend='author_GIN_exact_block_reconstruction_all_algorithm_device',
            progressive_ACN_enabled=True, ground_truth_used_during_inference=False,
            matching_budget_source='min_graph_order_only', matching_budget=min(n1, n2),
            bootstrap='all topology nodes when no initial seeds; original frontier thereafter',
            rounds=trace, prediction_rule='gradual ACN stable greedy; no final Hungarian',
            loss_mode='exact_full_dense_objective_block_recomputed_backward',
            reconstruction_block_rows=self.block_rows, epoch_losses=loss_trace,
            final_acn_graph_only=dict(enabled=self.final_acn, graph_only=True, ground_truth_used=False),
            device_execution=dict(requested_device=str(self.device), actual_device=str(self.device),
                cuda_active=True, execution_mode='all_algorithm_gpu', gpu_stages=GPU_STAGES,
                cpu_stages=CPU_STAGES, stage_elapsed_seconds=timings,
                full_algorithm_end_to_end_seconds=elapsed, all_stages_included_in_runtime=True,
                cpu_fallback=False, all_algorithm_decisions_on_gpu=True,
                host_dispatch='fixed budgets and GPU-computed dynamic tensor shapes',
                dynamic_shape_sync=True, cpu_numeric_algorithm=False,
                native_sort_tie_order_preserved=False,
                tie_policy='stable_score_descending_then_flatten_index_ascending',
                initialization='CUDA RNG; version change from author CPU RNG',
                objective_and_optimizer_budgets_preserved=True,
                numerical_difference='float32 block reduction order can perturb near-zero Adam gradients; no bitwise claim',
                bitwise_equivalence_to_previous_version=False))
        return AlignmentResult('GradAlign+', pairs, elapsed, score_matrix=score_numpy,
                               larger_is_better=True, metadata=metadata)


def construct(parameters, seed=2026, device='cuda:0', dataset_family=''):
    params = dict(parameters)
    params.pop('device', None)
    params.pop('seed', None)
    final_acn = bool(params.pop('final_acn', False))
    selected = 'cuda:0' if device == 'auto' else str(device)
    cfg = AuthorGradConfig(**dict(params, seed=seed, device=selected))
    model = TensorGradAlign(cfg, selected, final_acn)
    return model, dict(asdict(cfg), final_acn=final_acn, implementation_revision=IMPLEMENTATION,
        gpu_revision='261103_all_algorithm_device_v3', cpu_fallback=False,
        requested_device=selected, execution_mode='all_algorithm_gpu',
        gpu_stages=GPU_STAGES, cpu_stages=CPU_STAGES)
