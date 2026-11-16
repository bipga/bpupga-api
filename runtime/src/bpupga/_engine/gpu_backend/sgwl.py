"""Author recursive S-GWL with device graph partition and decoder state.

Partition membership, target probabilities, barycenters, leaf eligibility and
ordered greedy matching are CUDA operations. The host schedules the nonempty
subgraphs selected by device masks, reading their tensor shapes solely to
allocate buffers and launch the native graph-order iteration budgets.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import time
import torch

from gpu_backend.iterative import (Graph, execution_metadata, export, pairs_export)
from . import REVISION
from .ordered_sparse import mm
from common.result import AlignmentResult


def target_distribution(probabilities, count):
    result = torch.zeros((count, 1), dtype=probabilities[0].dtype,
                         device=probabilities[0].device)
    for p in probabilities:
        sorted_p = torch.sort(p[:, 0], descending=True, stable=True).values
        positions = torch.linspace(0, p.shape[0] - 1, count, device=p.device,
                                   dtype=p.dtype)
        lo, hi = positions.floor().long(), positions.ceil().long()
        interpolated = sorted_p[lo] + (sorted_p[hi] - sorted_p[lo]) * (positions - lo)
        result[:, 0] += interpolated
    return result / result.sum()


def squared_product(matrix, probability):
    # scipy sparse **2 is matrix power; dense learned barycenters use elementwise
    # power in the author's original equations. This distinction is preserved.
    return mm(matrix, mm(matrix, probability)) if matrix.layout != torch.strided else matrix.square() @ probability


def gw_discrepancy(cs, ct, ps, pt, cfg, outer_iterations, plan=None,
                   log_plan=None, enabled=None):
    """Author float64 L2/proximal GW, with GPU convergence masks.

    Fixed host launch budgets preserve the same early-stopped iterate: after a
    device convergence test becomes false, subsequent launches freeze its state.
    No .item(), bool(tensor), host threshold or inter-iteration output transfer.
    """
    if plan is None:
        plan = ps @ pt.T
    if log_plan is None:
        log_plan = plan.log()
    active = torch.ones((), device=ps.device, dtype=torch.bool) if enabled is None else enabled.clone()
    invariant = squared_product(cs, ps) + squared_product(ct, pt).T
    invariant += 1e3 * (ps - pt.T).square()
    log_a = torch.full_like(ps, -math.log(len(ps)))
    log_ps, log_pt = ps.log(), pt.log()
    finite = torch.ones((), device=ps.device, dtype=torch.bool)
    iterations = torch.zeros((), device=ps.device, dtype=torch.int64)
    for _ in range(outer_iterations):
        cost = invariant - 2 * mm(ct, mm(cs, plan).T).T
        kernel = -cost / cfg.beta + log_plan
        sk_active = active.clone()
        log_b = torch.zeros_like(pt)
        for _ in range(cfg.inner_iterations):
            candidate_b = log_pt - torch.logsumexp(kernel + log_a, 0)[:, None]
            candidate_a = log_ps - torch.logsumexp(kernel + candidate_b.T, 1)[:, None]
            shift = torch.maximum(candidate_a.max(), log_a.max())
            old, new = (log_a - shift).exp(), (candidate_a - shift).exp()
            error = (new - old).abs().sum() / old.sum()
            center = candidate_a.max()
            candidate_a, candidate_b = candidate_a - center, candidate_b + center
            log_a = torch.where(sk_active, candidate_a, log_a)
            log_b = torch.where(sk_active, candidate_b, log_b)
            sk_active = sk_active & (error >= 1e-30)
        candidate_log = kernel + log_a + log_b.T
        finite = finite & (~active | torch.isfinite(candidate_log).all())
        candidate = candidate_log.exp()
        relative = (candidate - plan).abs().sum() / plan.abs().sum()
        plan = torch.where(active, candidate, plan)
        log_plan = torch.where(active, candidate_log, log_plan)
        iterations += active.to(torch.int64)
        active = active & (relative > 1e-30)
    cost = invariant - 2 * mm(ct, mm(cs, plan).T).T
    return plan, log_plan, (cost * plan).sum(), finite, iterations


def barycenter(costs, probabilities, center, cfg, outer_iterations):
    transports = [p @ center.T for p in probabilities]
    log_transports = [t.log() for t in transports]
    cost_center = torch.sparse_coo_tensor(
        torch.stack((torch.arange(len(center), device=center.device),) * 2),
        center[:, 0], (len(center), len(center)), device=center.device).coalesce()
    active = torch.ones((), device=center.device, dtype=torch.bool)
    finite = active.clone()
    diagnostics = []
    for _ in range(4):
        for i in range(len(costs)):
            transports[i], log_transports[i], _, valid, iterations = gw_discrepancy(
                costs[i], cost_center, probabilities[i], center, cfg,
                outer_iterations, transports[i], log_transports[i], active)
            finite = finite & valid
            diagnostics.append(iterations)
        average = torch.zeros((len(center), len(center)), device=center.device, dtype=torch.float64)
        for cost, transport in zip(costs, transports):
            average += (transport.T @ mm(cost, transport)) / len(costs)
        average /= center @ center.T
        old = cost_center.to_dense() if cost_center.layout != torch.strided else cost_center
        relative = (average - old).abs().sum() / old.abs().sum()
        cost_center = torch.where(active, average, old)
        active = active & (relative > 1e-26)
    return transports, finite, torch.stack(diagnostics)


def subset(matrix, probability, nodes, selected):
    """GPU induced sparse subgraph, retained probabilities and global node IDs."""
    if matrix.layout == torch.strided:
        result = matrix[selected[:, None], selected[None]]
    else:
        indices, values = matrix.indices(), matrix.values()
        inverse = torch.full((matrix.shape[0],), -1, device=matrix.device, dtype=torch.int64)
        inverse[selected] = torch.arange(selected.numel(), device=matrix.device)
        mapped = inverse[indices]
        keep = (mapped >= 0).all(0)
        result = torch.sparse_coo_tensor(mapped[:, keep], values[keep],
            (selected.numel(), selected.numel()), device=matrix.device).coalesce()
    p = probability[selected]
    return result, p / p.sum(), nodes[selected]


def leaf_greedy(transport, ps, pt):
    """Author column/row traversal and first unused maximum, all on CUDA.

    Equal conditional transports use increasing candidate node ID. This is an
    explicitly versioned deterministic replacement for NumPy quicksort ties.
    """
    from gpu_backend.assignment import greedy_from_order
    n, m = transport.shape
    if n >= m:
        # Each target's complete source ranking is contiguous in the order.
        rows = torch.argsort((transport / ps).T, dim=1, descending=True, stable=True)
        columns = torch.arange(m, device=transport.device)[:, None].expand_as(rows)
    else:
        columns = torch.argsort(transport / pt.T, dim=1, descending=True, stable=True)
        rows = torch.arange(n, device=transport.device)[:, None].expand_as(columns)
    order = (rows * m + columns).flatten()
    source, target = greedy_from_order(order, n, m, device=str(transport.device))
    return torch.stack((source, target), 1)


class TensorSGWL:
    def __init__(self, config, device):
        self.config, self.device = config, torch.device(device)
        if config.partition_levels < 0 or config.cluster_count < 2:
            raise ValueError('Invalid S-GWL partition levels or cluster count')
        if config.inner_iterations < 1 or config.beta <= 0 or config.probability_epsilon < 0:
            raise ValueError('Invalid S-GWL inner iteration, beta or probability epsilon')

    def fit(self, graph1, graph2):
        cfg, device = self.config, self.device
        started = time.perf_counter()
        swapped = len(graph1) < len(graph2)
        a, b = (graph2, graph1) if swapped else (graph1, graph2)
        ga, gb = Graph(a, device), Graph(b, device)
        probabilities = []
        for graph in (ga, gb):
            p = graph.degree[:, None] + cfg.probability_epsilon
            # Device-side zero-volume guard, including epsilon=0.
            p = torch.where(p.sum() > 0, p, torch.ones_like(p))
            probabilities.append(p / p.sum())
        queue = [(ga.adjacency, probabilities[0], torch.arange(ga.n, device=device),
                  gb.adjacency, probabilities[1], torch.arange(gb.n, device=device))]
        finite = torch.ones((), device=device, dtype=torch.bool)
        partition_info, partition_iterations = [], []
        for level in range(cfg.partition_levels):
            next_queue = []
            for cs, ps, ns, ct, pt, nt in queue:
                center = target_distribution([ps, pt], cfg.cluster_count)
                # This is exactly the author's partition budget override.
                outer = max(cs.shape[0], ct.shape[0], 200)
                transports, valid, iterations = barycenter([cs, ct], [ps, pt], center, cfg, outer)
                finite = finite & valid
                partition_iterations.append(iterations)
                label_s = (transports[0] / center.T).argmax(1)
                label_t = (transports[1] / center.T).argmax(1)
                sizes = []
                for cluster in range(cfg.cluster_count):
                    mask_s, mask_t = label_s == cluster, label_t == cluster
                    eligible = mask_s.any() & mask_t.any()
                    # GPU determines whether both graphs have this cluster.
                    isel = torch.nonzero(mask_s & eligible, as_tuple=True)[0]
                    jsel = torch.nonzero(mask_t & eligible, as_tuple=True)[0]
                    sizes.append(torch.stack((mask_s.sum(), mask_t.sum())))
                    # Empty-shape dispatch is host scheduling of GPU decisions;
                    # there is no host node assignment or scalar threshold.
                    if isel.numel() == 0:
                        continue
                    source = subset(cs, ps, ns, isel)
                    target = subset(ct, pt, nt, jsel)
                    next_queue.append((*source, *target))
                partition_info.append((level + 1, cs.shape[0], ct.shape[0], torch.stack(sizes)))
            queue = next_queue
        score = torch.zeros((ga.n, gb.n), device=device, dtype=torch.float64)
        pair_blocks, leaf_info, leaf_iterations = [], [], []
        for cs, ps, ns, ct, pt, nt in queue:
            # The author replaces the user maximum with each leaf graph order.
            outer = max(cs.shape[0], ct.shape[0])
            transport, _, _, valid, iterations = gw_discrepancy(cs, ct, ps, pt, cfg, outer)
            finite = finite & valid
            score[ns[:, None], nt[None]] = transport
            local = leaf_greedy(transport, ps, pt)
            pair_blocks.append(torch.stack((ns[local[:, 0]], nt[local[:, 1]]), 1))
            leaf_info.append(dict(source_nodes=ns.numel(), target_nodes=nt.numel(),
                                  pairs=local.shape[0], outer_iterations=outer))
            leaf_iterations.append(iterations)
        pairs = torch.cat(pair_blocks) if pair_blocks else torch.empty((0, 2), device=device, dtype=torch.int64)
        if swapped:
            pairs, score = pairs.flip(1), score.T
        validity = torch.stack((finite, torch.isfinite(score).all(), (score >= 0).all()))
        isolates = torch.stack(((ga.unweighted_degree == 0).sum(), (gb.unweighted_degree == 0).sum()))
        # Only final artifacts/diagnostics cross the device boundary below.
        valid = export(validity).tolist()
        if not all(valid):
            raise FloatingPointError('Non-finite or negative device S-GWL transport')
        isolates = export(isolates).tolist()
        if swapped:
            isolates.reverse()
        partition_calls = [dict(level=level, input_orders={'0': n, '1': m},
            output_clusters=[{'0': int(s), '1': int(t)} for s, t in export(sizes).tolist()])
            for level, n, m, sizes in partition_info]
        execution = execution_metadata(device, ['GPU degree probabilities and target distribution interpolation',
            'GPU L2 GW/proximal log Sinkhorn with device convergence masks',
            'GPU barycenter averaging, partition labels and induced subgraph construction',
            'GPU recursive cluster eligibility and index maps', 'GPU native ordered leaf greedy decoder'])
        execution.update(recursion_control='GPU labels and cluster eligibility; host schedules GPU-selected dynamic shapes',
                         cpu_recursive_partition_decisions=False)
        result = AlignmentResult('S-GWL-author-recursive', pairs_export(pairs), 0.,
            score_matrix=export(score), larger_is_better=True,
            metadata=dict(config=asdict(cfg), official_commit='d682bcd1729e2c76cb65cb53e7d38ee6914e4263',
                actual_backend='device_author_recursive_direct_graph_matching', leaf_count=len(leaf_info),
                leaves=leaf_info, partition_calls=partition_calls, transposed_for_size_order=swapped,
                numerical_adapter=dict(version='device_float64_ordered_sparse_gw_log_sinkhorn_v4',
                    sparse_reduction='sorted_CSR_sequential_float64_no_atomics',
                    deterministic_sparse_reductions=True,
                    same_iteration_budgets=True, kernel_clipping=False, raw_transport_floor=False,
                    sparse_matrix_power_semantics_preserved=True,
                    effective_partition_outer_iterations=[export(x).tolist() for x in partition_iterations],
                    effective_leaf_outer_iterations=[int(export(x)) for x in leaf_iterations]),
                probability_adapter=dict(epsilon=cfg.probability_epsilon,
                    source_isolates=int(isolates[0]), target_isolates=int(isolates[1])),
                ranking_representation='same-fit author leaf transport lifted to global block matrix; excluded cross-leaf pairs have zero mass',
                prediction_rule='author recursive leaf traversal with device conditional-transport greedy',
                tie_policy='stable lower candidate node ID for exactly equal conditional transport; NumPy unstable quicksort order is not claimed',
                implementation_revision=REVISION, gpu_revision=REVISION, device_execution=execution,
                ground_truth_used_during_inference=False))
        result.runtime_sec = time.perf_counter() - started
        execution['full_algorithm_end_to_end_seconds'] = result.runtime_sec
        return result
