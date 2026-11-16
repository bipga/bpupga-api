"""ULOT, OPGM-U and SLOTAlign with numerical work and decoding on CUDA.

Input parsing/checkpoint reads, host launches and final artifact serialization
are the only CPU stages. Native objectives, budgets and assignment rules remain
explicit; GPU RNG initialization/sampling has its own reproducible trajectory.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import time
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl
import baseline
from common.result import AlignmentResult

from . import REVISION
from .graph import DeviceGraph, topology_pair
from .assignment import linear_sum_assignment
from .iterative import export, pairs_export, execution_metadata


def finite_export(flag, message):
    """Final error reporting only; it never selects a numerical fallback."""
    if not bool(export(flag)):
        raise FloatingPointError(message)


def single_graph_matching(self, x1, x2, batch1, batch2, features1, features2):
    """Exact single-graph specialization of author ULOT matching_layer.

    Every node is real in a one-graph packet, so to_dense_batch and its hidden
    .item() shape decision reduce to unsqueeze and all-true masks. This also
    avoids creating padded NaNs solely to mask them back to zero.
    """
    a, b = x1.unsqueeze(0), x2.unsqueeze(0)
    fa, fb = features1.unsqueeze(0), features2.unsqueeze(0)
    normalized1 = F.normalize(self.linear(fa), p=2, dim=-1)
    normalized2 = F.normalize(self.linear(fb), p=2, dim=-1)
    similarity = torch.bmm(normalized1, normalized2.transpose(1, 2))
    scale = self.a.square()
    attention1 = torch.softmax(scale * similarity, dim=2)
    attention2 = torch.softmax(scale * similarity, dim=1)
    match1 = a - torch.bmm(attention1, b)
    match2 = b - torch.bmm(attention2.transpose(1, 2), a)
    volume1 = torch.sigmoid(self.linear_volume(fa))
    volume2 = torch.sigmoid(self.linear_volume(fb))
    plan = (attention1 * volume1 / len(x1) +
            attention2 * volume2.permute(0, 2, 1) / len(x2)) / 2
    mask1 = torch.ones((1, len(x1)), device=x1.device, dtype=torch.bool)
    mask2 = torch.ones((1, len(x2)), device=x2.device, dtype=torch.bool)
    return match1[0], match2[0], plan, mask1, mask2


class TensorULOT:
    def __init__(self, config):
        self.config = config
        self.config.validate()

    def fit(self, graph1, graph2):
        from baseline.ulot.model import ULOT, _sha256, OFFICIAL_COMMIT
        from baseline.vendor.ulot.ulot.architectures import ULOT_net
        cfg, started = self.config, time.perf_counter()
        device = torch.device(cfg.device)
        g1, g2 = DeviceGraph(graph1, device), DeviceGraph(graph2, device)
        if not g1.n or not g2.n:
            raise ValueError('ULOT cannot align empty graphs')
        x1, x2 = topology_pair(g1, g2, cfg.feature_adapter)
        torch.manual_seed(cfg.seed)
        # Allocate and initialize model tensors directly on the selected device.
        with torch.device(device):
            network = ULOT_net(3, cfg.hidden_channels, cfg.hidden_channels_message,
                cfg.out_channels, cfg.num_layers, cfg.temperature, cfg.alpha_encoding_dim)
        checkpoint_path = ULOT(cfg)._checkpoint_path()
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
        network.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)
        network.eval()
        for layer in (network.first_GMN_layer, network.GMN_layer):
            layer.match.forward = MethodType(single_graph_matching, layer.match)
        with torch.inference_mode():
            plan, _, _ = network(x1, x2, g1.adjacency.indices(), g2.adjacency.indices(),
                torch.zeros(g1.n, dtype=torch.long, device=device),
                torch.zeros(g2.n, dtype=torch.long, device=device),
                torch.full((g1.n, 1), cfg.rho, device=device),
                torch.full((g2.n, 1), cfg.rho, device=device),
                torch.full((g1.n, 1), cfg.alpha, device=device),
                torch.full((g2.n, 1), cfg.alpha, device=device))
            score = plan[0].float()
            finite = torch.isfinite(score).all() & (score >= 0).all()
            source, target = linear_sum_assignment(score, maximize=True, device=str(device))
            pairs = torch.stack((source, target), 1)
        # Final diagnostics/export only.
        finite_export(finite, 'ULOT produced an invalid transport plan')
        execution = execution_metadata(device, ['GPU degree quantiles and topology proxy features',
            'GPU checkpoint GCN/cross-attention', 'GPU single-graph dense-batch specialization',
            'GPU exact maximum-weight assignment'])
        result = AlignmentResult('ULOT', pairs_export(pairs), 0., score_matrix=export(score),
            larger_is_better=True, metadata=dict(config=asdict(cfg), device=str(device),
                core_method='published neural ULOT cross-attention plan predictor',
                official_commit=OFFICIAL_COMMIT, checkpoint=str(checkpoint_path),
                checkpoint_sha256=_sha256(checkpoint_path),
                checkpoint_git_blob='3a1c784afa19cf77c7e77bab03bb163e1b98db91',
                checkpoint_training_domain='released 50k random-size three-community SBM model',
                evaluation_graphs_used_for_training=False, feature_adapter=cfg.feature_adapter,
                feature_adapter_scope='BPUPGA plain-graph adaptation; not in ULOT paper',
                prediction_rule='maximum-weight exact assignment on predicted transport',
                partial_adaptation='supplementary method-aware decoder only; native uses Hungarian',
                single_graph_specialization_mathematically_equivalent=True,
                implementation_revision=REVISION, gpu_revision=REVISION, device_execution=execution,
                ground_truth_used_during_inference=False,
                ground_truth_cardinality_used_during_inference=False))
        result.runtime_sec = time.perf_counter() - started
        execution['full_algorithm_end_to_end_seconds'] = result.runtime_sec
        return result


def sinkhorn_single(scores, iterations, temperature):
    """pygm 0.5.3 log Sinkhorn with dummy_row=True, one unpadded instance."""
    transposed = scores.shape[0] > scores.shape[1]
    matrix = scores.T if transposed else scores
    n, m = matrix.shape
    log_s = matrix / temperature
    if n < m:
        log_s = torch.cat((log_s, torch.full((m - n, m), -100., device=matrix.device,
                                           dtype=matrix.dtype)), 0)
    for iteration in range(iterations):
        dim = 1 if iteration % 2 == 0 else 0
        log_s = log_s - torch.logsumexp(log_s, dim=dim, keepdim=True)
    result = log_s[:n].exp()
    return result.T if transposed else result


def fixed_bias_assignment(scores, penalty):
    """Unchanged author PMA: capped square cost, exact LAP, feasibility filter.

    In float32 sigmoid(1e6*1) is exactly one. Real-real thresholds are 2*rho;
    native padded rows/columns have bias 1.1 and threshold 2.1*rho.
    """
    n, m = scores.shape
    maximum = max(n, m)
    cost = 1 - scores.float()
    row_bias = torch.ones(maximum, device=scores.device, dtype=torch.float32)
    col_bias = torch.ones_like(row_bias)
    if n < maximum:
        row_bias[n:] = 1.1
    if m < maximum:
        col_bias[m:] = 1.1
    # Preserve the author's float32 addition before multiplying by rho.
    baseline = (row_bias[:, None] + col_bias[None]) * penalty
    padded = torch.full_like(baseline, 10000.)
    padded[:n, :m] = cost
    capped = torch.where(padded <= baseline, padded, baseline)
    row, col = linear_sum_assignment(capped, device=str(scores.device))
    valid = (row < n) & (col < m)
    # Only valid real indices are used to test author feasibility.
    real_row, real_col = row[valid], col[valid]
    feasible = cost[real_row, real_col] <= baseline[real_row, real_col]
    return torch.stack((real_row[feasible], real_col[feasible]), 1)


@triton.jit
def _nonedge_with_replacement(KEYS, OUTPUT, N: tl.constexpr, E: tl.constexpr,
                              TARGET: tl.constexpr, SEED,
                              BLOCK: tl.constexpr):
    """Persistent GPU rejection sampling and binary-search edge membership."""
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < TARGET
    attempt = tl.full((BLOCK,), 0, tl.int32)
    while tl.sum(active.to(tl.int32), 0) > 0:
        offset = index.to(tl.uint32) * 0x9e3779b9 + attempt.to(tl.uint32) * 0x85ebca6b
        u = tl.minimum((tl.rand(SEED, offset) * N).to(tl.int64), N - 1)
        v = tl.minimum((tl.rand(SEED ^ 0x68bc21eb, offset) * N).to(tl.int64), N - 1)
        key = tl.minimum(u, v) * N + tl.maximum(u, v)
        lo = tl.full((BLOCK,), 0, tl.int32)
        hi = tl.full((BLOCK,), E, tl.int32)
        searching = active & (lo < hi)
        while tl.sum(searching.to(tl.int32), 0) > 0:
            middle = (lo + hi) // 2
            value = tl.load(KEYS + middle, searching & (middle < E), other=0x7fffffffffffffff)
            less = value < key
            lo = tl.where(searching & less, middle + 1, lo)
            hi = tl.where(searching & ~less, middle, hi)
            searching = active & (lo < hi)
        found = tl.load(KEYS + lo, active & (lo < E), other=-1) == key
        accepted = active & (u != v) & ~found
        tl.store(OUTPUT + 2 * index, tl.minimum(u, v), accepted)
        tl.store(OUTPUT + 2 * index + 1, tl.maximum(u, v), accepted)
        active = active & ~accepted
        attempt += 1


class DeviceEdgeSampler:
    def __init__(self, graph, batch, seed):
        self.graph, self.batch, self.seed = graph, batch, seed
        edges = graph.raw.T
        # Native positive records are lexicographically sorted undirected edges.
        edges = torch.sort(edges, dim=1).values
        order = torch.argsort(edges[:, 0] * graph.n + edges[:, 1], stable=True)
        self.positive = edges[order]
        self.keys = self.positive[:, 0] * graph.n + self.positive[:, 1]
        self.available = graph.n * (graph.n - 1) // 2 - edges.shape[0]
        self.target = min(batch, self.available)
        # This branch reads only imported graph shapes, exactly the native
        # n<=128 / density>0.5 sampling policy. All membership/sampling is GPU.
        self.without_replacement = graph.n <= 128 or edges.shape[0] > graph.n * (graph.n - 1) / 4
        self.pool = None
        if self.target and self.without_replacement:
            ij = torch.triu_indices(graph.n, graph.n, offset=1, device=graph.device)
            keys = ij[0] * graph.n + ij[1]
            location = torch.searchsorted(self.keys, keys)
            found = (location < self.keys.numel()) & (self.keys[location.clamp_max(max(0, self.keys.numel() - 1))] == keys) if self.keys.numel() else torch.zeros_like(keys, dtype=torch.bool)
            self.pool = ij[:, ~found].T.contiguous()

    def sample(self, epoch, generator):
        size = min(self.batch, self.positive.shape[0])
        positive = self.positive[torch.randperm(self.positive.shape[0], device=self.graph.device,
                                              generator=generator)[:size]]
        if not self.target:
            negative = torch.empty((0, 2), device=self.graph.device, dtype=torch.int64)
        elif self.without_replacement:
            negative = self.pool[torch.randperm(self.pool.shape[0], device=self.graph.device,
                                                generator=generator)[:self.target]]
        else:
            negative = torch.empty((self.target, 2), device=self.graph.device, dtype=torch.int64)
            if self.graph.device.type != 'cuda':
                raise RuntimeError('Large-graph rejection sampler requires CUDA; no CPU fallback')
            _nonedge_with_replacement[(triton.cdiv(self.target, 256),)](
                self.keys, negative, self.graph.n, self.keys.numel(), self.target,
                (self.seed + 1000003 * epoch) & 0xffffffff, 256)
        return positive, negative


class TensorOPGM:
    def __init__(self, config):
        self.config, self.network, self.training_history = config, None, []

    def export_checkpoint(self):
        # Called after fit: this is a final artifact export.
        return dict(config=asdict(self.config),
            state_dict={key: value.detach().cpu() for key, value in self.network.state_dict().items()},
            training_history=self.training_history, supervision='within_graph_edges_and_degree_features_only',
            cross_graph_targets_used=False)

    def fit(self, graph1, graph2):
        from baseline.opgm_unsupervised.model import Encoder, uniform_lift
        cfg, started = self.config, time.perf_counter()
        device = torch.device(cfg.device)
        graphs = [DeviceGraph(g, device) for g in (graph1, graph2)]
        if any(not g.n for g in graphs):
            raise ValueError('Nonempty simple graphs required')
        # Input validity is a GPU-derived shape check, not a CPU graph routine.
        if any(torch.nonzero(g.raw[0] == g.raw[1]).numel() for g in graphs):
            raise ValueError('Nonempty simple graphs without self loops required')
        torch.manual_seed(cfg.seed)
        generator = torch.Generator(device=device).manual_seed(cfg.seed)
        raw = [g.degree_features() for g in graphs]
        pooled = torch.cat(raw)
        center, scale = pooled.mean(0), pooled.std(0, correction=0).clamp_min(1e-6)
        features = [(x - center) / scale for x in raw]
        matrices = [g.sparse(torch.float32) for g in graphs]
        samplers = [DeviceEdgeSampler(g, cfg.edge_batch_size, cfg.seed + i * 1000033)
                    for i, g in enumerate(graphs)]
        with torch.device(device):
            self.network = Encoder(cfg)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay, capturable=device.type == 'cuda')
        initial = {key: value.detach().clone() for key, value in self.network.named_parameters()}
        self.network.train()
        valid = torch.ones((), device=device, dtype=torch.bool)
        history, sample_counts = [], []
        for epoch in range(cfg.epochs):
            optimizer.zero_grad(set_to_none=True)
            losses, components, counts = [], [], []
            for g, x, matrix, sampler in zip(graphs, features, matrices, samplers):
                hidden, _ = self.network(x, matrix)
                positive, negative = sampler.sample(epoch, generator)
                edge_terms = []
                for pairs, label in ((positive, 1.), (negative, 0.)):
                    if pairs.shape[0]:
                        logits = self.network.edge_logits(hidden, pairs)
                        edge_terms.append(F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, label)))
                edge_loss = torch.stack(edge_terms).mean() if edge_terms else hidden.sum() * 0
                feature_loss = F.mse_loss(self.network.feature_decoder(hidden), x)
                variance = F.relu(.5 - hidden.std(0, correction=0)).mean() if g.n > 1 else hidden.sum() * 0
                loss = edge_loss + cfg.feature_loss_weight * feature_loss + cfg.variance_loss_weight * variance
                losses.append(loss)
                components.append(torch.stack((edge_loss.detach(), feature_loss.detach(), variance.detach())))
                counts.append((positive.shape[0], negative.shape[0]))
            loss = torch.stack(losses).mean()
            valid = valid & torch.isfinite(loss)
            loss.backward()
            gradient_valid = torch.stack([torch.isfinite(p.grad).all() for p in self.network.parameters()
                                          if p.grad is not None]).all()
            valid = valid & gradient_valid
            gradient_norm = nn.utils.clip_grad_norm_(self.network.parameters(), 5.)
            optimizer.step()
            history.append(torch.cat((torch.stack((loss.detach(), gradient_norm.detach())),
                                      torch.cat(components))))
            sample_counts.append(counts)
        self.network.eval()
        with torch.no_grad():
            change = torch.stack([(p - initial[key]).square().sum()
                                  for key, p in self.network.named_parameters()]).sum().sqrt()
            states = []
            for x, matrix in zip(features, matrices):
                _, layers = self.network(x, matrix)
                states.append(torch.cat([x] + layers, 1))
            all_states = torch.cat(states)
            mean, std = all_states.mean(0, keepdim=True), all_states.std(0, keepdim=True, correction=0).clamp_min(1e-5)
            embeddings = [F.normalize((x - mean) / std, dim=1, eps=1e-8) for x in states]
            cosine = (embeddings[0] @ embeddings[1].T).clamp(-1, 1)
            transport = sinkhorn_single(cosine, cfg.sinkhorn_iterations, cfg.sinkhorn_temperature)
            score = uniform_lift(transport)
            valid = valid & torch.isfinite(score).all()
            pairs = fixed_bias_assignment(score, cfg.solver_penalty)
            isolates = torch.stack([(g.degree == 0).sum() for g in graphs])
        # Final serialization and numerical error reporting begin here.
        finite_export(valid, 'Non-finite unsupervised OPGM training or inference')
        values = export(torch.stack(history)).tolist()
        self.training_history = [dict(epoch=i + 1, loss=row[0], gradient_norm=row[1],
            graphs=[dict(edge=row[2 + 3 * j], feature=row[3 + 3 * j], variance=row[4 + 3 * j],
                         positive_samples=sample_counts[i][j][0], negative_samples=sample_counts[i][j][1])
                    for j in range(2)]) for i, row in enumerate(values)]
        digest = hashlib.sha256()
        for key, value in sorted(self.network.state_dict().items()):
            digest.update(key.encode())
            digest.update(export(value).tobytes())
        execution = execution_metadata(device, ['GPU degree moments and pooled feature normalization',
            'GPU seeded positive/negative edge sampling with persistent rejection kernel',
            'GPU five-layer GIN, Adam state, objective and gradient clipping',
            'GPU log Sinkhorn, uniform-lift calibration and exact partial PMA'])
        result = AlignmentResult('opgm_unsupervised', pairs_export(pairs), 0., score_matrix=export(score),
            larger_is_better=True, source_embeddings=export(embeddings[0]), target_embeddings=export(embeddings[1]),
            stage_score_matrices={'author_sinkhorn': export(transport), 'gin_cosine': export(cosine)},
            stage_diagnostics_metadata={'author_sinkhorn': {'larger_is_better': True}, 'gin_cosine': {'larger_is_better': True}},
            metadata=dict(implementation_scope='project_unsupervised_OPGM_fixed_bias_GIN_adaptation_device_v2',
                original_supervised_OPGMrs_reproduction=False, official_commit='39b5344c7a31c3a764f91be754932c7160a8bcfa',
                encoder='paper_PPI_five_layer_GIN_family_independently_implemented', config=asdict(cfg),
                loss='within_graph_balanced_edge_BCE_plus_degree_feature_reconstruction_plus_variance_regularizer',
                ground_truth_used_during_inference=False, cross_graph_training_targets=False, supervised_pretraining=False,
                known_K_used=False, true_overlap_used=False, optimizer_steps=cfg.epochs,
                parameter_update_l2=float(export(change)), first_training_loss=values[0][0], last_training_loss=values[-1][0],
                training_history=self.training_history, trained_state_sha256=digest.hexdigest(),
                score_calibration='uniform_null_bilateral_lift_S=max(0,1-1/sqrt(n2*P(j|i)*n1*P(i|j)))',
                score_calibration_is_author_original=False, calibrated_probability_claim=False,
                bias_mode='original_OPGM_fixed_alpha_beta_equal_one_not_learned_rs_bias',
                fixed_bias_implementation='author_sigmoid_with_os_ones_and_wm1_1e6_float32_saturation',
                solver='author_pma_cost_cap_exact_device_LSA_then_feasibility_filter',
                solver_penalty_is_not_node_overlap=True,
                prediction_rule='author_partial_assignment_on_explicitly_calibrated_same_fit_scores',
                no_minimum_prediction_count=True, empty_output_is_allowed=True, isolates=export(isolates).tolist(),
                numerical_safeguards=dict(feature_std_floor=1e-6, embedding_std_floor=1e-5,
                    normalization_epsilon=1e-8, gradient_norm_cap=5., conditional_mass_floor='float32_tiny'),
                gpu_rng_revision='torch_cuda_initialization_and_sampling_triton_rejection_v1',
                native_numpy_rng_trajectory_identical=False, same_native_sampling_replacement_policy=True,
                implementation_revision=REVISION, gpu_revision=REVISION, device_execution=execution))
        result.runtime_sec = time.perf_counter() - started
        execution['full_algorithm_end_to_end_seconds'] = result.runtime_sec
        return result


def project_simplex(vector):
    ordered = torch.sort(vector, descending=True).values
    cumulative = ordered.cumsum(0)
    ranks = torch.arange(1, vector.numel() + 1, device=vector.device, dtype=vector.dtype)
    valid = ordered * ranks > cumulative - 1
    rho = valid.sum().clamp_min(1)
    threshold = (cumulative.gather(0, (rho - 1).reshape(1))[0] - 1) / rho
    projected = (vector - threshold).clamp_min(0)
    feasible = torch.isclose(vector.sum(), torch.ones((), device=vector.device, dtype=vector.dtype),
                             rtol=1e-5, atol=1e-8) & (vector >= 0).all()
    return torch.where(feasible, vector, projected)


def structural_features(graph, rounds):
    adjacency = graph.sparse(torch.float32)
    degree = graph.neighbor_count.float()
    current = torch.stack((degree, degree.log1p(), (degree + 1e-6).sqrt(), degree.square()), 1)
    blocks = [current]
    denominator = degree.clamp_min(1)[:, None]
    for _ in range(rounds):
        summed = torch.sparse.mm(adjacency, current)
        mean = summed / denominator
        second = torch.sparse.mm(adjacency, current.square()) / denominator
        spread = ((second - mean.square()).clamp_min(0) + 1e-6).sqrt()
        current = torch.cat((mean, spread, summed), 1)
        current = ((current - current.mean(0, keepdim=True)) /
                   current.std(0, keepdim=True, correction=0).clamp_min(1e-6)).float()
        blocks.append(current)
    # This is the author's existing feature guard, performed on the device.
    return torch.nan_to_num(torch.cat(blocks, 1))


def normalize_pair(source, target, mode):
    if mode == 'none':
        return source, target
    if mode == 'center':
        return tuple((x - x.mean(0, keepdim=True)).float() for x in (source, target))
    if mode == 'separate':
        return tuple(((x - x.mean(0, keepdim=True)) /
            x.std(0, keepdim=True, correction=0).clamp_min(1e-6)).float() for x in (source, target))
    values = torch.cat((source, target))
    center, scale = values.mean(0, keepdim=True), values.std(0, keepdim=True, correction=0).clamp_min(1e-6)
    return (source - center) / scale, (target - center) / scale


def similarity_stack(graph, features, bases, dtype):
    adjacency = graph.dense(dtype)
    views = [adjacency]
    loops = adjacency + torch.eye(graph.n, device=graph.device, dtype=dtype)
    inv = loops.sum(1).clamp_min(1).rsqrt()
    propagation = inv[:, None] * loops * inv[None]
    current = features.to(dtype)
    for i in range(bases - 1):
        if i:
            current = propagation @ current
        normalized = current / current.norm(dim=1, keepdim=True).clamp_min(1e-12)
        views.append(normalized @ normalized.T)
    return torch.stack(views, 2)


SLOT_DIAGNOSTICS = ('updates', 'clip_entries', 'raw_nonfinite_exponent',
    'zero_column_denominator', 'zero_row_denominator', 'clamped_column_denominator',
    'clamped_row_denominator', 'inner_early_stops', 'nonfinite_return_previous',
    'floored_transport_entries', 'outer_early_stops')


@torch.no_grad()
def gw_step(source, target, transport, source_mass, target_mass, beta, iterations,
            diagnostics, enabled=None):
    active_outer = torch.ones((), device=transport.device, dtype=torch.bool) if enabled is None else enabled
    diagnostics['updates'] += active_outer.to(torch.int64)
    cost = -2. * source @ transport @ target.T
    raw = -(cost - cost.min()) / beta
    diagnostics['clip_entries'] += (((raw < -60) | (raw > 60)).sum() * active_outer)
    diagnostics['raw_nonfinite_exponent'] += ((~torch.isfinite(raw)).sum() * active_outer)
    diagnostics['floored_transport_entries'] += ((transport < 1e-30).sum() * active_outer)
    kernel = raw.clamp(-60, 60).exp() * transport.clamp_min(1e-30)
    left = torch.full_like(source_mass, 1. / source_mass.numel())
    right = torch.ones_like(target_mass)
    active = active_outer.clone()
    for _ in range(iterations):
        col_den = kernel.T @ left
        diagnostics['zero_column_denominator'] += ((col_den == 0).sum() * active)
        diagnostics['clamped_column_denominator'] += ((col_den < 1e-30).sum() * active)
        candidate_right = target_mass / col_den.clamp_min(1e-30)
        row_den = kernel @ candidate_right
        diagnostics['zero_row_denominator'] += ((row_den == 0).sum() * active)
        diagnostics['clamped_row_denominator'] += ((row_den < 1e-30).sum() * active)
        candidate_left = source_mass / row_den.clamp_min(1e-30)
        error = (candidate_left - left).abs().sum() / left.abs().sum()
        left = torch.where(active, candidate_left, left)
        right = torch.where(active, candidate_right, right)
        stopped = active & (error < 1e-10)
        diagnostics['inner_early_stops'] += stopped.to(torch.int64)
        active = active & ~stopped
    candidate = left[:, None] * kernel * right[None]
    finite = torch.isfinite(candidate).all()
    diagnostics['nonfinite_return_previous'] += ((~finite & active_outer).to(torch.int64))
    return torch.where(finite & active_outer, candidate, transport)


class TensorSLOTAlign:
    def __init__(self, config):
        self.config = config
        config.validate()

    def fit(self, graph1, graph2):
        cfg, started = self.config, time.perf_counter()
        device = torch.device(cfg.device)
        dtype = torch.float32 if cfg.dtype == 'float32' else torch.float64
        g1, g2 = DeviceGraph(graph1, device), DeviceGraph(graph2, device)
        if not g1.n or not g2.n:
            raise ValueError('SLOTAlign cannot align empty graphs')
        f1, f2 = normalize_pair(structural_features(g1, cfg.feature_rounds),
                                structural_features(g2, cfg.feature_rounds), cfg.feature_norm)
        views1, views2 = similarity_stack(g1, f1, cfg.bases, dtype), similarity_stack(g2, f2, cfg.bases, dtype)
        ps = torch.full((g1.n,), 1. / g1.n, device=device, dtype=dtype)
        pt = torch.full((g2.n,), 1. / g2.n, device=device, dtype=dtype)
        transport = ps[:, None] * pt[None]
        alpha = torch.full((cfg.bases,), 1. / cfg.bases, device=device, dtype=dtype)
        beta = torch.full_like(alpha, 1. / cfg.bases)
        source, target = (views1 * alpha).sum(2), (views2 * beta).sum(2)
        diagnostics = {name: torch.zeros((), device=device, dtype=torch.int64) for name in SLOT_DIAGNOSTICS}
        for _ in range(cfg.joint_epoch):
            a, b = alpha.detach().clone().requires_grad_(True), beta.detach().clone().requires_grad_(True)
            source, target = (views1 * a).sum(2), (views2 * b).sum(2)
            objective = source.square().mean() + target.square().mean() - torch.trace(source @ transport @ target @ transport.T)
            da, db = torch.autograd.grad(objective, (a, b))
            alpha, beta = project_simplex((a - cfg.step_size * da).detach()), project_simplex((b - cfg.step_size * db).detach())
            transport = gw_step(source.detach(), target.detach(), transport.detach(), ps, pt,
                                cfg.gw_beta, cfg.joint_sinkhorn_iters, diagnostics).detach()
        source, target, transport = source.detach(), target.detach(), transport.detach()
        active = torch.ones((), device=device, dtype=torch.bool)
        with torch.no_grad():
            for _ in range(cfg.epoch - cfg.joint_epoch):
                previous = transport
                transport = gw_step(source, target, transport, ps, pt, cfg.gw_beta,
                                    cfg.final_sinkhorn_iters, diagnostics, active)
                relative = (transport - previous).abs().sum() / previous.abs().sum()
                stopped = active & (relative < 1e-10)
                diagnostics['outer_early_stops'] += stopped.to(torch.int64)
                active = active & ~stopped
                # GPU computes convergence. The host only skips an empty launch
                # set, using the same dynamic-shape scheduling as sparse graphs.
                if torch.nonzero(active.reshape(1), as_tuple=True)[0].numel() == 0:
                    break
        pairs = torch.stack((torch.arange(g1.n, device=device), transport.argmax(1)), 1)
        finite = torch.isfinite(transport).all()
        finite_export(finite, 'Non-finite SLOTAlign transport')
        diagnostic_values = export(torch.stack(list(diagnostics.values()))).tolist()
        diagnostic_dict = dict(zip(SLOT_DIAGNOSTICS, diagnostic_values))
        execution = execution_metadata(device, ['GPU structural feature propagation and normalization',
            'GPU adjacency/feature similarity views and simplex projection',
            'GPU SLOT joint objective/autograd and GW/Sinkhorn convergence state',
            'GPU numerical diagnostics and row-maximum decoder'])
        result = AlignmentResult('SLOTAlign', pairs_export(pairs), 0., cost_matrix=export(-transport),
            larger_is_better=False, metadata=dict(config=asdict(cfg), device=str(device),
                alpha=export(alpha).tolist(), beta=export(beta).tolist(),
                feature_adapter='deterministic topology-only structural features',
                prediction_rule='row-wise maximum transport mass', numerical_guard_diagnostics=diagnostic_dict,
                nonfinite_guard_triggered=diagnostic_dict['nonfinite_return_previous'] > 0,
                author_update_coefficient=-1, author_code_priority=True,
                cost_matrix='negative transport mass (ranking-equivalent)',
                solver='exact dense SLOTAlign; no feature-OT fallback',
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
    params.update(seed=seed, device=device)
    if method == 'ulot':
        from baseline.ulot.model import ULOTConfig
        params['show_progress'] = False
        cfg = replace(ULOTConfig(**params), max_dense_entries=10 ** 18)
        model = TensorULOT(cfg)
    elif method == 'opgm_unsupervised':
        from baseline.opgm_unsupervised.model import Config
        if {'ground_truth', 'gt', 'true_k', 'true_overlap', 'rho', 'dataset', 'data_root'}.intersection(params):
            raise ValueError('Ground-truth and dataset inputs are forbidden')
        cfg, model = Config(**params), None
        model = TensorOPGM(cfg)
    elif method == 'slotalign':
        from baseline.slotalign.model import SLOTAlignConfig
        params['show_progress'] = False
        cfg = replace(SLOTAlignConfig(**params), max_dense_entries=10 ** 18)
        model = TensorSLOTAlign(cfg)
    else:
        raise ValueError('Unknown device neural method: ' + method)
    return model, dict(asdict(cfg), implementation_revision=REVISION, gpu_revision=REVISION,
                       cpu_fallback=False, execution_mode='device_resident_algorithm')
