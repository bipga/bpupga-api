"""OPGM-U: graph-only unsupervised GIN plus the author's partial assignment.

This is an explicitly changed method, not a reproduction of supervised
OPGM-rs. Cross-graph labels, true K, and true overlap are never accepted.
"""
from __future__ import annotations

import hashlib
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import baseline
from common.result import AlignmentResult

# The pinned wheel is extracted locally, without changing the running env.
DEPENDENCIES = Path(__file__).resolve().parent / "python_deps"
if str(DEPENDENCIES) not in sys.path:
    sys.path.insert(0, str(DEPENDENCIES))
from .vendor.pma import pma
from .vendor.sinkhorn import Sinkhorn


@dataclass(frozen=True)
class Config:
    layers: int = 5
    hidden_dim: int = 64
    epochs: int = 100
    learning_rate: float = .001
    weight_decay: float = .0001
    edge_batch_size: int = 8192
    feature_loss_weight: float = .1
    variance_loss_weight: float = .01
    sinkhorn_iterations: int = 20
    sinkhorn_temperature: float = .05
    solver_penalty: float = .4
    seed: int = 2026
    device: str = "cpu"

    def __post_init__(self):
        if self.layers != 5 or min(self.hidden_dim, self.epochs, self.edge_batch_size) < 1:
            raise ValueError("Use the declared five-layer GIN and positive budgets")
        if self.learning_rate <= 0 or self.sinkhorn_temperature <= 0 or self.sinkhorn_iterations < 1:
            raise ValueError("Invalid optimizer or Sinkhorn configuration")
        if not 0 < self.solver_penalty < .5:
            raise ValueError("Fixed-bias partial penalty must remain between0 and.5")


def degree_features(graph):
    """Permutation equivariant degree and neighbour-degree moments, no IDs."""
    n = len(graph)
    if set(graph) != set(range(n)):
        raise ValueError("Expected continuous local node indices")
    degree = np.asarray([graph.degree(u) for u in range(n)], dtype=np.float32)
    log_degree = np.log1p(degree)
    mean, spread = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for u in range(n):
        values = log_degree[list(graph.neighbors(u))]
        if len(values):
            mean[u], spread[u] = values.mean(), values.std()
    return np.column_stack((log_degree, mean, spread)).astype(np.float32)


def adjacency(graph, device):
    edges = np.asarray(sorted(tuple(sorted(e)) for e in graph.edges()), dtype=np.int64).reshape(-1, 2)
    if len(edges):
        directed = np.concatenate((edges, edges[:, ::-1]), axis=0).T.copy()
    else:
        directed = np.empty((2, 0), dtype=np.int64)
    indices = torch.as_tensor(directed, dtype=torch.long, device=device)
    matrix = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1], device=device),
                                     (len(graph), len(graph))).coalesce()
    return matrix, edges


class Encoder(nn.Module):
    """Five GIN sum-aggregation layers; dimensions are project choices."""
    def __init__(self, config):
        super().__init__()
        self.epsilon = nn.Parameter(torch.zeros(config.layers))
        self.layers = nn.ModuleList()
        for layer in range(config.layers):
            width = 3 if layer == 0 else config.hidden_dim
            self.layers.append(nn.Sequential(nn.Linear(width, config.hidden_dim), nn.ReLU(),
                              nn.Linear(config.hidden_dim, config.hidden_dim), nn.LayerNorm(config.hidden_dim), nn.Tanh()))
        self.feature_decoder = nn.Linear(config.hidden_dim, 3)
        self.edge_projection = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.edge_intercept = nn.Parameter(torch.tensor(-2.))

    def forward(self, features, graph):
        hidden, states = features, []
        for index, layer in enumerate(self.layers):
            hidden = layer((1 + self.epsilon[index]) * hidden + torch.sparse.mm(graph, hidden))
            states.append(hidden)
        return hidden, states

    def edge_logits(self, hidden, pairs):
        projected = F.normalize(self.edge_projection(hidden), dim=-1, eps=1e-8)
        return 8 * (projected[pairs[:, 0]] * projected[pairs[:, 1]]).sum(-1) + self.edge_intercept


def sample_edges(graph, positive, batch, rng):
    """Only within-graph observed edges/nonedges, never cross-graph pairs."""
    n = len(graph)
    chosen = positive[rng.choice(len(positive), min(batch, len(positive)), replace=False)] if len(positive) else positive
    negative = []
    available = n * (n - 1) // 2 - len(positive)
    target = min(batch, available)
    if target and (n <= 128 or nx.density(graph) > .5):
        pool = np.asarray([(u, v) for u in range(n) for v in range(u + 1, n) if not graph.has_edge(u, v)], dtype=np.int64)
        negative = pool[rng.choice(len(pool), target, replace=False)]
    elif target:
        # Sampling with replacement is intentional for large within-graph sets.
        while len(negative) < target:
            samples = rng.integers(0, n, size=(max(64, 2 * (target - len(negative))), 2))
            for u, v in samples:
                if u != v and not graph.has_edge(int(u), int(v)):
                    negative.append((int(u), int(v)))
                    if len(negative) == target:
                        break
    return chosen, np.asarray(negative, dtype=np.int64).reshape(-1, 2)


def uniform_lift(transport):
    """Dimensionless relative evidence above the independent uniform null.

    Explicit project calibration: it is not the author's raw Sinkhorn cost.
    This neither estimates K nor promises calibrated match probabilities.
    """
    n1, n2 = transport.shape
    eps = torch.finfo(transport.dtype).tiny
    row = transport.sum(1, keepdim=True).clamp_min(eps)
    col = transport.sum(0, keepdim=True).clamp_min(eps)
    lift = torch.sqrt((n2 * transport / row) * (n1 * transport / col))
    return (1 - 1 / lift.clamp_min(1)).clamp(0, 1)


def fixed_bias_assignment(scores, penalty):
    """Call unchanged author pma with alpha=beta=1 (the OPGM branch)."""
    cpu = scores.detach().to("cpu", dtype=torch.float32)
    # pma obtains biases by sigmoid. ones and scale1e6 yield exact1 in float32.
    shape = torch.ones_like(cpu)
    scale = torch.tensor([1e6], dtype=torch.float32)
    unused = torch.ones(1, dtype=torch.float32)
    assigned = pma(cpu, shape, scale, unused, unused, penalty, nproc=1)
    rows, columns = torch.nonzero(assigned > .5, as_tuple=True)
    return list(zip(rows.tolist(), columns.tolist()))


class OPGMUnsupervised:
    def __init__(self, config):
        self.config = config
        self.network = None
        self.training_history = []

    def export_checkpoint(self):
        return dict(config=asdict(self.config), state_dict={key: value.detach().cpu() for key, value in self.network.state_dict().items()},
                    training_history=self.training_history, supervision="within_graph_edges_and_degree_features_only",
                    cross_graph_targets_used=False)

    def fit(self, graph1, graph2):
        cfg, started = self.config, time.perf_counter()
        graphs = (graph1, graph2)
        if any(len(g) == 0 or nx.number_of_selfloops(g) for g in graphs):
            raise ValueError("Nonempty simple graphs required")
        device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        raw = [degree_features(g) for g in graphs]
        pooled = np.concatenate(raw)
        center = pooled.mean(0)
        scale = np.maximum(pooled.std(0), 1e-6)
        features = [torch.as_tensor((values - center) / scale, device=device) for values in raw]
        packets = [adjacency(g, device) for g in graphs]
        self.network = Encoder(cfg).to(device)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        initial = {key: value.detach().clone() for key, value in self.network.named_parameters()}
        self.training_history = []
        self.network.train()
        for epoch in range(cfg.epochs):
            optimizer.zero_grad(set_to_none=True)
            losses, items = [], []
            for graph, x, (matrix, positives) in zip(graphs, features, packets):
                hidden, _ = self.network(x, matrix)
                positive, negative = sample_edges(graph, positives, cfg.edge_batch_size, rng)
                edge_terms = []
                for pairs, label in ((positive, 1.), (negative, 0.)):
                    if len(pairs):
                        pair_tensor = torch.as_tensor(pairs, device=device)
                        logits = self.network.edge_logits(hidden, pair_tensor)
                        edge_terms.append(F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, label)))
                edge_loss = torch.stack(edge_terms).mean() if edge_terms else hidden.sum() * 0
                feature_loss = F.mse_loss(self.network.feature_decoder(hidden), x)
                variance_loss = F.relu(.5 - hidden.std(0, unbiased=False)).mean() if len(graph) > 1 else hidden.sum() * 0
                loss = edge_loss + cfg.feature_loss_weight * feature_loss + cfg.variance_loss_weight * variance_loss
                losses.append(loss)
                items.append(dict(edge=float(edge_loss.detach()), feature=float(feature_loss.detach()),
                                  variance=float(variance_loss.detach()), positive_samples=len(positive), negative_samples=len(negative)))
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite unsupervised training objective")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.network.parameters()):
                raise FloatingPointError("Non-finite unsupervised training gradient")
            gradient_norm = float(nn.utils.clip_grad_norm_(self.network.parameters(), 5.))
            optimizer.step()
            self.training_history.append(dict(epoch=epoch + 1, loss=float(loss.detach()), gradient_norm=gradient_norm, graphs=items))
        self.network.eval()
        with torch.no_grad():
            parameter_change = math.sqrt(sum(float((p - initial[key]).square().sum()) for key, p in self.network.named_parameters()))
            states = []
            for x, (matrix, _) in zip(features, packets):
                _, layers = self.network(x, matrix)
                states.append(torch.cat([x] + layers, 1))
            all_states = torch.cat(states)
            mean = all_states.mean(0, keepdim=True)
            std = all_states.std(0, keepdim=True, unbiased=False).clamp_min(1e-5)
            embeddings = [F.normalize((values - mean) / std, dim=1, eps=1e-8) for values in states]
            cosine = (embeddings[0] @ embeddings[1].T).clamp(-1, 1)
            normalizer = Sinkhorn(max_iter=cfg.sinkhorn_iterations, tau=cfg.sinkhorn_temperature, log_forward=True)
            orders = [torch.tensor([len(graph)], dtype=torch.long, device=device) for graph in graphs]
            transport = normalizer(cosine.unsqueeze(0), orders[0], orders[1], dummy_row=True).squeeze(0)
            score = uniform_lift(transport)
            if not torch.isfinite(score).all():
                raise FloatingPointError("Non-finite calibrated partial score")
            pairs = fixed_bias_assignment(score, cfg.solver_penalty)
            arrays = [values.cpu().numpy() for values in embeddings]
            score_array = score.cpu().numpy()
            transport_array = transport.cpu().numpy()
            cosine_array = cosine.cpu().numpy()
        digest = hashlib.sha256()
        for key, value in sorted(self.network.state_dict().items()):
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return AlignmentResult(method="opgm_unsupervised", predicted_pairs=pairs, runtime_sec=time.perf_counter() - started,
            score_matrix=score_array, larger_is_better=True, source_embeddings=arrays[0], target_embeddings=arrays[1],
            stage_score_matrices={"author_sinkhorn": transport_array, "gin_cosine": cosine_array},
            stage_diagnostics_metadata={"author_sinkhorn": {"larger_is_better": True}, "gin_cosine": {"larger_is_better": True}},
            metadata=dict(implementation_scope="project_unsupervised_OPGM_fixed_bias_GIN_adaptation_r1",
                original_supervised_OPGMrs_reproduction=False, official_commit="39b5344c7a31c3a764f91be754932c7160a8bcfa",
                encoder="paper_PPI_five_layer_GIN_family_independently_implemented", config=asdict(cfg),
                loss="within_graph_balanced_edge_BCE_plus_degree_feature_reconstruction_plus_variance_regularizer",
                ground_truth_used_during_inference=False, cross_graph_training_targets=False, supervised_pretraining=False,
                known_K_used=False, true_overlap_used=False, optimizer_steps=cfg.epochs, parameter_update_l2=parameter_change,
                first_training_loss=self.training_history[0]["loss"], last_training_loss=self.training_history[-1]["loss"],
                training_history=self.training_history, trained_state_sha256=digest.hexdigest(),
                score_calibration="uniform_null_bilateral_lift_S=max(0,1-1/sqrt(n2*P(j|i)*n1*P(i|j)))",
                score_calibration_is_author_original=False, calibrated_probability_claim=False,
                bias_mode="original_OPGM_fixed_alpha_beta_equal_one_not_learned_rs_bias",
                fixed_bias_implementation="author_sigmoid_with_os_ones_and_wm1_1e6_float32_saturation",
                solver="author_pma_cost_cap_LSA_then_feasibility_filter", solver_penalty_is_not_node_overlap=True,
                prediction_rule="author_partial_assignment_on_explicitly_calibrated_same_fit_scores",
                no_minimum_prediction_count=True, empty_output_is_allowed=True,
                isolates=[nx.number_of_isolates(g) for g in graphs],
                numerical_safeguards=dict(feature_std_floor=1e-6, embedding_std_floor=1e-5,
                    normalization_epsilon=1e-8, gradient_norm_cap=5., conditional_mass_floor="float32_tiny")))


def construct(parameters, seed=2026, device="cpu"):
    forbidden = {"ground_truth", "gt", "true_k", "true_overlap", "rho", "dataset", "data_root"}
    if forbidden.intersection(parameters):
        raise ValueError("Ground-truth and dataset inputs are forbidden")
    config = Config(**dict(parameters, seed=seed, device=device))
    return OPGMUnsupervised(config), asdict(config)
