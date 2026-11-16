"""Historical same-fit GRAMPA ranking transform, never used to refit/decode."""
import numpy as np
from .metrics import evaluate_ranking_from_matrix

POLICY = dict(revision="grampa_soft_v1", mode="grampa_sinkhorn_soft_score", temperature=0.3,
              sinkhorn_iterations=80, row_standardization_epsilon=1e-9, minimum_positive_mass=1e-300,
              ground_truth_used_for_transform=False, hard_outputs_changed=False,
              aggregation="bidirectional_macro_average", candidate_scope="all_opposite_input_nodes")


def sinkhorn_soft_scores(scores, tau=0.3, iters=80):
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite GRAMPA scores")
    tau = max(float(tau), 1e-12)
    z = np.asarray(scores, dtype=np.float64)
    z = (z - z.mean(axis=1, keepdims=True)) / (z.std(axis=1, keepdims=True) + 1e-9)
    z = z / tau
    z = z - z.max(axis=1, keepdims=True)
    soft = np.exp(z)
    soft = np.maximum(soft, 1e-300)
    n_rows, n_cols = soft.shape
    col_mass = float(n_rows) / float(max(1, n_cols))
    for _ in range(max(0, int(iters))):
        soft = soft / (soft.sum(axis=1, keepdims=True) + 1e-300)
        soft = soft * (col_mass / (soft.sum(axis=0, keepdims=True) + 1e-300))
    soft = soft / (soft.sum(axis=1, keepdims=True) + 1e-300)
    return soft


def rank_soft_scores(scores, ground_truth):
    soft = sinkhorn_soft_scores(scores)
    forward = evaluate_ranking_from_matrix(soft, ground_truth)
    reverse = evaluate_ranking_from_matrix(soft.T, {v:u for u,v in ground_truth.items()})
    return {"conditional_ranking": {"source_to_target": forward, "target_to_source": reverse},
            "bidirectional_conditional_ranking": {k:(forward[k]+reverse[k])/2 for k in ("hits@1","hits@5","hits@10","mrr")},
            "conditional_ranking_source": POLICY["mode"]}
