"""Reuse the frozen model with an explicit extension of centrality iterations."""
from types import FunctionType, MethodType
import torch
from gpu_backend import gradalign as previous
from gpu_backend.adapters import DeviceContract
MAX_ITER=1000
REVISION='centrality_budget'

CENTRALITY_TRACE = []


def centrality(graph, version):
    if version != 'eigenvector':
        return previous.centrality(graph, version)
    x = torch.full((graph.n,), 1. / graph.n, dtype=torch.float64, device=graph.device)
    active = torch.ones((), device=graph.device, dtype=torch.bool)
    first = torch.zeros((), device=graph.device, dtype=torch.int64)
    for start in range(0, MAX_ITER, 500):
        for step in range(start + 1, min(start + 500, MAX_ITER) + 1):
            candidate = x + torch.sparse.mm(graph.adjacency, x[:, None])[:, 0]
            candidate = candidate / torch.linalg.vector_norm(candidate).clamp_min(torch.finfo(torch.float64).tiny)
            delta = (candidate - x).abs().sum()
            converged = delta < graph.n * 1e-8
            first = torch.where((first == 0) & converged, torch.full_like(first, step), first)
            x = torch.where(active, candidate, x)
            active = active & ~converged
        # Only dispatch the extension if the unchanged original budget failed.
        if not bool(active.cpu()):
            break
    torch._assert_async(~active, f'GradAlign eigenvector centrality did not converge in {MAX_ITER} steps')
    CENTRALITY_TRACE.append(dict(nodes=graph.n, first_converged_iteration=int(first.cpu()),
        executed_iterations=step, max_iter=MAX_ITER, original_max_iter=500, tolerance=1e-8,
        L1_threshold=graph.n * 1e-8, final_checked_L1_change=float(delta.cpu())))
    return x


attribute_globals = dict(previous.augmented_attributes.__globals__, centrality=centrality)
augmented_attributes = FunctionType(previous.augmented_attributes.__code__, attribute_globals)
fit_globals = dict(previous.TensorGradAlign.fit.__globals__, augmented_attributes=augmented_attributes,
    IMPLEMENTATION=REVISION)
extended_fit = FunctionType(previous.TensorGradAlign.fit.__code__, fit_globals)


class ExtendedContract(DeviceContract):
    def fit(self, *graphs):
        CENTRALITY_TRACE.clear()
        result = super().fit(*graphs)
        result.metadata.update(implementation_revision=REVISION, gpu_revision=REVISION,
            centrality_iteration_extension=dict(max_iter=MAX_ITER, original_max_iter=500, tolerance=1e-8,
                only_unconverged_inputs_execute_extra_iterations=True, sides=list(CENTRALITY_TRACE),
                training_epochs_optimizer_and_decoder_unchanged=True))
        result.metadata['config']['centrality_max_iter'] = MAX_ITER
        result.metadata['device_execution']['implementation_revision'] = REVISION
        return result


def construct(method, parameters, seed=2026, device='cuda:0', dataset_family=''):
    assert method == 'gradalign_plus'
    params = dict(parameters)
    assert params.pop('centrality_max_iter') == MAX_ITER
    model, config = previous.construct(params, seed, device, dataset_family)
    model.fit = MethodType(extended_fit, model)
    config.update(centrality_max_iter=MAX_ITER, implementation_revision=REVISION, gpu_revision=REVISION)
    return ExtendedContract(method, model, device), config
