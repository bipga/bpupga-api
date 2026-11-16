"""Direct CPU execution of the archived, unmodified ALPINE author functions.

Only the two entry functions are loaded from pred.py; its unrelated imports and
module-level 40-thread setting are unnecessary for inference. Their function
bodies and the three helper source files are executed without rewriting.
"""
import ast
import importlib.util
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from numpy import linalg as LA
from sklearn.metrics.pairwise import euclidean_distances
from common.result import AlignmentResult

AUTHOR = Path(__file__).resolve().parent / 'vendor/alpine_author'


def load_author():
    helpers = {}
    for name in ('sinkhorn', 'hung_utils', 'feature_util'):
        spec = importlib.util.spec_from_file_location('_alpine_official_' + name, AUTHOR / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        helpers[name] = module
    namespace = dict(np=np, torch=torch, nx=nx, LA=LA,
                     euclidean_distances=euclidean_distances,
                     sinkhorn=helpers['sinkhorn'].sinkhorn,
                     convertToPermHungarian=helpers['hung_utils'].convertToPermHungarian,
                     convertToPermHungarian2new=helpers['hung_utils'].convertToPermHungarian2new,
                     feature_extraction=helpers['feature_util'].feature_extraction,
                     feature_extraction1=helpers['feature_util'].feature_extraction1)
    tree = ast.parse((AUTHOR / 'pred.py').read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {'Alpine', 'Alpine_pp_new'}]
    if len(tree.body) != 2:
        raise ValueError('Unexpected ALPINE author source')
    exec(compile(tree, str(AUTHOR / 'pred.py'), 'exec'), namespace)
    return namespace


class AuthorCPUAlpine:
    def __init__(self, parameters):
        self.parameters = dict(parameters)
        expected = dict(dtype='float64', inner_iterations=10, sinkhorn_iterations=500,
                        sinkhorn_reg=1., sinkhorn_tolerance=1e-5)
        for key, value in expected.items():
            if self.parameters.get(key, value) != value:
                raise ValueError(f'Author CPU ALPINE does not expose a different {key}')

    def fit(self, g1, g2):
        start = time.perf_counter()
        # The author's partial formulation requires query order <= target order.
        swapped = len(g1) > len(g2)
        query, target = (g2, g1) if swapped else (g1, g2)
        author = load_author()
        original = author['Alpine_pp_new']
        captured = {}
        def observe(*args, **kwargs):
            output = original(*args, **kwargs)
            captured['plan'] = output[0]
            return output
        author['Alpine_pp_new'] = observe
        pairs, _, objective = author['Alpine'](
            query.copy(), target.copy(), mu=self.parameters.get('mu', 1.),
            niter=self.parameters.get('niter', 10), weight=self.parameters.get('weight', 2))
        plan = captured['plan'].detach().cpu().numpy()
        if not np.isfinite(plan).all() or not np.isfinite(float(objective)):
            raise FloatingPointError('Non-finite original-author CPU ALPINE transport')
        pairs = [(int(v), int(u)) if swapped else (int(u), int(v)) for u, v in pairs]
        return AlignmentResult(
            'ALPINE', pairs, time.perf_counter() - start,
            score_matrix=plan.T if swapped else plan, larger_is_better=True,
            metadata=dict(actual_backend='original_author_Alpine_Alpine_pp_new_CPU',
                          official_commit='ce676811c34f5fbc972ce449e6163951f1a8b2ba',
                          original_function_bodies_unchanged=True,
                          dtype='float64', device='cpu', source_target_swapped=swapped,
                          author_outer_iterations=10, author_inner_iterations=10,
                          author_sinkhorn_iterations=500, author_sinkhorn_reg=1.,
                          author_sinkhorn_tolerance=1e-5,
                          author_reported_objective=float(objective),
                          thread_limit=torch.get_num_threads(),
                          prediction_rule='author_Hungarian_on_same_fit_transport',
                          ground_truth_used_during_inference=False))
