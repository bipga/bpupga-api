"""Meaningful contract/solver/training checks; no benchmark-GT tuning."""
import itertools
import sys
import unittest
import tempfile
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import networkx as nx
import numpy as np
import torch
from baseline.opgm_unsupervised.model import (Config, Encoder, adjacency, construct,
                                           degree_features, fixed_bias_assignment, uniform_lift)
from baseline.opgm_unsupervised.runtime.fit import validate_packet


class ContractTests(unittest.TestCase):
    def test_no_supervision_inputs(self):
        packet = dict(method="opgm_unsupervised", parameters={}, seed=2026,
                      graphs_sha256="x", source_hashes={}, gpu_uuid=None)
        validate_packet(packet)
        for key in ("ground_truth", "true_k", "true_overlap", "rho", "data_root"):
            with self.assertRaises(ValueError):
                validate_packet(dict(packet, **{key: 0}))
            with self.assertRaises(ValueError):
                construct({key: 0})

    def test_uniform_rejection_and_rectangular_solver_optimality(self):
        self.assertTrue(torch.equal(uniform_lift(torch.ones(3, 5) / 5), torch.zeros(3, 5)))
        self.assertEqual(fixed_bias_assignment(torch.zeros(3, 5), .4), [])
        rng = np.random.default_rng(8)
        for n1, n2 in ((2, 2), (3, 2), (2, 3), (1, 4)):
            score = torch.tensor(rng.uniform(0, 1, (n1, n2)), dtype=torch.float32)
            pairs = fixed_bias_assignment(score, .4)
            actual = sum(float(1 - score[u, v]) for u, v in pairs) + .4 * (n1 + n2 - 2 * len(pairs))
            optimum = float("inf")
            for k in range(min(n1, n2) + 1):
                for rows in itertools.combinations(range(n1), k):
                    for columns in itertools.permutations(range(n2), k):
                        objective = sum(float(1 - score[u, v]) for u, v in zip(rows, columns)) + .4 * (n1 + n2 - 2 * k)
                        optimum = min(optimum, objective)
            self.assertAlmostEqual(actual, optimum, places=6)
            self.assertEqual(len({u for u, _ in pairs}), len(pairs))
            self.assertEqual(len({v for _, v in pairs}), len(pairs))

    def test_real_training_and_checkpoint(self):
        torch.set_num_threads(1)
        model, _ = construct({"epochs": 4, "hidden_dim": 16}, seed=11)
        result = model.fit(nx.path_graph(9), nx.cycle_graph(13))
        self.assertEqual(result.score_matrix.shape, (9, 13))
        self.assertTrue(np.isfinite(result.score_matrix).all())
        self.assertGreater(result.metadata["parameter_update_l2"], 0)
        self.assertEqual(result.metadata["optimizer_steps"], 4)
        self.assertFalse(result.metadata["cross_graph_training_targets"])
        self.assertIn("state_dict", model.export_checkpoint())
        self.assertEqual(len(result.predicted_pairs), len(set(u for u, _ in result.predicted_pairs)))
        self.assertEqual(len(result.predicted_pairs), len(set(v for _, v in result.predicted_pairs)))

    def test_singletons_isolates_and_complete_graph(self):
        torch.set_num_threads(1)
        for first, second in ((nx.empty_graph(1), nx.empty_graph(3)),
                              (nx.complete_graph(6), nx.path_graph(7))):
            model, _ = construct({"epochs": 2, "hidden_dim": 8}, seed=12)
            result = model.fit(first, second)
            self.assertTrue(np.isfinite(result.score_matrix).all())

    def test_encoder_equivariance_and_ignored_attributes(self):
        graph = nx.barbell_graph(4, 2)
        permutation = np.random.default_rng(4).permutation(len(graph))
        remapped = nx.relabel_nodes(graph, dict(enumerate(permutation)))
        remapped.graph["ground_truth"] = {"unused": "metadata"}
        features = degree_features(graph)
        moved = degree_features(remapped)
        np.testing.assert_allclose(features, moved[permutation])
        torch.manual_seed(3)
        network = Encoder(Config(hidden_dim=8, epochs=1)).eval()
        with torch.no_grad():
            original, _ = network(torch.tensor(features), adjacency(graph, "cpu")[0])
            revised, _ = network(torch.tensor(moved), adjacency(remapped, "cpu")[0])
        torch.testing.assert_close(original, revised[permutation], atol=1e-5, rtol=1e-5)

    def test_report_waits_for_coordinator_terminal_state(self):
        from baseline.opgm_unsupervised.runtime.report import generate
        task = dict(id="one", dataset="example", method="opgm_unsupervised", condition="c", seed=2026,
                    relative_dir="example/opgm_unsupervised/c/seed_2026")
        plan = dict(datasets=["example"], tasks=[task])
        state = dict(tasks={"one": dict(status="RUNNING")})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = root / task["relative_dir"] / "attempt_0001/result.json"
            result.parent.mkdir(parents=True)
            result.write_text(json.dumps(dict(task=task, status="SUCCESS", variants={"native": {"pair": {"f1": 1}}})))
            generate(plan, state, root)
            self.assertIn("RUNNING", (root / "native_metrics.csv").read_text())
            self.assertNotIn("native.pair.f1", (root / "native_metrics.csv").read_text())

    def test_formal_report_uses_shared_dataset_layout(self):
        from baseline.opgm_unsupervised.runtime.report import generate
        task = dict(id="one", dataset="example", method="opgm_unsupervised", condition="c", seed=2026,
                    relative_dir="example/opgm_unsupervised/c/seed_2026")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata, results = root / "metadata", root / "output"
            metadata.mkdir()
            (results / "example").mkdir(parents=True)
            (results / "example/README.md").write_text("other methods")
            (results / "PROGRESS.md").write_text("unified report")
            generate(dict(datasets=["example"], tasks=[task], output_root=str(results)),
                     dict(tasks={"one": dict(status="PENDING")}), metadata)
            self.assertTrue((results / "example/opgm_unsupervised/README.md").exists())
            self.assertEqual((results / "example/README.md").read_text(), "other methods")
            self.assertEqual((results / "PROGRESS.md").read_text(), "unified report")
            self.assertFalse((metadata / "PROGRESS.md").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
