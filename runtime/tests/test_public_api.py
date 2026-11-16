"""Small contract checks: no benchmark dataset or frozen attempt is written."""
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import networkx as nx
import numpy as np

import bpupga as bp


class PublicAPIContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="bpupga-api-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def raw_dataset(self, *, incomplete=False):
        raw = self.root / "raw"
        raw.mkdir()
        # Three edge-connected nodes and four declared isolates on each side.
        (raw / "G1.edgelist").write_text("10 11\n11 12\n", encoding="utf-8")
        (raw / "G2.edgelist").write_text("20 21\n21 22\n", encoding="utf-8")
        (raw / "G1.nodes").write_text("\n".join(str(i) for i in range(10, 17)), encoding="utf-8")
        (raw / "G2.nodes").write_text("\n".join(str(i) for i in range(20, 27)), encoding="utf-8")
        (raw / "Alignment.json").write_text(json.dumps({str(i): i + 10 for i in range(10, 17)}), encoding="utf-8")
        if incomplete:
            (raw / "metadata.json").write_text(json.dumps({"groundtruth_limitations": "Test reference is incomplete."}), encoding="utf-8")
        return raw

    def test_construction_is_deterministic_preserves_nodes_and_refuses_overwrite(self):
        raw = self.raw_dataset()
        before = {p.name: p.read_bytes() for p in raw.iterdir()}
        dataset = bp.Dataset.load(raw)
        self.assertEqual(nx.number_of_isolates(dataset.graph1), 4)
        self.assertEqual(dataset.pairs_to_original([(0, 0)]), [(10, 20)])
        kwargs = dict(rho_source=0.6, rho_target=0.6, construction_seed=2026)
        first = bp.construct(raw, output_dir=self.root / "first", **kwargs)
        second = bp.construct(raw, output_dir=self.root / "second", **kwargs)
        self.assertEqual(first.ground_truth, second.ground_truth)
        self.assertEqual(first.graph1.number_of_nodes(), 5)
        self.assertEqual(first.graph2.number_of_nodes(), 5)
        self.assertEqual(len(first.ground_truth), 3)
        self.assertEqual({p.name: p.read_bytes() for p in raw.iterdir()}, before)
        self.assertEqual((first.root / "metadata.json").read_bytes(), (second.root / "metadata.json").read_bytes())
        with self.assertRaises(FileExistsError):
            bp.construct(raw, output_dir=self.root / "first", **kwargs)

    def test_custom_adapter_gets_only_topology_and_native_duplicates_are_disclosed(self):
        dataset = bp.Dataset.load(self.raw_dataset())
        dataset.graph1.graph["ground_truth"] = dataset.ground_truth
        dataset.graph1.nodes[0]["label"] = 999
        dataset.graph1.edges[0, 1]["weight"] = 8
        seen = {}

        class Identity:
            def fit(self, first, second):
                seen["graphs"] = (first, second)
                self.assert_topology(first)
                first.remove_node(6)
                return bp.AlignmentResult("api_test_identity", [(0, 0), (0, 0), (1, 0)], 0.0)

            @staticmethod
            def assert_topology(graph):
                assert not graph.graph
                assert all(not attributes for _, attributes in graph.nodes(data=True))
                assert all(not attributes for _, _, attributes in graph.edges(data=True))

        def factory(**kwargs):
            seen["factory"] = kwargs
            return Identity()

        bp.register_algorithm("api_test_identity", factory)
        result = bp.Algorithm("api_test_identity", {"alpha": 0.1}).align(dataset.graph1, dataset.graph2)
        self.assertEqual(set(seen["factory"]), {"parameters", "seed", "device", "dataset_family"})
        self.assertEqual(len(dataset.graph1), 7)
        report = bp.evaluate(result, dataset)
        self.assertEqual(report["pair"]["predicted"], 2)
        self.assertEqual(report["pair"]["true_positive"], 1)
        self.assertEqual(report["validity"]["duplicate_pair_count"], 1)
        self.assertFalse(report["validity"]["is_one_to_one"])
        self.assertIsNone(report["ranking"])
        with self.assertRaises(ValueError):
            bp.register_algorithm("api_test_identity", factory)
        with self.assertRaises(ValueError):
            bp.Algorithm("api_test_identity", {"ground_truth": {0: 0}})
        with self.assertRaises(ValueError):
            bp.Algorithm("opgm")

    def test_adapter_mutation_does_not_change_output_index_space(self):
        dataset = bp.Dataset.load(self.raw_dataset())

        class AddsNode:
            def fit(self, first, second):
                first.add_node(7)
                return bp.AlignmentResult("adds_node", [(7, 0)], 0.0)

        class RemovesNode:
            def fit(self, first, second):
                first.remove_node(6)
                return bp.AlignmentResult("removes_node", [(6, 6)], 0.0, score_matrix=np.eye(7))

        bp.register_algorithm("api_test_adds", lambda **kwargs: AddsNode())
        bp.register_algorithm("api_test_removes", lambda **kwargs: RemovesNode())
        with self.assertRaises(ValueError):
            bp.Algorithm("api_test_adds").align(dataset.graph1, dataset.graph2)
        result = bp.Algorithm("api_test_removes").align(dataset.graph1, dataset.graph2)
        self.assertEqual(result.score_matrix.shape, (7, 7))
        self.assertEqual(len(dataset.graph1), 7)

    def test_guarded_or_nonfinite_results_cannot_be_scored(self):
        dataset = bp.Dataset.load(self.raw_dataset())
        guarded = bp.AlignmentResult("guarded", [(0, 0)], 0.0,
                                     metadata={"nonfinite_guard_triggered": True})
        nonfinite = bp.AlignmentResult("nan", [(0, 0)], 0.0, score_matrix=np.full((7, 7), np.nan))
        for result in (guarded, nonfinite):
            for action in (bp.evaluate, bp.diagnose):
                with self.subTest(method=result.method, action=action.__name__):
                    with self.assertRaises(FloatingPointError):
                        action(result, dataset)

    def test_builtin_grampa_align_evaluate_and_diagnose_roundtrip(self):
        dataset = bp.Dataset.load(self.raw_dataset())
        result = bp.Algorithm("grampa", seed=2026, device="cpu").align(dataset.graph1, dataset.graph2)
        self.assertEqual(result.method.lower(), "grampa")
        scores = bp.evaluate(result, dataset)
        self.assertEqual(scores["ranking_policy"]["revision"], "grampa_soft_v1")
        self.assertIn("raw_ranking", scores)
        self.assertTrue(bp.diagnose(result, dataset, max_dense_entries=100)["score_space"]["available"])

    def test_alpine_cpu_backend_and_parameter_validation(self):
        first, second = nx.cycle_graph(5), nx.cycle_graph(7)
        before = tuple((tuple(graph.nodes()), tuple(graph.edges())) for graph in (first, second))
        method = bp.Algorithm("alpine", device="cpu")
        result = method.align(first, second)
        self.assertEqual(result.metadata["actual_backend"], "original_author_Alpine_Alpine_pp_new_CPU")
        self.assertTrue(result.metadata["original_function_bodies_unchanged"])
        self.assertEqual(result.metadata["dtype"], "float64")
        self.assertEqual(method.config["device"], "cpu")
        self.assertEqual(result.score_matrix.shape, (5, 7))
        self.assertEqual(tuple((tuple(graph.nodes()), tuple(graph.edges())) for graph in (first, second)), before)
        self.assertEqual(bp.Algorithm("alpine", device="cuda:0").device, "cuda:0")
        with self.assertRaises(ValueError):
            bp.Algorithm("alpine", {"niter": 20}).align(first, second)

    def test_sgwl_uses_stable_proximal_state_and_restores_the_numerical_hook(self):
        from baseline import sgwl_numerics
        original = sgwl_numerics.log_sinkhorn_knopp
        method = bp.Algorithm("s_gwl", {"partition_levels": 1}, seed=2026)
        result = method.align(nx.path_graph(6), nx.path_graph(6))
        self.assertEqual(result.metadata["numerical_revision"], "retained_log_proximal_state_and_dual_gauge_v2")
        self.assertGreater(result.metadata["numerical_revision_diagnostics"]["scaling_gauge_normalizations"], 0)
        self.assertTrue(np.isfinite(result.score_matrix).all())
        self.assertIs(sgwl_numerics.log_sinkhorn_knopp, original)

    def test_sgwl_restores_the_hook_after_a_failed_fit(self):
        from baseline import sgwl_numerics
        original = sgwl_numerics.log_sinkhorn_knopp
        test = self
        class Fails:
            def fit(self, first, second):
                test.assertIsNot(sgwl_numerics.log_sinkhorn_knopp, original)
                raise RuntimeError("deliberate inference failure")
        with patch("bpupga._engine.cpu_registry.construct", return_value=(Fails(), {})):
            with self.assertRaisesRegex(RuntimeError, "deliberate inference failure"):
                bp.Algorithm("s_gwl").align(nx.path_graph(3), nx.path_graph(3))
        self.assertIs(sgwl_numerics.log_sinkhorn_knopp, original)

    def test_node_classification_separates_endpoint_selection_from_pair_identity(self):
        raw = self.raw_dataset()
        (raw / "Alignment.json").write_text(json.dumps({"10": 20, "11": 21, "12": 22}), encoding="utf-8")
        dataset = bp.Dataset.load(raw)
        # One correct pair; two predicted matched endpoints on each side are
        # reference-covered, even though their pairing identities are wrong.
        result = bp.AlignmentResult("confusion_example", [(0, 0), (2, 6), (4, 1)], 0.0)
        report = bp.evaluate(result, dataset)
        self.assertEqual(report["pair"]["true_positive"], 1)
        self.assertAlmostEqual(report["matchability"]["source"]["f1"], 2 / 3)
        for side in ("source", "target"):
            counts = report["node_classification"][side]
            self.assertEqual({name: counts[name] for name in ("tp", "fp", "fn", "tn")},
                             {"tp": 2, "fp": 1, "fn": 1, "tn": 3})
            self.assertAlmostEqual(counts["accuracy"], 5 / 7)
            self.assertAlmostEqual(counts["balanced_accuracy"], (2 / 3 + 3 / 4) / 2)
            self.assertAlmostEqual(counts["unmatched_f1"], 3 / 4)
        dataset.ground_truth.update({i: i for i in range(7)})
        full = bp.evaluate(result, dataset)
        self.assertIsNone(full["node_classification"]["source"]["balanced_accuracy"])
        self.assertFalse(full["node_classification"]["source"]["negative_class_present"])

    def test_diagnostics_honors_score_direction_without_mutation_or_writes(self):
        dataset = bp.Dataset.load(self.raw_dataset())
        costs = np.ones((7, 7)) - np.eye(7)
        result = bp.AlignmentResult("cost_test", [(i, i) for i in range(7)], 0.0,
                                    score_matrix=costs, larger_is_better=False)
        scores = bp.evaluate(result, dataset, ks=(1, 3))
        self.assertEqual(scores["ranking"]["bidirectional"]["hits@1"], 1.0)
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        diagnostics = bp.diagnose(result, dataset, ks=(1, 3), max_dense_entries=100)
        self.assertEqual(diagnostics["score_space"]["directional_correspondence"]["source_to_target"]["all_candidates_ranking"]["hits@1"], 1.0)
        self.assertFalse(result.larger_is_better)
        np.testing.assert_array_equal(result.score_matrix, costs)
        self.assertEqual({str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}, before)

    def test_incomplete_reference_skips_private_and_oracle_claims(self):
        dataset = bp.Dataset.load(self.raw_dataset(incomplete=True))
        result = bp.AlignmentResult("partial_reference", [(0, 0)], 0.0)
        scores = bp.evaluate(result, dataset)
        self.assertEqual(scores["pair"]["true_positive"], 1)
        self.assertIsNone(scores["matchability"])
        self.assertIsNone(scores["overlap_estimation"])
        self.assertIsNone(scores["node_classification"])
        self.assertIn("node_classification", scores["skipped_metrics"])
        diagnostics = bp.diagnose(result, dataset)
        self.assertIn("oracle_decoding_ladder", diagnostics["skipped_diagnostics"])
        self.assertIn("Test reference is incomplete.", diagnostics["reference_note"])

    def test_runlog_is_read_only_and_failures_remain_missing_scores(self):
        attempt = self.root / "attempt_0001"
        attempt.mkdir()
        result_path = attempt / "result.json"
        result_path.write_text(json.dumps({"status": "SUCCESS", "variants": {"native": {"pair": {"f1": 0.75}}},
                                           "runtime": {"fit_wall_seconds": 2.5}}), encoding="utf-8")
        before = hashlib.sha256(result_path.read_bytes()).hexdigest()
        log = bp.RunLog.open(attempt)
        self.assertEqual(log.native_metrics["pair"]["f1"], 0.75)
        self.assertEqual(log.runtime["fit_wall_seconds"], 2.5)
        copy_of_metrics = log.native_metrics
        copy_of_metrics["pair"]["f1"] = -1
        self.assertEqual(log.native_metrics["pair"]["f1"], 0.75)
        self.assertEqual(hashlib.sha256(result_path.read_bytes()).hexdigest(), before)
        result_path.write_text(json.dumps({"status": "TIMEOUT", "error": "budget exceeded",
                                           "variants": {"native": {"pair": {"f1": 0.0}}}}), encoding="utf-8")
        failed = bp.RunLog.open(attempt)
        self.assertIsNone(failed.native_metrics)
        self.assertEqual(failed.errors[0]["status"], "TIMEOUT")
        result_path.unlink()
        (attempt / "started.json").write_text('{"status": "RUNNING"}', encoding="utf-8")
        self.assertIsNone(bp.RunLog.open(attempt).native_metrics)


if __name__ == "__main__":
    unittest.main()
