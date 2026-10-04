import unittest
from dataclasses import replace

import numpy as np

import bipartite_scope
from bipartite_scope import (
    AcademicExplorerAdapter,
    AffinityBuilder,
    AffinityConfig,
    BuildConfig,
    CanonicalBipartiteGraph,
    EncoderConfig,
    QueryConfig,
    QueryEngine,
    RecommendationAdapter,
    build_snapshot,
)
from bipartite_scope.core import rowwise_top_k


def graph() -> CanonicalBipartiteGraph:
    return CanonicalBipartiteGraph.from_edges_and_features(
        [("u1", "i1"), ("u1", "i2"), ("u2", "i1"), ("u2", "i2"), ("u3", "i3")],
        {"u1": [1, 0], "u2": [1, 0], "u3": [0, 1]},
        feature_names=("f1", "f2"),
    )


def config(epochs: int = 2) -> BuildConfig:
    return BuildConfig(
        AffinityConfig(top_k=2, steps=2),
        EncoderConfig(hidden_dim=4, layers=1, latent_groups=2, epochs=epochs),
        semantic_recall_budget=2,
    )


class CoreTests(unittest.TestCase):
    def test_version_and_supported_imports(self) -> None:
        self.assertEqual(bipartite_scope.__version__, "4.0.0")
        self.assertTrue(callable(bipartite_scope.create_app))

    def test_graph_is_binary_sparse_and_deterministic(self) -> None:
        value = CanonicalBipartiteGraph.from_edges_and_features(
            [("u2", "i2"), ("u1", "i1"), ("u1", "i1")],
            {"u1": [1, 0], "u2": [0, 2]},
        )
        self.assertEqual(value.u_ids, ("u1", "u2"))
        self.assertEqual(value.v_ids, ("i1", "i2"))
        self.assertEqual(value.incidence.nnz, 2)
        np.testing.assert_array_equal(value.incidence.data, 1)
        with self.assertRaises(ValueError):
            CanonicalBipartiteGraph.from_edges_and_features([("u", "i")], {"u": [-1]})

    def test_affinity_preserves_v1_invariants(self) -> None:
        result = AffinityBuilder(AffinityConfig(top_k=2, steps=2)).build(graph())
        np.testing.assert_allclose(np.asarray(result.structural.sum(axis=1)).ravel(), 1)
        np.testing.assert_allclose(np.asarray(result.attribute.sum(axis=1)).ravel(), 1)
        self.assertEqual((result.affinity - result.affinity.T).nnz, 0)
        np.testing.assert_allclose(result.affinity.diagonal(), 0)
        matrix = np.array([[1, 0.5, 0.5], [0, 1, 0], [0, 0, 1]], dtype=float)
        selected = rowwise_top_k(matrix, top_k=1)
        self.assertEqual(selected[0, 1], 0.5)
        self.assertEqual(selected[0, 2], 0)

    def test_build_query_and_warm_start(self) -> None:
        first = build_snapshot(graph(), config())
        result = QueryEngine(first).search("u1", QueryConfig(size_budget=2, delta_0=1.0))
        self.assertEqual(result.query_entity, "u1")
        self.assertIn("u1", result.members)
        self.assertTrue(result.support_entities)
        second_graph = CanonicalBipartiteGraph.from_edges_and_features(
            [
                ("u1", "i1"),
                ("u1", "i2"),
                ("u2", "i1"),
                ("u2", "i2"),
                ("u3", "i3"),
                ("u3", "i4"),
            ],
            {"u1": [1, 0], "u2": [1, 0], "u3": [0, 1]},
            u_order=first.graph.u_ids,
            v_order=(*first.graph.v_ids, "i4"),
        )
        second = build_snapshot(
            second_graph,
            replace(config(), encoder=replace(config().encoder, epochs=1)),
            initial_state=first.model_state,
            parent_snapshot_id=first.snapshot_id,
            build_mode="incremental",
        )
        self.assertIn("v_embedding", second.update_stats["warm_started_parameters"])
        self.assertEqual(second.item_embeddings.shape[0], 4)

    def test_adapters_keep_generic_boundaries(self) -> None:
        recommendation = RecommendationAdapter().to_graph(
            [{"user_id": "u", "item_id": "i"}],
            {"u": [1]},
        )
        academic = AcademicExplorerAdapter().to_graph(
            [{"researcher_id": "r", "paper_id": "p"}],
            {"r": [1, 0]},
        )
        self.assertEqual(recommendation.u_ids, ("u",))
        self.assertEqual(academic.v_ids, ("p",))


if __name__ == "__main__":
    unittest.main()
