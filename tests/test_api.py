import importlib.util
import tempfile
import unittest

from bipartite_scope import create_app


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "FastAPI is not installed")
class ApiTests(unittest.TestCase):
    def test_v2_endpoints_and_workspace_validation(self) -> None:
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as directory:  # noqa: SIM117
            with TestClient(create_app(directory)) as client:
                health = client.get("/health")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json()["version"], "4.0.0")
                payload = {
                    "edges": [
                        ["u1", "i1"],
                        ["u1", "i2"],
                        ["u1", "i3"],
                        ["u2", "i1"],
                        ["u2", "i3"],
                        ["u2", "i4"],
                        ["u3", "i2"],
                        ["u3", "i4"],
                    ],
                    "features": {"u1": [1, 0], "u2": [1, 0], "u3": [0, 1]},
                    "config": {
                        "affinity": {"top_k": 2},
                        "encoder": {"hidden_dim": 4, "layers": 1, "latent_groups": 2, "epochs": 1},
                        "semantic_recall_budget": 2,
                    },
                }
                built = client.post("/workspaces/demo/build", json=payload)
                self.assertEqual(built.status_code, 200, built.text)
                initial_id = built.json()["snapshot_id"]
                query = client.get(
                    f"/workspaces/demo/snapshots/{initial_id}/query/u1?size_budget=2"
                )
                self.assertEqual(query.status_code, 200, query.text)
                self.assertIn("u1", query.json()["members"])
                events = [
                    {
                        "event_id": f"e{index}",
                        "user_id": user,
                        "item_id": item,
                        "event_type": "click",
                        "event_time": timestamp,
                    }
                    for index, (user, item, timestamp) in enumerate(
                        (
                            ("u1", "i1", "2026-09-01T00:00:00Z"),
                            ("u1", "i2", "2026-09-02T00:00:00Z"),
                            ("u1", "i3", "2026-09-03T00:00:00Z"),
                            ("u2", "i1", "2026-09-01T00:00:00Z"),
                            ("u2", "i3", "2026-09-02T00:00:00Z"),
                            ("u2", "i4", "2026-09-03T00:00:00Z"),
                        ),
                        1,
                    )
                ]
                updated = client.post("/workspaces/demo/update", json={"events": events})
                self.assertEqual(updated.status_code, 200, updated.text)
                candidate_id = updated.json()["snapshot_id"]
                recommendation = client.post(
                    "/workspaces/demo/recommend",
                    json={"user_id": "u1", "top_n": 2},
                )
                self.assertEqual(recommendation.status_code, 200, recommendation.text)
                feedback = client.post(
                    "/workspaces/demo/feedback",
                    json={"user_id": "u1", "item_id": "i4", "event_type": "dislike"},
                )
                self.assertEqual(feedback.status_code, 200, feedback.text)
                self.assertFalse(feedback.json()["retraining_triggered"])
                evaluation = client.post("/workspaces/demo/evaluate", json={"k": 2})
                self.assertEqual(evaluation.status_code, 200, evaluation.text)
                snapshots = client.get("/workspaces/demo/snapshots")
                self.assertEqual(snapshots.status_code, 200, snapshots.text)
                self.assertEqual(len(snapshots.json()["snapshots"]), 2)
                verified = client.post(f"/workspaces/demo/snapshots/{candidate_id}/verify")
                self.assertEqual(verified.status_code, 200, verified.text)
                activated = client.post(f"/workspaces/demo/snapshots/{candidate_id}/activate")
                self.assertEqual(activated.status_code, 200, activated.text)
                invalid = client.post("/workspaces/bad!id/build", json=payload)
                self.assertEqual(invalid.status_code, 422)


if __name__ == "__main__":
    unittest.main()
