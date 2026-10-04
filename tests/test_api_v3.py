import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from bipartite_scope.config import ServiceSettings
from bipartite_scope.database import ServiceDatabase
from bipartite_scope.interface import create_app


class RecordingDispatcher:
    def __init__(self) -> None:
        self.submitted: list[str] = []
        self.cancelled: list[str] = []

    def submit(self, job_id: str) -> str:
        self.submitted.append(job_id)
        return f"recorded-{job_id}"

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)


class V3ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "workspaces"
        self.root.mkdir()
        self.settings = ServiceSettings.model_validate(
            {
                "service": {"workspace_root": str(self.root)},
                "database": {
                    "url": f"sqlite+pysqlite:///{Path(self.temporary.name) / 'service.sqlite3'}"
                },
                "redis": {
                    "broker_url": "redis://127.0.0.1:1/0",
                    "result_url": "redis://127.0.0.1:1/1",
                    "progress_url": "redis://127.0.0.1:1/2",
                },
                "worker": {"always_eager": False},
                "api": {"enable_legacy_v2_routes": False},
            }
        )
        self.database = ServiceDatabase(self.settings)
        self.dispatcher = RecordingDispatcher()
        self.client = TestClient(
            create_app(
                settings=self.settings,
                database=self.database,
                dispatcher=self.dispatcher,
            )
        )

    def tearDown(self) -> None:
        self.client.close()
        self.database.dispose()
        self.temporary.cleanup()

    def test_version_liveness_readiness_and_openapi_security(self) -> None:
        version = self.client.get("/api/v3/version", headers={"X-Request-ID": "request-123"})
        self.assertEqual(version.status_code, 200)
        self.assertEqual(version.headers["X-Request-ID"], "request-123")
        self.assertEqual(
            version.json(),
            {
                "name": "BipartiteScope",
                "version": "4.0.0",
                "api_version": "v3",
                "authentication": False,
            },
        )
        self.assertEqual(self.client.get("/api/v3/health/live").status_code, 200)
        readiness = self.client.get("/api/v3/health/ready")
        self.assertEqual(readiness.status_code, 503)
        self.assertEqual(readiness.headers["content-type"], "application/problem+json")
        self.assertEqual(readiness.json()["code"], "service_not_ready")

        schema = self.client.get("/openapi.json").json()
        self.assertNotIn("securitySchemes", schema.get("components", {}))
        for path in schema["paths"].values():
            for operation in path.values():
                self.assertNotIn("security", operation)
        self.assertEqual(self.client.get("/health").status_code, 404)

    def test_workspace_validation_prevents_path_escape(self) -> None:
        invalid = self.client.post(
            "/api/v3/workspaces", json={"workspace_id": "../escape", "create": True}
        )
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(invalid.json()["code"], "invalid_workspace_id")
        self.assertFalse((Path(self.temporary.name) / "escape").exists())

        created = self.client.post(
            "/api/v3/workspaces", json={"workspace_id": "demo", "create": True}
        )
        self.assertEqual(created.status_code, 201, created.text)
        self.assertEqual(created.json()["workspace_id"], "demo")
        self.assertTrue((self.root / "demo" / "bipartitescope.toml").is_file())

    def test_async_submission_is_durable_and_idempotent(self) -> None:
        created = self.client.post(
            "/api/v3/workspaces", json={"workspace_id": "demo", "create": True}
        )
        self.assertEqual(created.status_code, 201)
        headers = {"Idempotency-Key": "validation-request"}
        first = self.client.post("/api/v3/workspaces/demo/validate", json={}, headers=headers)
        self.assertEqual(first.status_code, 202, first.text)
        job_id = first.json()["job_id"]
        self.assertEqual(first.headers["Location"], f"/api/v3/jobs/{job_id}")
        self.assertEqual(first.json()["status"], "queued")
        self.assertEqual(self.dispatcher.submitted, [job_id])

        repeated = self.client.post("/api/v3/workspaces/demo/validate", json={}, headers=headers)
        self.assertEqual(repeated.status_code, 202, repeated.text)
        self.assertEqual(repeated.json()["job_id"], job_id)
        self.assertEqual(self.dispatcher.submitted, [job_id])

        conflict = self.client.post(
            "/api/v3/workspaces/demo/validate",
            json={"delimiter": ";"},
            headers=headers,
        )
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.json()["code"], "idempotency_conflict")
        persisted = self.client.get(f"/api/v3/jobs/{job_id}")
        self.assertEqual(persisted.status_code, 200)
        self.assertEqual(persisted.json()["input_hash"], first.json()["input_hash"])

    def test_missing_job_artifacts_return_problem_details(self) -> None:
        for path in (
            "/api/v3/jobs/missing/artifacts",
            "/api/v3/jobs/missing/artifacts/report.json",
        ):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.headers["content-type"], "application/problem+json")
            self.assertEqual(response.json()["code"], "job_not_found")


if __name__ == "__main__":
    unittest.main()
