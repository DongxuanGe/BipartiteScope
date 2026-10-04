import json
import logging
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from bipartite_scope.config import ServiceSettings
from bipartite_scope.core import (
    AffinityConfig,
    BuildConfig,
    CanonicalBipartiteGraph,
    EncoderConfig,
    build_snapshot,
)
from bipartite_scope.database import JobRecord, ServiceDatabase
from bipartite_scope.interface import create_app
from bipartite_scope.reliability import (
    ExecutionLostError,
    JobHeartbeat,
    WorkspaceBusyError,
    set_service_maintenance,
    workspace_write_lock,
)
from bipartite_scope.storage import SnapshotStore, init_workspace
from bipartite_scope.tasks import execute_job_record


class RecordingDispatcher:
    def __init__(self):
        self.submitted = []

    def submit(self, job_id):
        self.submitted.append(job_id)
        return f"recorded-{job_id}"

    def cancel(self, job_id):
        return None


class V4IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name).resolve()
        self.root = self.directory / "workspaces"
        self.workspace = self.root / "personal"
        init_workspace(self.workspace)
        (self.workspace / "data" / "edges.csv").write_text(
            "u_id,v_id\nu1,i1\nu1,i2\nu2,i1\nu3,i3\n", encoding="utf-8"
        )
        (self.workspace / "data" / "features.csv").write_text(
            "u_id,f1,f2\nu1,1,0\nu2,1,0\nu3,0,1\n", encoding="utf-8"
        )
        self.settings = ServiceSettings.model_validate(
            {
                "service": {"workspace_root": str(self.root), "request_max_bytes": 4096},
                "database": {"url": f"sqlite+pysqlite:///{self.directory / 'service.sqlite3'}"},
                "redis": {
                    "broker_url": "redis://127.0.0.1:1/0",
                    "result_url": "redis://127.0.0.1:1/1",
                    "progress_url": "redis://127.0.0.1:1/2",
                },
                "api": {"enable_legacy_v2_routes": False},
                "resources": {"min_free_bytes": 0},
                "retention": {"candidate_snapshots": 0, "rollback_snapshots": 1},
                "backup": {"root": str(self.directory / "backups"), "wait_seconds": 1},
            }
        )
        self.db = ServiceDatabase(self.settings)
        self.dispatcher = RecordingDispatcher()
        self.client = TestClient(
            create_app(
                settings=self.settings,
                database=self.db,
                dispatcher=self.dispatcher,
            )
        )
        self.db.register_workspace("personal", self.workspace)

    def tearDown(self):
        self.client.close()
        for component in ("api", "worker"):
            logger = logging.getLogger(f"bipartite_scope.{component}")
            for handler in tuple(logger.handlers):
                logger.removeHandler(handler)
                handler.close()
        self.db.dispose()
        self.temporary.cleanup()

    def run_job(self, job_id):
        with patch("bipartite_scope.tasks._publish"):
            return execute_job_record(job_id, self.settings, worker_name="integration-worker")

    def save_snapshots(self, count=1):
        graph = CanonicalBipartiteGraph.from_edges_and_features(
            [("u1", "i1"), ("u1", "i2"), ("u2", "i1"), ("u3", "i3")],
            {"u1": [1, 0], "u2": [1, 0], "u3": [0, 1]},
        )
        snapshot = build_snapshot(
            graph,
            BuildConfig(
                AffinityConfig(top_k=2, steps=2),
                EncoderConfig(hidden_dim=4, layers=1, latent_groups=2, epochs=1),
            ),
        )
        store = SnapshotStore(self.workspace / "artifacts")
        identifiers = []
        for index in range(count):
            identifier = f"snapshot-{index}"
            store.save(
                replace(
                    snapshot,
                    snapshot_id=identifier,
                    created_at=(
                        datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index)
                    ).isoformat(),
                )
            )
            identifiers.append(identifier)
        store.activate(identifiers[-1])
        return identifiers

    def test_operational_endpoints_expose_durable_metrics_audit_and_alert_lifecycle(self):
        submitted = self.client.post("/api/v3/workspaces/personal/validate", json={})
        self.assertEqual(submitted.status_code, 202, submitted.text)
        job_id = submitted.json()["job_id"]
        summary = self.client.get("/api/v4/operations/summary")
        self.assertEqual(summary.status_code, 200)
        self.assertEqual(summary.json()["jobs"]["by_status"]["queued"], 1)
        self.assertGreaterEqual(summary.json()["http"]["requests"], 1)
        metrics = self.client.get("/api/v4/operations/metrics")
        self.assertEqual(metrics.status_code, 200)
        self.assertIn("text/plain", metrics.headers["content-type"])
        self.assertIn('bipartite_scope_jobs{status="queued"} 1', metrics.text)
        self.assertIn('route="/api/v3/workspaces/{workspace_id}/validate"', metrics.text)
        self.assertNotIn(job_id, metrics.text)
        audit = self.client.get("/api/v4/operations/audit?limit=1&offset=0").json()
        self.assertEqual(audit["records"][0]["payload"]["job_id"], job_id)
        with self.db.sessions.begin() as session:
            record = session.get(JobRecord, job_id)
            record.queued_at = datetime.now(UTC) - timedelta(hours=2)
        alerts = self.client.get("/api/v4/operations/alerts").json()["alerts"]
        alert = next(item for item in alerts if item["key"] == "queue_wait")
        acknowledged = self.client.post(f"/api/v4/operations/alerts/{alert['id']}/acknowledge")
        self.assertEqual(acknowledged.status_code, 200)
        self.assertEqual(acknowledged.json()["payload"]["status"], "acknowledged")
        self.db.request_cancel(job_id)
        alerts = self.client.get("/api/v4/operations/alerts").json()["alerts"]
        self.assertEqual(
            next(item for item in alerts if item["id"] == alert["id"])["payload"]["status"],
            "resolved",
        )
        self.assertEqual(
            self.client.post("/api/v4/operations/alerts/absent/acknowledge").status_code, 404
        )

    def test_storage_pin_and_nonempty_retention_run_preserve_live_assets(self):
        identifiers = self.save_snapshots(4)
        pinned = self.client.post(f"/api/v4/workspaces/personal/snapshots/{identifiers[0]}/pin")
        self.assertEqual(pinned.status_code, 200, pinned.text)
        report = self.workspace / "reports" / "evaluations" / "obsolete" / "report.md"
        report.parent.mkdir(parents=True)
        report.write_text("Obsolete generated report.", encoding="utf-8")
        old = (datetime.now(UTC) - timedelta(days=60)).timestamp()
        os.utime(report, (old, old))
        os.utime(report.parent, (old, old))
        before = self.client.get("/api/v4/workspaces/personal/storage").json()
        self.assertGreater(before["total_bytes"], 0)
        plan = self.client.post("/api/v4/workspaces/personal/retention/plan").json()
        targets = {item["path"] for item in plan["entries"]}
        self.assertIn("artifacts/snapshot-1", targets)
        self.assertIn("reports/evaluations/obsolete", targets)
        self.assertIn("pinned_snapshot", plan["protected"]["artifacts/snapshot-0"])
        submitted = self.client.post("/api/v4/workspaces/personal/retention/apply", json=plan)
        self.assertEqual(submitted.status_code, 202, submitted.text)
        result = self.run_job(submitted.json()["job_id"])
        self.assertEqual(result["status"], "succeeded", result)
        self.assertEqual(result["result"]["moved_count"], 2)
        self.assertTrue(result["result"]["recoverable"])
        self.assertFalse(report.exists())
        self.assertFalse((self.workspace / "artifacts" / "snapshot-1").exists())
        for identifier in ("snapshot-0", "snapshot-2", "snapshot-3"):
            self.assertTrue(
                SnapshotStore(self.workspace / "artifacts").verify(identifier)["verified"]
            )
        self.assertTrue((self.workspace / "data" / "edges.csv").exists())
        trash = Path(result["result"]["trash_path"])
        self.assertTrue((trash / "reports" / "evaluations" / "obsolete" / "report.md").exists())

    def test_malformed_reserved_and_oversized_requests_return_problems(self):
        malformed = self.client.post(
            "/api/v3/workspaces", content="{invalid", headers={"Content-Type": "application/json"}
        )
        self.assertEqual(malformed.status_code, 422)
        self.assertEqual(malformed.json()["code"], "request_validation_failed")
        self.assertNotIn("input", json.dumps(malformed.json()["errors"]))
        oversized = self.client.post(
            "/api/v3/workspaces", content="x" * 4097, headers={"X-Request-ID": "large-request"}
        )
        self.assertEqual(oversized.status_code, 413)
        self.assertEqual(oversized.json()["code"], "request_too_large")
        self.assertEqual(oversized.headers["X-Request-ID"], "large-request")
        reserved = self.client.post(
            "/api/v3/workspaces/personal/validate", json={"_provenance": {}}
        )
        self.assertEqual(reserved.status_code, 422)
        self.assertEqual(reserved.json()["code"], "reserved_input_field")
        self.assertEqual(self.db.list_jobs(), [])

    def test_queue_admission_preserves_idempotent_replay(self):
        self.db.settings = self.settings.model_copy(
            update={
                "resources": self.settings.resources.model_copy(
                    update={"max_workspace_queued_jobs": 1}
                ),
            }
        )
        headers = {"Idempotency-Key": "one-request"}
        first = self.client.post("/api/v3/workspaces/personal/validate", json={}, headers=headers)
        self.assertEqual(first.status_code, 202)
        repeated = self.client.post(
            "/api/v3/workspaces/personal/validate", json={}, headers=headers
        )
        self.assertEqual(repeated.status_code, 202)
        self.assertEqual(first.json()["job_id"], repeated.json()["job_id"])
        blocked = self.client.post("/api/v3/workspaces/personal/validate", json={})
        self.assertEqual(blocked.status_code, 429)
        self.assertEqual(blocked.json()["code"], "workspace_queue_full")
        self.assertEqual(self.dispatcher.submitted, [first.json()["job_id"]])

    def test_global_maintenance_blocks_write_admission_and_worker_claims(self):
        queued, _ = self.db.create_job("personal", "validate", {})
        set_service_maintenance(self.db, True)
        try:
            operations = [
                self.client.post("/api/v3/workspaces/personal/validate", json={}),
                self.client.post("/api/v3/workspaces", json={"workspace_id": "new-workspace"}),
                self.client.post(
                    "/api/v3/workspaces/personal/feedback",
                    json={"user_id": "u1", "item_id": "i1", "event_type": "click"},
                ),
                self.client.post(
                    "/api/v3/workspaces/personal/uploads",
                    files={"file": ("events.csv", b"event_id,user_id\n", "text/csv")},
                ),
            ]
            self.assertTrue(
                all(response.status_code == 409 for response in operations),
                [response.text for response in operations],
            )
            claimed, attempt = self.db.claim_job(queued["job_id"], "blocked-worker")
            self.assertEqual(claimed["status"], "queued")
            self.assertIsNone(attempt)
            self.assertFalse((self.root / "new-workspace").exists())
            self.assertFalse((self.workspace / "data" / "feedback.jsonl").exists())
        finally:
            set_service_maintenance(self.db, False)
        self.assertIsNotNone(self.db.claim_job(queued["job_id"], "ready-worker")[1])

    def test_changed_queued_input_fails_before_operation_and_replay_keeps_original_job(self):
        headers = {"Idempotency-Key": "stable-input-request"}
        submitted = self.client.post(
            "/api/v3/workspaces/personal/validate", json={}, headers=headers
        )
        original = submitted.json()
        (self.workspace / "data" / "edges.csv").write_text(
            "u_id,v_id\nu1,new-item\n", encoding="utf-8"
        )
        repeated = self.client.post(
            "/api/v3/workspaces/personal/validate", json={}, headers=headers
        )
        self.assertEqual(repeated.status_code, 202)
        self.assertEqual(repeated.json()["job_id"], original["job_id"])
        self.assertEqual(repeated.json()["input_hash"], original["input_hash"])
        with patch("bipartite_scope.tasks._operation") as operation:
            failed = self.run_job(original["job_id"])
            operation.assert_not_called()
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error_code"], "invalid_job_input")
        self.assertIn("changed after submission", failed["error_message"])

    def test_changed_active_parent_fails_before_incremental_operation(self):
        identifiers = self.save_snapshots(2)
        store = SnapshotStore(self.workspace / "artifacts")
        store.activate(identifiers[0])
        submitted = self.client.post("/api/v3/workspaces/personal/update", json={})
        self.assertEqual(submitted.status_code, 202, submitted.text)
        store.activate(identifiers[1])
        with patch("bipartite_scope.tasks._operation") as operation:
            failed = self.run_job(submitted.json()["job_id"])
            operation.assert_not_called()
        self.assertEqual(failed["status"], "failed")
        self.assertIn("active snapshot changed", failed["error_message"])
        self.assertEqual(store.latest_id(), identifiers[1])

    def test_concurrent_heavy_claims_obey_global_capacity(self):
        jobs = [self.db.create_job("personal", "benchmark", {})[0] for _ in range(2)]
        with ThreadPoolExecutor(max_workers=2) as executor:
            claims = list(
                executor.map(lambda job: self.db.claim_job(job["job_id"], "parallel-worker"), jobs)
            )
        self.assertEqual(sum(attempt is not None for _, attempt in claims), 1)
        running = next(job for job, attempt in claims if attempt is not None)
        waiting = next(job for job, attempt in claims if attempt is None)
        self.assertEqual(waiting["status"], "queued")
        self.db.transition_job(
            running["job_id"], "succeeded", expected_generation=running["execution_generation"]
        )
        self.assertIsNotNone(self.db.claim_job(waiting["job_id"], "next-worker")[1])

    def test_real_heartbeat_refreshes_without_progress_and_nested_owner_survives_maintenance(self):
        job, _ = self.db.create_job("personal", "validate", {})
        running, _ = self.db.claim_job(job["job_id"], "heartbeat-worker")
        generation = running["execution_generation"]
        with self.db.sessions.begin() as session:
            record = session.get(JobRecord, job["job_id"])
            record.heartbeat_at = datetime.now(UTC) - timedelta(hours=1)
            record.updated_at = record.heartbeat_at
        observed = threading.Event()
        original = self.db.heartbeat_job

        def heartbeat(job_id, execution_generation):
            result = original(job_id, execution_generation)
            observed.set()
            return result

        with (
            patch.object(self.db, "heartbeat_job", side_effect=heartbeat),
            JobHeartbeat(self.db, job["job_id"], generation, 0.01) as monitor,
        ):
            self.assertTrue(observed.wait(2), "heartbeat thread did not report its refresh")
            self.assertEqual(self.db.recover_stale_jobs(), [])
            self.assertEqual(self.db.get_job(job["job_id"])["progress"], running["progress"])
            with self.db.sessions.begin() as session:
                record = session.get(JobRecord, job["job_id"])
                record.execution_generation += 1
            with self.assertRaises(ExecutionLostError):
                monitor.check()
        marker = self.root / ".operations" / "maintenance.json"
        with workspace_write_lock(self.workspace):
            marker.write_text(json.dumps({"source": "test"}), encoding="utf-8")
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:

                    def separate_writer():
                        with workspace_write_lock(self.workspace):
                            raise AssertionError("another writer entered during maintenance")

                    blocked = executor.submit(separate_writer)
                    with self.assertRaises(WorkspaceBusyError):
                        blocked.result()
                with workspace_write_lock(self.workspace):
                    (self.workspace / "owner-finished.txt").write_text("Owner completed.")
            finally:
                marker.unlink()
        self.assertTrue((self.workspace / "owner-finished.txt").exists())

    def test_workspace_service_backup_and_diagnostic_artifacts_run_end_to_end(self):
        validated = self.client.post("/api/v3/workspaces/personal/validate", json={}).json()
        self.assertEqual(self.run_job(validated["job_id"])["status"], "succeeded")
        completed = []
        for scope in ("workspace", "service"):
            submitted = self.client.post(
                "/api/v4/backups", json={"workspace_id": "personal", "scope": scope}
            )
            self.assertEqual(submitted.status_code, 202, submitted.text)
            job = self.run_job(submitted.json()["job_id"])
            self.assertEqual(job["status"], "succeeded", job)
            self.assertEqual(job["result"]["scope"], scope)
            completed.append(job["result"])
        listed = self.client.get("/api/v4/backups").json()["backups"]
        self.assertEqual(len(listed), 2)
        backup = completed[0]
        verification = self.client.post(
            f"/api/v4/backups/{backup['backup_id']}/verify", json={"workspace_id": "personal"}
        )
        self.assertEqual(verification.status_code, 202, verification.text)
        self.assertEqual(self.run_job(verification.json()["job_id"])["status"], "succeeded")
        diagnostic = self.client.post(
            "/api/v4/workspaces/personal/diagnostics", json={"job_id": validated["job_id"]}
        )
        self.assertEqual(diagnostic.status_code, 202, diagnostic.text)
        job = self.run_job(diagnostic.json()["job_id"])
        self.assertEqual(job["status"], "succeeded", job)
        artifacts = self.client.get(f"/api/v3/jobs/{job['job_id']}/artifacts")
        self.assertEqual(artifacts.status_code, 200, artifacts.text)
        name = artifacts.json()["artifacts"][0]["name"]
        download = self.client.get(f"/api/v3/jobs/{job['job_id']}/artifacts/{name}")
        self.assertEqual(download.status_code, 200, download.text)
        self.assertEqual(download.json()["jobs"][0]["job_id"], validated["job_id"])
        summary = self.client.get("/api/v4/operations/summary").json()
        self.assertEqual(summary["backups"]["completed"], 2)
        self.assertEqual(summary["backups"]["last_verification"]["payload"]["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
