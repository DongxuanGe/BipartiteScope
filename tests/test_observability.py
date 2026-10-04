import json
import logging
import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from bipartite_scope.config import ServiceSettings
from bipartite_scope.database import JobRecord, ServiceDatabase
from bipartite_scope.observability import (
    acknowledge_alert,
    audit,
    check_alerts,
    configure_logging,
    correlated_log,
    export_diagnostics,
    operations_summary,
    prometheus_metrics,
    record_request,
    redact,
)


class ObservabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "workspaces"
        self.root.mkdir()
        self.settings = ServiceSettings.model_validate(
            {
                "service": {"workspace_root": str(self.root)},
                "database": {
                    "url": f"sqlite+pysqlite:///{Path(self.temporary.name) / 'control.sqlite3'}"
                },
                "redis": {"broker_url": "redis://example:private-password@localhost:6379/0"},
            }
        )
        self.db = ServiceDatabase(self.settings)
        self.db.initialize()
        self.workspace = self.root / "personal"
        self.workspace.mkdir()
        self.db.register_workspace("personal", self.workspace)

    def tearDown(self) -> None:
        for component in ("tests", "diagnostics"):
            logger = logging.getLogger(f"bipartite_scope.{component}")
            for handler in tuple(logger.handlers):
                logger.removeHandler(handler)
                handler.close()
        self.db.dispose()
        self.temporary.cleanup()

    def test_logs_rotate_and_redact_nested_fields_and_exception_messages(self) -> None:
        options = self.settings.observability.model_copy(update={"log_max_bytes": 600})
        settings = self.settings.model_copy(update={"observability": options})
        logger = configure_logging(settings, "tests")
        for number in range(8):
            correlated_log(
                logger,
                "INFO",
                "request.completed",
                request_id=f"request-{number}",
                job_id="test-job",
                database_url="postgresql://user:log-password@localhost/db",
                details={"password": "nested-password", "token": "nested-token"},
            )
        try:
            raise RuntimeError("connection redis://user:exception-password@localhost/0 token=value")
        except RuntimeError:
            logger.exception("worker.failed", extra={"fields": {"job_id": "test-job"}})
        directory = self.root / ".operations" / "logs"
        files = list(directory.glob("tests-*.log*"))
        self.assertGreater(len(files), 1)
        encoded = "".join(path.read_text() for path in files)
        for secret in (
            "log-password",
            "nested-password",
            "nested-token",
            "exception-password",
            "token=value",
        ):
            self.assertNotIn(secret, encoded)
        records = [json.loads(line) for path in files for line in path.read_text().splitlines()]
        self.assertTrue(all(record["job_id"] == "test-job" for record in records))
        self.assertTrue(all(record["component"] == "tests" for record in records))
        self.assertEqual(os.stat(directory / f"tests-{os.getpid()}.log").st_mode & 0o777, 0o600)
        with self.assertRaises(ValueError):
            configure_logging(settings, "../escape")

    def test_audit_is_durable_and_sanitizes_details(self) -> None:
        record = audit(
            self.db,
            "snapshot.activate",
            source="cli",
            workspace_id="personal",
            details={
                "before": "old",
                "after": "new",
                "authorization": "secret",
                "connection": "postgresql://user:audit-password@localhost/db",
            },
        )
        self.assertEqual(record["payload"]["source"], "cli")
        self.assertEqual(record["payload"]["details"]["before"], "old")
        self.assertNotIn("audit-password", json.dumps(record))
        second = ServiceDatabase(self.settings)
        try:
            self.assertEqual(second.list_operations("audit")[0]["id"], record["id"])
        finally:
            second.dispose()
        with self.assertRaises(ValueError):
            audit(self.db, "snapshot.activate", source="unknown-user")

    def test_metrics_count_all_jobs_without_identifier_labels(self) -> None:
        for _ in range(205):
            job, _ = self.db.create_job("personal", "validate", {})
            self.db.transition_job(job["job_id"], "running")
            self.db.transition_job(job["job_id"], "succeeded")
        (self.workspace / "data").mkdir()
        (self.workspace / "data" / "owned.dat").write_bytes(b"abc")
        outside = Path(self.temporary.name) / "outside.dat"
        outside.write_bytes(b"x" * 50)
        (self.workspace / "data" / "external").symlink_to(outside)
        summary = operations_summary(self.db)
        self.assertEqual(summary["jobs"]["total"], 205)
        self.assertEqual(summary["jobs"]["by_status"]["succeeded"], 205)
        self.assertEqual(summary["storage"]["workspaces"][0]["data"], 3)
        metrics = prometheus_metrics(self.db)
        self.assertIn('bipartite_scope_jobs{status="succeeded"} 205', metrics)
        self.assertNotIn(job["job_id"], metrics)
        self.assertNotIn("personal", metrics)
        self.assertNotIn("workspace_id=", metrics)

    def test_alerts_deduplicate_acknowledge_resolve_and_reopen(self) -> None:
        job, _ = self.db.create_job("personal", "validate", {})
        now = datetime.now(UTC)
        with self.db.sessions.begin() as session:
            record = session.get(JobRecord, job["job_id"])
            record.queued_at = now - timedelta(hours=2)
        self.db.record_operation("backup", {"status": "completed", "completed_at": now.isoformat()})
        with patch(
            "bipartite_scope.observability.shutil.disk_usage", return_value=shutil_disk(2**40)
        ):
            check_alerts(self.db)
            first = self.db.get_operation("alert", "queue_wait")
            check_alerts(self.db)
            second = self.db.get_operation("alert", "queue_wait")
            self.assertEqual(first["id"], second["id"])
            self.assertEqual(second["payload"]["occurrences"], 1)
            acknowledged = acknowledge_alert(self.db, first["id"])
            self.assertEqual(acknowledged["payload"]["status"], "acknowledged")
            check_alerts(self.db)
            self.assertEqual(
                self.db.get_operation("alert", "queue_wait")["payload"]["status"], "acknowledged"
            )
            self.db.request_cancel(job["job_id"])
            check_alerts(self.db)
            self.assertEqual(
                self.db.get_operation("alert", "queue_wait")["payload"]["status"], "resolved"
            )
            with self.assertRaises(ValueError):
                acknowledge_alert(self.db, first["id"])
            retried = self.db.retry_job(job["job_id"])
            with self.db.sessions.begin() as session:
                record = session.get(JobRecord, retried["job_id"])
                record.queued_at = now - timedelta(hours=2)
            check_alerts(self.db)
            reopened = self.db.get_operation("alert", "queue_wait")
            self.assertEqual(reopened["id"], first["id"])
            self.assertEqual(reopened["payload"]["status"], "active")
            self.assertEqual(reopened["payload"]["occurrences"], 2)

    def test_independent_heartbeat_drives_stale_worker_alert(self) -> None:
        job, _ = self.db.create_job("personal", "validate", {})
        self.db.transition_job(job["job_id"], "running")
        now = datetime.now(UTC)
        with self.db.sessions.begin() as session:
            record = session.get(JobRecord, job["job_id"])
            record.updated_at = now - timedelta(hours=1)
            record.heartbeat_at = now
        check_alerts(self.db)
        self.assertIsNone(self.db.get_operation("alert", "worker_stale"))
        with self.db.sessions.begin() as session:
            record = session.get(JobRecord, job["job_id"])
            record.heartbeat_at = now - timedelta(hours=1)
        check_alerts(self.db)
        self.assertEqual(
            self.db.get_operation("alert", "worker_stale")["payload"]["status"], "active"
        )

    def test_failed_backup_verification_and_restore_drill_alert_then_resolve(self) -> None:
        self.db.record_operation("backup_verify", {"status": "failed", "backup_id": "backup-1"})
        self.db.record_operation("restore_drill", {"status": "failed", "backup_id": "backup-1"})
        check_alerts(self.db)
        self.assertEqual(
            self.db.get_operation("alert", "backup_verification_failed")["payload"]["status"],
            "active",
        )
        self.assertEqual(
            self.db.get_operation("alert", "restore_drill_failed")["payload"]["status"], "active"
        )
        self.db.record_operation("backup_verify", {"status": "succeeded", "backup_id": "backup-2"})
        self.db.record_operation("restore_drill", {"status": "succeeded", "backup_id": "backup-2"})
        check_alerts(self.db)
        self.assertEqual(
            self.db.get_operation("alert", "backup_verification_failed")["payload"]["status"],
            "resolved",
        )
        self.assertEqual(
            self.db.get_operation("alert", "restore_drill_failed")["payload"]["status"], "resolved"
        )

    def test_diagnostics_include_history_but_never_credentials(self) -> None:
        job, _ = self.db.create_job("personal", "validate", {})
        self.db.transition_job(job["job_id"], "running")
        self.db.begin_attempt(job["job_id"], "test-worker")
        self.db.update_progress(job["job_id"], 50, "validation", "Checking inputs")
        logger = configure_logging(self.settings, "diagnostics")
        correlated_log(
            logger,
            "ERROR",
            "task.failed",
            job_id=job["job_id"],
            message="postgresql://user:diagnostic-password@localhost/db",
        )
        path = export_diagnostics(self.db, job["job_id"])
        payload = json.loads(path.read_text())
        self.assertEqual(payload["jobs"][0]["job_id"], job["job_id"])
        self.assertEqual(payload["attempts"][0]["worker_name"], "test-worker")
        self.assertGreater(len(payload["events"][job["job_id"]]), 1)
        self.assertEqual(payload["logs"][0]["job_id"], job["job_id"])
        self.assertNotIn("private-password", path.read_text())
        self.assertNotIn("diagnostic-password", path.read_text())
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        self.assertEqual(path.parent, self.root.resolve() / ".operations" / "diagnostics")
        self.assertTrue(self.db.list_operations("audit"))

    def test_redaction_handles_embedded_urls_and_query_secrets(self) -> None:
        value = redact(
            {
                "api_key": "abc",
                "message": "https://user:secret@site/path?token=private&x=1",
                "nested": [{"password": "secret"}],
            }
        )
        self.assertEqual(value["api_key"], "***")
        self.assertNotIn("secret", value["message"])
        self.assertNotIn("private", value["message"])
        self.assertEqual(value["nested"][0]["password"], "***")
        message = redact('password="private phrase" authorization: Bearer abcdef "token": "hidden"')
        self.assertNotIn("private phrase", message)
        self.assertNotIn("abcdef", message)
        self.assertNotIn("hidden", message)

    def test_request_metrics_are_persistent_bounded_and_thread_safe(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(
                executor.map(
                    lambda _: record_request(self.db, "GET", "/api/v4/jobs/{job_id}", 200, 0.1),
                    range(20),
                )
            )
        summary = operations_summary(self.db)
        self.assertEqual(summary["http"]["requests"], 20)
        self.assertAlmostEqual(summary["http"]["duration_sum_seconds"], 2.0)
        metric = self.db.list_operations("http_metric")[0]["payload"]
        self.assertEqual(metric["buckets"]["0.1"], 20)
        self.assertEqual(metric["buckets"]["0.05"], 0)
        for number in range(110):
            record_request(self.db, "GET", f"/route-{number}", 404, 0.01)
        self.assertLessEqual(len(self.db.list_operations("http_metric", limit=200)), 101)
        metrics = prometheus_metrics(self.db)
        self.assertIn('route="/api/v4/jobs/{job_id}"', metrics)
        self.assertIn('le="+Inf"', metrics)
        self.assertEqual(operations_summary(self.db)["http"]["requests"], 130)
        with self.assertRaises(ValueError):
            record_request(self.db, "GET", "/invalid", 200, float("nan"))


def shutil_disk(free: int):
    from collections import namedtuple

    return namedtuple("DiskUsage", "total used free")(2**41, 2**41 - free, free)


if __name__ == "__main__":
    unittest.main()
