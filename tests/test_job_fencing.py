import json
import logging
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from bipartite_scope.config import ServiceSettings
from bipartite_scope.database import (
    JobAttemptRecord,
    JobRecord,
    JobStateError,
    ServiceDatabase,
    utcnow,
)
from bipartite_scope.reliability import (
    ExecutionLostError,
    execution_commit,
    execution_guard,
    maintenance_mode,
    recover_maintenance,
)
from bipartite_scope.storage import init_workspace
from bipartite_scope.tasks import execute_job_record


class JobFencingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        directory = Path(self.temporary.name).resolve()
        self.root = directory / "workspaces"
        self.workspace = self.root / "personal"
        init_workspace(self.workspace)
        self.settings = ServiceSettings.model_validate(
            {
                "service": {"workspace_root": str(self.root)},
                "database": {"url": f"sqlite+pysqlite:///{directory / 'service.sqlite3'}"},
                "resources": {"min_free_bytes": 0},
                "worker": {"stale_job_seconds": 30, "heartbeat_seconds": 10},
            }
        )
        self.database = ServiceDatabase(self.settings)
        self.database.initialize()
        self.database.register_workspace("personal", self.workspace)

    def tearDown(self):
        logger = logging.getLogger("bipartite_scope.worker")
        for handler in tuple(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        self.database.dispose()
        self.temporary.cleanup()

    def recover_and_reclaim(self, job_id):
        with self.database.sessions.begin() as session:
            record = session.get(JobRecord, job_id)
            record.heartbeat_at = utcnow() - timedelta(minutes=1)
        self.assertEqual(self.database.recover_stale_jobs(), [job_id])
        self.database.retry_job(job_id)
        reclaimed, attempt = self.database.claim_job(job_id, "replacement-worker")
        return reclaimed, attempt

    def previous_execution(self):
        job, _ = self.database.create_job("personal", "validate", {})
        original, attempt = self.database.claim_job(job["job_id"], "original-worker")
        reclaimed, replacement_attempt = self.recover_and_reclaim(job["job_id"])
        self.assertEqual(original["execution_generation"], 1)
        self.assertEqual(reclaimed["execution_generation"], 3)
        return original, attempt, reclaimed, replacement_attempt

    def test_stale_progress_cannot_change_replacement_execution(self):
        original, _, reclaimed, _ = self.previous_execution()
        with self.assertRaises(ExecutionLostError):
            self.database.update_progress(
                original["job_id"],
                70,
                "old-stage",
                "Stale worker progress",
                expected_generation=original["execution_generation"],
            )
        current = self.database.get_job(original["job_id"])
        self.assertEqual(current["status"], "running")
        self.assertEqual(current["stage"], reclaimed["stage"])
        self.assertEqual(current["progress"], reclaimed["progress"])

    def test_asset_commit_rechecks_ownership_after_acquiring_fence(self):
        from contextlib import contextmanager

        original, _, _, _ = self.previous_execution()
        order = []

        @contextmanager
        def fence():
            order.append("locked")
            yield

        def check():
            order.append("checked")
            if not self.database.execution_owned(
                original["job_id"], original["execution_generation"]
            ):
                raise ExecutionLostError("stale execution")

        with (
            execution_guard(check, fence),
            self.assertRaises(ExecutionLostError),
            execution_commit(),
        ):
            self.fail("a stale execution must not publish its asset")
        self.assertEqual(order, ["locked", "checked"])

    def test_recovery_clears_only_orphaned_operation_markers(self):
        operations = self.root / ".operations"
        global_marker = operations / "maintenance.json"
        global_marker.write_text(json.dumps({"reason": "backup", "id": "orphan"}))
        marker = self.workspace / ".maintenance.json"
        marker.write_text(json.dumps({"token": "a" * 32}))
        self.assertEqual(len(recover_maintenance(self.database)), 2)
        self.assertFalse(global_marker.exists())
        self.assertFalse(marker.exists())
        global_marker.write_text(json.dumps({"source": "manual", "token": "manual"}))
        self.assertEqual(recover_maintenance(self.database), [])
        self.assertTrue(global_marker.exists())
        global_marker.unlink()

    def test_recovery_preserves_live_workspace_maintenance(self):
        from concurrent.futures import ThreadPoolExecutor

        with maintenance_mode(self.workspace), ThreadPoolExecutor(max_workers=1) as executor:
            self.assertEqual(
                executor.submit(recover_maintenance, self.database).result(timeout=5), []
            )
            self.assertTrue((self.workspace / ".maintenance.json").exists())

    def test_stale_attempt_finish_preserves_recovery_and_replacement_attempts(self):
        original, attempt_id, reclaimed, replacement_attempt = self.previous_execution()
        with self.database.sessions() as session:
            previous = session.get(JobAttemptRecord, attempt_id)
            before = (previous.status, previous.error_code, previous.completed_at)
        with self.assertRaises(ExecutionLostError):
            self.database.finish_attempt(
                attempt_id,
                "failed",
                error_code="old_failure",
                expected_generation=original["execution_generation"],
            )
        with self.database.sessions() as session:
            previous = session.get(JobAttemptRecord, attempt_id)
            replacement = session.get(JobAttemptRecord, replacement_attempt)
            self.assertEqual((previous.status, previous.error_code, previous.completed_at), before)
            self.assertEqual(previous.error_code, "worker_lost")
            self.assertEqual(replacement.status, "running")
            self.assertIsNone(replacement.completed_at)
        self.assertEqual(
            self.database.get_job(original["job_id"])["execution_generation"],
            reclaimed["execution_generation"],
        )

    def test_stale_terminal_transition_is_rejected_atomically(self):
        original, _, reclaimed, _ = self.previous_execution()
        with self.assertRaises(JobStateError):
            self.database.transition_job(
                original["job_id"],
                "failed",
                error_code="old_failure",
                expected_generation=original["execution_generation"],
            )
        current = self.database.get_job(original["job_id"])
        self.assertEqual(current["status"], "running")
        self.assertEqual(current["execution_generation"], reclaimed["execution_generation"])

    def test_exception_check_write_interleaving_does_not_fail_replacement_execution(self):
        job, _ = self.database.create_job("personal", "validate", {})
        state = {"operation_failed": False, "interleaved": False}
        original_owned = ServiceDatabase.execution_owned

        def failing_operation(*args, **kwargs):
            state["operation_failed"] = True
            raise ValueError("Original operation failed")

        def interleaved_ownership(database, job_id, generation):
            formerly_owned = original_owned(database, job_id, generation)
            if state["operation_failed"] and not state["interleaved"]:
                state["interleaved"] = True
                reclaimed, replacement = self.recover_and_reclaim(job_id)
                state["generation"] = reclaimed["execution_generation"]
                state["attempt"] = replacement
            return formerly_owned

        with (
            patch("bipartite_scope.tasks._operation", side_effect=failing_operation),
            patch("bipartite_scope.tasks._publish"),
            patch.object(ServiceDatabase, "execution_owned", interleaved_ownership),
        ):
            result = execute_job_record(job["job_id"], self.settings, worker_name="old-worker")
        self.assertTrue(state["interleaved"])
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["execution_generation"], state["generation"])
        self.assertIsNone(result["error_code"])
        with self.database.sessions() as session:
            replacement = session.get(JobAttemptRecord, state["attempt"])
            self.assertEqual(replacement.status, "running")
            self.assertIsNone(replacement.completed_at)


if __name__ == "__main__":
    unittest.main()
