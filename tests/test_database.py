import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from sqlalchemy import create_engine, inspect, select, text

from bipartite_scope.config import ServiceSettings
from bipartite_scope.database import (
    IdempotencyConflictError,
    JobRecord,
    JobStateError,
    ServiceDatabase,
    WorkspaceLeaseRecord,
    upgrade_database,
    utcnow,
)


def service_settings(root: Path, database: Path, **worker: object) -> ServiceSettings:
    return ServiceSettings.model_validate(
        {
            "service": {"workspace_root": str(root)},
            "database": {"url": f"sqlite+pysqlite:///{database}"},
            "redis": {
                "broker_url": "redis://127.0.0.1:1/0",
                "result_url": "redis://127.0.0.1:1/1",
                "progress_url": "redis://127.0.0.1:1/2",
            },
            "worker": worker,
        }
    )


class ServiceDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "workspaces"
        self.root.mkdir()
        self.settings = service_settings(
            self.root,
            Path(self.temporary.name) / "service.sqlite3",
            stale_job_seconds=30,
        )
        self.database = ServiceDatabase(self.settings)
        self.database.initialize()
        self.workspace = self.root / "demo"
        self.workspace.mkdir()
        self.database.register_workspace("demo", self.workspace)

    def tearDown(self) -> None:
        self.database.dispose()
        self.temporary.cleanup()

    def test_durable_job_lifecycle_events_and_idempotency(self) -> None:
        job, created = self.database.create_job(
            "demo", "validate", {"delimiter": ","}, idempotency_key="request-1"
        )
        repeated, repeated_created = self.database.create_job(
            "demo", "validate", {"delimiter": ","}, idempotency_key="request-1"
        )
        self.assertTrue(created)
        self.assertFalse(repeated_created)
        self.assertEqual(repeated["job_id"], job["job_id"])
        with self.assertRaises(IdempotencyConflictError):
            self.database.create_job(
                "demo", "validate", {"delimiter": ";"}, idempotency_key="request-1"
            )

        running = self.database.transition_job(
            job["job_id"], "running", stage="starting", progress=1
        )
        self.assertEqual(running["status"], "running")
        attempt = self.database.begin_attempt(job["job_id"], "test-worker")
        progress = self.database.update_progress(
            job["job_id"], 50, "validation", "Inputs validated"
        )
        self.assertEqual(progress["progress"], 50)
        self.database.finish_attempt(attempt, "succeeded")
        completed = self.database.transition_job(
            job["job_id"],
            "succeeded",
            stage="completed",
            progress=100,
            result={"valid": True},
        )
        self.assertEqual(completed["result"], {"valid": True})
        self.assertEqual(completed["attempt_count"], 1)
        events = self.database.events_after(job["job_id"])
        self.assertEqual([event["id"] for event in events], list(range(1, len(events) + 1)))
        self.assertEqual(events[-1]["event"], "succeeded")

    def test_mutation_exclusion_lease_cancel_and_retry(self) -> None:
        build, _ = self.database.create_job("demo", "build", {})
        with self.assertRaises(JobStateError):
            self.database.create_job("demo", "update", {})
        self.assertTrue(self.database.acquire_workspace_lease(build["job_id"]))
        self.assertFalse(self.database.acquire_workspace_lease(build["job_id"]))
        self.database.release_workspace_lease(build["job_id"])

        queued, _ = self.database.create_job("demo", "validate", {})
        cancelled = self.database.request_cancel(queued["job_id"])
        self.assertEqual(cancelled["status"], "cancelled")
        retried = self.database.retry_job(queued["job_id"])
        self.assertEqual(retried["status"], "queued")
        self.assertEqual(retried["progress"], 0)

    def test_stale_running_job_is_failed_and_lease_is_released(self) -> None:
        job, _ = self.database.create_job("demo", "build", {})
        self.database.transition_job(job["job_id"], "running", progress=1)
        self.assertTrue(self.database.acquire_workspace_lease(job["job_id"]))
        with self.database.sessions.begin() as session:
            record = session.get(JobRecord, job["job_id"])
            self.assertIsNotNone(record)
            record.updated_at = utcnow() - timedelta(seconds=60)

        self.assertEqual(self.database.recover_stale_jobs(), [job["job_id"]])
        recovered = self.database.get_job(job["job_id"])
        self.assertEqual(recovered["status"], "failed")
        self.assertEqual(recovered["error_code"], "worker_lost")
        with self.database.sessions() as session:
            lease = session.scalar(
                select(WorkspaceLeaseRecord).where(WorkspaceLeaseRecord.job_id == job["job_id"])
            )
        self.assertIsNone(lease)

    def test_job_claim_is_atomic_and_duplicate_delivery_is_ignored(self) -> None:
        job, _ = self.database.create_job("demo", "validate", {})
        claimed, attempt_id = self.database.claim_job(job["job_id"], "worker-one")
        duplicate, duplicate_attempt = self.database.claim_job(job["job_id"], "worker-two")
        self.assertEqual(claimed["status"], "running")
        self.assertIsNotNone(attempt_id)
        self.assertEqual(duplicate["status"], "running")
        self.assertIsNone(duplicate_attempt)
        self.assertEqual(self.database.get_job(job["job_id"])["attempt_count"], 1)

    def test_independent_heartbeat_and_execution_generation_fence_stale_results(self) -> None:
        job, _ = self.database.create_job("demo", "validate", {})
        claimed, _ = self.database.claim_job(job["job_id"], "worker-one")
        original_generation = claimed["execution_generation"]
        with self.database.sessions.begin() as session:
            record = session.get(JobRecord, job["job_id"])
            record.updated_at = utcnow() - timedelta(seconds=60)
            record.heartbeat_at = utcnow()
        self.assertEqual(self.database.recover_stale_jobs(), [])
        self.assertTrue(self.database.execution_owned(job["job_id"], original_generation))
        with self.database.sessions.begin() as session:
            record = session.get(JobRecord, job["job_id"])
            record.heartbeat_at = utcnow() - timedelta(seconds=60)
        self.assertEqual(self.database.recover_stale_jobs(), [job["job_id"]])
        self.assertFalse(self.database.heartbeat_job(job["job_id"], original_generation))
        self.database.retry_job(job["job_id"])
        reclaimed, _ = self.database.claim_job(job["job_id"], "worker-two")
        self.assertGreater(reclaimed["execution_generation"], original_generation)
        with self.assertRaises(JobStateError):
            self.database.transition_job(
                job["job_id"],
                "succeeded",
                result={"stale": True},
                expected_generation=original_generation,
            )
        self.assertEqual(self.database.get_job(job["job_id"])["status"], "running")
        completed = self.database.transition_job(
            job["job_id"],
            "succeeded",
            result={"stale": False},
            expected_generation=reclaimed["execution_generation"],
        )
        self.assertEqual(completed["result"], {"stale": False})


class AlembicMigrationTests(unittest.TestCase):
    def test_upgrade_creates_the_control_plane_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspaces"
            root.mkdir()
            database_path = Path(directory) / "migrated.sqlite3"
            settings = service_settings(root, database_path)
            self.assertEqual(upgrade_database(settings), "0002_v4_operations")
            database = ServiceDatabase(settings)
            try:
                tables = set(inspect(database.engine).get_table_names())
                self.assertTrue(
                    {
                        "alembic_version",
                        "workspaces",
                        "jobs",
                        "job_attempts",
                        "job_events",
                        "workspace_leases",
                        "operations",
                    }.issubset(tables)
                )
                workspace = root / "demo"
                workspace.mkdir()
                database.register_workspace("demo", workspace)
                job, created = database.create_job("demo", "validate", {})
                self.assertTrue(created)
                self.assertEqual(job["status"], "queued")
            finally:
                database.dispose()

    def test_v3_database_upgrade_preserves_existing_workspace_and_job(self) -> None:
        from alembic import command
        from alembic.config import Config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "workspaces"
            workspace = root / "demo"
            workspace.mkdir(parents=True)
            settings = service_settings(root, Path(directory).resolve() / "service.sqlite3")
            repository = Path(__file__).resolve().parents[1]
            migration = Config(str(repository / "alembic.ini"))
            migration.set_main_option("script_location", str(repository / "migrations"))
            migration.set_main_option("sqlalchemy.url", settings.database.url)
            migration.attributes["explicit_url"] = True
            command.upgrade(migration, "0001_v3_control_plane")
            engine = create_engine(settings.database.url)
            try:
                with engine.begin() as connection:
                    connection.execute(
                        text(
                            "INSERT INTO workspaces(id,workspace_key,storage_path) VALUES(:id,:key,:path)"
                        ),
                        {"id": "existing-workspace", "key": "demo", "path": str(workspace)},
                    )
                    connection.execute(
                        text(
                            "INSERT INTO jobs(id,workspace_id,kind,input_hash,input_payload,configuration_snapshot,configuration_hash) VALUES(:id,:workspace,:kind,:hash,:payload,:configuration,:hash)"
                        ),
                        {
                            "id": "existing-job",
                            "workspace": "existing-workspace",
                            "kind": "validate",
                            "hash": "0" * 64,
                            "payload": "{}",
                            "configuration": "{}",
                        },
                    )
                self.assertEqual(upgrade_database(settings), "0002_v4_operations")
                self.assertTrue(
                    {"heartbeat_at", "execution_generation"}
                    <= {column["name"] for column in inspect(engine).get_columns("jobs")}
                )
            finally:
                engine.dispose()
            database = ServiceDatabase(settings)
            try:
                self.assertEqual(database.get_workspace("demo")["storage_path"], str(workspace))
                self.assertEqual(database.get_job("existing-job")["status"], "queued")
                self.assertEqual(database.get_job("existing-job")["execution_generation"], 0)
                operation = database.record_operation(
                    "audit", {"action": "migration.verify"}, "migration"
                )
                self.assertEqual(operation["kind"], "audit")
            finally:
                database.dispose()


if __name__ == "__main__":
    unittest.main()
