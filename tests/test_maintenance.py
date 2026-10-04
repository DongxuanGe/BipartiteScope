import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import select

from bipartite_scope import Event, SnapshotStore, build_snapshot
from bipartite_scope.config import ServiceSettings
from bipartite_scope.database import JobStateError, ServiceDatabase, WorkspaceLeaseRecord
from bipartite_scope.maintenance import (
    BackupIntegrityError,
    _pg_run,
    create_backup,
    drill_backup,
    list_backups,
    restore_backup,
    service_maintenance,
    verify_backup,
)
from bipartite_scope.reliability import WorkspaceBusyError, workspace_write_lock
from bipartite_scope.storage import (
    init_workspace,
    load_workspace,
    mark_events_applied,
    pending_events,
    register_events,
)
from tests.test_core import config, graph


class MaintenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "workspaces"
        self.workspace = load_workspace(init_workspace(self.root / "demo"))
        self.settings = ServiceSettings.model_validate(
            {
                "service": {"workspace_root": str(self.root)},
                "database": {"url": f"sqlite+pysqlite:///{self.base / 'service.sqlite3'}"},
                "backup": {"wait_seconds": 1},
                "resources": {"min_free_bytes": 0},
            }
        )
        self.database = ServiceDatabase(self.settings)
        self.database.initialize()
        self.database.register_workspace("demo", self.workspace.root)

    def tearDown(self) -> None:
        self.database.dispose()
        self.temporary.cleanup()

    def test_workspace_backup_preserves_active_snapshot_ledger_and_pending_feedback(self) -> None:
        snapshot = build_snapshot(graph(), config(1))
        SnapshotStore(self.workspace.artifacts).save(snapshot, activate=True)
        applied = Event("applied", "u1", "i1", "click", 1, "2026-09-01T00:00:00Z")
        feedback = Event("pending", "u2", "i3", "favorite", 1, "2026-09-02T00:00:00Z")
        register_events(self.workspace, (applied,), source="test")
        mark_events_applied(self.workspace, (applied.event_id,), snapshot.snapshot_id, "batch-one")
        register_events(
            self.workspace, (feedback,), source="feedback", ledger_name="feedback.jsonl"
        )
        (self.workspace.root / ".env").write_text("SECRET=not-for-backup", encoding="utf-8")
        (self.workspace.data / "external").symlink_to(self.base / "service.sqlite3")
        backup = create_backup(self.workspace.root, self.settings, self.database)
        self.assertTrue(verify_backup(backup["path"])["verified"])
        restored = restore_backup(backup["path"], self.base / "restored-workspace")
        target = load_workspace(restored["target_root"])
        self.assertEqual(SnapshotStore(target.artifacts).latest_id(), snapshot.snapshot_id)
        self.assertEqual(pending_events(target), (feedback,))
        accepted, duplicate = register_events(target, (feedback,), source="test")
        self.assertEqual(accepted, ())
        self.assertEqual(duplicate, (feedback.event_id,))
        self.assertFalse((target.root / ".env").exists())
        self.assertFalse((target.data / "external").exists())
        self.assertEqual(pending_events(self.workspace), (feedback,))
        drill = drill_backup(backup["path"])
        self.assertEqual(drill["status"], "succeeded")
        self.assertEqual(
            drill["workspaces"][0]["representative_query"]["snapshot_id"], snapshot.snapshot_id
        )
        self.assertEqual(list_backups(self.settings)[0]["backup_id"], backup["backup_id"])

    def test_tamper_and_unlisted_files_are_rejected_before_restore(self) -> None:
        backup = create_backup(self.workspace.root, self.settings)
        bundle = Path(backup["path"])
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        asset = bundle / manifest["files"][0]["path"]
        original = asset.read_bytes()
        asset.write_bytes(original + b"tampered")
        target = self.base / "restore-invalid"
        with self.assertRaises(BackupIntegrityError):
            restore_backup(bundle, target)
        self.assertFalse(target.exists())
        asset.write_bytes(original)
        (bundle / "unexpected.txt").write_text("unexpected", encoding="utf-8")
        with self.assertRaises(BackupIntegrityError):
            verify_backup(bundle)

    def test_restore_requires_new_nonoverlapping_target_and_respects_size_limit(self) -> None:
        backup = create_backup(self.workspace.root, self.settings)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            restore_backup(backup["path"], self.workspace.root)
        target = self.base / "occupied"
        target.mkdir()
        (target / "preserved").write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            restore_backup(backup["path"], target)
        self.assertEqual((target / "preserved").read_text(encoding="utf-8"), "keep")
        with self.assertRaisesRegex(ValueError, "size limit"):
            restore_backup(backup["path"], self.base / "too-large", max_restore_bytes=1)

    def test_service_fence_blocks_new_jobs_and_writes_and_cleans_up(self) -> None:
        with service_maintenance(self.settings, self.database):
            with self.assertRaises(JobStateError):
                self.database.create_job("demo", "validate", {})

            def write() -> None:
                with workspace_write_lock(self.workspace.root):
                    self.fail("maintenance allowed a writer")

            with (
                ThreadPoolExecutor(max_workers=1) as executor,
                self.assertRaises(WorkspaceBusyError),
            ):
                executor.submit(write).result()
        self.assertFalse((self.root / ".operations" / "maintenance.json").exists())
        self.database.create_job("demo", "validate", {})

    def test_service_backup_restore_remaps_paths_preserves_history_and_interrupts_jobs(
        self,
    ) -> None:
        complete, _ = self.database.create_job(
            "demo", "validate", {"path": str(self.workspace.data)}
        )
        self.database.transition_job(complete["job_id"], "running")
        self.database.transition_job(
            complete["job_id"],
            "succeeded",
            result={"path": str(self.workspace.exports / "result.json")},
        )
        waiting, _ = self.database.create_job(
            "demo", "validate", {"path": str(self.workspace.data)}
        )
        running, _ = self.database.create_job("demo", "build", {})
        claimed, attempt = self.database.claim_job(running["job_id"], "test-worker")
        self.assertTrue(self.database.acquire_workspace_lease(running["job_id"]))
        self.database.mark_published(running["job_id"], "old-celery-id")
        self.database.record_operation(
            "audit", {"operation": "test", "path": str(self.workspace.root)}, "audit-one"
        )
        backup = create_backup(
            self.root, self.settings, self.database, "service", exclude_job_id=claimed["job_id"]
        )
        restored_database_path = self.base / "restored-service.sqlite3"
        target_root = self.base / "restored-service"
        restored = restore_backup(
            backup["path"], target_root, f"sqlite+pysqlite:///{restored_database_path}"
        )
        target_settings = self.settings.model_copy(
            update={
                "service": self.settings.service.model_copy(update={"workspace_root": target_root}),
                "database": self.settings.database.model_copy(
                    update={"url": f"sqlite+pysqlite:///{restored_database_path}"}
                ),
            }
        )
        database = ServiceDatabase(target_settings)
        try:
            self.assertEqual(
                database.get_workspace("demo")["storage_path"], str(target_root / "demo")
            )
            self.assertEqual(database.get_job(complete["job_id"])["status"], "succeeded")
            self.assertEqual(
                database.get_job(complete["job_id"])["result"]["path"],
                str(target_root / "demo" / "exports" / "result.json"),
            )
            self.assertEqual(
                database.get_job(waiting["job_id"])["error_code"], "restore_interrupted"
            )
            restored_job = database.get_job(running["job_id"])
            self.assertEqual(restored_job["status"], "failed")
            self.assertIsNone(restored_job["celery_task_id"])
            with database.sessions() as session:
                self.assertEqual(list(session.scalars(select(WorkspaceLeaseRecord))), [])
            self.assertEqual(
                database.get_operation("audit", "audit-one")["payload"]["path"],
                str(target_root / "demo"),
            )
            self.assertEqual(restored["interrupted_jobs"], 2)
            self.assertFalse(restored["automatic_task_replay"])
            with closing(sqlite3.connect(restored_database_path)) as connection:
                status = connection.execute(
                    "SELECT status FROM job_attempts WHERE id=?", (attempt,)
                ).fetchone()[0]
            self.assertEqual(status, "failed")
        finally:
            database.dispose()
        self.assertEqual(self.database.get_job(running["job_id"])["status"], "running")
        self.assertEqual(drill_backup(backup["path"])["status"], "succeeded")
        with self.assertRaisesRegex(FileExistsError, "database is not empty"):
            restore_backup(
                backup["path"], self.base / "restore-refused", self.settings.database.url
            )

    def test_service_backup_times_out_without_leaving_a_maintenance_marker(self) -> None:
        job, _ = self.database.create_job("demo", "validate", {})
        self.database.claim_job(job["job_id"], "worker")
        with self.assertRaises(TimeoutError):
            create_backup(self.root, self.settings, self.database, "service")
        self.assertFalse((self.root / ".operations" / "maintenance.json").exists())
        self.assertEqual(list_backups(self.settings), [])

    def test_manifest_traversal_and_symbolic_bundle_are_rejected(self) -> None:
        backup = create_backup(self.workspace.root, self.settings)
        bundle = Path(backup["path"])
        symbolic = self.base / "symbolic-bundle"
        symbolic.symlink_to(bundle, target_is_directory=True)
        with self.assertRaisesRegex(BackupIntegrityError, "symbolic"):
            verify_backup(symbolic)
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][0]["path"] = "../source-file"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(BackupIntegrityError, "unsafe"):
            restore_backup(bundle, self.base / "rejected")

    def test_postgresql_tools_keep_passwords_out_of_arguments_and_errors(self) -> None:
        from sqlalchemy.engine import make_url

        url = make_url("postgresql+psycopg://owner:secret-password@localhost:5432/control")
        with patch(
            "bipartite_scope.maintenance.subprocess.run", return_value=SimpleNamespace(returncode=0)
        ) as run:
            _pg_run(["pg_dump", "--format=custom"], url)
            self.assertNotIn("secret-password", " ".join(run.call_args.args[0]))
            self.assertEqual(run.call_args.kwargs["env"]["PGPASSWORD"], "secret-password")
            self.assertEqual(run.call_args.kwargs["env"]["PGDATABASE"], "control")
        with patch(
            "bipartite_scope.maintenance.subprocess.run", return_value=SimpleNamespace(returncode=1)
        ):
            with self.assertRaises(RuntimeError) as error:
                _pg_run(["pg_dump"], url)
            self.assertNotIn("secret-password", str(error.exception))


if __name__ == "__main__":
    unittest.main()
