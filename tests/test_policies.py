import hashlib
import json
import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from bipartite_scope.config import ServiceSettings
from bipartite_scope.database import ServiceDatabase
from bipartite_scope.policies import (
    ResourceLimitError,
    apply_backup_retention,
    apply_retention,
    check_admission,
    heavy_slot_available,
    pin_snapshot,
    plan_backup_retention,
    plan_retention,
    purge_backup_trash,
    purge_trash,
    storage_usage,
)
from bipartite_scope.storage import SnapshotStore, init_workspace


def snapshot(root: Path, identifier: str, index: int) -> None:
    directory = root / "artifacts" / identifier
    directory.mkdir()
    assets = {}
    for name in SnapshotStore._V2_ASSETS:
        data = f"asset-{identifier}-{name}".encode()
        (directory / name).write_bytes(data)
        assets[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {
        "snapshot_id": identifier,
        "complete": True,
        "core_version": "4.0.0",
        "asset_schema_version": 2,
        "created_at": (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index)).isoformat(),
        "parent_snapshot_id": f"s{index - 1}" if index else None,
        "build_mode": "incremental",
        "assets": assets,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def aged_file(path: Path, days: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("retained runtime asset", encoding="utf-8")
    when = (datetime.now(UTC) - timedelta(days=days)).timestamp()
    os.utime(path, (when, when))
    os.utime(path.parent, (when, when))


class PoliciesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "workspaces"
        self.workspace = self.root / "demo"
        init_workspace(self.workspace)
        self.settings = ServiceSettings.model_validate(
            {
                "service": {"workspace_root": str(self.root)},
                "database": {"url": f"sqlite+pysqlite:///{self.root.parent / 'service.sqlite3'}"},
                "resources": {
                    "max_queued_jobs": 2,
                    "max_workspace_queued_jobs": 1,
                    "min_free_bytes": 0,
                },
                "retention": {"candidate_snapshots": 0, "rollback_snapshots": 1},
            }
        )
        self.database = ServiceDatabase(self.settings)
        self.database.initialize()
        self.database.register_workspace("demo", self.workspace)

    def tearDown(self) -> None:
        self.database.dispose()
        self.temporary.cleanup()

    def test_admission_enforces_queue_disk_and_storage(self) -> None:
        check_admission(self.database, "demo", "validate", {})
        self.database.create_job("demo", "validate", {})
        with self.assertRaises(ResourceLimitError) as caught:
            check_admission(self.database, "demo", "validate", {})
        self.assertEqual(caught.exception.code, "workspace_queue_full")
        job = self.database.list_jobs()[0]
        self.database.request_cancel(job["job_id"])
        with (
            patch(
                "bipartite_scope.policies.storage_usage",
                return_value={"disk_free_bytes": -1, "total_bytes": 0},
            ),
            self.assertRaises(ResourceLimitError) as caught,
        ):
            check_admission(self.database, "demo", "build", {})
        self.assertEqual(caught.exception.code, "insufficient_disk_space")
        self.database.settings = self.settings.model_copy(
            update={
                "resources": self.settings.resources.model_copy(update={"workspace_max_bytes": 1})
            }
        )
        with self.assertRaises(ResourceLimitError) as caught:
            check_admission(self.database, "demo", "build", {})
        self.assertEqual(caught.exception.code, "workspace_storage_full")

    def test_heavy_limit_and_symlink_storage_accounting(self) -> None:
        first, _ = self.database.create_job("demo", "evaluate", {})
        self.database.transition_job(first["job_id"], "running")
        second, _ = self.database.create_job("demo", "evaluate", {})
        self.assertFalse(heavy_slot_available(self.database, second["job_id"]))
        external = self.root.parent / "external.txt"
        external.write_bytes(b"x" * 10000)
        (self.workspace / "linked").symlink_to(external)
        usage = storage_usage(self.workspace)
        self.assertEqual(usage["symlink_count"], 1)
        self.assertNotIn("linked", usage["categories"])

    def test_global_queue_limit_applies_across_workspaces(self) -> None:
        for key in ("other", "third"):
            path = self.root / key
            init_workspace(path)
            self.database.register_workspace(key, path)
        self.database.create_job("demo", "validate", {})
        self.database.create_job("other", "validate", {})
        with self.assertRaises(ResourceLimitError) as caught:
            self.database.create_job("third", "validate", {})
        self.assertEqual(caught.exception.code, "global_queue_full")

    def test_protection_quarantine_lineage_and_runtime_references(self) -> None:
        for index in range(4):
            snapshot(self.workspace, f"s{index}", index)
        SnapshotStore(self.workspace / "artifacts").activate("s3")
        pin_snapshot(self.workspace, "s0")
        upload = self.workspace / "data" / "uploads" / "u1" / "events.csv"
        aged_file(upload)
        report = self.workspace / "reports" / "evaluations" / "e1" / "report.md"
        aged_file(report)
        job, _ = self.database.create_job("demo", "update", {"events_upload_id": "u1"})
        ledger = self.workspace / "data" / "events.jsonl"
        aged_file(ledger)
        state = self.workspace / "state.sqlite3"
        aged_file(state)
        plan = plan_retention(self.workspace, self.database)
        self.assertEqual(
            {item["path"] for item in plan["entries"]}, {"artifacts/s1", "reports/evaluations/e1"}
        )
        self.assertIn("pinned_snapshot", plan["protected"]["artifacts/s0"])
        self.assertIn("rollback_snapshot", plan["protected"]["artifacts/s2"])
        self.assertIn("pending_job_reference", plan["protected"]["data/uploads/u1"])
        result = apply_retention(self.workspace, plan, self.database)
        self.assertEqual(result["moved_count"], 2)
        self.assertEqual(result["reclaimed_disk_bytes"], 0)
        self.assertTrue(ledger.is_file())
        self.assertTrue(state.is_file())
        self.assertTrue(upload.is_file())
        self.assertTrue((self.workspace / "artifacts" / "lineage" / "s1.json").is_file())
        self.assertTrue(
            (Path(result["trash_path"]) / "artifacts" / "s1" / "manifest.json").is_file()
        )
        self.database.request_cancel(job["job_id"])

    def test_stale_modified_and_newly_protected_plans_are_rejected(self) -> None:
        for index in range(3):
            snapshot(self.workspace, f"s{index}", index)
        SnapshotStore(self.workspace / "artifacts").activate("s2")
        plan = plan_retention(self.workspace, self.database)
        pin_snapshot(self.workspace, "s0")
        with self.assertRaisesRegex(ValueError, "stale"):
            apply_retention(self.workspace, plan, self.database)
        report = self.workspace / "reports" / "evaluations" / "old" / "report.md"
        aged_file(report)
        plan = plan_retention(self.workspace, self.database)
        report.write_text("changed after planning", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stale"):
            apply_retention(self.workspace, plan, self.database)
        self.assertTrue(report.exists())
        self.assertFalse((self.workspace / ".trash").exists())

    def test_tampered_plan_symlink_and_invalid_pointer_protect_data(self) -> None:
        for index in range(3):
            snapshot(self.workspace, f"s{index}", index)
        SnapshotStore(self.workspace / "artifacts").activate("s2")
        plan = plan_retention(self.workspace, self.database)
        plan["entries"].append({"path": "../external.txt"})
        with self.assertRaisesRegex(ValueError, "invalid"):
            apply_retention(self.workspace, plan, self.database)
        (self.workspace / "artifacts" / "latest.json").write_text("invalid", encoding="utf-8")
        protected = plan_retention(self.workspace, self.database)
        self.assertFalse(protected["entries"])
        self.assertIn("invalid_active_pointer", protected["protected"]["artifacts/s0"])
        external = self.root.parent / "external"
        external.mkdir()
        (self.workspace / "data" / "uploads").mkdir(exist_ok=True)
        (self.workspace / "data" / "uploads" / "unsafe").symlink_to(external)
        protected = plan_retention(self.workspace, self.database)
        self.assertIn("unsafe_or_unreadable_path", protected["protected"]["data/uploads/unsafe"])

    def test_trash_purge_is_explicit_and_rejects_modified_bundles(self) -> None:
        report = self.workspace / "reports" / "evaluations" / "old" / "report.md"
        aged_file(report)
        result = apply_retention(
            self.workspace, plan_retention(self.workspace, self.database), self.database
        )
        trash = Path(result["trash_path"])
        dry = purge_trash(self.workspace, older_than_days=0, db=self.database)
        self.assertEqual(len(dry["entries"]), 1)
        self.assertEqual(dry["reclaimed_disk_bytes"], 0)
        self.assertTrue(trash.exists())
        (trash / "unexpected.txt").write_text("unexpected", encoding="utf-8")
        protected = purge_trash(self.workspace, older_than_days=0, dry_run=False, db=self.database)
        self.assertFalse(protected["entries"])
        self.assertTrue(trash.exists())
        (trash / "unexpected.txt").unlink()
        purged = purge_trash(self.workspace, older_than_days=0, dry_run=False, db=self.database)
        self.assertGreater(purged["reclaimed_disk_bytes"], 0)
        self.assertFalse(trash.exists())

    def test_backup_retention_preserves_newest_verified_and_rejects_stale(self) -> None:
        from bipartite_scope.maintenance import create_backup

        settings = self.settings.model_copy(
            update={"retention": self.settings.retention.model_copy(update={"backups_keep": 1})}
        )
        backups = [create_backup(self.workspace, settings=settings) for _ in range(3)]
        plan = plan_backup_retention(settings)
        self.assertEqual(len(plan["entries"]), 2)
        self.assertIn(backups[-1]["backup_id"], plan["protected"])
        changed = Path(backups[0]["path"]) / "manifest.json"
        original = changed.read_text(encoding="utf-8")
        changed.write_text(original + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stale"):
            apply_backup_retention(settings, plan)
        result = apply_backup_retention(settings, plan_backup_retention(settings))
        self.assertEqual(result["moved_count"], 2)
        self.assertTrue(Path(backups[-1]["path"]).is_dir())
        self.assertFalse(Path(backups[0]["path"]).exists())
        self.assertTrue((Path(result["trash_path"]) / backups[0]["backup_id"]).is_dir())
        dry = purge_backup_trash(settings, older_than_days=0)
        self.assertEqual(len(dry["entries"]), 1)
        self.assertTrue(Path(result["trash_path"]).is_dir())
        purged = purge_backup_trash(settings, older_than_days=0, dry_run=False)
        self.assertGreater(purged["reclaimed_disk_bytes"], 0)
        self.assertFalse(Path(result["trash_path"]).exists())
        self.assertTrue(Path(backups[-1]["path"]).is_dir())


if __name__ == "__main__":
    unittest.main()
