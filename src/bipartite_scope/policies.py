from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from .storage import SnapshotIntegrityError, SnapshotStore

HEAVY_JOB_KINDS = frozenset({"build", "update", "evaluate", "benchmark", "backup", "restore"})
PENDING_JOB_STATES = ("queued", "retrying", "running", "cancelling")
_RETENTION = {
    "candidate_snapshots": 10,
    "rollback_snapshots": 2,
    "reports_days": 30,
    "uploads_days": 7,
    "trash_days": 7,
    "backups_keep": 7,
}


class ResourceLimitError(ValueError):
    def __init__(self, message: str, code: str = "resource_limit"):
        super().__init__(message)
        self.code = code


def _root(workspace: Any) -> Path:
    value = workspace if isinstance(workspace, (str, Path)) else workspace.root
    return Path(value).expanduser().resolve()


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _files(path: Path) -> list[Path]:
    if path.is_symlink():
        raise ValueError("symbolic links are not eligible for retention")
    if path.is_file():
        return [path]
    result: list[Path] = []
    for directory, names, filenames in os.walk(path, followlinks=False):
        base = Path(directory)
        for name in names + filenames:
            if (base / name).is_symlink():
                raise ValueError("symbolic links are not eligible for retention")
        result.extend(base / name for name in filenames)
    return sorted(result)


def _fingerprint(path: Path) -> dict[str, Any]:
    files = []
    for item in _files(path):
        digest = hashlib.sha256()
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        files.append(
            {
                "path": item.relative_to(path).as_posix() if path.is_dir() else item.name,
                "bytes": item.stat().st_size,
                "mtime_ns": item.stat().st_mtime_ns,
                "sha256": digest.hexdigest(),
            }
        )
    return {"sha256": _hash(files), "bytes": sum(item["bytes"] for item in files)}


def storage_usage(path: str | Path) -> dict[str, Any]:
    root = _root(path)
    categories: dict[str, int] = {}
    files = 0
    symlinks = 0
    if root.exists():
        for directory, names, filenames in os.walk(root, followlinks=False):
            base = Path(directory)
            for name in names + filenames:
                if (base / name).is_symlink():
                    symlinks += 1
            names[:] = [name for name in names if not (base / name).is_symlink()]
            for name in filenames:
                item = base / name
                if item.is_symlink():
                    continue
                category = item.relative_to(root).parts[0]
                categories[category] = categories.get(category, 0) + item.stat().st_size
                files += 1
    existing = root
    while not existing.exists() and existing.parent != existing:
        existing = existing.parent
    disk = shutil.disk_usage(existing)
    return {
        "path": str(root),
        "total_bytes": sum(categories.values()),
        "file_count": files,
        "symlink_count": symlinks,
        "categories": categories,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "trash_bytes": categories.get(".trash", 0),
    }


def check_admission(
    db: Any, workspace_id: str, kind: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    from sqlalchemy import func, select

    from .database import JobRecord

    workspace = db.get_workspace(workspace_id)
    resources = db.settings.resources
    queued = ("queued", "retrying")
    with db.sessions() as session:
        global_count = (
            session.scalar(select(func.count(JobRecord.id)).where(JobRecord.status.in_(queued)))
            or 0
        )
        workspace_count = (
            session.scalar(
                select(func.count(JobRecord.id)).where(
                    JobRecord.workspace_id == workspace["id"], JobRecord.status.in_(queued)
                )
            )
            or 0
        )
    if global_count >= resources.max_queued_jobs:
        raise ResourceLimitError("global queued job limit reached", "global_queue_full")
    if workspace_count >= resources.max_workspace_queued_jobs:
        raise ResourceLimitError("workspace queued job limit reached", "workspace_queue_full")
    usage = check_storage_capacity(workspace["storage_path"], resources)
    return {
        "admitted": True,
        "kind": kind,
        "global_queued_jobs": global_count,
        "workspace_queued_jobs": workspace_count,
        "storage": usage,
    }


def check_storage_capacity(
    path: str | Path, resources: Any, additional_bytes: int = 0
) -> dict[str, Any]:
    if additional_bytes < 0:
        raise ValueError("additional storage bytes must be nonnegative")
    usage = storage_usage(path)
    if usage["disk_free_bytes"] - additional_bytes < resources.min_free_bytes:
        raise ResourceLimitError("insufficient free disk space", "insufficient_disk_space")
    if (
        resources.workspace_max_bytes
        and usage["total_bytes"] + additional_bytes >= resources.workspace_max_bytes
    ):
        raise ResourceLimitError("workspace storage limit reached", "workspace_storage_full")
    return usage


def heavy_slot_available(db: Any, job_id: str) -> bool:
    from sqlalchemy import func, select

    from .database import JobRecord

    with db.sessions() as session:
        job = session.get(JobRecord, job_id)
        if job is None:
            raise ValueError("job does not exist")
        if job.kind not in HEAVY_JOB_KINDS:
            return True
        count = (
            session.scalar(
                select(func.count(JobRecord.id)).where(
                    JobRecord.kind.in_(tuple(HEAVY_JOB_KINDS)),
                    JobRecord.status.in_(("running", "cancelling")),
                    JobRecord.id != job_id,
                )
            )
            or 0
        )
    return count < db.settings.resources.max_running_heavy_jobs


def _pins(root: Path) -> set[str]:
    path = root / "artifacts" / "pins.json"
    if path.is_symlink():
        raise ValueError("snapshot pins file must not be a symbolic link")
    if not path.exists():
        return set()
    value = json.loads(path.read_text(encoding="utf-8"))
    ids = value.get("snapshot_ids") if isinstance(value, dict) else None
    if not isinstance(ids, list) or any(
        not isinstance(item, str) or not item or Path(item).name != item for item in ids
    ):
        raise ValueError("snapshot pins file is invalid")
    return set(ids)


def pin_snapshot(workspace: Any, snapshot_id: str, pinned: bool = True) -> dict[str, Any]:
    from .reliability import workspace_write_lock

    root = _root(workspace)
    with workspace_write_lock(root):
        SnapshotStore(root / "artifacts").verify(snapshot_id)
        ids = _pins(root)
        if pinned:
            ids.add(snapshot_id)
        else:
            ids.discard(snapshot_id)
        payload = {"snapshot_ids": sorted(ids)}
        target = root / "artifacts" / "pins.json"
        temporary = target.with_name(f".pins.{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, target)
    return {"snapshot_id": snapshot_id, "pinned": pinned}


def _job_protections(root: Path, db: Any) -> tuple[set[str], set[str]]:
    from sqlalchemy import select

    from .database import JobRecord, WorkspaceRecord

    snapshots: set[str] = set()
    uploads: set[str] = set()
    if db is None:
        return snapshots, uploads
    with db.sessions() as session:
        records = session.scalars(
            select(JobRecord)
            .join(WorkspaceRecord, WorkspaceRecord.id == JobRecord.workspace_id)
            .where(
                WorkspaceRecord.storage_path == str(root),
                JobRecord.status.in_(PENDING_JOB_STATES),
                JobRecord.kind != "retention",
            )
        ).all()
        values = [(record.input_payload, record.result_payload) for record in records]

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(item, str) and Path(item).name == item:
                    if "snapshot" in key and key.endswith("id"):
                        snapshots.add(item)
                    if "upload" in key and key.endswith("id"):
                        uploads.add(item)
                if isinstance(item, str) and Path(item).is_absolute():
                    candidate = Path(item)
                    upload_root = root / "data" / "uploads"
                    if upload_root in candidate.parents:
                        uploads.add(candidate.relative_to(upload_root).parts[0])
                    if root / "artifacts" in candidate.parents:
                        snapshots.add(candidate.relative_to(root / "artifacts").parts[0])
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    for value in values:
        collect(value)
    return snapshots, uploads


def _settings(db: Any) -> dict[str, int]:
    settings = getattr(getattr(db, "settings", None), "retention", None)
    return {key: int(getattr(settings, key, default)) for key, default in _RETENTION.items()}


def plan_retention(workspace: Any, db: Any = None) -> dict[str, Any]:
    root = _root(workspace)
    if not root.is_dir():
        raise ValueError("workspace does not exist")
    now = datetime.now(UTC)
    settings = _settings(db)
    store = SnapshotStore(root / "artifacts")
    protected: dict[str, list[str]] = {}
    entries: list[dict[str, Any]] = []
    snapshot_refs, upload_refs = _job_protections(root, db)
    pins = _pins(root)
    pointer_invalid = False
    try:
        active = store.pointer_id()
    except SnapshotIntegrityError:
        active = None
        pointer_invalid = (store.root / "latest.json").exists()
    snapshots = [
        item
        for item in store.list()
        if item["snapshot_id"] != "lineage" and (store.root / item["snapshot_id"]).is_dir()
    ]
    snapshots.sort(
        key=lambda item: (item.get("created_at") or "", item["snapshot_id"]), reverse=True
    )
    rollback = {
        item["snapshot_id"]
        for item in [item for item in snapshots if item["verified"] and not item["active"]][
            : settings["rollback_snapshots"]
        ]
    }
    candidates = [
        item["snapshot_id"]
        for item in snapshots
        if item["verified"] and item["snapshot_id"] not in rollback and not item["active"]
    ][: settings["candidate_snapshots"]]

    def consider(path: Path, category: str, reasons: list[str], reason: str) -> None:
        relative = path.relative_to(root).as_posix()
        try:
            fingerprint = _fingerprint(path)
        except (OSError, ValueError):
            protected[relative] = sorted(set(reasons + ["unsafe_or_unreadable_path"]))
            return
        if reasons:
            protected[relative] = sorted(set(reasons))
        else:
            entries.append(
                {"path": relative, "category": category, "reason": reason, **fingerprint}
            )

    for item in snapshots:
        identifier = item["snapshot_id"]
        reasons = []
        if identifier == active:
            reasons.append("active_snapshot")
        if pointer_invalid:
            reasons.append("invalid_active_pointer")
        if identifier in pins:
            reasons.append("pinned_snapshot")
        if identifier in rollback:
            reasons.append("rollback_snapshot")
        if identifier in candidates:
            reasons.append("retained_candidate")
        if identifier in snapshot_refs:
            reasons.append("pending_job_reference")
        if not item["verified"]:
            reasons.append("snapshot_integrity_unverified")
        consider(store.root / identifier, "snapshot", reasons, "candidate_limit_exceeded")

    for category, parent, days in (
        ("upload", root / "data" / "uploads", settings["uploads_days"]),
        ("report", root / "reports" / "evaluations", settings["reports_days"]),
    ):
        if not parent.is_dir() or parent.is_symlink():
            continue
        cutoff = (now - timedelta(days=days)).timestamp()
        for path in sorted(parent.iterdir()):
            reasons = []
            if category == "upload" and path.name in upload_refs:
                reasons.append("pending_job_reference")
            try:
                modified = max(
                    [path.stat().st_mtime] + [item.stat().st_mtime for item in _files(path)]
                )
            except (OSError, ValueError):
                modified = now.timestamp()
                reasons.append("unsafe_or_unreadable_path")
            if modified >= cutoff:
                reasons.append("retention_period")
            consider(path, category, reasons, "retention_period_expired")
    trash = root / ".trash"
    expired_trash = []
    if trash.is_dir() and not trash.is_symlink():
        cutoff = (now - timedelta(days=settings["trash_days"])).timestamp()
        expired_trash = [
            path.name for path in sorted(trash.iterdir()) if path.stat().st_mtime < cutoff
        ]
    payload = {
        "schema_version": 1,
        "workspace_path": str(root),
        "created_at": now.isoformat(),
        "policy": settings,
        "entries": sorted(entries, key=lambda item: item["path"]),
        "protected": protected,
        "estimated_bytes": sum(item["bytes"] for item in entries),
        "expired_trash": expired_trash,
        "action": "quarantine",
    }
    return {**payload, "plan_hash": _hash(payload)}


retention_plan = plan_retention


def apply_retention(workspace: Any, plan: Mapping[str, Any], db: Any = None) -> dict[str, Any]:
    from .reliability import workspace_write_lock

    root = _root(workspace)
    supplied = dict(plan)
    claimed_hash = supplied.pop("plan_hash", None)
    if (
        claimed_hash != _hash(supplied)
        or supplied.get("workspace_path") != str(root)
        or supplied.get("schema_version") != 1
        or supplied.get("action") != "quarantine"
    ):
        raise ValueError("retention plan is invalid or belongs to another workspace")
    with workspace_write_lock(root):
        current = plan_retention(root, db)
        allowed = {item["path"]: item for item in current["entries"]}
        targets = []
        for entry in supplied.get("entries", []):
            relative = entry.get("path")
            if relative not in allowed or allowed[relative] != entry:
                raise ValueError("retention plan is stale; generate a new plan")
            target = root / relative
            if target.resolve() != target or root not in target.parents:
                raise ValueError("retention target escapes the workspace")
            targets.append((target, entry))
        if len(targets) != len({target for target, _ in targets}):
            raise ValueError("retention plan has duplicate targets")
        operation_id = uuid4().hex
        destination = root / ".trash" / operation_id
        if destination.parent.is_symlink():
            raise ValueError("trash path must not be a symbolic link")
        destination.mkdir(parents=True, exist_ok=False)
        moved = []
        tombstone = {
            "operation_id": operation_id,
            "plan_hash": claimed_hash,
            "created_at": datetime.now(UTC).isoformat(),
            "entries": moved,
            "status": "running",
        }
        marker = destination / "tombstone.json"
        marker.write_text(json.dumps(tombstone, indent=2), encoding="utf-8")
        try:
            for target, entry in targets:
                output = destination / entry["path"]
                output.parent.mkdir(parents=True, exist_ok=True)
                if entry["category"] == "snapshot":
                    lineage = root / "artifacts" / "lineage"
                    if lineage.is_symlink():
                        raise ValueError("snapshot lineage path must not be a symbolic link")
                    lineage.mkdir(exist_ok=True)
                    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
                    manifest["retention"] = {
                        "operation_id": operation_id,
                        "quarantined_at": datetime.now(UTC).isoformat(),
                        "original_path": entry["path"],
                    }
                    temporary = lineage / f".{target.name}.{uuid4().hex}.tmp"
                    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    os.replace(temporary, lineage / f"{target.name}.json")
                os.replace(target, output)
                moved.append(entry)
                marker.write_text(json.dumps(tombstone, indent=2), encoding="utf-8")
            tombstone["status"] = "completed"
        except Exception:
            tombstone["status"] = "partial"
            raise
        finally:
            marker.write_text(json.dumps(tombstone, indent=2), encoding="utf-8")
    return {
        "operation_id": operation_id,
        "status": "completed",
        "moved_count": len(moved),
        "quarantined_bytes": sum(item["bytes"] for item in moved),
        "reclaimed_disk_bytes": 0,
        "trash_path": str(destination),
        "recoverable": True,
    }


def purge_trash(
    workspace: Any,
    older_than_days: int | None = None,
    dry_run: bool = True,
    db: Any = None,
) -> dict[str, Any]:
    from .reliability import workspace_write_lock

    root = _root(workspace)
    days = _settings(db)["trash_days"] if older_than_days is None else older_than_days
    if days < 0:
        raise ValueError("trash retention days must be nonnegative")
    with workspace_write_lock(root):
        trash = root / ".trash"
        if trash.is_symlink():
            raise ValueError("trash path must not be a symbolic link")
        cutoff = datetime.now(UTC) - timedelta(days=days)
        snapshot_refs, upload_refs = _job_protections(root, db)
        pins = _pins(root)
        try:
            pins.add(SnapshotStore(root / "artifacts").pointer_id())
        except SnapshotIntegrityError:
            if (root / "artifacts" / "latest.json").exists():
                raise ValueError("invalid active pointer prevents trash purge") from None
        entries = []
        protected = {}
        if not trash.is_dir():
            return {"dry_run": dry_run, "entries": [], "protected": {}, "reclaimed_disk_bytes": 0}
        for bundle in sorted(trash.iterdir()):
            if not bundle.is_dir() or bundle.is_symlink():
                protected[bundle.name] = "unsafe_trash_bundle"
                continue
            marker = bundle / "tombstone.json"
            try:
                value = json.loads(marker.read_text(encoding="utf-8"))
                created = datetime.fromisoformat(value["created_at"])
                if created.tzinfo is None or created > cutoff:
                    protected[bundle.name] = "retention_period"
                    continue
                if value["status"] != "completed" or value["operation_id"] != bundle.name:
                    raise ValueError("invalid trash operation")
                expected = {"tombstone.json"}
                for item in value["entries"]:
                    relative = Path(item["path"])
                    parts = relative.parts
                    category = item["category"]
                    prefixes = {
                        "snapshot": ("artifacts",),
                        "upload": ("data", "uploads"),
                        "report": ("reports", "evaluations"),
                    }
                    prefix = prefixes.get(category)
                    if (
                        prefix is None
                        or parts[: len(prefix)] != prefix
                        or len(parts) != len(prefix) + 1
                        or relative.is_absolute()
                        or any(part in {".", ".."} for part in parts)
                    ):
                        raise ValueError("invalid trash target")
                    identifier = parts[-1]
                    if category == "snapshot" and identifier in pins | snapshot_refs:
                        raise ValueError("protected snapshot reference")
                    if category == "upload" and identifier in upload_refs:
                        raise ValueError("pending upload reference")
                    target = bundle / relative
                    if target.resolve() != target or bundle not in target.parents:
                        raise ValueError("unsafe trash target")
                    if _fingerprint(target) != {"sha256": item["sha256"], "bytes": item["bytes"]}:
                        raise ValueError("trash target was modified")
                    expected.update(file.relative_to(bundle).as_posix() for file in _files(target))
                actual = {file.relative_to(bundle).as_posix() for file in _files(bundle)}
                if expected != actual:
                    raise ValueError("trash bundle has unexpected files")
                entries.append(
                    {"operation_id": bundle.name, "bytes": _fingerprint(bundle)["bytes"]}
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                protected[bundle.name] = str(exc)
        reclaimed = 0
        if not dry_run:
            for item in entries:
                target = trash / item["operation_id"]
                shutil.rmtree(target)
                reclaimed += item["bytes"]
        return {
            "dry_run": dry_run,
            "entries": entries,
            "protected": protected,
            "estimated_bytes": sum(item["bytes"] for item in entries),
            "reclaimed_disk_bytes": reclaimed,
            "recoverable": dry_run,
        }


def _backup_root(settings: Any) -> Path:
    configured = settings.backup.root
    source = settings.workspace_root
    raw = (
        Path(configured).expanduser().absolute()
        if configured
        else source.parent / f"{source.name}-backups"
    )
    root = raw.resolve()
    if (
        raw.is_symlink()
        or root == Path("/")
        or root in source.parents
        or root == source
        or source in root.parents
    ):
        raise ValueError("backup retention root must be a separate safe directory")
    return root


def plan_backup_retention(settings: Any, db: Any = None) -> dict[str, Any]:
    from .maintenance import BackupIntegrityError, verify_backup

    root = _backup_root(settings)
    candidates = []
    protected = {}
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        if path.name.startswith("."):
            continue
        if not re.fullmatch(r"backup-\d{8}T\d{6}-[0-9a-f]{12}", path.name):
            protected[path.name] = "unrecognized_backup_path"
            continue
        try:
            if not path.is_dir() or path.is_symlink():
                raise ValueError("backup path must be a real directory")
            verify_backup(path)
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            if manifest["backup_id"] != path.name:
                raise ValueError("backup identifier does not match its directory")
            if not isinstance(manifest["source_root"], str) or not manifest["source_root"]:
                raise ValueError("backup source root is invalid")
            created = datetime.fromisoformat(manifest["created_at"])
            if created.tzinfo is None:
                raise ValueError("backup timestamp must include a timezone")
            candidates.append(
                {
                    "path": path.name,
                    "scope": manifest["scope"],
                    "source_root": manifest["source_root"],
                    "created_at": manifest["created_at"],
                    **_fingerprint(path),
                }
            )
        except (OSError, ValueError, KeyError, BackupIntegrityError) as exc:
            protected[path.name] = f"backup_integrity_unverified: {exc}"
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in candidates:
        groups.setdefault((item["scope"], item["source_root"]), []).append(item)
    entries = []
    for group in groups.values():
        group.sort(key=lambda item: (item["created_at"], item["path"]), reverse=True)
        for item in group[: settings.retention.backups_keep]:
            protected[item["path"]] = "retained_verified_backup"
        for item in group[settings.retention.backups_keep :]:
            entries.append({**item, "reason": "backup_limit_exceeded"})
    payload = {
        "schema_version": 1,
        "backup_root": str(root),
        "created_at": datetime.now(UTC).isoformat(),
        "backups_keep": settings.retention.backups_keep,
        "entries": sorted(entries, key=lambda item: item["path"]),
        "protected": protected,
        "estimated_bytes": sum(item["bytes"] for item in entries),
        "action": "quarantine",
    }
    return {**payload, "plan_hash": _hash(payload)}


def apply_backup_retention(
    settings: Any, plan: Mapping[str, Any], db: Any = None
) -> dict[str, Any]:
    from .reliability import workspace_write_lock

    root = _backup_root(settings)
    supplied = dict(plan)
    checksum = supplied.pop("plan_hash", None)
    if (
        checksum != _hash(supplied)
        or supplied.get("backup_root") != str(root)
        or supplied.get("schema_version") != 1
        or supplied.get("action") != "quarantine"
    ):
        raise ValueError("backup retention plan is invalid")
    with workspace_write_lock(root):
        current = plan_backup_retention(settings, db)
        allowed = {item["path"]: item for item in current["entries"]}
        for entry in supplied.get("entries", []):
            if allowed.get(entry.get("path")) != entry:
                raise ValueError("backup retention plan is stale; generate a new plan")
        if len(supplied["entries"]) != len({item["path"] for item in supplied["entries"]}):
            raise ValueError("backup retention plan has duplicate targets")
        trash = root / ".trash"
        if trash.is_symlink():
            raise ValueError("backup trash directory must not be a symbolic link")
        operation = uuid4().hex
        destination = trash / operation
        destination.mkdir(parents=True, exist_ok=False)
        moved = []
        tombstone = {
            "operation_id": operation,
            "plan_hash": checksum,
            "created_at": datetime.now(UTC).isoformat(),
            "entries": moved,
            "status": "running",
        }
        marker = destination / "tombstone.json"
        marker.write_text(json.dumps(tombstone, indent=2), encoding="utf-8")
        try:
            for entry in supplied["entries"]:
                os.replace(root / entry["path"], destination / entry["path"])
                moved.append(entry)
                marker.write_text(json.dumps(tombstone, indent=2), encoding="utf-8")
            tombstone["status"] = "completed"
        except Exception:
            tombstone["status"] = "partial"
            raise
        finally:
            marker.write_text(json.dumps(tombstone, indent=2), encoding="utf-8")
        result = {
            "operation_id": operation,
            "moved_count": len(moved),
            "quarantined_bytes": sum(item["bytes"] for item in moved),
            "reclaimed_disk_bytes": 0,
            "trash_path": str(destination),
            "recoverable": True,
            "status": "completed",
        }
        if db is not None:
            db.record_operation("backup_retention", result, key=operation)
    return result


def purge_backup_trash(
    settings: Any,
    older_than_days: int | None = None,
    dry_run: bool = True,
    db: Any = None,
) -> dict[str, Any]:
    from .maintenance import BackupIntegrityError, verify_backup
    from .reliability import workspace_write_lock

    root = _backup_root(settings)
    days = settings.retention.trash_days if older_than_days is None else older_than_days
    if days < 0:
        raise ValueError("backup trash retention days must be nonnegative")
    entries = []
    protected = {}
    reclaimed = 0
    with workspace_write_lock(root):
        trash = root / ".trash"
        if trash.is_symlink():
            raise ValueError("backup trash directory must not be a symbolic link")
        cutoff = datetime.now(UTC) - timedelta(days=days)
        for bundle in sorted(trash.iterdir()) if trash.is_dir() else []:
            try:
                if not bundle.is_dir() or bundle.is_symlink():
                    raise ValueError("unsafe backup trash bundle")
                value = json.loads((bundle / "tombstone.json").read_text(encoding="utf-8"))
                created = datetime.fromisoformat(value["created_at"])
                if created.tzinfo is None or created > cutoff:
                    protected[bundle.name] = "retention_period"
                    continue
                if value["operation_id"] != bundle.name or value["status"] != "completed":
                    raise ValueError("incomplete backup quarantine")
                expected = {"tombstone.json"}
                for item in value["entries"]:
                    name = item["path"]
                    if not re.fullmatch(r"backup-\d{8}T\d{6}-[0-9a-f]{12}", name):
                        raise ValueError("invalid quarantined backup identifier")
                    target = bundle / name
                    if target.is_symlink() or not target.is_dir():
                        raise ValueError("unsafe quarantined backup path")
                    verify_backup(target)
                    if _fingerprint(target) != {"bytes": item["bytes"], "sha256": item["sha256"]}:
                        raise ValueError("quarantined backup was modified")
                    expected.update(file.relative_to(bundle).as_posix() for file in _files(target))
                if expected != {file.relative_to(bundle).as_posix() for file in _files(bundle)}:
                    raise ValueError("backup trash contains unexpected files")
                entries.append(
                    {"operation_id": bundle.name, "bytes": _fingerprint(bundle)["bytes"]}
                )
            except (OSError, ValueError, KeyError, TypeError, BackupIntegrityError) as exc:
                protected[bundle.name] = str(exc)
        if not dry_run:
            for item in entries:
                shutil.rmtree(trash / item["operation_id"])
                reclaimed += item["bytes"]
        result = {
            "dry_run": dry_run,
            "entries": entries,
            "protected": protected,
            "estimated_bytes": sum(item["bytes"] for item in entries),
            "reclaimed_disk_bytes": reclaimed,
            "recoverable": dry_run,
        }
        if db is not None and not dry_run:
            db.record_operation("backup_trash_purge", result)
    return result
