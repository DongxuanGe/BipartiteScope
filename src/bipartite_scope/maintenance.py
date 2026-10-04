from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, closing, contextmanager
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from .core import QueryEngine
from .storage import CONFIG_FILENAME, SnapshotStore, load_workspace, normalize_event

_SKIP_NAMES = {".env", ".trash", ".operations", "__pycache__", "logs", "tmp", "temp"}
_ROOT_FILES = {CONFIG_FILENAME, "state.sqlite3"}
_ROOT_DIRS = {"data", "artifacts", "exports", "reports", "uploads"}
_MAX_RESTORE_BYTES = 1024**4


class BackupIntegrityError(ValueError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or any(part in {".", ".."} for part in value.split("/"))
        or str(path) != value
    ):
        raise BackupIntegrityError("backup contains an unsafe relative path")
    return path


def _backup_root(settings: Any, source: Path) -> Path:
    configured = getattr(getattr(settings, "backup", None), "root", None)
    service_root = settings.workspace_root if settings is not None else source
    target = (
        Path(configured).expanduser().resolve()
        if configured
        else service_root.parent / f"{service_root.name}-backups"
    )
    if (
        target == source
        or source in target.parents
        or target == service_root
        or service_root in target.parents
    ):
        raise ValueError("backup root must be outside the source directory")
    return target


def _option(settings: Any, name: str, default: Any) -> Any:
    return getattr(getattr(settings, "backup", None), name, default)


def _bundle_locked(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(path: str | Path, *args: Any, **kwargs: Any) -> Any:
        from .reliability import workspace_write_lock

        root = Path(path).expanduser().resolve().parent
        for ancestor in (root, *root.parents):
            if ancestor.name == ".trash":
                root = ancestor.parent
                break
        with workspace_write_lock(root, blocking=True):
            return function(path, *args, **kwargs)

    return wrapped


@contextmanager
def service_maintenance(
    settings: Any, database: Any = None, *, exclude_job_id: str | None = None
) -> Iterator[dict[str, Any]]:
    """Fence service writes and wait for existing execution before taking writer locks."""
    from .reliability import workspace_write_lock

    root = settings.workspace_root
    operations = root / ".operations"
    operations.mkdir(parents=True, exist_ok=True)
    marker = operations / "maintenance.json"
    owned_database = False
    if database is None:
        from .database import ServiceDatabase

        database = ServiceDatabase(settings)
        owned_database = True
    with (operations / "maintenance.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        marker_id = str(uuid4())
        try:
            with workspace_write_lock(operations, allow_maintenance=True, blocking=True):
                if marker.exists():
                    raise RuntimeError("service is already in maintenance mode")
                _json(marker, {"id": marker_id, "created_at": _now(), "reason": "backup"})
            deadline = time.monotonic() + _option(settings, "wait_seconds", 60)
            from sqlalchemy import func, select

            from .database import JobRecord

            while True:
                with database.sessions() as session:
                    statement = select(func.count(JobRecord.id)).where(
                        JobRecord.status.in_(("running", "cancelling"))
                    )
                    if exclude_job_id:
                        statement = statement.where(JobRecord.id != exclude_job_id)
                    running = session.scalar(statement)
                if not running:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("running jobs did not finish before maintenance timeout")
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            workspaces = _service_workspaces(root, database)
            with ExitStack() as stack:
                for workspace in workspaces:
                    stack.enter_context(
                        workspace_write_lock(workspace["source_path"], allow_maintenance=True)
                    )
                yield {"id": marker_id, "workspaces": workspaces}
        finally:
            with workspace_write_lock(operations, allow_maintenance=True, blocking=True):
                if marker.is_file():
                    try:
                        if json.loads(marker.read_text(encoding="utf-8")).get("id") == marker_id:
                            marker.unlink()
                    except (OSError, ValueError):
                        pass
            fcntl.flock(handle, fcntl.LOCK_UN)
            if owned_database:
                database.dispose()


def _service_workspaces(root: Path, database: Any) -> list[dict[str, str]]:
    entries: dict[Path, str] = {}
    for record in database.list_workspaces():
        path = Path(record["storage_path"]).resolve()
        if path != root and root not in path.parents:
            raise ValueError("registered workspace escapes the service root")
        entries[path] = record["workspace_id"]
    for candidate in root.rglob(CONFIG_FILENAME):
        if not candidate.is_symlink() and not any(
            part.startswith(".") for part in candidate.relative_to(root).parts
        ):
            entries.setdefault(candidate.parent, candidate.parent.name)
    output = []
    for source in sorted(entries):
        load_workspace(source)
        relative = source.relative_to(root).as_posix()
        if relative == ".":
            relative = ""
        output.append(
            {"key": entries[source], "source_path": str(source), "relative_path": relative}
        )
    return output


def _copy_sqlite(source: Path, target: Path) -> None:
    with (
        closing(sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)) as original,
        closing(sqlite3.connect(target)) as destination,
    ):
        original.backup(destination)
        destination.execute("PRAGMA journal_mode=DELETE")
        if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise BackupIntegrityError("SQLite integrity verification failed")


def _copy_workspace(source: Path, target: Path, settings: Any) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.iterdir()):
        if child.is_symlink():
            continue
        if child.is_file() and child.name in _ROOT_FILES:
            if child.name == "state.sqlite3":
                _copy_sqlite(child, target / child.name)
            else:
                shutil.copy2(child, target / child.name)
        elif child.is_dir() and child.name in _ROOT_DIRS:
            if child.name == "reports" and not _option(settings, "include_reports", True):
                continue
            if child.name == "uploads" and not _option(settings, "include_uploads", True):
                continue
            for directory, names, filenames in os.walk(child, followlinks=False):
                current = Path(directory)
                names[:] = [
                    name
                    for name in names
                    if name not in _SKIP_NAMES
                    and not name.startswith(".")
                    and not (current / name).is_symlink()
                    and (name != "uploads" or _option(settings, "include_uploads", True))
                ]
                destination = target / current.relative_to(source)
                destination.mkdir(parents=True, exist_ok=True)
                for name in sorted(filenames):
                    original = current / name
                    if (
                        original.is_symlink()
                        or name in _SKIP_NAMES
                        or name.startswith(".")
                        or name.endswith((".lock", "-wal", "-shm", ".tmp"))
                    ):
                        continue
                    shutil.copy2(original, destination / name)


def _pg_environment(url: Any) -> dict[str, str]:
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith("PG"):
            environment.pop(key)
    environment["PGCONNECT_TIMEOUT"] = "10"
    for name, value in (
        ("PGHOST", url.host),
        ("PGPORT", url.port),
        ("PGUSER", url.username),
        ("PGPASSWORD", url.password),
        ("PGDATABASE", url.database),
    ):
        if value is not None:
            environment[name] = str(value)
    for key in ("sslmode", "sslrootcert", "sslcert", "sslkey"):
        if key in url.query:
            environment["PG" + key.upper()] = str(url.query[key])
    return environment


def _pg_run(command: list[str], url: Any) -> None:
    try:
        result = subprocess.run(
            command,
            env=_pg_environment(url),
            capture_output=True,
            check=False,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("PostgreSQL backup tools are unavailable or timed out") from exc
    if result.returncode:
        raise RuntimeError("PostgreSQL backup tool failed; verify database access and tool version")


def _export_database(database: Any, target: Path) -> dict[str, str]:
    url = database.engine.url
    if database.engine.dialect.name == "sqlite":
        connection = database.engine.raw_connection()
        try:
            with closing(sqlite3.connect(target / "service.sqlite3")) as destination:
                connection.driver_connection.backup(destination)
                destination.execute("PRAGMA journal_mode=DELETE")
                if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise BackupIntegrityError("control database integrity verification failed")
        finally:
            connection.close()
        return {"kind": "sqlite", "path": "control/service.sqlite3"}
    if database.engine.dialect.name == "postgresql":
        _pg_run(
            [
                "pg_dump",
                "--format=custom",
                "--no-owner",
                "--no-acl",
                "--file",
                str(target / "service.dump"),
            ],
            url,
        )
        return {"kind": "postgresql", "path": "control/service.dump"}
    raise ValueError("unsupported control database backend")


def _workspace_checks(root: Path, *, query: bool = False) -> dict[str, Any]:
    workspace = load_workspace(root)
    store = SnapshotStore(workspace.artifacts)
    snapshots = store.list()
    for snapshot in snapshots:
        store.verify(snapshot["snapshot_id"])
    active = store.latest_id() if (workspace.artifacts / "latest.json").exists() else None
    ledger: dict[str, Any] = {}
    ledger_path = workspace.data / "events.jsonl"
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                event = normalize_event(payload)
                if event.event_id in ledger and ledger[event.event_id] != payload:
                    raise BackupIntegrityError("event ledger contains conflicting identifiers")
                ledger[event.event_id] = payload
    index: dict[str, Any] = {}
    pending = 0
    if workspace.state_database.exists():
        with closing(
            sqlite3.connect(f"file:{workspace.state_database.as_posix()}?mode=ro", uri=True)
        ) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise BackupIntegrityError("workspace event index failed integrity verification")
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "events" in tables:
                for event_id, payload, applied in connection.execute(
                    "SELECT event_id,payload,applied_snapshot FROM events"
                ):
                    index[event_id] = json.loads(payload)
                    pending += applied is None
    if ledger != index:
        raise BackupIntegrityError("event ledger and idempotency index disagree")
    feedback = workspace.data / "feedback.jsonl"
    if feedback.exists():
        for line in feedback.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                event = normalize_event(value)
                if ledger.get(event.event_id) != value:
                    raise BackupIntegrityError(
                        "feedback is absent from the normalized event ledger"
                    )
    result: dict[str, Any] = {
        "snapshots": len(snapshots),
        "active_snapshot_id": active,
        "events": len(index),
        "pending_events": pending,
    }
    if query and snapshots:
        snapshot = store.load(active or snapshots[0]["snapshot_id"])
        if snapshot.graph.u_ids:
            community = QueryEngine(snapshot).search(
                snapshot.graph.u_ids[0], workspace.query_config()
            )
            result["representative_query"] = {
                "user_id": community.query_entity,
                "members": len(community.members),
                "snapshot_id": snapshot.snapshot_id,
            }
    return result


def _create_backup(
    workspace_or_root: str | Path,
    settings: Any = None,
    database: Any = None,
    scope: str = "workspace",
    *,
    exclude_job_id: str | None = None,
) -> dict[str, Any]:
    """Create a verified immutable directory bundle under a writer fence."""
    from .reliability import maintenance_mode

    source = Path(workspace_or_root).expanduser().resolve()
    if scope not in {"workspace", "service"}:
        raise ValueError("backup scope must be workspace or service")
    if scope == "service" and (settings is None or source != settings.workspace_root):
        raise ValueError("service backup requires the configured workspace root")
    backup_root = _backup_root(settings, source)
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_id = f"backup-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:12]}"
    stage = Path(tempfile.mkdtemp(prefix=f".{backup_id}-", dir=backup_root))
    target = backup_root / backup_id
    owned_database = False
    if scope == "service" and database is None:
        from .database import ServiceDatabase

        database = ServiceDatabase(settings)
        owned_database = True
    try:
        fence = (
            service_maintenance(settings, database, exclude_job_id=exclude_job_id)
            if scope == "service"
            else maintenance_mode(source)
        )
        with fence as maintenance:
            if scope == "service":
                workspaces = maintenance["workspaces"]
            else:
                load_workspace(source)
                workspaces = [{"key": source.name, "source_path": str(source), "relative_path": ""}]
            records = []
            for index, entry in enumerate(workspaces):
                prefix = f"workspaces/{index}"
                _copy_workspace(Path(entry["source_path"]), stage / prefix, settings)
                checks = _workspace_checks(stage / prefix)
                records.append({**entry, "payload_prefix": prefix, **checks})
            control = None
            if scope == "service":
                (stage / "control").mkdir()
                control = _export_database(database, stage / "control")
            files = [
                {
                    "path": path.relative_to(stage).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _digest(path),
                }
                for path in sorted(stage.rglob("*"))
                if path.is_file()
            ]
            manifest = {
                "backup_id": backup_id,
                "schema_version": 4,
                "application_version": "4.0.0",
                "created_at": _now(),
                "scope": scope,
                "source_root": str(source),
                "workspaces": records,
                "control_database": control,
                "files": files,
                "size_bytes": sum(entry["size"] for entry in files),
                "include_uploads": _option(settings, "include_uploads", True),
                "include_reports": _option(settings, "include_reports", True),
            }
            _json(stage / "manifest.json", manifest)
            verify_backup(stage)
            os.replace(stage, target)
        result = {
            "backup_id": backup_id,
            "scope": scope,
            "status": "succeeded",
            "path": str(target),
            "created_at": manifest["created_at"],
            "completed_at": _now(),
            "size_bytes": manifest["size_bytes"],
            "workspace_count": len(records),
            "verified": True,
        }
        if database is not None and hasattr(database, "record_operation"):
            database.record_operation(kind="backup", payload=result, key=backup_id)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if owned_database:
            database.dispose()


def list_backups(settings: Any) -> list[dict[str, Any]]:
    root = _backup_root(settings, settings.workspace_root)
    if not root.exists():
        return []
    result = []
    for path in sorted(root.iterdir(), reverse=True):
        if path.is_dir() and not path.is_symlink() and not path.name.startswith("."):
            try:
                manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                result.append(
                    {
                        "backup_id": manifest["backup_id"],
                        "path": str(path),
                        "scope": manifest["scope"],
                        "created_at": manifest["created_at"],
                        "size_bytes": manifest["size_bytes"],
                    }
                )
            except (OSError, ValueError, KeyError):
                continue
    return result


@_bundle_locked
def verify_backup(path: str | Path) -> dict[str, Any]:
    if Path(path).expanduser().is_symlink():
        raise BackupIntegrityError("backup root must not be a symbolic link")
    root = Path(path).expanduser().resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BackupIntegrityError("backup manifest is missing or invalid") from exc
    if manifest.get("schema_version") != 4 or manifest.get("scope") not in {"workspace", "service"}:
        raise BackupIntegrityError("unsupported backup schema or scope")
    if not isinstance(manifest.get("files"), list) or not isinstance(
        manifest.get("workspaces"), list
    ):
        raise BackupIntegrityError("backup manifest omits its file or workspace inventory")
    expected = set()
    total = 0
    for entry in manifest["files"]:
        try:
            relative = str(_relative(entry["path"]))
            size = entry["size"]
            checksum = entry["sha256"]
        except (TypeError, KeyError) as exc:
            raise BackupIntegrityError("invalid backup file inventory") from exc
        asset = root / relative
        if (
            relative in expected
            or relative == "manifest.json"
            or not isinstance(size, int)
            or size < 0
        ):
            raise BackupIntegrityError("invalid or duplicate backup file inventory")
        expected.add(relative)
        if (
            asset.is_symlink()
            or not asset.is_file()
            or any(parent.is_symlink() for parent in asset.parents if parent != root.parent)
        ):
            raise BackupIntegrityError("backup asset is missing or symbolic")
        if asset.stat().st_size != size or _digest(asset) != checksum:
            raise BackupIntegrityError(f"backup asset failed integrity verification: {relative}")
        total += size
    actual = set()
    for asset in root.rglob("*"):
        if asset.is_symlink():
            raise BackupIntegrityError("backup contains symbolic links")
        if asset.is_file() and asset != root / "manifest.json":
            actual.add(asset.relative_to(root).as_posix())
    if actual != expected or total != manifest.get("size_bytes"):
        raise BackupIntegrityError("backup file allowlist or total size does not match")
    relative_roots = []
    checks = []
    if manifest["scope"] == "workspace" and len(manifest["workspaces"]) != 1:
        raise BackupIntegrityError("workspace backup must contain exactly one workspace")
    for entry in manifest["workspaces"]:
        prefix = str(_relative(entry["payload_prefix"]))
        relative = entry.get("relative_path", "")
        if relative:
            _relative(relative)
        if manifest["scope"] == "workspace" and relative:
            raise BackupIntegrityError("workspace backup destination must be its target root")
        if any(
            relative == previous
            or not relative
            or not previous
            or PurePosixPath(relative) in PurePosixPath(previous).parents
            or PurePosixPath(previous) in PurePosixPath(relative).parents
            for previous in relative_roots
        ):
            raise BackupIntegrityError("backup contains overlapping workspace destinations")
        relative_roots.append(relative)
        workspace_check = _workspace_checks(root / prefix)
        if any(
            workspace_check.get(name) != entry.get(name)
            for name in ("snapshots", "active_snapshot_id", "events", "pending_events")
        ):
            raise BackupIntegrityError("workspace state disagrees with the backup manifest")
        checks.append(workspace_check)
    control = manifest.get("control_database")
    if manifest["scope"] == "service":
        if not control or control.get("kind") not in {"sqlite", "postgresql"}:
            raise BackupIntegrityError("service backup has no supported control database")
        control_path = str(_relative(control["path"]))
        if control_path not in expected:
            raise BackupIntegrityError("control database is absent from the backup inventory")
        if control["kind"] == "sqlite":
            with closing(
                sqlite3.connect(f"file:{(root / control_path).as_posix()}?mode=ro", uri=True)
            ) as connection:
                if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise BackupIntegrityError("control database integrity verification failed")
    return {
        "backup_id": manifest["backup_id"],
        "verified": True,
        "scope": manifest["scope"],
        "file_count": len(expected),
        "size_bytes": total,
        "workspaces": checks,
    }


def _rewrite(
    value: Any, source_root: str, target_root: str, workspaces: list[dict[str, Any]]
) -> Any:
    if isinstance(value, dict):
        return {
            key: _rewrite(item, source_root, target_root, workspaces) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite(item, source_root, target_root, workspaces) for item in value]
    if isinstance(value, str):
        for workspace in sorted(
            workspaces, key=lambda entry: len(entry["source_path"]), reverse=True
        ):
            original = workspace["source_path"]
            destination = str(Path(target_root) / workspace["relative_path"])
            if value == original or value.startswith(original + os.sep):
                return destination + value[len(original) :]
        if value == source_root or value.startswith(source_root + os.sep):
            return target_root + value[len(source_root) :]
    return value


def _reconcile_database(engine: Any, manifest: Mapping[str, Any], target: Path) -> int:
    from sqlalchemy import JSON, MetaData, delete, select, update

    metadata = MetaData()
    metadata.reflect(engine)
    interrupted = 0
    now = datetime.now(UTC)
    with engine.begin() as connection:
        for name, table in metadata.tables.items():
            json_columns = [column for column in table.columns if isinstance(column.type, JSON)]
            paths = [column for column in table.columns if column.name == "storage_path"]
            if json_columns or paths:
                primary = list(table.primary_key.columns)
                if not primary:
                    continue
                for row in connection.execute(select(table)).mappings():
                    changed = {}
                    for column in (*json_columns, *paths):
                        value = row[column.name]
                        updated = _rewrite(
                            value, manifest["source_root"], str(target), manifest["workspaces"]
                        )
                        if updated != value:
                            changed[column.name] = updated
                    if changed:
                        if name == "jobs":
                            for payload, digest in (
                                ("input_payload", "input_hash"),
                                ("configuration_snapshot", "configuration_hash"),
                            ):
                                if payload in changed and digest in table.columns:
                                    changed[digest] = hashlib.sha256(
                                        json.dumps(
                                            changed[payload],
                                            sort_keys=True,
                                            separators=(",", ":"),
                                            default=str,
                                        ).encode()
                                    ).hexdigest()
                        connection.execute(
                            update(table)
                            .where(*(column == row[column.name] for column in primary))
                            .values(**changed)
                        )
            if name == "jobs":
                columns = set(table.columns.keys())
                interrupted = connection.execute(
                    select(table.c.id).where(
                        table.c.status.not_in(("succeeded", "failed", "cancelled"))
                    )
                ).fetchall()
                values = {
                    "status": "failed",
                    "stage": "restore_interrupted",
                    "error_code": "restore_interrupted",
                    "error_message": "Execution was interrupted by backup restoration; explicit resubmission is required.",
                    "completed_at": now,
                    "updated_at": now,
                    "published_at": None,
                    "celery_task_id": None,
                    "heartbeat_at": None,
                }
                connection.execute(
                    update(table)
                    .where(table.c.status.not_in(("succeeded", "failed", "cancelled")))
                    .values(**{key: value for key, value in values.items() if key in columns})
                )
                if "execution_generation" in columns:
                    connection.execute(
                        update(table).values(execution_generation=table.c.execution_generation + 1)
                    )
            elif name == "job_attempts":
                connection.execute(
                    update(table)
                    .where(table.c.status.in_(("running", "retrying", "cancelling")))
                    .values(
                        status="failed",
                        completed_at=now,
                        error_code="restore_interrupted",
                        error_message="Execution was interrupted by backup restoration.",
                    )
                )
            elif name == "workspace_leases":
                connection.execute(delete(table))
    return len(interrupted) if isinstance(interrupted, list) else interrupted


def _restore_database(bundle: Path, manifest: dict[str, Any], url_value: str, target: Path) -> int:
    from sqlalchemy import create_engine, inspect
    from sqlalchemy.engine import make_url

    url = make_url(url_value)
    kind = manifest["control_database"]["kind"]
    if (kind == "sqlite") != url.drivername.startswith("sqlite"):
        raise ValueError("restore requires the same control database backend")
    if kind == "sqlite":
        if not url.database or url.database == ":memory:":
            raise ValueError("restore requires a new file-backed control database")
        destination = Path(url.database).expanduser().resolve()
        if Path(url.database).expanduser().is_symlink():
            raise ValueError("restore database must not be a symbolic link")
        for protected in (Path(manifest["source_root"]).resolve(), bundle, target):
            if destination == protected or protected in destination.parents:
                raise ValueError(
                    "restore database must be outside source, backup, and target workspace directories"
                )
        if destination.exists() and destination.stat().st_size:
            engine = create_engine(url)
            try:
                if inspect(engine).get_table_names():
                    raise FileExistsError("restore control database is not empty")
            finally:
                engine.dispose()
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".restore-database-", dir=destination.parent
        )
        os.close(descriptor)
        staged = Path(temporary)
        try:
            _copy_sqlite(bundle / manifest["control_database"]["path"], staged)
            engine = create_engine(url.set(database=str(staged)))
            try:
                interrupted = _reconcile_database(engine, manifest, target)
            finally:
                engine.dispose()
            os.replace(staged, destination)
            return interrupted
        finally:
            if staged.exists():
                staged.unlink()
    engine = create_engine(url)
    try:
        inspector = inspect(engine)
        for schema in inspector.get_schema_names():
            if schema == "information_schema" or schema.startswith("pg_"):
                continue
            if inspector.get_table_names(schema=schema) or inspector.get_view_names(schema=schema):
                raise FileExistsError("restore control database is not empty")
        _pg_run(
            [
                "pg_restore",
                "--dbname",
                url.database or "",
                "--no-owner",
                "--no-acl",
                "--single-transaction",
                "--exit-on-error",
                str(bundle / manifest["control_database"]["path"]),
            ],
            url,
        )
        return _reconcile_database(engine, manifest, target)
    finally:
        engine.dispose()


@_bundle_locked
def restore_backup(
    path: str | Path,
    target_root: str | Path,
    database_url: str | None = None,
    *,
    max_restore_bytes: int = _MAX_RESTORE_BYTES,
) -> dict[str, Any]:
    """Restore into a new empty target; historical executions are never replayed."""
    verified = verify_backup(path)
    bundle = Path(path).expanduser().resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if Path(target_root).expanduser().is_symlink():
        raise ValueError("restore target must not be a symbolic link")
    target = Path(target_root).expanduser().resolve()
    source = Path(manifest["source_root"]).resolve()
    if target == source or source in target.parents or target in source.parents:
        raise ValueError("restore target overlaps the original source")
    if target == bundle or bundle in target.parents or target in bundle.parents:
        raise ValueError("restore target overlaps the backup bundle")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError("restore target must be new or empty")
    if verified["size_bytes"] > max_restore_bytes:
        raise ValueError("backup exceeds the configured restoration size limit")
    if manifest["scope"] == "service" and not database_url:
        raise ValueError("service restore requires an empty target database URL")
    if manifest["scope"] == "workspace" and database_url:
        raise ValueError("workspace restore does not include a control database")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".restore-workspaces-", dir=target.parent))
    try:
        for entry in manifest["workspaces"]:
            destination = stage / entry["relative_path"]
            shutil.copytree(bundle / entry["payload_prefix"], destination, dirs_exist_ok=True)
            _workspace_checks(destination)
        interrupted = (
            _restore_database(bundle, manifest, database_url, target) if database_url else 0
        )
        if target.exists():
            target.rmdir()
        os.replace(stage, target)
        return {
            "backup_id": manifest["backup_id"],
            "restored": True,
            "target_root": str(target),
            "workspace_count": len(manifest["workspaces"]),
            "interrupted_jobs": interrupted,
            "automatic_task_replay": False,
        }
    finally:
        if stage.exists():
            shutil.rmtree(stage)


@_bundle_locked
def drill_backup(path: str | Path, *, database_url: str | None = None) -> dict[str, Any]:
    verified = verify_backup(path)
    bundle = Path(path).expanduser().resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("control_database") or {}).get("kind") == "postgresql" and not database_url:
        raise ValueError("PostgreSQL service drills require an explicit empty target database")
    with tempfile.TemporaryDirectory(prefix="bipartite-scope-drill-") as temporary:
        root = Path(temporary)
        target_database = database_url
        if manifest["scope"] == "service" and not target_database:
            target_database = f"sqlite+pysqlite:///{root / 'service.sqlite3'}"
        restored = restore_backup(bundle, root / "workspaces", target_database)
        checks = [
            _workspace_checks(root / "workspaces" / entry["relative_path"], query=True)
            for entry in manifest["workspaces"]
        ]
    return {
        "backup_id": verified["backup_id"],
        "status": "succeeded",
        "verified": True,
        "restored": restored["restored"],
        "workspaces": checks,
        "completed_at": _now(),
    }


def create_backup(
    workspace_or_root: str | Path,
    settings: Any = None,
    database: Any = None,
    scope: str = "workspace",
    *,
    exclude_job_id: str | None = None,
) -> dict[str, Any]:
    from .reliability import workspace_write_lock

    source = Path(workspace_or_root).expanduser().resolve()
    with workspace_write_lock(_backup_root(settings, source), blocking=True):
        return _create_backup(source, settings, database, scope, exclude_job_id=exclude_job_id)
