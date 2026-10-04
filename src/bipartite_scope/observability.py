from __future__ import annotations

import json
import logging
import math
import os
import platform
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select

from .database import JobAttemptRecord, JobRecord, ServiceDatabase, WorkspaceRecord

_SECRET_KEY = re.compile(
    r"password|passwd|secret|token|authorization|cookie|credential|api[_-]?key", re.IGNORECASE
)
_URL_CREDENTIALS = re.compile(r"([a-z][a-z0-9+.-]*://)([^\s/@:]+):([^\s/@]+)@", re.IGNORECASE)
_PASSWORD_ASSIGNMENT = re.compile(
    r"((?:password|passwd|secret|token|api[_-]?key)[\"']?\s*[=:]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s&;,]+)",
    re.IGNORECASE,
)
_BEARER_TOKEN = re.compile(r"(\bBearer\s+)[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_STATUSES = {"queued", "running", "retrying", "cancelling", "succeeded", "failed", "cancelled"}
_LATENCY_BUCKETS = (0.05, 0.1, 0.5, 1.0, 5.0, 30.0)


def redact(value: Any, key: str = "") -> Any:
    if _SECRET_KEY.search(key):
        return "***"
    if isinstance(value, Mapping):
        return {str(name): redact(item, str(name)) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        value = _URL_CREDENTIALS.sub(r"\1\2:***@", value)
        value = _PASSWORD_ASSIGNMENT.sub(r"\1***", value)
        return _BEARER_TOKEN.sub(r"\1***", value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact(str(value))


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "fields", {})
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "component": record.name.rsplit(".", 1)[-1],
            "event": record.getMessage(),
            **{
                name: None
                for name in (
                    "request_id",
                    "workspace_id",
                    "job_id",
                    "attempt_id",
                    "snapshot_id",
                    "duration_ms",
                )
            },
            **{
                key: value
                for key, value in fields.items()
                if key not in {"timestamp", "level", "component", "event"}
            },
        }
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(redact(payload), ensure_ascii=True, separators=(",", ":"))


class _PrivateRotatingFileHandler(RotatingFileHandler):
    def _open(self):
        handle = super()._open()
        os.chmod(self.baseFilename, 0o600)
        return handle


def _runtime_directory(root: Path, name: str) -> Path:
    operations = root / ".operations"
    directory = operations / name
    if operations.is_symlink() or directory.is_symlink():
        raise ValueError("operations directories cannot be symbolic links")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory


def configure_logging(settings: Any, component: str = "service") -> logging.Logger:
    """Configure correlated JSON logs without writing credentials."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", component):
        raise ValueError("invalid logging component")
    directory = _runtime_directory(settings.workspace_root, "logs")
    path = directory / f"{component}-{os.getpid()}.log"
    if path.is_symlink():
        raise ValueError("log files cannot be symbolic links")
    options = settings.observability
    logger = logging.getLogger(f"bipartite_scope.{component}")
    logger.disabled = False
    logger.setLevel(options.log_level)
    logger.propagate = False
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = _PrivateRotatingFileHandler(
        path, maxBytes=options.log_max_bytes, backupCount=options.log_backups, encoding="utf-8"
    )
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
    os.chmod(path, 0o600)
    return logger


def correlated_log(logger: logging.Logger, level: int | str, event: str, **fields: Any) -> None:
    numeric_level = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if not isinstance(numeric_level, int):
        raise TypeError("invalid logging level")
    logger.log(numeric_level, redact(event), extra={"fields": redact(fields)})


def audit(
    db: ServiceDatabase,
    action: str,
    source: str = "api",
    workspace_id: str | None = None,
    job_id: str | None = None,
    details: Mapping[str, Any] | None = None,
    outcome: str = "succeeded",
) -> dict[str, Any]:
    """Persist a redacted operation record without inventing an authenticated actor."""
    if source not in {"api", "cli", "worker", "scheduler", "system"}:
        raise ValueError("unsupported audit source")
    return db.record_operation(
        "audit",
        redact(
            {
                "action": action,
                "source": source,
                "workspace_id": workspace_id,
                "job_id": job_id,
                "outcome": outcome,
                "details": dict(details or {}),
            }
        ),
    )


def _time(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    result = datetime.fromisoformat(value) if isinstance(value, str) else value
    return result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)


def _operations(db: ServiceDatabase, kind: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    while True:
        batch = db.list_operations(kind, limit=100, offset=len(records))
        records.extend(batch)
        if len(batch) < 100:
            return records


def _tree_bytes(path: Path) -> int:
    total = 0
    if not path.is_dir() or path.is_symlink():
        return total
    for directory, subdirectories, filenames in os.walk(path, followlinks=False):
        subdirectories[:] = [
            name for name in subdirectories if not (Path(directory) / name).is_symlink()
        ]
        for filename in filenames:
            candidate = Path(directory) / filename
            if candidate.is_symlink():
                continue
            try:
                total += candidate.stat().st_size
            except OSError:
                continue
    return total


def _disk_usage(root: Path) -> shutil._ntuple_diskusage:
    while not root.exists() and root != root.parent:
        root = root.parent
    return shutil.disk_usage(root)


def _operation_lock(db: ServiceDatabase, name: str):
    from .reliability import workspace_write_lock

    return workspace_write_lock(
        _runtime_directory(db.settings.workspace_root, name),
        allow_maintenance=True,
        blocking=True,
    )


def record_request(
    db: ServiceDatabase, method: str, route: str, status: int, duration_seconds: float
) -> None:
    """Aggregate canonical routes atomically with a bounded persistent series count."""
    if not math.isfinite(duration_seconds) or duration_seconds < 0:
        raise ValueError("request duration must be finite and nonnegative")
    method = (
        method.upper()
        if method.upper() in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
        else "OTHER"
    )
    if len(route) > 200 or not re.fullmatch(r"/[A-Za-z0-9/_{}.-]*", route):
        route = "/unmatched"
    status_family = f"{status // 100}xx" if 100 <= status < 600 else "other"
    key = f"{method}:{route}:{status_family}"
    with _operation_lock(db, "metrics"):
        existing = db.get_operation("http_metric", key)
        if existing is None and len(db.list_operations("http_metric", limit=101)) >= 100:
            route, method, status_family = "/unmatched", "OTHER", "other"
            key = "OTHER:/unmatched:other"
            existing = db.get_operation("http_metric", key)
        payload = (
            dict(existing["payload"])
            if existing
            else {
                "method": method,
                "route": route,
                "status": status_family,
                "count": 0,
                "duration_sum_seconds": 0.0,
                "buckets": {str(bound): 0 for bound in _LATENCY_BUCKETS},
            }
        )
        payload["count"] += 1
        payload["duration_sum_seconds"] = round(
            payload["duration_sum_seconds"] + duration_seconds, 6
        )
        payload["buckets"] = {
            str(bound): payload["buckets"].get(str(bound), 0) + int(duration_seconds <= bound)
            for bound in _LATENCY_BUCKETS
        }
        db.upsert_operation("http_metric", key, payload)


def operations_summary(db: ServiceDatabase) -> dict[str, Any]:
    """Read durable job totals and a bounded timing sample from the control plane."""
    now = datetime.now(UTC)
    with db.sessions() as session:
        counts = session.execute(
            select(JobRecord.kind, JobRecord.status, func.count(JobRecord.id)).group_by(
                JobRecord.kind, JobRecord.status
            )
        ).all()
        attempts = session.scalar(select(func.sum(JobRecord.attempt_count))) or 0
        retried = (
            session.scalar(select(func.count(JobRecord.id)).where(JobRecord.attempt_count > 1)) or 0
        )
        sample = session.execute(
            select(JobRecord.queued_at, JobRecord.started_at, JobRecord.completed_at)
            .order_by(JobRecord.created_at.desc(), JobRecord.id)
            .limit(1000)
        ).all()
        heartbeat_column = getattr(JobRecord, "heartbeat_at", JobRecord.updated_at)
        heartbeat = session.scalar(select(func.max(heartbeat_column)))
        workspace_count = session.scalar(select(func.count(WorkspaceRecord.id))) or 0
    statuses: Counter[str] = Counter()
    by_kind: dict[str, dict[str, int]] = {}
    for kind, status, count in counts:
        statuses[status] += count
        by_kind.setdefault(kind, {})[status] = count
    waits: list[float] = []
    durations: list[float] = []
    for queued_at, started_at, completed_at in sample:
        queued, started, completed = _time(queued_at), _time(started_at), _time(completed_at)
        if queued and started:
            waits.append(max(0.0, (started - queued).total_seconds()))
        if started and completed:
            durations.append(max(0.0, (completed - started).total_seconds()))
    root = db.settings.workspace_root
    storage = []
    for workspace in db.list_workspaces():
        path = Path(workspace["storage_path"]).resolve()
        if path == root or root not in path.parents:
            continue
        categories = {
            name: _tree_bytes(path / name) for name in ("data", "artifacts", "reports", "uploads")
        }
        storage.append(
            {"workspace_id": workspace["workspace_id"], "bytes": _tree_bytes(path), **categories}
        )
    backups = _operations(db, "backup")
    completed_backups = [
        record
        for record in backups
        if record["payload"].get("status") in {"completed", "succeeded"}
    ]
    last_backup = max(
        (
            _time(record["payload"].get("completed_at") or record["updated_at"])
            for record in completed_backups
        ),
        default=None,
    )
    alerts = _operations(db, "alert")
    requests = [record["payload"] for record in _operations(db, "http_metric")]
    verifications = db.list_operations("backup_verify", limit=1)
    drills = db.list_operations("restore_drill", limit=1)
    disk = _disk_usage(root)
    return {
        "generated_at": now.isoformat(),
        "database_available": db.health(),
        "workspace_count": workspace_count,
        "jobs": {
            "total": sum(statuses.values()),
            "by_status": {status: statuses.get(status, 0) for status in sorted(_STATUSES)},
            "by_kind": by_kind,
            "attempts": attempts,
            "retried_jobs": retried,
            "timing_sample_limit": 1000,
            "queue_wait_samples": len(waits),
            "queue_wait_mean_seconds": sum(waits) / len(waits) if waits else 0.0,
            "duration_samples": len(durations),
            "duration_mean_seconds": sum(durations) / len(durations) if durations else 0.0,
        },
        "last_worker_heartbeat_at": _time(heartbeat).isoformat() if heartbeat else None,
        "storage": {
            "workspaces": storage,
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
        },
        "backups": {
            "count": len(backups),
            "completed": len(completed_backups),
            "last_completed_at": last_backup.isoformat() if last_backup else None,
            "last_completed_age_seconds": max(0.0, (now - last_backup).total_seconds())
            if last_backup
            else None,
            "last_verification": verifications[0] if verifications else None,
            "last_restore_drill": drills[0] if drills else None,
        },
        "alerts": dict(Counter(record["payload"].get("status", "active") for record in alerts)),
        "http": {
            "requests": sum(record["count"] for record in requests),
            "errors": sum(
                record["count"] for record in requests if record["status"] in {"4xx", "5xx"}
            ),
            "duration_sum_seconds": sum(record["duration_sum_seconds"] for record in requests),
            "series": requests,
        },
    }


def prometheus_metrics(db: ServiceDatabase) -> str:
    summary = operations_summary(db)
    jobs = summary["jobs"]
    metrics = [
        "# TYPE bipartite_scope_jobs gauge",
        *[
            f'bipartite_scope_jobs{{status="{status}"}} {count}'
            for status, count in jobs["by_status"].items()
        ],
        "# TYPE bipartite_scope_job_attempts_total counter",
        f"bipartite_scope_job_attempts_total {jobs['attempts']}",
        "# TYPE bipartite_scope_queue_wait_mean_seconds gauge",
        f"bipartite_scope_queue_wait_mean_seconds {jobs['queue_wait_mean_seconds']:.6f}",
        "# TYPE bipartite_scope_job_duration_mean_seconds gauge",
        f"bipartite_scope_job_duration_mean_seconds {jobs['duration_mean_seconds']:.6f}",
        "# TYPE bipartite_scope_workspaces gauge",
        f"bipartite_scope_workspaces {summary['workspace_count']}",
        "# TYPE bipartite_scope_disk_free_bytes gauge",
        f"bipartite_scope_disk_free_bytes {summary['storage']['disk_free_bytes']}",
        "# TYPE bipartite_scope_workspace_storage_bytes gauge",
        f"bipartite_scope_workspace_storage_bytes {sum(item['bytes'] for item in summary['storage']['workspaces'])}",
        "# TYPE bipartite_scope_alerts gauge",
        *[
            f'bipartite_scope_alerts{{status="{status}"}} {summary["alerts"].get(status, 0)}'
            for status in ("active", "acknowledged", "resolved")
        ],
    ]
    metrics.extend(
        [
            "# TYPE bipartite_scope_http_requests_total counter",
            "# TYPE bipartite_scope_http_request_duration_seconds histogram",
        ]
    )
    for record in summary["http"]["series"]:
        labels = (
            f'method="{record["method"]}",route="{record["route"]}",status="{record["status"]}"'
        )
        metrics.append(f"bipartite_scope_http_requests_total{{{labels}}} {record['count']}")
        for bound, count in record["buckets"].items():
            metrics.append(
                f'bipartite_scope_http_request_duration_seconds_bucket{{{labels},le="{bound}"}} {count}'
            )
        metrics.extend(
            [
                f'bipartite_scope_http_request_duration_seconds_bucket{{{labels},le="+Inf"}} {record["count"]}',
                f"bipartite_scope_http_request_duration_seconds_count{{{labels}}} {record['count']}",
                f"bipartite_scope_http_request_duration_seconds_sum{{{labels}}} {record['duration_sum_seconds']:.6f}",
            ]
        )
    return "\n".join(metrics) + "\n"


def check_alerts(db: ServiceDatabase) -> list[dict[str, Any]]:
    """Persist deduplicated local conditions and resolve conditions that cleared."""
    with _operation_lock(db, "alerts"):
        return _check_alerts_locked(db)


def _check_alerts_locked(db: ServiceDatabase) -> list[dict[str, Any]]:
    now = datetime.now(UTC)
    options = db.settings.observability
    with db.sessions() as session:
        waiting = list(
            session.scalars(
                select(JobRecord.id).where(
                    JobRecord.status.in_(("queued", "retrying")),
                    JobRecord.queued_at < now - timedelta(seconds=options.queue_wait_seconds),
                )
            )
        )
        heartbeat = getattr(JobRecord, "heartbeat_at", JobRecord.updated_at)
        stale = list(
            session.scalars(
                select(JobRecord.id).where(
                    JobRecord.status.in_(("running", "cancelling")),
                    func.coalesce(heartbeat, JobRecord.started_at, JobRecord.updated_at)
                    < now - timedelta(seconds=db.settings.worker.stale_job_seconds),
                )
            )
        )
        failures = (
            session.scalar(
                select(func.count(JobRecord.id)).where(
                    JobRecord.status == "failed", JobRecord.updated_at >= now - timedelta(hours=1)
                )
            )
            or 0
        )
    summary = operations_summary(db)
    backup_age = summary["backups"]["last_completed_age_seconds"]
    min_free = db.settings.resources.min_free_bytes
    findings = {
        "queue_wait": (
            bool(waiting),
            "warning",
            "Tasks exceeded the queue wait threshold",
            {"job_ids": waiting[:20], "count": len(waiting)},
        ),
        "worker_stale": (
            bool(stale),
            "error",
            "Running tasks have stale worker heartbeats",
            {"job_ids": stale[:20], "count": len(stale)},
        ),
        "repeated_failures": (
            failures >= options.alert_failure_count,
            "error",
            "Recent task failures exceeded the threshold",
            {"count": failures, "window_seconds": 3600},
        ),
        "disk_low": (
            summary["storage"]["disk_free_bytes"] < min_free,
            "error",
            "Available disk space is below the configured minimum",
            {"free_bytes": summary["storage"]["disk_free_bytes"], "minimum_bytes": min_free},
        ),
        "backup_overdue": (
            backup_age is None or backup_age > options.backup_max_age_seconds,
            "warning",
            "No recent verified backup is available",
            {"age_seconds": backup_age, "maximum_age_seconds": options.backup_max_age_seconds},
        ),
    }
    for category, summary_field, message in (
        (
            "backup_verification_failed",
            "last_verification",
            "The latest backup verification failed",
        ),
        ("restore_drill_failed", "last_restore_drill", "The latest restore drill failed"),
    ):
        record = summary["backups"][summary_field]
        payload = record["payload"] if record else {}
        findings[category] = (
            payload.get("status") == "failed",
            "error",
            message,
            redact(
                {
                    "operation_id": record["id"] if record else None,
                    "backup_id": payload.get("backup_id"),
                    "error": payload.get("error"),
                }
            ),
        )
    for key, (present, severity, message, details) in findings.items():
        existing = db.get_operation("alert", key)
        payload = dict(existing["payload"]) if existing else {}
        if present:
            reopened = not existing or payload.get("status") == "resolved"
            payload.update(
                {
                    "condition": key,
                    "severity": severity,
                    "message": message,
                    "details": details,
                    "last_seen_at": now.isoformat(),
                }
            )
            if reopened:
                payload.update(
                    {
                        "status": "active",
                        "first_seen_at": now.isoformat(),
                        "resolved_at": None,
                        "acknowledged_at": None,
                        "occurrences": int(payload.get("occurrences", 0)) + 1,
                    }
                )
            db.upsert_operation("alert", key, payload)
        elif existing and payload.get("status") != "resolved":
            payload.update(
                {"status": "resolved", "resolved_at": now.isoformat(), "details": details}
            )
            db.upsert_operation("alert", key, payload)
    return _operations(db, "alert")


def acknowledge_alert(db: ServiceDatabase, alert_id: str) -> dict[str, Any]:
    with _operation_lock(db, "alerts"):
        return _acknowledge_alert_locked(db, alert_id)


def _acknowledge_alert_locked(db: ServiceDatabase, alert_id: str) -> dict[str, Any]:
    existing = db.get_operation("alert", alert_id)
    if existing is None:
        existing = next(
            (record for record in _operations(db, "alert") if record["id"] == alert_id), None
        )
    if existing is None:
        raise ValueError("alert does not exist")
    payload = dict(existing["payload"])
    if payload.get("status") == "resolved":
        raise ValueError("a resolved alert cannot be acknowledged")
    payload.update({"status": "acknowledged", "acknowledged_at": datetime.now(UTC).isoformat()})
    result = db.upsert_operation("alert", existing["key"], payload)
    audit(db, "alert.acknowledge", details={"alert_id": result["id"]})
    return result


def _log_excerpt(root: Path, job_id: str | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    directory = root / ".operations" / "logs"
    if not directory.is_dir() or directory.is_symlink() or directory.parent.is_symlink():
        return records
    for path in sorted(directory.glob("*.log"))[:8]:
        if path.is_symlink() or not path.is_file():
            continue
        with path.open("rb") as handle:
            length = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, length - 256 * 1024))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
        for line in lines[-1000:]:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and (job_id is None or value.get("job_id") == job_id):
                records.append(redact(value))
    return records[-1000:]


def export_diagnostics(db: ServiceDatabase, job_id: str | None = None) -> Path:
    """Export bounded, redacted state and history to a private local JSON file."""
    jobs = [db.get_job(job_id)] if job_id else db.list_jobs(limit=100)
    identifiers = [job["job_id"] for job in jobs]
    with db.sessions() as session:
        attempts = session.scalars(
            select(JobAttemptRecord)
            .where(JobAttemptRecord.job_id.in_(identifiers))
            .order_by(JobAttemptRecord.started_at, JobAttemptRecord.id)
        )
        attempt_history = [
            {
                "attempt_id": item.id,
                "job_id": item.job_id,
                "attempt_number": item.attempt_number,
                "worker_name": item.worker_name,
                "status": item.status,
                "started_at": _time(item.started_at).isoformat(),
                "completed_at": _time(item.completed_at).isoformat() if item.completed_at else None,
                "error_code": item.error_code,
                "error_message": item.error_message,
            }
            for item in attempts
        ]
    payload = redact(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "python_version": platform.python_version(),
            "configuration": db.settings.public_dict(),
            "summary": operations_summary(db),
            "jobs": jobs,
            "attempts": attempt_history,
            "events": {identifier: db.events_after(identifier) for identifier in identifiers},
            "alerts": _operations(db, "alert"),
            "logs": _log_excerpt(db.settings.workspace_root, job_id),
        }
    )
    directory = _runtime_directory(db.settings.workspace_root, "diagnostics")
    path = directory / f"diagnostic-{uuid4().hex}.json"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".diagnostic-", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    audit(db, "diagnostic.export", job_id=job_id, details={"filename": path.name})
    return path
