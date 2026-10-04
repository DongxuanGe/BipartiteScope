from __future__ import annotations

import hashlib
import json
import re
import sysconfig
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from .config import ServiceSettings
from .reliability import ExecutionLostError, service_transaction

JOB_KINDS = frozenset(
    {
        "validate",
        "build",
        "update",
        "evaluate",
        "snapshot_verify",
        "benchmark",
        "backup",
        "backup_verify",
        "retention",
        "diagnostics",
    }
)
HEAVY_JOB_KINDS = frozenset({"build", "update", "evaluate", "benchmark", "backup"})
JOB_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
MUTATING_JOB_KINDS = frozenset({"build", "update"})
_TRANSITIONS = {
    "queued": {"running", "cancelling", "cancelled", "failed"},
    "running": {"succeeded", "failed", "retrying", "cancelling", "cancelled"},
    "retrying": {"queued", "running", "failed", "cancelling", "cancelled"},
    "cancelling": {"cancelled", "failed"},
    "succeeded": set(),
    "failed": {"retrying"},
    "cancelled": {"retrying"},
}


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class WorkspaceRecord(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    storage_path: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ready")
    last_observed_snapshot_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class JobRecord(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id", "kind", "idempotency_key", name="uq_jobs_workspace_kind_key"
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="api")
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stage: Mapped[str] = mapped_column(String(64), nullable=False, default="queued")
    idempotency_key: Mapped[str | None] = mapped_column(String(255))
    celery_task_id: Mapped[str | None] = mapped_column(String(255))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    dispatch_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    configuration_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    configuration_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    result_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    queued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    execution_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class JobAttemptRecord(Base):
    __tablename__ = "job_attempts"
    __table_args__ = (UniqueConstraint("job_id", "attempt_number", name="uq_attempt_job_number"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    worker_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)


class JobEventRecord(Base):
    __tablename__ = "job_events"
    __table_args__ = (UniqueConstraint("job_id", "sequence", name="uq_event_job_sequence"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    progress: Mapped[float] = mapped_column(Float, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WorkspaceLeaseRecord(Base):
    __tablename__ = "workspace_leases"

    workspace_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ServiceDatabaseError(RuntimeError):
    pass


class OperationRecord(Base):
    __tablename__ = "operations"
    __table_args__ = (UniqueConstraint("kind", "key", name="uq_operation_kind_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    key: Mapped[str | None] = mapped_column(String(255))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class JobNotFoundError(ServiceDatabaseError):
    pass


class WorkspaceNotFoundError(ServiceDatabaseError):
    pass


class JobStateError(ServiceDatabaseError):
    pass


class _ExecutionConflict(JobStateError, ExecutionLostError):
    pass


class IdempotencyConflictError(ServiceDatabaseError):
    pass


def upgrade_database(settings: ServiceSettings) -> str:
    try:
        from alembic import command
        from alembic.config import Config
    except ImportError as exc:
        raise RuntimeError("database upgrade requires the service optional dependency") from exc
    source_root = Path(__file__).resolve().parents[2]
    installed_root = Path(sysconfig.get_path("data")) / "share" / "bipartite-scope"
    migration_root = next(
        (
            root
            for root in (source_root, installed_root)
            if (root / "alembic.ini").is_file() and (root / "migrations" / "env.py").is_file()
        ),
        None,
    )
    if migration_root is None:
        raise RuntimeError("BipartiteScope migration resources are unavailable")
    configuration = Config(str(migration_root / "alembic.ini"))
    configuration.set_main_option("script_location", str(migration_root / "migrations"))
    configuration.set_main_option("sqlalchemy.url", settings.database.url.replace("%", "%%"))
    configuration.attributes["explicit_url"] = True
    command.upgrade(configuration, "head")
    return "0002_v4_operations"


def _payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_error(value: BaseException | str) -> str:
    from .observability import redact

    text_value = redact(str(value)).replace("\n", " ").strip()
    text_value = re.sub(
        r"(?P<scheme>(?:postgresql|postgres|redis|rediss)(?:\+[^:]+)?://)"
        r"(?P<user>[^:@/\s]+):[^@/\s]+@",
        r"\g<scheme>\g<user>:***@",
        text_value,
    )
    return text_value[:2000] or "operation failed"


class ServiceDatabase:
    def __init__(self, settings: ServiceSettings, engine: Engine | None = None):
        self.settings = settings
        if engine is None:
            options: dict[str, Any] = {"pool_pre_ping": True}
            if settings.database.url.startswith("sqlite"):
                options["connect_args"] = {"check_same_thread": False}
                if ":memory:" in settings.database.url:
                    options["poolclass"] = StaticPool
            else:
                options.update(
                    {
                        "pool_size": settings.database.pool_size,
                        "max_overflow": settings.database.max_overflow,
                        "pool_timeout": settings.database.pool_timeout_seconds,
                    }
                )
            engine = create_engine(settings.database.url, **options)
        self.engine = engine
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    @service_transaction
    def initialize(self) -> None:
        if ":memory:" in self.settings.database.url:
            Base.metadata.create_all(self.engine)
            return
        tables = set(inspect(self.engine).get_table_names())
        if "alembic_version" in tables:
            with self.engine.connect() as connection:
                revision = connection.execute(
                    text("SELECT version_num FROM alembic_version")
                ).scalar()
            if revision == "0002_v4_operations":
                return
        elif "jobs" in tables:
            raise RuntimeError("unversioned service tables require an explicit migration review")
        upgrade_database(self.settings)

    def dispose(self) -> None:
        self.engine.dispose()

    def health(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001
            return False

    @service_transaction
    def register_workspace(self, workspace_key: str, storage_path: str | Path) -> dict[str, Any]:
        path = Path(storage_path).expanduser().resolve()
        root = self.settings.workspace_root
        if path != root and root not in path.parents:
            raise ValueError("workspace path escapes the configured service root")
        now = utcnow()
        with self.sessions.begin() as session:
            existing = session.scalar(
                select(WorkspaceRecord).where(WorkspaceRecord.workspace_key == workspace_key)
            )
            if existing:
                if Path(existing.storage_path) != path:
                    raise IdempotencyConflictError("workspace key is registered to another path")
                existing.updated_at = now
                return self.workspace_dict(existing)
            from .reliability import WorkspaceBusyError, maintenance_active

            if maintenance_active(root):
                raise WorkspaceBusyError("workspace registration is paused for maintenance")
            record = WorkspaceRecord(
                id=str(uuid4()),
                workspace_key=workspace_key,
                storage_path=str(path),
                schema_version=3,
                status="ready",
                created_at=now,
                updated_at=now,
            )
            session.add(record)
        return self.workspace_dict(record)

    def get_workspace(self, workspace_key: str) -> dict[str, Any]:
        with self.sessions() as session:
            record = session.scalar(
                select(WorkspaceRecord).where(WorkspaceRecord.workspace_key == workspace_key)
            )
            if record is None:
                raise WorkspaceNotFoundError(f"workspace is not registered: {workspace_key}")
            return self.workspace_dict(record)

    def list_workspaces(self) -> list[dict[str, Any]]:
        with self.sessions() as session:
            records = session.scalars(
                select(WorkspaceRecord).order_by(WorkspaceRecord.workspace_key)
            )
            return [self.workspace_dict(record) for record in records]

    @staticmethod
    def workspace_dict(record: WorkspaceRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "workspace_id": record.workspace_key,
            "storage_path": record.storage_path,
            "schema_version": record.schema_version,
            "status": record.status,
            "last_observed_snapshot_id": record.last_observed_snapshot_id,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        }

    @service_transaction
    def create_job(
        self,
        workspace_key: str,
        kind: str,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
        source: str = "api",
    ) -> tuple[dict[str, Any], bool]:
        if kind not in JOB_KINDS:
            raise ValueError(f"unsupported job kind: {kind}")
        from .reliability import maintenance_active

        if maintenance_active(self.settings.workspace_root):
            raise JobStateError("new jobs are paused for service maintenance")
        input_payload = dict(payload)
        input_hash = _payload_hash(input_payload)
        now = utcnow()
        try:
            with self.sessions.begin() as session:
                workspace = session.scalar(
                    select(WorkspaceRecord)
                    .where(WorkspaceRecord.workspace_key == workspace_key)
                    .with_for_update()
                )
                if workspace is None:
                    raise WorkspaceNotFoundError(f"workspace is not registered: {workspace_key}")
                if idempotency_key:
                    existing = session.scalar(
                        select(JobRecord).where(
                            JobRecord.workspace_id == workspace.id,
                            JobRecord.kind == kind,
                            JobRecord.idempotency_key == idempotency_key,
                        )
                    )
                    if existing:
                        if _payload_hash(
                            {
                                key: value
                                for key, value in existing.input_payload.items()
                                if key != "_provenance"
                            }
                        ) != _payload_hash(
                            {
                                key: value
                                for key, value in input_payload.items()
                                if key != "_provenance"
                            }
                        ):
                            raise IdempotencyConflictError(
                                "idempotency key was already used with a different payload"
                            )
                        return self.job_dict(existing, workspace.workspace_key), False
                from .policies import check_admission

                check_admission(self, workspace_key, kind, input_payload)
                if kind in MUTATING_JOB_KINDS:
                    active = session.scalar(
                        select(func.count(JobRecord.id)).where(
                            JobRecord.workspace_id == workspace.id,
                            JobRecord.kind.in_(tuple(MUTATING_JOB_KINDS)),
                            JobRecord.status.in_(("queued", "running", "retrying", "cancelling")),
                        )
                    )
                    if active:
                        raise JobStateError(
                            "another mutation job is queued or running for this workspace"
                        )
                record = JobRecord(
                    id=str(uuid4()),
                    workspace_id=workspace.id,
                    kind=kind,
                    source=source,
                    status="queued",
                    progress=0.0,
                    stage="queued",
                    idempotency_key=idempotency_key,
                    configuration_snapshot=self.settings.public_dict(),
                    configuration_hash=self.settings.fingerprint(),
                    input_payload=input_payload,
                    input_hash=input_hash,
                    queued_at=now,
                    created_at=now,
                    updated_at=now,
                )
                session.add(record)
                session.flush()
                self._append_event(session, record, "queued", "Job accepted for execution")
            return self.job_dict(record, workspace_key), True
        except IntegrityError as exc:
            raise IdempotencyConflictError("job idempotency constraint failed") from exc

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.sessions() as session:
            row = session.execute(
                select(JobRecord, WorkspaceRecord.workspace_key)
                .join(WorkspaceRecord, WorkspaceRecord.id == JobRecord.workspace_id)
                .where(JobRecord.id == job_id)
            ).one_or_none()
            if row is None:
                raise JobNotFoundError(f"job does not exist: {job_id}")
            return self.job_dict(row[0], row[1])

    def list_jobs(
        self,
        *,
        workspace_key: str | None = None,
        kind: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        statement = select(JobRecord, WorkspaceRecord.workspace_key).join(
            WorkspaceRecord, WorkspaceRecord.id == JobRecord.workspace_id
        )
        if workspace_key:
            statement = statement.where(WorkspaceRecord.workspace_key == workspace_key)
        if kind:
            statement = statement.where(JobRecord.kind == kind)
        if status:
            statement = statement.where(JobRecord.status == status)
        statement = statement.order_by(JobRecord.created_at.desc()).limit(max(1, min(limit, 200)))
        with self.sessions() as session:
            return [self.job_dict(record, key) for record, key in session.execute(statement)]

    @staticmethod
    def job_dict(record: JobRecord, workspace_key: str | None = None) -> dict[str, Any]:
        return {
            "job_id": record.id,
            "workspace_id": workspace_key,
            "kind": record.kind,
            "source": record.source,
            "status": record.status,
            "progress": record.progress,
            "stage": record.stage,
            "result": record.result_payload,
            "error_code": record.error_code,
            "error_message": record.error_message,
            "attempt_count": record.attempt_count,
            "celery_task_id": record.celery_task_id,
            "published_at": record.published_at.isoformat() if record.published_at else None,
            "dispatch_attempts": record.dispatch_attempts,
            "configuration_hash": record.configuration_hash,
            "input_hash": record.input_hash,
            "version": record.version,
            "execution_generation": record.execution_generation,
            "heartbeat_at": record.heartbeat_at.isoformat() if record.heartbeat_at else None,
            "created_at": record.created_at.isoformat(),
            "queued_at": record.queued_at.isoformat(),
            "started_at": record.started_at.isoformat() if record.started_at else None,
            "completed_at": record.completed_at.isoformat() if record.completed_at else None,
            "updated_at": record.updated_at.isoformat(),
            "links": {
                "self": f"/api/v3/jobs/{record.id}",
                "events": f"/api/v3/jobs/{record.id}/events",
            },
        }

    def has_active_mutation_job(self, workspace_key: str) -> bool:
        with self.sessions() as session:
            count = session.scalar(
                select(func.count(JobRecord.id))
                .join(WorkspaceRecord, WorkspaceRecord.id == JobRecord.workspace_id)
                .where(
                    WorkspaceRecord.workspace_key == workspace_key,
                    JobRecord.kind.in_(tuple(MUTATING_JOB_KINDS)),
                    JobRecord.status.in_(("queued", "running", "retrying", "cancelling")),
                )
            )
            return bool(count)

    def _locked_job(self, session: Session, job_id: str) -> JobRecord:
        record = session.scalar(select(JobRecord).where(JobRecord.id == job_id).with_for_update())
        if record is None:
            raise JobNotFoundError(f"job does not exist: {job_id}")
        return record

    def _append_event(
        self,
        session: Session,
        job: JobRecord,
        event_type: str,
        message: str,
        payload: Mapping[str, Any] | None = None,
    ) -> JobEventRecord:
        sequence = (
            session.scalar(
                select(func.max(JobEventRecord.sequence)).where(JobEventRecord.job_id == job.id)
            )
            or 0
        ) + 1
        event = JobEventRecord(
            id=str(uuid4()),
            job_id=job.id,
            sequence=sequence,
            event_type=event_type,
            stage=job.stage,
            progress=job.progress,
            message=message,
            payload=dict(payload or {}),
            created_at=utcnow(),
        )
        session.add(event)
        return event

    def events_after(self, job_id: str, sequence: int = 0) -> list[dict[str, Any]]:
        with self.sessions() as session:
            if session.get(JobRecord, job_id) is None:
                raise JobNotFoundError(f"job does not exist: {job_id}")
            events = session.scalars(
                select(JobEventRecord)
                .where(JobEventRecord.job_id == job_id, JobEventRecord.sequence > sequence)
                .order_by(JobEventRecord.sequence)
            )
            return [
                {
                    "id": event.sequence,
                    "event": event.event_type,
                    "stage": event.stage,
                    "progress": event.progress,
                    "message": event.message,
                    "payload": event.payload,
                    "created_at": event.created_at.isoformat(),
                }
                for event in events
            ]

    @service_transaction
    def transition_job(
        self,
        job_id: str,
        status: str,
        *,
        stage: str | None = None,
        progress: float | None = None,
        result: Mapping[str, Any] | None = None,
        error_code: str | None = None,
        error_message: BaseException | str | None = None,
        message: str | None = None,
        expected_generation: int | None = None,
    ) -> dict[str, Any]:
        now = utcnow()
        with self.sessions.begin() as session:
            record = self._locked_job(session, job_id)
            if (
                expected_generation is not None
                and record.execution_generation != expected_generation
            ):
                raise _ExecutionConflict("the task no longer owns this execution generation")
            if status != record.status and status not in _TRANSITIONS.get(record.status, set()):
                raise JobStateError(f"job cannot transition from {record.status} to {status}")
            record.status = status
            record.stage = stage or status
            if progress is not None:
                if progress < record.progress and status not in {"retrying", "queued"}:
                    raise JobStateError("job progress cannot decrease within an attempt")
                record.progress = max(0.0, min(float(progress), 100.0))
            if status == "running" and record.started_at is None:
                record.started_at = now
            if status in JOB_TERMINAL_STATES:
                record.completed_at = now
            if result is not None:
                record.result_payload = dict(result)
            record.error_code = error_code
            record.error_message = _safe_error(error_message) if error_message is not None else None
            record.updated_at = now
            record.version += 1
            self._append_event(
                session,
                record,
                status if status in JOB_TERMINAL_STATES else "progress",
                message or f"Job entered {status} state",
            )
            workspace_key = session.scalar(
                select(WorkspaceRecord.workspace_key).where(
                    WorkspaceRecord.id == record.workspace_id
                )
            )
        return self.job_dict(record, workspace_key)

    @service_transaction
    def update_progress(
        self,
        job_id: str,
        progress: float,
        stage: str,
        message: str,
        *,
        expected_generation: int | None = None,
    ) -> dict[str, Any]:
        with self.sessions.begin() as session:
            record = self._locked_job(session, job_id)
            if (
                expected_generation is not None
                and record.execution_generation != expected_generation
            ):
                raise _ExecutionConflict("the task no longer owns this execution generation")
            if record.status not in {"running", "cancelling"}:
                raise JobStateError("progress can only be recorded for a running job")
            if progress < record.progress:
                raise JobStateError("job progress cannot decrease within an attempt")
            record.progress = min(float(progress), 99.0)
            record.stage = stage
            record.updated_at = utcnow()
            record.version += 1
            self._append_event(session, record, "progress", message)
            workspace_key = session.scalar(
                select(WorkspaceRecord.workspace_key).where(
                    WorkspaceRecord.id == record.workspace_id
                )
            )
        return self.job_dict(record, workspace_key)

    @service_transaction
    def mark_published(self, job_id: str, celery_task_id: str) -> None:
        with self.sessions.begin() as session:
            record = self._locked_job(session, job_id)
            record.celery_task_id = celery_task_id
            record.published_at = utcnow()
            record.dispatch_attempts += 1
            record.updated_at = utcnow()

    @service_transaction
    def mark_dispatch_failed(self, job_id: str, error: BaseException | str) -> None:
        with self.sessions.begin() as session:
            record = self._locked_job(session, job_id)
            record.dispatch_attempts += 1
            record.error_code = "dispatch_unavailable"
            record.error_message = _safe_error(error)
            record.updated_at = utcnow()
            self._append_event(session, record, "dispatch_failed", "Task dispatch failed")

    @service_transaction
    def request_cancel(self, job_id: str) -> dict[str, Any]:
        with self.sessions.begin() as session:
            record = self._locked_job(session, job_id)
            if record.status == "queued":
                record.status = "cancelled"
                record.stage = "cancelled"
                record.progress = 0.0
                record.completed_at = utcnow()
            elif record.status == "running":
                record.status = "cancelling"
                record.stage = "cancelling"
                record.cancel_requested_at = utcnow()
            else:
                raise JobStateError(f"job in {record.status} state cannot be cancelled")
            record.updated_at = utcnow()
            record.version += 1
            self._append_event(session, record, record.status, "Cancellation requested")
            workspace_key = session.scalar(
                select(WorkspaceRecord.workspace_key).where(
                    WorkspaceRecord.id == record.workspace_id
                )
            )
        return self.job_dict(record, workspace_key)

    @service_transaction
    def retry_job(self, job_id: str) -> dict[str, Any]:
        with self.sessions.begin() as session:
            record = self._locked_job(session, job_id)
            if record.status not in {"failed", "cancelled"}:
                raise JobStateError(f"job in {record.status} state cannot be retried")
            from .policies import check_admission
            from .reliability import maintenance_active

            if maintenance_active(self.settings.workspace_root):
                raise JobStateError("new jobs are paused for maintenance")
            workspace_key = session.scalar(
                select(WorkspaceRecord.workspace_key).where(
                    WorkspaceRecord.id == record.workspace_id
                )
            )
            check_admission(self, workspace_key, record.kind, record.input_payload)
            if record.kind in MUTATING_JOB_KINDS and self.has_active_mutation_job(workspace_key):
                raise JobStateError("another mutation job is queued or running for this workspace")
            record.status = "retrying"
            record.stage = "retrying"
            record.progress = 0.0
            record.completed_at = None
            record.cancel_requested_at = None
            record.error_code = None
            record.error_message = None
            record.celery_task_id = None
            record.published_at = None
            record.updated_at = utcnow()
            record.version += 1
            self._append_event(session, record, "retrying", "Job queued for another attempt")
            record.status = "queued"
            record.stage = "queued"
            self._append_event(session, record, "queued", "Retry accepted for execution")
            workspace_key = session.scalar(
                select(WorkspaceRecord.workspace_key).where(
                    WorkspaceRecord.id == record.workspace_id
                )
            )
        return self.job_dict(record, workspace_key)

    @service_transaction
    def begin_attempt(self, job_id: str, worker_name: str) -> str:
        with self.sessions.begin() as session:
            job = self._locked_job(session, job_id)
            job.attempt_count += 1
            attempt = JobAttemptRecord(
                id=str(uuid4()),
                job_id=job.id,
                attempt_number=job.attempt_count,
                worker_name=worker_name,
                status="running",
                started_at=utcnow(),
            )
            session.add(attempt)
            job.updated_at = utcnow()
        return attempt.id

    @service_transaction
    def claim_job(self, job_id: str, worker_name: str) -> tuple[dict[str, Any], str | None]:
        now = utcnow()
        with self.sessions.begin() as session:
            job = self._locked_job(session, job_id)
            workspace_key = session.scalar(
                select(WorkspaceRecord.workspace_key).where(WorkspaceRecord.id == job.workspace_id)
            )
            if job.status != "queued":
                return self.job_dict(job, workspace_key), None
            from .reliability import maintenance_active

            if maintenance_active(self.settings.workspace_root):
                return self.job_dict(job, workspace_key), None
            if job.kind in HEAVY_JOB_KINDS:
                count = (
                    session.scalar(
                        select(func.count(JobRecord.id)).where(
                            JobRecord.kind.in_(tuple(HEAVY_JOB_KINDS)),
                            JobRecord.status.in_(("running", "cancelling")),
                        )
                    )
                    or 0
                )
                if count >= self.settings.resources.max_running_heavy_jobs:
                    return self.job_dict(job, workspace_key), None
            job.status = "running"
            job.stage = "starting"
            job.progress = 1.0
            job.started_at = job.started_at or now
            job.updated_at = now
            job.heartbeat_at = now
            job.execution_generation += 1
            job.version += 1
            job.attempt_count += 1
            attempt = JobAttemptRecord(
                id=str(uuid4()),
                job_id=job.id,
                attempt_number=job.attempt_count,
                worker_name=worker_name,
                status="running",
                started_at=now,
            )
            session.add(attempt)
            self._append_event(session, job, "progress", "Worker accepted the job")
        return self.job_dict(job, workspace_key), attempt.id

    @service_transaction
    def heartbeat_job(self, job_id: str, generation: int) -> bool:
        with self.sessions.begin() as session:
            job = self._locked_job(session, job_id)
            if job.execution_generation != generation or job.status not in {
                "running",
                "cancelling",
            }:
                return False
            now = utcnow()
            job.heartbeat_at = now
            job.updated_at = now
            lease = session.scalar(
                select(WorkspaceLeaseRecord).where(WorkspaceLeaseRecord.job_id == job_id)
            )
            if lease:
                lease.heartbeat_at = now
            return True

    def execution_owned(self, job_id: str, generation: int) -> bool:
        with self.sessions() as session:
            job = session.get(JobRecord, job_id)
            return bool(
                job
                and job.execution_generation == generation
                and job.status in {"running", "cancelling"}
            )

    @service_transaction
    def finish_attempt(
        self,
        attempt_id: str,
        status: str,
        *,
        error_code: str | None = None,
        error_message: BaseException | str | None = None,
        expected_generation: int | None = None,
    ) -> None:
        with self.sessions.begin() as session:
            attempt = session.get(JobAttemptRecord, attempt_id)
            if attempt is None:
                raise JobNotFoundError(f"job attempt does not exist: {attempt_id}")
            job = self._locked_job(session, attempt.job_id)
            if expected_generation is not None and job.execution_generation != expected_generation:
                raise _ExecutionConflict("the attempt no longer owns this execution generation")
            if attempt.status in JOB_TERMINAL_STATES:
                return
            attempt.status = status
            attempt.completed_at = utcnow()
            attempt.error_code = error_code
            attempt.error_message = (
                _safe_error(error_message) if error_message is not None else None
            )

    @service_transaction
    def acquire_workspace_lease(self, job_id: str) -> bool:
        now = utcnow()
        try:
            with self.sessions.begin() as session:
                job = self._locked_job(session, job_id)
                session.add(
                    WorkspaceLeaseRecord(
                        workspace_id=job.workspace_id,
                        job_id=job.id,
                        acquired_at=now,
                        heartbeat_at=now,
                    )
                )
            return True
        except IntegrityError:
            return False

    @service_transaction
    def refresh_workspace_lease(self, job_id: str) -> None:
        with self.sessions.begin() as session:
            lease = session.scalar(
                select(WorkspaceLeaseRecord).where(WorkspaceLeaseRecord.job_id == job_id)
            )
            if lease:
                lease.heartbeat_at = utcnow()

    @service_transaction
    def release_workspace_lease(self, job_id: str, generation: int | None = None) -> None:
        with self.sessions.begin() as session:
            job = self._locked_job(session, job_id)
            if generation is not None and job.execution_generation != generation:
                return
            session.execute(
                delete(WorkspaceLeaseRecord).where(WorkspaceLeaseRecord.job_id == job_id)
            )

    def cancellation_requested(self, job_id: str) -> bool:
        with self.sessions() as session:
            status = session.scalar(select(JobRecord.status).where(JobRecord.id == job_id))
            if status is None:
                raise JobNotFoundError(f"job does not exist: {job_id}")
            return status in {"cancelling", "cancelled"}

    @service_transaction
    def recover_stale_jobs(self) -> list[str]:
        threshold = utcnow() - timedelta(seconds=self.settings.worker.stale_job_seconds)
        recovered: list[str] = []
        with self.sessions.begin() as session:
            jobs = session.scalars(
                select(JobRecord)
                .where(JobRecord.status.in_(("running", "cancelling")))
                .where(func.coalesce(JobRecord.heartbeat_at, JobRecord.updated_at) < threshold)
                .with_for_update()
            )
            for job in jobs:
                job.status = "failed"
                job.stage = "failed"
                job.error_code = "worker_lost"
                job.error_message = (
                    "Worker heartbeat expired before the job reached a terminal state"
                )
                job.completed_at = utcnow()
                job.updated_at = utcnow()
                job.version += 1
                job.execution_generation += 1
                for attempt in session.scalars(
                    select(JobAttemptRecord).where(
                        JobAttemptRecord.job_id == job.id, JobAttemptRecord.status == "running"
                    )
                ):
                    attempt.status = "failed"
                    attempt.completed_at = job.completed_at
                    attempt.error_code = "worker_lost"
                    attempt.error_message = job.error_message
                self._append_event(session, job, "worker_recovery", "Stale job marked as failed")
                session.execute(
                    delete(WorkspaceLeaseRecord).where(WorkspaceLeaseRecord.job_id == job.id)
                )
                recovered.append(job.id)
        return recovered

    @staticmethod
    def operation_dict(record: OperationRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "operation_id": record.id,
            "kind": record.kind,
            "key": record.key,
            "payload": record.payload,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        }

    @service_transaction
    def record_operation(
        self, kind: str, payload: dict[str, Any], key: str | None = None
    ) -> dict[str, Any]:
        now = utcnow()
        with self.sessions.begin() as session:
            record = OperationRecord(
                id=str(uuid4()), kind=kind, key=key, payload=payload, created_at=now, updated_at=now
            )
            session.add(record)
        return self.operation_dict(record)

    @service_transaction
    def upsert_operation(self, kind: str, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.sessions.begin() as session:
            record = session.scalar(
                select(OperationRecord)
                .where(OperationRecord.kind == kind, OperationRecord.key == key)
                .with_for_update()
            )
            now = utcnow()
            if record is None:
                record = OperationRecord(
                    id=str(uuid4()),
                    kind=kind,
                    key=key,
                    payload=payload,
                    created_at=now,
                    updated_at=now,
                )
                session.add(record)
            else:
                record.payload = payload
                record.updated_at = now
        return self.operation_dict(record)

    def get_operation(self, kind: str, key: str) -> dict[str, Any] | None:
        with self.sessions() as session:
            record = session.scalar(
                select(OperationRecord).where(
                    OperationRecord.kind == kind, OperationRecord.key == key
                )
            )
            return self.operation_dict(record) if record else None

    def list_operations(self, kind: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        with self.sessions() as session:
            records = session.scalars(
                select(OperationRecord)
                .where(OperationRecord.kind == kind)
                .order_by(OperationRecord.created_at.desc(), OperationRecord.id)
                .limit(max(1, min(limit, 1000)))
                .offset(max(0, offset))
            )
            return [self.operation_dict(record) for record in records]

    def unpublished_jobs(self) -> Iterable[str]:
        with self.sessions() as session:
            return tuple(
                session.scalars(
                    select(JobRecord.id)
                    .where(JobRecord.status == "queued", JobRecord.published_at.is_(None))
                    .order_by(JobRecord.queued_at)
                )
            )
