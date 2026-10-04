from __future__ import annotations

import json
import socket
import time
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from celery import Celery
from redis import Redis
from redis.exceptions import RedisError
from sqlalchemy.exc import OperationalError

from .config import ServiceSettings, load_service_settings
from .core import (
    AffinityConfig,
    BuildConfig,
    CanonicalBipartiteGraph,
    EncoderConfig,
    build_snapshot,
)
from .database import (
    JOB_TERMINAL_STATES,
    MUTATING_JOB_KINDS,
    JobStateError,
    ServiceDatabase,
)
from .recommendation import evaluate, update_from_events
from .reliability import (
    ExecutionLostError,
    JobHeartbeat,
    assert_execution,
    execution_guard,
    recover_maintenance,
    service_transaction,
    workspace_write_lock,
)
from .storage import (
    InputValidationError,
    SnapshotIntegrityError,
    SnapshotStore,
    load_events,
    load_feature_rows,
    load_workspace,
    normalize_event,
    pending_events,
    validate_csv_graph,
)


class TaskCancelled(RuntimeError):
    pass


class TransientTaskError(RuntimeError):
    pass


class TaskDeferred(RuntimeError):
    pass


def _create_celery(settings: ServiceSettings) -> Celery:
    app = Celery(
        "bipartite_scope",
        broker=settings.redis.broker_url,
        backend=settings.redis.result_url,
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=settings.worker.prefetch_multiplier,
        task_soft_time_limit=settings.worker.task_soft_time_limit_seconds,
        task_time_limit=settings.worker.task_hard_time_limit_seconds,
        task_always_eager=settings.worker.always_eager,
        broker_transport_options={
            "visibility_timeout": settings.worker.task_hard_time_limit_seconds + 300
        },
        timezone="UTC",
        enable_utc=True,
    )
    if settings.scheduler.enabled:
        app.conf.beat_schedule = {
            "bipartite-scope-service-recovery": {
                "task": "bipartite_scope.recover_service_jobs",
                "schedule": min(60, settings.worker.stale_job_seconds),
            },
            "bipartite-scope-incremental-dispatch": {
                "task": "bipartite_scope.schedule_incremental_updates",
                "schedule": settings.scheduler.incremental_update_seconds,
            },
        }
    return app


_worker_settings = load_service_settings()
celery_app = _create_celery(_worker_settings)


def _publish(settings: ServiceSettings, job_id: str, payload: dict[str, Any]) -> None:
    try:
        client = Redis.from_url(settings.redis.progress_url, decode_responses=True)
        client.publish(f"bipartite-scope:jobs:{job_id}", json.dumps(payload))
        client.close()
    except (RedisError, OSError):
        return


def _progress(
    database: ServiceDatabase,
    settings: ServiceSettings,
    job_id: str,
    value: float,
    stage: str,
    message: str,
    generation: int | None = None,
) -> None:
    assert_execution()
    database.update_progress(job_id, value, stage, message, expected_generation=generation)
    _publish(
        settings,
        job_id,
        {"event": "progress", "stage": stage, "progress": value, "message": message},
    )
    if database.cancellation_requested(job_id):
        raise TaskCancelled("job cancellation was requested")


def _build_config(raw: dict[str, Any], users: int) -> BuildConfig:
    encoder = dict(raw.get("encoder", {}))
    encoder.setdefault("latent_groups", min(8, users))
    return BuildConfig(
        AffinityConfig(**raw.get("affinity", {})),
        EncoderConfig(**encoder),
        int(raw.get("semantic_recall_budget", 100)),
        2,
    )


def _upload_path(workspace_root: Path, upload_id: str) -> Path:
    if not upload_id or Path(upload_id).name != upload_id:
        raise ValueError("invalid upload identifier")
    directory = (workspace_root / "data" / "uploads" / upload_id).resolve()
    uploads = (workspace_root / "data" / "uploads").resolve()
    if uploads not in directory.parents:
        raise ValueError("upload path escapes the workspace staging directory")
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"upload does not exist: {upload_id}")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    file_path = (directory / value["stored_name"]).resolve()
    if directory not in file_path.parents or not file_path.is_file():
        raise ValueError("upload manifest references an invalid file")
    return file_path


def _run_validate(workspace: Any, payload: dict[str, Any], progress: Any) -> dict[str, Any]:
    progress(20, "validation", "Resolving validation inputs")
    if payload.get("edges_upload_id") and payload.get("features_upload_id"):
        report = validate_csv_graph(
            _upload_path(workspace.root, payload["edges_upload_id"]),
            _upload_path(workspace.root, payload["features_upload_id"]),
            delimiter=str(payload.get("delimiter", ",")),
        )
    else:
        report = workspace.validate()
    progress(90, "validation", "Validation completed")
    return report.to_dict()


def _run_build(workspace: Any, payload: dict[str, Any], progress: Any) -> dict[str, Any]:
    progress(15, "loading", "Loading graph inputs")
    if "edges" in payload and "features" in payload:
        graph = CanonicalBipartiteGraph.from_edges_and_features(
            payload["edges"], payload["features"]
        )
        config = _build_config(payload.get("config", {}), len(graph.u_ids))
    elif payload.get("edges_upload_id") and payload.get("features_upload_id"):
        from .storage import load_csv_graph

        graph = load_csv_graph(
            _upload_path(workspace.root, payload["edges_upload_id"]),
            _upload_path(workspace.root, payload["features_upload_id"]),
            delimiter=str(payload.get("delimiter", ",")),
        )
        config = workspace.build_config()
    else:
        report = workspace.validate()
        if not report.valid:
            raise InputValidationError(report)
        graph = workspace.load_graph()
        config = workspace.build_config()
    progress(35, "building", "Building the full model snapshot")
    snapshot = build_snapshot(graph, config)
    progress(85, "persisting", "Persisting the immutable snapshot")
    store = SnapshotStore(workspace.artifacts)
    try:
        store.latest_id()
        activate = False
    except SnapshotIntegrityError:
        activate = True
    path = store.save(snapshot, activate=activate)
    return {
        "snapshot_id": snapshot.snapshot_id,
        "path": str(path),
        "active": activate,
        "build_mode": snapshot.build_mode,
    }


def _run_update(workspace: Any, payload: dict[str, Any], progress: Any) -> dict[str, Any]:
    progress(15, "loading", "Loading the incremental event batch")
    if payload.get("events_upload_id"):
        events = load_events(_upload_path(workspace.root, payload["events_upload_id"]))
    else:
        events = tuple(normalize_event(value) for value in payload.get("events", ()))
    feature_names = tuple(payload.get("feature_names", ()))
    feature_updates = dict(payload.get("features", {}))
    if payload.get("features_upload_id"):
        feature_names, feature_updates = load_feature_rows(
            _upload_path(workspace.root, payload["features_upload_id"])
        )
    progress(35, "updating", "Applying events and warm-starting the model")
    result = update_from_events(
        workspace,
        events,
        feature_names=feature_names,
        feature_updates=feature_updates,
        source="service",
    )
    progress(90, "persisting", "Candidate snapshot persisted")
    return asdict(result)


def _run_evaluate(workspace: Any, payload: dict[str, Any], progress: Any) -> dict[str, Any]:
    progress(20, "evaluation", "Preparing chronological evaluation")
    result = evaluate(
        workspace,
        payload.get("snapshot_id"),
        workspace.evaluation_config(payload.get("k")),
    )
    progress(90, "reporting", "Evaluation report persisted")
    return asdict(result)


def _run_verify(workspace: Any, payload: dict[str, Any], progress: Any) -> dict[str, Any]:
    progress(30, "verification", "Verifying snapshot integrity")
    result = SnapshotStore(workspace.artifacts).verify(str(payload["snapshot_id"]))
    progress(90, "verification", "Snapshot integrity verified")
    return result


def _run_benchmark(payload: dict[str, Any], progress: Any) -> dict[str, Any]:
    progress(15, "benchmark", "Generating a sparse benchmark graph")
    from .interface import _benchmark

    result = _benchmark(
        Namespace(
            users=int(payload.get("users", 100)),
            items=int(payload.get("items", 100)),
            events=int(payload.get("events", 1000)),
            delta_ratio=float(payload.get("delta_ratio", 0.05)),
        )
    )
    progress(90, "benchmark", "Sparse benchmark completed")
    return result


def _operation(
    kind: str,
    workspace: Any,
    payload: dict[str, Any],
    progress: Any,
    database: ServiceDatabase | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    if kind == "validate":
        return _run_validate(workspace, payload, progress)
    if kind == "build":
        return _run_build(workspace, payload, progress)
    if kind == "update":
        return _run_update(workspace, payload, progress)
    if kind == "evaluate":
        return _run_evaluate(workspace, payload, progress)
    if kind == "snapshot_verify":
        return _run_verify(workspace, payload, progress)
    if kind == "benchmark":
        return _run_benchmark(payload, progress)
    if kind == "backup":
        from .maintenance import create_backup

        progress(10, "backup", "Preparing a consistent backup")
        return create_backup(
            database.settings.workspace_root
            if payload.get("scope") == "service"
            else workspace.root,
            settings=database.settings,
            database=database,
            scope=payload.get("scope", "workspace"),
            exclude_job_id=job_id,
        )
    if kind == "backup_verify":
        from .maintenance import list_backups, verify_backup

        candidates = {item["backup_id"]: item for item in list_backups(database.settings)}
        backup = candidates.get(payload.get("backup_id"))
        if backup is None:
            raise ValueError("backup does not exist")
        try:
            result = verify_backup(backup["path"])
        except Exception:
            database.record_operation(
                "backup_verify", {"backup_id": backup["backup_id"], "status": "failed"}
            )
            raise
        database.record_operation(
            "backup_verify", {"backup_id": backup["backup_id"], "status": "succeeded"}
        )
        return result
    if kind == "retention":
        from .policies import apply_retention

        return apply_retention(workspace, payload["plan"], database)
    if kind == "diagnostics":
        from .observability import export_diagnostics

        output = export_diagnostics(database, payload.get("job_id"))
        return {"output": str(output)}
    raise ValueError(f"unsupported job kind: {kind}")


def execute_job_record(
    job_id: str,
    settings: ServiceSettings | None = None,
    *,
    worker_name: str | None = None,
) -> dict[str, Any]:
    effective = settings or load_service_settings()
    database = ServiceDatabase(effective)
    database.initialize()
    lease = False
    attempt_id: str | None = None
    generation: int | None = None
    try:
        job = database.get_job(job_id)
        if job["status"] in JOB_TERMINAL_STATES:
            return job
        job, attempt_id = database.claim_job(job_id, worker_name or socket.gethostname())
        if attempt_id is None:
            if job["status"] == "queued":
                raise TaskDeferred("the task is waiting for maintenance or compute capacity")
            return job
        generation = job["execution_generation"]
        if job["kind"] in MUTATING_JOB_KINDS:
            lease = database.acquire_workspace_lease(job_id)
            if not lease:
                raise JobStateError("another mutation job holds the workspace lease")
        workspace_record = database.get_workspace(str(job["workspace_id"]))
        workspace = load_workspace(workspace_record["storage_path"])

        def report(value: float, stage: str, message: str) -> None:
            _progress(database, effective, job_id, value, stage, message, generation)

        from contextlib import nullcontext

        from .observability import audit, configure_logging, correlated_log

        logger = configure_logging(effective, "worker")
        with (
            JobHeartbeat(
                database, job_id, generation, effective.worker.heartbeat_seconds
            ) as heartbeat,
            execution_guard(heartbeat.check, heartbeat.commit_fence),
        ):
            guard = (
                workspace_write_lock(workspace)
                if job["kind"] in {"build", "update", "retention"}
                else nullcontext()
            )
            with guard:
                payload = job_input(database, job_id)
                _verify_inputs(workspace, payload)
                correlated_log(
                    logger,
                    "INFO",
                    "job_started",
                    job_id=job_id,
                    attempt_id=attempt_id,
                    workspace_id=job["workspace_id"],
                )
                result = _operation(job["kind"], workspace, payload, report, database, job_id)
                heartbeat.check()
        completed = _finish_success(database, job_id, attempt_id, generation, result)
        audit(
            database,
            "job.completed",
            source="worker",
            workspace_id=job["workspace_id"],
            job_id=job_id,
            details={"attempt_id": attempt_id, "result": result},
        )
        _publish(effective, job_id, {"event": "complete", "result": result})
        return completed
    except TaskCancelled as exc:
        cancelled, changed = _finish_error(
            database, job_id, attempt_id, generation, exc, "cancelled"
        )
        if not changed:
            return cancelled
        from .observability import audit

        audit(database, "job.cancelled", source="worker", job_id=job_id, outcome="cancelled")
        _publish(effective, job_id, {"event": "cancelled"})
        return cancelled
    except TaskDeferred:
        raise
    except ExecutionLostError:
        return database.get_job(job_id)
    except (OperationalError, ConnectionError, TimeoutError) as exc:
        current, changed = _finish_error(database, job_id, attempt_id, generation, exc, "retrying")
        if not changed:
            return current
        raise TransientTaskError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        failed, changed = _finish_error(database, job_id, attempt_id, generation, exc, "failed")
        if changed:
            from .observability import audit

            audit(
                database,
                "job.failed",
                source="worker",
                job_id=job_id,
                outcome="failed",
                details={"error_code": failed["error_code"]},
            )
            _publish(
                effective,
                job_id,
                {"event": "failed", "code": failed["error_code"]},
            )
            return failed
        return failed
    finally:
        if lease:
            database.release_workspace_lease(job_id, generation)
        database.dispose()


@service_transaction
def _finish_success(
    database: ServiceDatabase,
    job_id: str,
    attempt_id: str | None,
    generation: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    if database.cancellation_requested(job_id):
        raise TaskCancelled("job cancellation was requested")
    if attempt_id:
        database.finish_attempt(attempt_id, "succeeded", expected_generation=generation)
    return database.transition_job(
        job_id,
        "succeeded",
        stage="completed",
        progress=100,
        result=result,
        message="Job completed successfully",
        expected_generation=generation,
    )


@service_transaction
def _finish_error(
    database: ServiceDatabase,
    job_id: str,
    attempt_id: str | None,
    generation: int | None,
    error: BaseException,
    status: str,
) -> tuple[dict[str, Any], bool]:
    try:
        current = database.get_job(job_id)
        if current["status"] in JOB_TERMINAL_STATES or (
            generation is not None and not database.execution_owned(job_id, generation)
        ):
            return current, False
        code = (
            "cancelled"
            if status == "cancelled"
            else "transient_failure"
            if status == "retrying"
            else _error_code(error)
        )
        if attempt_id:
            database.finish_attempt(
                attempt_id,
                status,
                error_code=code,
                error_message=error,
                expected_generation=generation,
            )
        if status == "cancelled" and current["status"] == "running":
            database.transition_job(
                job_id,
                "cancelling",
                message="Cancellation acknowledged",
                expected_generation=generation,
            )
        if status == "retrying":
            if current["status"] == "running":
                database.transition_job(
                    job_id,
                    "retrying",
                    stage="retrying",
                    error_code=code,
                    error_message=error,
                    message="Transient failure scheduled for retry",
                    expected_generation=generation,
                )
                current = database.transition_job(
                    job_id, "queued", stage="queued", progress=0, expected_generation=generation
                )
            elif current["status"] == "cancelling":
                current = database.transition_job(
                    job_id, "cancelled", stage="cancelled", expected_generation=generation
                )
                return current, False
            return current, True
        current = database.transition_job(
            job_id,
            status,
            stage=status,
            error_code=code,
            error_message=error,
            message=f"Job {status}",
            expected_generation=generation,
        )
        return current, True
    except ExecutionLostError:
        return database.get_job(job_id), False


def job_input(database: ServiceDatabase, job_id: str) -> dict[str, Any]:
    from .database import JobRecord

    with database.sessions() as session:
        record = session.get(JobRecord, job_id)
        if record is None:
            raise ValueError(f"job does not exist: {job_id}")
        return dict(record.input_payload)


def _error_code(error: BaseException) -> str:
    if isinstance(error, InputValidationError):
        return "input_validation_failed"
    if isinstance(error, SnapshotIntegrityError):
        return "snapshot_integrity_failed"
    if isinstance(error, (KeyError, ValueError)):
        return "invalid_job_input"
    if isinstance(error, JobStateError):
        return "workspace_busy"
    return "job_failed"


class CeleryDispatcher:
    def __init__(
        self,
        settings: ServiceSettings,
        database: ServiceDatabase,
        app: Celery | None = None,
    ):
        self.settings = settings
        self.database = database
        self.app = app or _create_celery(settings)

    def submit(self, job_id: str) -> str:
        try:
            result = self.app.send_task("bipartite_scope.execute_job", args=[job_id])
        except Exception as exc:
            self.database.mark_dispatch_failed(job_id, exc)
            raise TransientTaskError("task broker is unavailable") from exc
        self.database.mark_published(job_id, result.id)
        return str(result.id)

    def cancel(self, job_id: str) -> None:
        job = self.database.get_job(job_id)
        task_id = job.get("celery_task_id")
        if task_id:
            self.app.control.revoke(task_id, terminate=False)


class InlineDispatcher:
    def __init__(self, settings: ServiceSettings, database: ServiceDatabase):
        self.settings = settings
        self.database = database

    def submit(self, job_id: str) -> str:
        task_id = f"inline-{uuid4()}"
        self.database.mark_published(job_id, task_id)
        try:
            execute_job_record(job_id, self.settings, worker_name="inline-test-worker")
        except TaskDeferred:
            pass
        return task_id

    def cancel(self, job_id: str) -> None:
        return None


@celery_app.task(bind=True, name="bipartite_scope.execute_job", max_retries=None)
def execute_job_task(self: Any, job_id: str) -> dict[str, Any]:
    try:
        return execute_job_record(job_id, _worker_settings, worker_name=self.request.hostname)
    except TaskDeferred as exc:
        raise self.retry(exc=exc, countdown=5, max_retries=None)
    except TransientTaskError as exc:
        database = ServiceDatabase(_worker_settings)
        database.initialize()
        try:
            current = database.get_job(job_id)
            retries = max(0, current["attempt_count"] - 1)
            if retries >= _worker_settings.worker.maximum_retries:
                if current["status"] == "queued":
                    current = database.transition_job(
                        job_id,
                        "failed",
                        stage="failed",
                        error_code="retry_exhausted",
                        error_message=exc,
                        message="Transient failure retry limit exhausted",
                    )
                return current
        finally:
            database.dispose()
        raise self.retry(
            exc=exc,
            countdown=_worker_settings.worker.retry_backoff_seconds * (2**retries),
            max_retries=None,
        )


@celery_app.task(name="bipartite_scope.schedule_incremental_updates")
def schedule_incremental_updates() -> dict[str, int]:
    settings = _worker_settings
    if not settings.scheduler.incremental_update_enabled:
        return {"created": 0, "skipped": 0}
    database = ServiceDatabase(settings)
    database.initialize()
    dispatcher = CeleryDispatcher(settings, database, celery_app)
    created = 0
    skipped = 0
    try:
        from .observability import check_alerts

        check_alerts(database)
        for record in database.list_workspaces():
            workspace = load_workspace(record["storage_path"])
            if not pending_events(workspace) or database.has_active_mutation_job(
                record["workspace_id"]
            ):
                skipped += 1
                continue
            bucket = int(time.time() // settings.scheduler.incremental_update_seconds)
            job, was_created = database.create_job(
                record["workspace_id"],
                "update",
                capture_inputs(workspace, "update", {}),
                idempotency_key=f"scheduled-update-{bucket}",
                source="scheduler",
            )
            if was_created:
                dispatcher.submit(job["job_id"])
                created += 1
            else:
                skipped += 1
    finally:
        database.dispose()
    return {"created": created, "skipped": skipped}


@celery_app.task(name="bipartite_scope.recover_service_jobs")
def recover_service_jobs() -> dict[str, int]:
    settings = _worker_settings
    database = ServiceDatabase(settings)
    database.initialize()
    dispatcher = CeleryDispatcher(settings, database, celery_app)
    stale = database.recover_stale_jobs()
    maintenance = recover_maintenance(database)
    republished = 0
    unavailable = 0
    try:
        from .observability import check_alerts

        check_alerts(database)
        for job_id in database.unpublished_jobs():
            try:
                dispatcher.submit(job_id)
                republished += 1
            except TransientTaskError:
                unavailable += 1
    finally:
        database.dispose()
    return {
        "stale_failed": len(stale),
        "maintenance_recovered": len(maintenance),
        "republished": republished,
        "unavailable": unavailable,
    }


def configure_worker(settings: ServiceSettings) -> Celery:
    global _worker_settings

    _worker_settings = settings
    celery_app.conf.update(_create_celery(settings).conf)
    return celery_app


def capture_inputs(workspace: Any, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    from .storage import _sha256

    files = {"configuration": (workspace.root / "bipartitescope.toml")}
    for key, value in payload.items():
        if key.endswith("_upload_id"):
            files[key] = _upload_path(workspace.root, str(value))
    if (
        kind in {"validate", "build"}
        and not payload.get("edges")
        and not payload.get("edges_upload_id")
    ):
        edges, features, _ = workspace.data_paths
        files.update({"edges": edges, "features": features})
    provenance = {
        key: {"path": str(path.relative_to(workspace.root)), "sha256": _sha256(path)}
        for key, path in files.items()
        if path.is_file()
    }
    if kind == "update":
        provenance["parent_snapshot_id"] = SnapshotStore(workspace.artifacts).latest_id()
    return {**payload, "_provenance": provenance}


def _verify_inputs(workspace: Any, payload: dict[str, Any]) -> None:
    from .storage import _sha256

    for key, value in payload.get("_provenance", {}).items():
        if key == "parent_snapshot_id":
            if SnapshotStore(workspace.artifacts).latest_id() != value:
                raise ValueError("the active snapshot changed after this task was submitted")
            continue
        path = (workspace.root / value["path"]).resolve()
        if (
            workspace.root not in path.parents
            or not path.is_file()
            or _sha256(path) != value["sha256"]
        ):
            raise ValueError("task input changed after submission")
