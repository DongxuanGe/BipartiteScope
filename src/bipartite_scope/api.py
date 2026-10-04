from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import FastAPI, File, Header, Query, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import RedisError

from .config import ServiceSettings, load_service_settings
from .core import QueryEngine
from .database import (
    IdempotencyConflictError,
    JobNotFoundError,
    JobStateError,
    ServiceDatabase,
)
from .policies import ResourceLimitError
from .recommendation import recommend, record_feedback
from .reliability import WorkspaceBusyError, workspace_write_lock
from .storage import (
    SnapshotIntegrityError,
    SnapshotStore,
    all_events,
    init_workspace,
    load_workspace,
    normalize_event,
)
from .tasks import CeleryDispatcher, InlineDispatcher, TransientTaskError, capture_inputs

WORKSPACE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
TERMINAL_JOB_STATES = frozenset({"succeeded", "failed", "cancelled"})


class _ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkspaceCreate(_ApiModel):
    workspace_id: str
    create: bool = True


class FeedbackRequest(_ApiModel):
    user_id: str
    item_id: str
    event_type: str
    event_value: float = 1.0
    event_id: str | None = None
    event_time: str | None = None


class RecommendationRequest(_ApiModel):
    user_id: str
    snapshot_id: str | None = None
    top_n: int | None = Field(None, ge=1)


class ProblemError(RuntimeError):
    def __init__(self, status: int, code: str, title: str, detail: str):
        self.status = status
        self.code = code
        self.title = title
        self.detail = detail
        super().__init__(detail)


def _problem(status: int, code: str, title: str, detail: str) -> ProblemError:
    return ProblemError(status, code, title, detail)


def _workspace_path(settings: ServiceSettings, workspace_id: str) -> Path:
    if not WORKSPACE_PATTERN.fullmatch(workspace_id):
        raise _problem(
            422,
            "invalid_workspace_id",
            "Invalid workspace identifier",
            "Workspace identifiers may contain letters, digits, dots, underscores, and hyphens.",
        )
    root = settings.workspace_root
    path = (root / workspace_id).resolve()
    if path != root and root not in path.parents:
        raise _problem(
            422,
            "workspace_path_escape",
            "Invalid workspace path",
            "The resolved workspace path escapes the configured service root.",
        )
    return path


def _workspace(
    settings: ServiceSettings,
    database: ServiceDatabase,
    workspace_id: str,
    *,
    create: bool = False,
) -> Any:
    path = _workspace_path(settings, workspace_id)
    if create and not path.exists():
        with workspace_write_lock(settings.workspace_root):
            init_workspace(path)
    try:
        value = load_workspace(path)
    except FileNotFoundError as exc:
        raise _problem(404, "workspace_not_found", "Workspace not found", str(exc)) from exc
    database.register_workspace(workspace_id, path)
    return value


def _job_response(job: Mapping[str, Any]) -> dict[str, Any]:
    return dict(job)


def install_v3_routes(
    app: FastAPI,
    settings: ServiceSettings,
    *,
    database: ServiceDatabase | None = None,
    dispatcher: Any | None = None,
) -> None:
    owns_database = database is None
    database = database or ServiceDatabase(settings)
    database.initialize()
    dispatcher = dispatcher or (
        InlineDispatcher(settings, database)
        if settings.worker.always_eager
        else CeleryDispatcher(settings, database)
    )
    app.state.v3_settings = settings
    app.state.v3_database = database
    app.state.v3_dispatcher = dispatcher
    from .observability import audit, configure_logging, correlated_log, record_request

    logger = configure_logging(settings, "api")
    if owns_database:
        app.router.add_event_handler("shutdown", database.dispose)

    @app.middleware("http")
    async def request_identifier(request: Request, call_next: Any) -> Response:
        supplied_id = request.headers.get("X-Request-ID", "")
        request_id = (
            supplied_id if re.fullmatch(r"[A-Za-z0-9._-]{1,128}", supplied_id) else str(uuid4())
        )
        request.state.request_id = request_id
        started = time.monotonic()
        chunks: list[bytes] = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > settings.service.request_max_bytes:
                return JSONResponse(
                    status_code=413,
                    media_type="application/problem+json",
                    headers={"X-Request-ID": request_id},
                    content={
                        "type": "about:blank",
                        "title": "Request is too large",
                        "status": 413,
                        "code": "request_too_large",
                        "detail": "The request exceeds the configured byte limit.",
                        "request_id": request_id,
                    },
                )
            chunks.append(chunk)
        request._body = b"".join(chunks)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        route = getattr(request.scope.get("route"), "path", "/unmatched")
        duration = time.monotonic() - started
        correlated_log(
            logger,
            "INFO",
            "http_request",
            request_id=request_id,
            route=route,
            method=request.method,
            status=response.status_code,
            duration_ms=round(duration * 1000, 3),
        )
        try:
            await asyncio.to_thread(
                record_request, database, request.method, route, response.status_code, duration
            )
        except Exception:  # noqa: BLE001
            correlated_log(logger, "ERROR", "request_metric_unavailable", request_id=request_id)
        return response

    @app.exception_handler(WorkspaceBusyError)
    async def busy_handler(request: Request, error: WorkspaceBusyError) -> JSONResponse:
        return await problem_handler(
            request, _problem(409, "workspace_busy", "Workspace is busy", str(error))
        )

    @app.exception_handler(ResourceLimitError)
    async def resource_handler(request: Request, error: ResourceLimitError) -> JSONResponse:
        return await problem_handler(
            request, _problem(429, error.code, "Resource limit reached", str(error))
        )

    @app.exception_handler(ValueError)
    async def value_handler(request: Request, error: ValueError) -> JSONResponse:
        return await problem_handler(
            request, _problem(422, "invalid_operation", "Invalid operation", str(error))
        )

    @app.exception_handler(SnapshotIntegrityError)
    async def integrity_handler(request: Request, error: SnapshotIntegrityError) -> JSONResponse:
        return await problem_handler(
            request,
            _problem(
                422, "snapshot_integrity_failed", "Snapshot integrity check failed", str(error)
            ),
        )

    from sqlalchemy.exc import OperationalError

    @app.exception_handler(OperationalError)
    async def dependency_handler(request: Request, error: OperationalError) -> JSONResponse:
        return await problem_handler(
            request,
            _problem(
                503,
                "database_unavailable",
                "Database unavailable",
                "The control database is unavailable. Retry after it recovers.",
            ),
        )

    @app.exception_handler(ProblemError)
    async def problem_handler(request: Request, error: ProblemError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status,
            media_type="application/problem+json",
            content={
                "type": f"https://github.com/DongxuanGe/BipartiteScope/problems/{error.code}",
                "title": error.title,
                "status": error.status,
                "code": error.code,
                "detail": error.detail,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(Exception)
    async def internal_handler(request: Request, error: Exception) -> JSONResponse:
        correlated_log(
            logger,
            "ERROR",
            "internal_error",
            request_id=getattr(request.state, "request_id", None),
            error_type=type(error).__name__,
        )
        response = await problem_handler(
            request,
            _problem(
                500,
                "internal_error",
                "Internal error",
                "The operation failed unexpectedly. Inspect the local diagnostic log.",
            ),
        )
        response.headers["X-Request-ID"] = getattr(request.state, "request_id", "")
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            media_type="application/problem+json",
            content={
                "type": "https://github.com/DongxuanGe/BipartiteScope/problems/request-validation",
                "title": "Request validation failed",
                "status": 422,
                "code": "request_validation_failed",
                "detail": "The request does not match the API contract.",
                "request_id": getattr(request.state, "request_id", None),
                "errors": [
                    {"loc": entry["loc"], "type": entry["type"], "msg": entry["msg"]}
                    for entry in error.errors()
                ],
            },
        )

    def registered_workspace(workspace_id: str) -> Any:
        return _workspace(settings, database, workspace_id)

    def submit_job(
        workspace_id: str,
        kind: str,
        payload: Mapping[str, Any],
        idempotency_key: str | None,
        response: Response,
    ) -> dict[str, Any]:
        workspace = registered_workspace(workspace_id)
        if "_provenance" in payload:
            raise _problem(
                422,
                "reserved_input_field",
                "Invalid job input",
                "Input provenance is assigned by the server.",
            )
        if kind in {"validate", "build", "update", "evaluate", "snapshot_verify"}:
            try:
                payload = capture_inputs(workspace, kind, dict(payload))
            except (ValueError, FileNotFoundError, SnapshotIntegrityError) as exc:
                raise _problem(422, "invalid_job_input", "Invalid job input", str(exc)) from exc
        try:
            job, created = database.create_job(
                workspace_id,
                kind,
                payload,
                idempotency_key=idempotency_key,
                source="api",
            )
        except IdempotencyConflictError as exc:
            raise _problem(409, "idempotency_conflict", "Idempotency conflict", str(exc)) from exc
        except JobStateError as exc:
            raise _problem(409, "workspace_busy", "Workspace is busy", str(exc)) from exc
        if created:
            audit(
                database,
                "job.submitted",
                workspace_id=workspace_id,
                job_id=job["job_id"],
                details={"kind": kind, "input_hash": job["input_hash"]},
            )
            try:
                dispatcher.submit(job["job_id"])
            except TransientTaskError as exc:
                raise _problem(
                    503,
                    "task_dispatch_unavailable",
                    "Task dispatch unavailable",
                    "The job is durable and can be republished after the broker recovers.",
                ) from exc
            job = database.get_job(job["job_id"])
        response.status_code = 202
        response.headers["Location"] = f"{settings.service.api_prefix}/jobs/{job['job_id']}"
        return _job_response(job)

    prefix = settings.service.api_prefix.rstrip("/")

    @app.get(f"{prefix}/version", tags=["system"])
    def version() -> dict[str, Any]:
        return {
            "name": "BipartiteScope",
            "version": "4.0.0",
            "api_version": "v3",
            "authentication": False,
        }

    @app.get(f"{prefix}/health/live", tags=["system"])
    def live() -> dict[str, str]:
        return {"status": "ok", "version": "4.0.0"}

    @app.get(f"{prefix}/health/ready", tags=["system"])
    async def ready() -> dict[str, Any]:
        database_ready = await asyncio.to_thread(database.health)
        storage_ready = settings.workspace_root.is_dir() and os.access(
            settings.workspace_root, os.W_OK
        )
        redis_ready = False
        client: AsyncRedis | None = None
        try:
            client = AsyncRedis.from_url(settings.redis.broker_url)
            redis_ready = bool(await client.ping())
        except (RedisError, OSError):
            redis_ready = False
        finally:
            if client is not None:
                await client.aclose()
        checks = {"database": database_ready, "redis": redis_ready, "storage": storage_ready}
        if not all(checks.values()):
            raise _problem(
                503,
                "service_not_ready",
                "Service is not ready",
                "One or more required service dependencies are unavailable.",
            )
        return {"status": "ready", "checks": checks}

    @app.get(f"{prefix}/config", tags=["system"])
    def public_config() -> dict[str, Any]:
        return {"configuration": settings.public_dict(), "fingerprint": settings.fingerprint()}

    @app.post(f"{prefix}/workspaces", status_code=201, tags=["workspaces"])
    def create_workspace(payload: WorkspaceCreate) -> dict[str, Any]:
        workspace = _workspace(settings, database, payload.workspace_id, create=payload.create)
        return {
            **database.get_workspace(payload.workspace_id),
            "configuration": str(workspace.root / "bipartitescope.toml"),
        }

    @app.get(f"{prefix}/workspaces", tags=["workspaces"])
    def list_workspaces() -> dict[str, Any]:
        return {"workspaces": database.list_workspaces()}

    @app.get(f"{prefix}/workspaces/{{workspace_id}}", tags=["workspaces"])
    def get_workspace(workspace_id: str) -> dict[str, Any]:
        registered_workspace(workspace_id)
        return database.get_workspace(workspace_id)

    @app.post(f"{prefix}/workspaces/{{workspace_id}}/reconcile", tags=["workspaces"])
    def reconcile_workspace(workspace_id: str) -> dict[str, Any]:
        workspace = registered_workspace(workspace_id)
        store = SnapshotStore(workspace.artifacts)
        try:
            active = store.latest_id()
        except SnapshotIntegrityError:
            active = None
        return {
            "workspace": database.get_workspace(workspace_id),
            "active_snapshot_id": active,
            "snapshots": store.list(),
        }

    @app.post(f"{prefix}/workspaces/{{workspace_id}}/uploads", status_code=201, tags=["workspaces"])
    async def upload_file(workspace_id: str, file: Annotated[UploadFile, File()]) -> dict[str, Any]:
        workspace = registered_workspace(workspace_id)
        from .policies import check_storage_capacity

        check_storage_capacity(workspace.root, settings.resources)
        supplied = file.filename or "upload"
        if supplied != Path(supplied).name or "/" in supplied or "\\" in supplied:
            raise _problem(
                422,
                "invalid_upload_name",
                "Invalid upload filename",
                "Upload filenames cannot contain path separators.",
            )
        suffix = Path(supplied).suffix.lower()
        if suffix not in {".csv", ".jsonl", ".ndjson"}:
            raise _problem(
                422,
                "unsupported_upload_type",
                "Unsupported upload type",
                "Uploads must be CSV, JSONL, or NDJSON files.",
            )
        upload_id = str(uuid4())
        directory = workspace.data / "uploads" / upload_id
        with workspace_write_lock(workspace):
            directory.mkdir(parents=True, exist_ok=False)
        stored_name = f"payload{suffix}"
        temporary = directory / f".{stored_name}.tmp"
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    check_storage_capacity(workspace.root, settings.resources, len(chunk))
                    size += len(chunk)
                    if size > settings.service.request_max_bytes:
                        raise _problem(
                            413,
                            "upload_too_large",
                            "Upload is too large",
                            "The upload exceeds the configured request size limit.",
                        )
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            manifest = {
                "upload_id": upload_id,
                "original_name": supplied,
                "stored_name": stored_name,
                "bytes": size,
                "sha256": digest.hexdigest(),
                "created_at": datetime.now(UTC).isoformat(),
            }
            with workspace_write_lock(workspace):
                os.replace(temporary, directory / stored_name)
                (directory / "manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
            audit(database, "upload.completed", workspace_id=workspace_id, details=manifest)
            return manifest
        except Exception:
            temporary.unlink(missing_ok=True)
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
            raise
        finally:
            await file.close()

    def job_endpoint(kind: str) -> Any:
        def endpoint(
            workspace_id: str,
            payload: dict[str, Any],
            response: Response,
            idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
        ) -> dict[str, Any]:
            return submit_job(workspace_id, kind, payload, idempotency_key, response)

        return endpoint

    for job_kind in ("validate", "build", "update", "evaluate", "benchmark"):
        app.add_api_route(
            f"{prefix}/workspaces/{{workspace_id}}/{job_kind}",
            job_endpoint(job_kind),
            methods=["POST"],
            status_code=202,
            tags=["jobs"],
            name=f"submit_{job_kind}_job",
        )

    @app.post(
        f"{prefix}/workspaces/{{workspace_id}}/snapshots/{{snapshot_id}}/verify",
        status_code=202,
        tags=["jobs"],
    )
    def verify_snapshot(
        workspace_id: str,
        snapshot_id: str,
        response: Response,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return submit_job(
            workspace_id,
            "snapshot_verify",
            {"snapshot_id": snapshot_id},
            idempotency_key,
            response,
        )

    @app.get(f"{prefix}/jobs", tags=["jobs"])
    def list_jobs(
        workspace_id: str | None = None,
        kind: str | None = None,
        status: str | None = None,
        limit: int = Query(50, ge=1, le=200),
    ) -> dict[str, Any]:
        return {
            "jobs": database.list_jobs(
                workspace_key=workspace_id, kind=kind, status=status, limit=limit
            )
        }

    @app.get(f"{prefix}/jobs/{{job_id}}", tags=["jobs"])
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return database.get_job(job_id)
        except JobNotFoundError as exc:
            raise _problem(404, "job_not_found", "Job not found", str(exc)) from exc

    @app.post(f"{prefix}/jobs/{{job_id}}/cancel", tags=["jobs"])
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            job = database.request_cancel(job_id)
            audit(database, "job.cancelled", job_id=job_id)
            dispatcher.cancel(job_id)
            return job
        except JobNotFoundError as exc:
            raise _problem(404, "job_not_found", "Job not found", str(exc)) from exc
        except JobStateError as exc:
            raise _problem(409, "invalid_job_state", "Invalid job state", str(exc)) from exc

    @app.post(f"{prefix}/jobs/{{job_id}}/retry", status_code=202, tags=["jobs"])
    def retry_job(job_id: str, response: Response) -> dict[str, Any]:
        try:
            database.retry_job(job_id)
            audit(database, "job.retried", job_id=job_id)
            dispatcher.submit(job_id)
            response.headers["Location"] = f"{prefix}/jobs/{job_id}"
            return database.get_job(job_id)
        except JobNotFoundError as exc:
            raise _problem(404, "job_not_found", "Job not found", str(exc)) from exc
        except JobStateError as exc:
            raise _problem(409, "invalid_job_state", "Invalid job state", str(exc)) from exc
        except TransientTaskError as exc:
            raise _problem(
                503, "task_dispatch_unavailable", "Task dispatch unavailable", str(exc)
            ) from exc

    async def event_stream(job_id: str, after: int) -> AsyncIterator[str]:
        sequence = after
        client: AsyncRedis | None = None
        pubsub: Any | None = None
        last_heartbeat = time.monotonic()
        try:
            client = AsyncRedis.from_url(settings.redis.progress_url, decode_responses=True)
            pubsub = client.pubsub()
            await pubsub.subscribe(f"bipartite-scope:jobs:{job_id}")
        except (RedisError, OSError):
            pubsub = None
        try:
            while True:
                events = await asyncio.to_thread(database.events_after, job_id, sequence)
                for event in events:
                    sequence = int(event["id"])
                    yield (
                        f"id: {sequence}\n"
                        f"event: {event['event']}\n"
                        f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
                    )
                job = await asyncio.to_thread(database.get_job, job_id)
                if job["status"] in TERMINAL_JOB_STATES and not events:
                    break
                if time.monotonic() - last_heartbeat >= 15:
                    yield ": heartbeat\n\n"
                    last_heartbeat = time.monotonic()
                if pubsub is not None:
                    try:
                        await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    except (RedisError, OSError):
                        pubsub = None
                else:
                    await asyncio.sleep(0.5)
        finally:
            if pubsub is not None:
                await pubsub.aclose()
            if client is not None:
                await client.aclose()

    @app.get(f"{prefix}/jobs/{{job_id}}/events", tags=["jobs"])
    async def job_events(
        job_id: str,
        request: Request,
        after: int = Query(0, ge=0),
    ) -> StreamingResponse:
        try:
            database.get_job(job_id)
        except JobNotFoundError as exc:
            raise _problem(404, "job_not_found", "Job not found", str(exc)) from exc
        last_event = request.headers.get("Last-Event-ID")
        if last_event:
            try:
                after = max(after, int(last_event))
            except ValueError as exc:
                raise _problem(
                    400,
                    "invalid_event_id",
                    "Invalid event identifier",
                    "Last-Event-ID must be an integer.",
                ) from exc
        return StreamingResponse(
            event_stream(job_id, after),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get(f"{prefix}/jobs/{{job_id}}/artifacts", tags=["jobs"])
    def list_artifacts(job_id: str) -> dict[str, Any]:
        try:
            job = database.get_job(job_id)
        except JobNotFoundError as exc:
            raise _problem(404, "job_not_found", "Job not found", str(exc)) from exc
        result = job.get("result") or {}
        output = result.get("output") or result.get("path")
        if not output:
            return {"artifacts": []}
        path = _validated_artifact_root(database, job, output)
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(value for value in path.iterdir() if value.is_file())
        else:
            files = []
        return {
            "artifacts": [{"name": value.name, "bytes": value.stat().st_size} for value in files]
        }

    @app.get(f"{prefix}/jobs/{{job_id}}/artifacts/{{artifact_name}}", tags=["jobs"])
    def download_artifact(job_id: str, artifact_name: str) -> FileResponse:
        if artifact_name != Path(artifact_name).name:
            raise _problem(
                422,
                "invalid_artifact_name",
                "Invalid artifact name",
                "Artifact names cannot contain a path.",
            )
        try:
            job = database.get_job(job_id)
        except JobNotFoundError as exc:
            raise _problem(404, "job_not_found", "Job not found", str(exc)) from exc
        result = job.get("result") or {}
        output = result.get("output") or result.get("path")
        if not output:
            raise _problem(
                404,
                "artifact_not_found",
                "Artifact not found",
                "The job has no downloadable artifacts.",
            )
        base = _validated_artifact_root(database, job, output)
        target = (
            base
            if base.is_file() and base.name == artifact_name
            else (base / artifact_name).resolve()
        )
        if base.is_dir() and base not in target.parents:
            raise _problem(
                422,
                "artifact_path_escape",
                "Invalid artifact path",
                "The artifact path escapes the job output directory.",
            )
        if not target.is_file():
            raise _problem(
                404,
                "artifact_not_found",
                "Artifact not found",
                "The requested artifact does not exist.",
            )
        return FileResponse(target)

    @app.get(f"{prefix}/workspaces/{{workspace_id}}/query/{{entity_id}}", tags=["inference"])
    def query_community(
        workspace_id: str,
        entity_id: str,
        snapshot_id: str | None = None,
        size_budget: int | None = Query(None, ge=1),
    ) -> dict[str, Any]:
        workspace = registered_workspace(workspace_id)
        store = SnapshotStore(workspace.artifacts)
        try:
            snapshot = store.load(snapshot_id or store.latest_id())
            return asdict(
                QueryEngine(snapshot).search(entity_id, workspace.query_config(size_budget))
            )
        except (KeyError, ValueError, SnapshotIntegrityError) as exc:
            raise _problem(422, "query_failed", "Query failed", str(exc)) from exc

    @app.post(f"{prefix}/workspaces/{{workspace_id}}/recommend", tags=["inference"])
    def recommend_items(workspace_id: str, payload: RecommendationRequest) -> dict[str, Any]:
        workspace = registered_workspace(workspace_id)
        store = SnapshotStore(workspace.artifacts)
        try:
            snapshot = store.load(payload.snapshot_id or store.latest_id())
            return asdict(
                recommend(
                    snapshot,
                    payload.user_id,
                    workspace.recommendation_config(payload.top_n),
                    events=all_events(workspace),
                )
            )
        except (KeyError, ValueError, SnapshotIntegrityError) as exc:
            raise _problem(422, "recommendation_failed", "Recommendation failed", str(exc)) from exc

    @app.post(f"{prefix}/workspaces/{{workspace_id}}/feedback", tags=["inference"])
    def submit_feedback(workspace_id: str, payload: FeedbackRequest) -> dict[str, Any]:
        workspace = registered_workspace(workspace_id)
        from .policies import check_storage_capacity

        check_storage_capacity(workspace.root, settings.resources, 4096)
        try:
            event = normalize_event(
                {
                    **payload.model_dump(),
                    "event_id": payload.event_id or str(uuid4()),
                    "event_time": payload.event_time or datetime.now(UTC).isoformat(),
                }
            )
            result = record_feedback(workspace, event)
            audit(
                database,
                "feedback.recorded",
                workspace_id=workspace_id,
                details={"event_id": event.event_id, **result},
            )
            return result
        except (KeyError, ValueError) as exc:
            raise _problem(422, "invalid_feedback", "Invalid feedback", str(exc)) from exc

    @app.get(f"{prefix}/workspaces/{{workspace_id}}/snapshots", tags=["snapshots"])
    def list_snapshots(workspace_id: str) -> dict[str, Any]:
        workspace = registered_workspace(workspace_id)
        return {"snapshots": SnapshotStore(workspace.artifacts).list()}

    @app.post(
        f"{prefix}/workspaces/{{workspace_id}}/snapshots/{{snapshot_id}}/activate",
        tags=["snapshots"],
    )
    def activate_snapshot(workspace_id: str, snapshot_id: str) -> dict[str, Any]:
        workspace = registered_workspace(workspace_id)
        try:
            value = SnapshotStore(workspace.artifacts).activate(snapshot_id)
            audit(
                database,
                "snapshot.activated",
                workspace_id=workspace_id,
                details={"snapshot_id": value},
            )
            return {"snapshot_id": value, "active": True}
        except SnapshotIntegrityError as exc:
            raise _problem(
                422, "snapshot_integrity_failed", "Snapshot activation failed", str(exc)
            ) from exc

    @app.post(
        f"{prefix}/workspaces/{{workspace_id}}/snapshots/{{snapshot_id}}/rollback",
        tags=["snapshots"],
    )
    def rollback_snapshot(workspace_id: str, snapshot_id: str) -> dict[str, Any]:
        workspace = registered_workspace(workspace_id)
        try:
            value = SnapshotStore(workspace.artifacts).rollback(snapshot_id)
            audit(
                database,
                "snapshot.rolled_back",
                workspace_id=workspace_id,
                details={"snapshot_id": value},
            )
            return {"snapshot_id": value, "active": True, "rollback": True}
        except SnapshotIntegrityError as exc:
            raise _problem(
                422, "snapshot_integrity_failed", "Snapshot rollback failed", str(exc)
            ) from exc

    install_v4_routes(app, settings, database, submit_job, registered_workspace)


def install_v4_routes(
    app: FastAPI,
    settings: ServiceSettings,
    database: ServiceDatabase,
    submit_job: Any,
    registered_workspace: Any,
) -> None:
    from .maintenance import list_backups
    from .observability import (
        acknowledge_alert,
        check_alerts,
        operations_summary,
        prometheus_metrics,
    )
    from .policies import pin_snapshot, plan_retention, storage_usage

    prefix = "/api/v4"

    @app.get(f"{prefix}/version", tags=["operations"])
    def version_v4() -> dict[str, Any]:
        return {
            "name": "BipartiteScope",
            "version": "4.0.0",
            "api_version": "v4",
            "authentication": False,
        }

    @app.get(f"{prefix}/operations/summary", tags=["operations"])
    def summary() -> dict[str, Any]:
        return operations_summary(database)

    @app.get(f"{prefix}/operations/metrics", tags=["operations"])
    def metrics() -> Response:
        return Response(prometheus_metrics(database), media_type="text/plain; version=0.0.4")

    @app.get(f"{prefix}/operations/audit", tags=["operations"])
    def audits(
        limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)
    ) -> dict[str, Any]:
        return {"records": database.list_operations("audit", limit, offset), "offset": offset}

    @app.get(f"{prefix}/operations/alerts", tags=["operations"])
    def alerts() -> dict[str, Any]:
        return {"alerts": check_alerts(database)}

    @app.post(f"{prefix}/operations/alerts/{{alert_id}}/acknowledge", tags=["operations"])
    def acknowledge(alert_id: str) -> dict[str, Any]:
        try:
            return acknowledge_alert(database, alert_id)
        except (KeyError, ValueError) as exc:
            raise _problem(404, "alert_not_found", "Alert not found", str(exc)) from exc

    @app.get(f"{prefix}/workspaces/{{workspace_id}}/storage", tags=["operations"])
    def storage(workspace_id: str) -> dict[str, Any]:
        return storage_usage(registered_workspace(workspace_id).root)

    @app.post(f"{prefix}/workspaces/{{workspace_id}}/retention/plan", tags=["operations"])
    def retention_plan(workspace_id: str) -> dict[str, Any]:
        return plan_retention(registered_workspace(workspace_id), database)

    @app.post(
        f"{prefix}/workspaces/{{workspace_id}}/retention/apply",
        status_code=202,
        tags=["operations"],
    )
    def retention_apply(
        workspace_id: str,
        payload: dict[str, Any],
        response: Response,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return submit_job(workspace_id, "retention", {"plan": payload}, idempotency_key, response)

    @app.post(
        f"{prefix}/workspaces/{{workspace_id}}/snapshots/{{snapshot_id}}/pin", tags=["operations"]
    )
    def pin(workspace_id: str, snapshot_id: str, pinned: bool = True) -> dict[str, Any]:
        return pin_snapshot(registered_workspace(workspace_id), snapshot_id, pinned)

    @app.get(f"{prefix}/backups", tags=["operations"])
    def backups() -> dict[str, Any]:
        return {"backups": list_backups(settings)}

    @app.post(f"{prefix}/backups", status_code=202, tags=["operations"])
    def backup(
        payload: dict[str, Any],
        response: Response,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        workspace_id = payload.get("workspace_id")
        if not isinstance(workspace_id, str) or payload.get("scope", "workspace") not in {
            "workspace",
            "service",
        }:
            raise _problem(
                422,
                "invalid_backup_scope",
                "Invalid backup request",
                "Provide workspace_id and a workspace or service scope.",
            )
        return submit_job(
            workspace_id,
            "backup",
            {"scope": payload.get("scope", "workspace")},
            idempotency_key,
            response,
        )

    @app.post(f"{prefix}/backups/{{backup_id}}/verify", status_code=202, tags=["operations"])
    def backup_verify(
        backup_id: str, payload: dict[str, Any], response: Response
    ) -> dict[str, Any]:
        workspace_id = payload.get("workspace_id")
        if not isinstance(workspace_id, str):
            raise _problem(
                422, "invalid_backup_request", "Invalid backup request", "Provide workspace_id."
            )
        return submit_job(workspace_id, "backup_verify", {"backup_id": backup_id}, None, response)

    @app.post(
        f"{prefix}/workspaces/{{workspace_id}}/diagnostics", status_code=202, tags=["operations"]
    )
    def diagnostics(
        workspace_id: str, payload: dict[str, Any], response: Response
    ) -> dict[str, Any]:
        return submit_job(
            workspace_id, "diagnostics", {"job_id": payload.get("job_id")}, None, response
        )


def create_service_app(settings: ServiceSettings | None = None) -> FastAPI:
    from .interface import create_app

    effective = settings or load_service_settings()
    return create_app(effective.workspace_root, settings=effective)


def _validated_artifact_root(
    database: ServiceDatabase, job: Mapping[str, Any], output: str
) -> Path:
    workspace = database.get_workspace(str(job["workspace_id"]))
    root = Path(workspace["storage_path"]).resolve()
    path = Path(output).resolve()
    if job["kind"] == "diagnostics":
        root = (database.settings.workspace_root / ".operations" / "diagnostics").resolve()
    elif job["kind"] == "backup":
        from .maintenance import _backup_root

        root = _backup_root(database.settings, database.settings.workspace_root).resolve()
    if path != root and root not in path.parents:
        raise _problem(
            422,
            "artifact_path_escape",
            "Invalid artifact path",
            "The job output path escapes its registered workspace.",
        )
    return path
