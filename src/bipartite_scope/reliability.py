from __future__ import annotations

import fcntl
import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path
from typing import Any, Self
from uuid import uuid4


class WorkspaceBusyError(RuntimeError):
    pass


class ExecutionLostError(RuntimeError):
    pass


_state = threading.local()
_locks: dict[str, threading.RLock] = {}
_registry_lock = threading.Lock()


def _root(value: Any) -> Path:
    return Path(value if isinstance(value, (str, Path)) else value.root).resolve()


def maintenance_active(root: Any) -> bool:
    path = _root(root)
    return (path / ".maintenance.json").exists() or any(
        (parent / ".operations" / "maintenance.json").exists() for parent in (path, *path.parents)
    )


def assert_execution() -> None:
    guard = getattr(_state, "execution_guard", None)
    if guard is not None:
        guard()


@contextmanager
def execution_guard(
    check: Callable[[], None], fence: Callable[[], Any] | None = None
) -> Iterator[None]:
    previous = getattr(_state, "execution_guard", None)
    previous_fence = getattr(_state, "execution_fence", None)
    _state.execution_guard = check
    _state.execution_fence = fence
    try:
        yield
    finally:
        _state.execution_guard = previous
        _state.execution_fence = previous_fence


@contextmanager
def execution_commit() -> Iterator[None]:
    fence = getattr(_state, "execution_fence", None)
    with fence() if fence is not None else nullcontext():
        assert_execution()
        yield


def execution_mutation(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with execution_commit():
            return function(*args, **kwargs)

    return wrapped


@contextmanager
def workspace_write_lock(
    root: Any, allow_maintenance: bool = False, *, blocking: bool = False
) -> Iterator[None]:
    path = _root(root)
    key = str(path)
    held = getattr(_state, "held", {})
    allowed = allow_maintenance or getattr(_state, "allow_maintenance", False) or key in held
    if maintenance_active(path) and not allowed:
        raise WorkspaceBusyError("workspace writes are paused for maintenance")
    path.mkdir(parents=True, exist_ok=True)
    with _registry_lock:
        lock = _locks.setdefault(key, threading.RLock())
    if not lock.acquire(blocking=blocking):
        raise WorkspaceBusyError("another operation holds the workspace write lock")
    _state.held = held
    first = key not in held
    handle = None
    previous = getattr(_state, "allow_maintenance", False)
    try:
        if first:
            lock_path = path / ".write.lock"
            if lock_path.is_symlink():
                raise WorkspaceBusyError("workspace lock cannot be a symbolic link")
            handle = lock_path.open("a+b")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError as exc:
                raise WorkspaceBusyError("another process holds the workspace write lock") from exc
            held[key] = handle
        if maintenance_active(path) and not allowed:
            raise WorkspaceBusyError("workspace writes are paused for maintenance")
        _state.allow_maintenance = allowed
        assert_execution()
        yield
    finally:
        _state.allow_maintenance = previous
        if first and handle is not None:
            held.pop(key, None)
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        lock.release()


def set_service_maintenance(database: Any, enabled: bool) -> dict[str, Any]:
    from sqlalchemy import func, select

    from .database import JobRecord

    root = database.settings.workspace_root / ".operations"
    with workspace_write_lock(root, allow_maintenance=True, blocking=True):
        marker = root / "maintenance.json"
        if enabled:
            if marker.exists():
                raise WorkspaceBusyError("maintenance is already active")
            with database.sessions() as session:
                running = session.scalar(
                    select(func.count(JobRecord.id)).where(
                        JobRecord.status.in_(("running", "cancelling"))
                    )
                )
            if running:
                raise WorkspaceBusyError("wait for running jobs before entering manual maintenance")
            marker.write_text(
                json.dumps({"token": uuid4().hex, "source": "manual"}), encoding="utf-8"
            )
        elif marker.exists():
            value = json.loads(marker.read_text(encoding="utf-8"))
            if value.get("source") != "manual":
                raise WorkspaceBusyError("an active operation owns the maintenance marker")
            marker.unlink()
    return {"maintenance": enabled}


def recover_maintenance(database: Any) -> list[str]:
    from sqlalchemy import func, select

    from .database import JobRecord

    operations = database.settings.workspace_root / ".operations"
    recovered = []
    with workspace_write_lock(operations, allow_maintenance=True, blocking=True):
        with database.sessions() as session:
            if session.scalar(
                select(func.count(JobRecord.id)).where(
                    JobRecord.status.in_(("running", "cancelling"))
                )
            ):
                return recovered
        marker = operations / "maintenance.json"
        if marker.is_file() and not marker.is_symlink():
            value = json.loads(marker.read_text(encoding="utf-8"))
            if value.get("reason") == "backup" and value.get("id"):
                with (operations / "maintenance.lock").open("a+b") as handle:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        marker.unlink()
                        recovered.append(str(marker))
                        fcntl.flock(handle, fcntl.LOCK_UN)
        for entry in database.list_workspaces():
            root = Path(entry["storage_path"])
            marker = root / ".maintenance.json"
            if not marker.is_file() or marker.is_symlink():
                continue
            try:
                with workspace_write_lock(root, allow_maintenance=True):
                    value = json.loads(marker.read_text(encoding="utf-8"))
                    if isinstance(value.get("token"), str) and len(value["token"]) == 32:
                        marker.unlink()
                        recovered.append(str(marker))
            except WorkspaceBusyError:
                continue
        if recovered:
            database.record_operation(
                "audit",
                {
                    "action": "maintenance.recovered",
                    "source": "scheduler",
                    "outcome": "succeeded",
                    "details": {"markers": recovered},
                },
            )
    return recovered


def workspace_mutation(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(workspace: Any, *args: Any, **kwargs: Any) -> Any:
        with workspace_write_lock(workspace):
            return function(workspace, *args, **kwargs)

    return wrapped


def snapshot_mutation(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(store: Any, *args: Any, **kwargs: Any) -> Any:
        with workspace_write_lock(store.root.parent):
            return function(store, *args, **kwargs)

    return wrapped


def service_transaction(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(database: Any, *args: Any, **kwargs: Any) -> Any:
        with workspace_write_lock(
            database.settings.workspace_root / ".operations", allow_maintenance=True, blocking=True
        ):
            return function(database, *args, **kwargs)

    return wrapped


@contextmanager
def maintenance_mode(root: Any) -> Iterator[None]:
    path = _root(root)
    token = uuid4().hex
    with workspace_write_lock(path, allow_maintenance=True):
        marker = path / ".maintenance.json"
        if marker.exists():
            raise WorkspaceBusyError("workspace already has an active maintenance marker")
        marker.write_text(json.dumps({"token": token}), encoding="utf-8")
        try:
            yield
        finally:
            if marker.is_file() and json.loads(marker.read_text()).get("token") == token:
                marker.unlink()


class JobHeartbeat:
    def __init__(self, database: Any, job_id: str, generation: int, interval: float):
        self.database = database
        self.job_id = job_id
        self.generation = generation
        self.interval = interval
        self.stop = threading.Event()
        self.lost = threading.Event()
        self.thread = threading.Thread(target=self._run, name="job-heartbeat", daemon=True)

    def _run(self) -> None:
        while not self.stop.wait(self.interval):
            try:
                if not self.database.heartbeat_job(self.job_id, self.generation):
                    self.lost.set()
                    return
            except Exception:  # noqa: BLE001
                self.lost.set()
                return

    def check(self) -> None:
        if self.lost.is_set() or not self.database.execution_owned(self.job_id, self.generation):
            raise ExecutionLostError("the task execution no longer owns this attempt")

    def commit_fence(self) -> Any:
        return workspace_write_lock(
            self.database.settings.workspace_root / ".operations",
            allow_maintenance=True,
            blocking=True,
        )

    def __enter__(self) -> Self:
        self.check()
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join(timeout=max(1.0, self.interval))
