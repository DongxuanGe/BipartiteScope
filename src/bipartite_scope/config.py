from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeSettings(_SettingsModel):
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    api_prefix: str = "/api/v3"
    workspace_root: Path = Path("workspaces")
    request_max_bytes: int = Field(256 * 1024 * 1024, ge=1)
    allow_insecure_remote: bool = False


class DatabaseSettings(_SettingsModel):
    url: str = Field("sqlite+pysqlite:///bipartite_scope_service.sqlite3", repr=False)
    pool_size: int = Field(5, ge=1)
    max_overflow: int = Field(5, ge=0)
    pool_timeout_seconds: int = Field(30, ge=1)


class RedisSettings(_SettingsModel):
    broker_url: str = Field("redis://127.0.0.1:6379/0", repr=False)
    result_url: str = Field("redis://127.0.0.1:6379/1", repr=False)
    progress_url: str = Field("redis://127.0.0.1:6379/2", repr=False)


class WorkerSettings(_SettingsModel):
    concurrency: int = Field(2, ge=1)
    prefetch_multiplier: int = Field(1, ge=1)
    task_soft_time_limit_seconds: int = Field(3600, ge=1)
    task_hard_time_limit_seconds: int = Field(3900, ge=1)
    maximum_retries: int = Field(3, ge=0)
    retry_backoff_seconds: int = Field(5, ge=1)
    stale_job_seconds: int = Field(300, ge=30)
    always_eager: bool = False
    heartbeat_seconds: int = Field(10, ge=1)

    @model_validator(mode="after")
    def validate_limits(self) -> WorkerSettings:
        if self.task_hard_time_limit_seconds <= self.task_soft_time_limit_seconds:
            raise ValueError("worker hard time limit must exceed the soft time limit")
        if self.stale_job_seconds < self.heartbeat_seconds * 3:
            raise ValueError("worker stale timeout must span at least three heartbeats")
        return self


class SchedulerSettings(_SettingsModel):
    enabled: bool = True
    incremental_update_enabled: bool = False
    incremental_update_seconds: int = Field(3600, ge=60)


class ApiSettings(_SettingsModel):
    enable_legacy_v2_routes: bool = True
    enable_swagger: bool = True
    enable_redoc: bool = True
    cors_origins: tuple[str, ...] = ()


class ResourceSettings(_SettingsModel):
    max_queued_jobs: int = Field(100, ge=1)
    max_workspace_queued_jobs: int = Field(10, ge=1)
    max_running_heavy_jobs: int = Field(1, ge=1)
    min_free_bytes: int = Field(268435456, ge=0)
    workspace_max_bytes: int = Field(0, ge=0)


class RetentionSettings(_SettingsModel):
    candidate_snapshots: int = Field(10, ge=0)
    rollback_snapshots: int = Field(2, ge=1)
    reports_days: int = Field(30, ge=1)
    uploads_days: int = Field(7, ge=1)
    trash_days: int = Field(7, ge=1)
    backups_keep: int = Field(7, ge=1)


class BackupSettings(_SettingsModel):
    root: Path | None = None
    wait_seconds: int = Field(60, ge=1)
    include_uploads: bool = True
    include_reports: bool = True
    max_restore_bytes: int = Field(1099511627776, ge=1)


class ObservabilitySettings(_SettingsModel):
    log_level: str = "INFO"
    log_max_bytes: int = Field(5242880, ge=1024)
    log_backups: int = Field(3, ge=1)
    queue_wait_seconds: int = Field(600, ge=1)
    backup_max_age_seconds: int = Field(86400, ge=1)
    alert_failure_count: int = Field(3, ge=1)

    @model_validator(mode="after")
    def validate_log_level(self) -> ObservabilitySettings:
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("log_level must be a standard uppercase logging level")
        return self


class ServiceSettings(_SettingsModel):
    service: RuntimeSettings = RuntimeSettings()
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    worker: WorkerSettings = WorkerSettings()
    scheduler: SchedulerSettings = SchedulerSettings()
    api: ApiSettings = ApiSettings()
    resources: ResourceSettings = ResourceSettings()
    retention: RetentionSettings = RetentionSettings()
    backup: BackupSettings = BackupSettings()
    observability: ObservabilitySettings = ObservabilitySettings()

    @model_validator(mode="after")
    def validate_service(self) -> ServiceSettings:
        if not re.fullmatch(r"/(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+", self.service.api_prefix):
            raise ValueError("service api_prefix must be a normalized absolute route prefix")
        host = self.service.host.strip().lower()
        loopback = {"127.0.0.1", "localhost", "::1"}
        if host not in loopback and not self.service.allow_insecure_remote:
            raise ValueError(
                "an unauthenticated service may bind outside loopback only when "
                "allow_insecure_remote is explicitly enabled"
            )
        if not self.database.url.startswith(
            ("postgresql+psycopg://", "postgresql://", "sqlite+pysqlite://", "sqlite://")
        ):
            raise ValueError("database URL must use PostgreSQL or SQLite")
        for value in (
            self.redis.broker_url,
            self.redis.result_url,
            self.redis.progress_url,
        ):
            if not value.startswith(("redis://", "rediss://")):
                raise ValueError("Redis URLs must use redis:// or rediss://")
        return self

    @property
    def workspace_root(self) -> Path:
        return self.service.workspace_root.expanduser().resolve()

    def public_dict(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["database"]["url"] = _redact_url(self.database.url)
        for key in ("broker_url", "result_url", "progress_url"):
            payload["redis"][key] = _redact_url(payload["redis"][key])
        return payload

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


_SECTIONS = {
    "service",
    "database",
    "redis",
    "worker",
    "scheduler",
    "api",
    "resources",
    "retention",
    "backup",
    "observability",
}
_FLAT_ENV = {
    "WORKSPACE_ROOT": ("service", "workspace_root"),
    "HOST": ("service", "host"),
    "PORT": ("service", "port"),
    "ALLOW_INSECURE_REMOTE": ("service", "allow_insecure_remote"),
    "DATABASE_URL": ("database", "url"),
    "REDIS_BROKER_URL": ("redis", "broker_url"),
    "REDIS_RESULT_URL": ("redis", "result_url"),
    "REDIS_PROGRESS_URL": ("redis", "progress_url"),
    "WORKER_CONCURRENCY": ("worker", "concurrency"),
    "WORKER_ALWAYS_EAGER": ("worker", "always_eager"),
    "SCHEDULER_ENABLED": ("scheduler", "enabled"),
    "INCREMENTAL_UPDATE_ENABLED": ("scheduler", "incremental_update_enabled"),
    "INCREMENTAL_UPDATE_SECONDS": ("scheduler", "incremental_update_seconds"),
}


def _redact_url(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.password:
        return value
    hostname = parsed.hostname or ""
    if parsed.port:
        hostname = f"{hostname}:{parsed.port}"
    username = f"{parsed.username}:***@" if parsed.username else "***@"
    return urlunsplit(
        (parsed.scheme, username + hostname, parsed.path, parsed.query, parsed.fragment)
    )


def _merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        elif isinstance(value, Mapping):
            target[key] = dict(value)
        else:
            target[key] = value


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _environment_payload(environment: Mapping[str, str]) -> dict[str, Any]:
    prefix = "BIPARTITE_SCOPE_"
    payload: dict[str, Any] = {}
    for name, value in environment.items():
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        path = _FLAT_ENV.get(suffix)
        if path is None:
            parts = suffix.lower().split("__")
            if len(parts) == 1:
                first, separator, rest = parts[0].partition("_")
                parts = [first, rest] if separator and first in _SECTIONS else []
            path = tuple(parts) if len(parts) == 2 and parts[0] in _SECTIONS else None
        if path is None:
            continue
        payload.setdefault(path[0], {})[path[1]] = _coerce(value)
    return payload


def load_service_settings(
    config_path: str | Path | None = None,
    *,
    env_path: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ServiceSettings:
    config_file = Path(config_path) if config_path else Path("bipartite-scope.toml")
    values: dict[str, Any] = {}
    if config_file.is_file():
        with config_file.open("rb") as handle:
            loaded = tomllib.load(handle)
        _merge(values, {key: value for key, value in loaded.items() if key in _SECTIONS})
    dotenv = _parse_env_file(Path(env_path) if env_path else config_file.parent / ".env")
    _merge(values, _environment_payload(dotenv))
    _merge(values, _environment_payload(os.environ if environment is None else environment))
    if overrides:
        _merge(values, overrides)
    return ServiceSettings.model_validate(values)
