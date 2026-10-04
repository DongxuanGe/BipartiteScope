from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .core import (
    AffinityConfig,
    BuildConfig,
    CanonicalBipartiteGraph,
    EncoderConfig,
    Event,
    IncrementalConfig,
    QueryEngine,
    build_snapshot,
)
from .recommendation import (
    _apply_events,
    evaluate,
    recommend,
    record_feedback,
    update_from_events,
    update_snapshot,
)
from .reliability import WorkspaceBusyError, workspace_write_lock
from .storage import (
    InputValidationError,
    SnapshotIntegrityError,
    SnapshotStore,
    all_events,
    init_workspace,
    load_workspace,
    normalize_event,
    validate_csv_graph,
)


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _write_payload(path: str | Path, payload: Any) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(output)


def _workspace_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True, help="path to a BipartiteScope workspace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bipartite-scope", description="BipartiteScope graph intelligence engine"
    )
    parser.add_argument("--version", action="version", version="BipartiteScope 4.0.0")
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="create an empty local workspace")
    initialize.add_argument("workspace")
    validate = commands.add_parser("validate", help="validate workspace or direct CSV input")
    validate.add_argument("--workspace")
    validate.add_argument("--edges")
    validate.add_argument("--features")
    validate.add_argument("--delimiter", default=",")
    validate.add_argument("--report")
    build = commands.add_parser("build", help="build an immutable full snapshot")
    _workspace_argument(build)
    query = commands.add_parser("query", help="run BLC community search")
    _workspace_argument(query)
    query.add_argument("--entity", required=True)
    query.add_argument("--snapshot")
    query.add_argument("--size", type=int)
    query.add_argument("--output")
    export = commands.add_parser("export", help="export a community search result")
    _workspace_argument(export)
    export.add_argument("--entity", required=True)
    export.add_argument("--snapshot")
    export.add_argument("--size", type=int)
    export.add_argument("--name", default="community.json")
    update = commands.add_parser("update", help="build a candidate snapshot from an event batch")
    _workspace_argument(update)
    update.add_argument("--events", required=True)
    update.add_argument("--features")
    recommendation = commands.add_parser("recommend", help="generate ranked item recommendations")
    _workspace_argument(recommendation)
    recommendation.add_argument("--user", required=True)
    recommendation.add_argument("--snapshot")
    recommendation.add_argument("--top-n", type=int)
    recommendation.add_argument("--output")
    feedback = commands.add_parser("feedback", help="append normalized feedback without retraining")
    _workspace_argument(feedback)
    feedback.add_argument("--user", required=True)
    feedback.add_argument("--item", required=True)
    feedback.add_argument("--event-type", required=True)
    feedback.add_argument("--value", type=float, default=1.0)
    feedback.add_argument("--event-id")
    feedback.add_argument("--event-time")
    evaluation = commands.add_parser("evaluate", help="run chronological offline evaluation")
    _workspace_argument(evaluation)
    evaluation.add_argument("--snapshot")
    evaluation.add_argument("--k", type=int)
    snapshot = commands.add_parser("snapshot", help="manage immutable snapshots")
    snapshot_commands = snapshot.add_subparsers(dest="snapshot_command", required=True)
    snapshot_list = snapshot_commands.add_parser("list")
    _workspace_argument(snapshot_list)
    for name in ("verify", "activate", "rollback", "pin", "unpin"):
        operation = snapshot_commands.add_parser(name)
        _workspace_argument(operation)
        operation.add_argument("--snapshot", required=True)
    benchmark = commands.add_parser("benchmark", help="run a generated sparse build benchmark")
    benchmark.add_argument("--users", type=int, required=True)
    benchmark.add_argument("--items", type=int, required=True)
    benchmark.add_argument("--events", type=int, required=True)
    benchmark.add_argument("--delta-ratio", type=float, required=True)
    benchmark.add_argument("--output")
    configuration = commands.add_parser("config", help="inspect service configuration")
    configuration.add_argument("--file")
    configuration_commands = configuration.add_subparsers(dest="config_command", required=True)
    for name in ("show", "validate", "fingerprint"):
        configuration_commands.add_parser(name)
    doctor = commands.add_parser("doctor", help="check service dependencies")
    doctor.add_argument("--config")
    database = commands.add_parser("database", help="manage the control database")
    database.add_argument("--config")
    database_commands = database.add_subparsers(dest="database_command", required=True)
    database_commands.add_parser("status")
    database_commands.add_parser("upgrade")
    workspace = commands.add_parser("workspace", help="register existing local workspaces")
    workspace.add_argument("--config")
    workspace_commands = workspace.add_subparsers(dest="workspace_command", required=True)
    for name in ("register", "reconcile"):
        operation = workspace_commands.add_parser(name)
        operation.add_argument("--workspace", required=True)
        operation.add_argument("--id")
    job = commands.add_parser("job", help="inspect and control asynchronous jobs")
    job.add_argument("--config")
    job_commands = job.add_subparsers(dest="job_command", required=True)
    job_list = job_commands.add_parser("list")
    job_list.add_argument("--workspace")
    job_list.add_argument("--kind")
    job_list.add_argument("--status")
    job_list.add_argument("--limit", type=int, default=50)
    for name in ("status", "watch", "cancel", "retry"):
        operation = job_commands.add_parser(name)
        operation.add_argument("--job", required=True)
    for name in ("serve", "worker", "scheduler"):
        service = commands.add_parser(name, help=f"start the {name} process")
        service.add_argument("--config")
    operations = commands.add_parser("operations", help="inspect local operational state")
    operations.add_argument("--config")
    operation_commands = operations.add_subparsers(dest="operations_command", required=True)
    for name in ("summary", "metrics", "audit", "alerts", "diagnostics"):
        operation = operation_commands.add_parser(name)
        if name == "diagnostics":
            operation.add_argument("--job")
    backup = commands.add_parser("backup", help="manage consistent local backups")
    backup.add_argument("--config")
    backups = backup.add_subparsers(dest="backup_command", required=True)
    create = backups.add_parser("create")
    create.add_argument("--workspace", required=True)
    create.add_argument("--scope", choices=("workspace", "service"), default="workspace")
    backups.add_parser("list")
    backups.add_parser("retention-plan")
    apply = backups.add_parser("retention-apply")
    apply.add_argument("--plan", required=True)
    purge = backups.add_parser("purge")
    purge.add_argument("--apply", action="store_true")
    for name in ("verify", "restore", "drill"):
        operation = backups.add_parser(name)
        operation.add_argument("--backup", required=True)
        if name in {"restore", "drill"}:
            operation.add_argument("--database-url")
        if name == "restore":
            operation.add_argument("--target", required=True)
    retention = commands.add_parser("retention", help="plan and quarantine expired artifacts")
    retention.add_argument("--config")
    retention_commands = retention.add_subparsers(dest="retention_command", required=True)
    for name in ("plan", "apply", "purge"):
        operation = retention_commands.add_parser(name)
        _workspace_argument(operation)
        if name == "apply":
            operation.add_argument("--plan", required=True)
        if name == "purge":
            operation.add_argument("--apply", action="store_true")
    maintenance = commands.add_parser("maintenance", help="pause service writes safely")
    maintenance.add_argument("--config")
    maintenance_commands = maintenance.add_subparsers(dest="maintenance_command", required=True)
    for name in ("status", "enter", "exit"):
        maintenance_commands.add_parser(name)
    return parser


def _validate(args: argparse.Namespace) -> int:
    if args.workspace:
        if args.edges or args.features:
            raise ValueError("use either --workspace or --edges and --features")
        report = load_workspace(args.workspace).validate()
    elif args.edges and args.features:
        report = validate_csv_graph(args.edges, args.features, delimiter=args.delimiter)
    else:
        raise ValueError("validate requires --workspace or both --edges and --features")
    payload = report.to_dict()
    if args.report:
        payload["report"] = _write_payload(args.report, payload)
    _json(payload)
    return 0 if report.valid else 2


def _query(args: argparse.Namespace, export_name: str | None = None) -> int:
    workspace = load_workspace(args.workspace)
    store = SnapshotStore(workspace.artifacts)
    snapshot_id = args.snapshot or store.latest_id()
    result = QueryEngine(store.load(snapshot_id)).search(
        args.entity,
        workspace.query_config(size_budget=args.size),
    )
    payload = asdict(result)
    output = getattr(args, "output", None)
    if export_name:
        output = workspace.exports / export_name
    if output:
        payload["output"] = _write_payload(output, payload)
    _json(payload)
    return 0


def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.users, args.items, args.events) < 1 or not 0 <= args.delta_ratio <= 1:
        raise ValueError("benchmark sizes must be positive and delta ratio must be in [0, 1]")
    rng = np.random.default_rng(17)
    edges = {
        (f"u{int(rng.integers(args.users))}", f"i{int(rng.integers(args.items))}")
        for _ in range(args.events * 2)
    }
    edges = set(sorted(edges)[: args.events])
    for user in range(args.users):
        edges.add((f"u{user}", f"i{user % args.items}"))
    features = {f"u{user}": [1.0, float(user % 3), float(user % 5)] for user in range(args.users)}
    graph = CanonicalBipartiteGraph.from_edges_and_features(edges, features)
    config = BuildConfig(
        AffinityConfig(top_k=min(16, args.users)),
        EncoderConfig(hidden_dim=8, latent_groups=min(4, args.users), epochs=2),
        semantic_recall_budget=min(16, args.users),
    )
    started = time.perf_counter()
    parent = build_snapshot(graph, config)
    full_seconds = time.perf_counter() - started
    count = max(1, round(args.events * args.delta_ratio))
    timestamp = datetime.now(UTC)
    events = tuple(
        Event(
            f"benchmark-{index}",
            f"u{index % args.users}",
            f"i{(index * 7 + 1) % args.items}",
            "click",
            1.0,
            (timestamp + timedelta(seconds=index)).isoformat(),
        )
        for index in range(count)
    )
    started = time.perf_counter()
    updated_graph, weights, affinity, stats = _apply_events(
        parent,
        events,
        (),
        {},
        IncrementalConfig(max_affected_ratio=max(args.delta_ratio, 0.01), warm_start_epochs=1),
    )
    warm_config = replace(config, encoder=replace(config.encoder, epochs=1))
    build_snapshot(
        updated_graph,
        warm_config,
        interaction_weights=weights,
        initial_state=parent.model_state,
        affinity_override=affinity,
        parent_snapshot_id=parent.snapshot_id,
        build_mode=stats["build_mode"],
    )
    incremental_seconds = time.perf_counter() - started
    return {
        "users": args.users,
        "items": args.items,
        "events": len(edges),
        "delta_events": count,
        "full_build_seconds": full_seconds,
        "incremental_build_seconds": incremental_seconds,
        "affected_ratio": stats["affected_ratio"],
        "build_mode": stats["build_mode"],
        "dense_user_by_user_matrix_materialized": False,
    }


def _service_settings(args: argparse.Namespace) -> Any:
    from .config import load_service_settings

    path = getattr(args, "config", None) or getattr(args, "file", None)
    return load_service_settings(path)


def _run_service_command(args: argparse.Namespace) -> int:
    from .database import ServiceDatabase

    settings = _service_settings(args)
    if args.command == "config":
        if args.config_command == "show":
            _json({"configuration": settings.public_dict()})
        elif args.config_command == "fingerprint":
            _json({"fingerprint": settings.fingerprint()})
        else:
            _json({"valid": True, "fingerprint": settings.fingerprint()})
        return 0
    database = ServiceDatabase(settings)
    if args.command == "doctor":
        from redis import Redis
        from redis.exceptions import RedisError

        settings.workspace_root.mkdir(parents=True, exist_ok=True)
        storage = settings.workspace_root.is_dir() and os.access(settings.workspace_root, os.W_OK)
        redis_ready = False
        try:
            client = Redis.from_url(settings.redis.broker_url)
            redis_ready = bool(client.ping())
            client.close()
        except (RedisError, OSError):
            redis_ready = False
        checks = {"database": database.health(), "redis": redis_ready, "storage": storage}
        _json({"ready": all(checks.values()), "checks": checks})
        database.dispose()
        return 0 if all(checks.values()) else 2
    if args.command == "database":
        if args.database_command == "upgrade":
            from .database import upgrade_database

            revision = upgrade_database(settings)
            _json(
                {
                    "upgraded": True,
                    "revision": revision,
                    "database": settings.public_dict()["database"]["url"],
                }
            )
            database.dispose()
            return 0
        ready = database.health()
        _json({"ready": ready, "database": settings.public_dict()["database"]["url"]})
        database.dispose()
        return 0 if ready else 2
    database.initialize()
    if args.command in {"operations", "backup", "retention", "maintenance"}:
        try:
            return _run_operations_command(args, settings, database)
        finally:
            database.dispose()
    if args.command == "workspace":
        workspace = load_workspace(args.workspace)
        workspace_id = args.id or workspace.root.name
        record = database.register_workspace(workspace_id, workspace.root)
        payload: dict[str, Any] = {"workspace": record}
        if args.workspace_command == "reconcile":
            store = SnapshotStore(workspace.artifacts)
            try:
                active = store.latest_id()
            except SnapshotIntegrityError:
                active = None
            payload.update({"active_snapshot_id": active, "snapshots": store.list()})
        _json(payload)
        database.dispose()
        return 0
    if args.command == "job":
        from .tasks import CeleryDispatcher

        dispatcher = CeleryDispatcher(settings, database)
        if args.job_command == "list":
            _json(
                {
                    "jobs": database.list_jobs(
                        workspace_key=args.workspace,
                        kind=args.kind,
                        status=args.status,
                        limit=args.limit,
                    )
                }
            )
        elif args.job_command == "status":
            _json(database.get_job(args.job))
        elif args.job_command == "watch":
            sequence = 0
            while True:
                events = database.events_after(args.job, sequence)
                for event in events:
                    sequence = event["id"]
                    _json(event)
                job = database.get_job(args.job)
                if job["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                time.sleep(0.5)
        elif args.job_command == "cancel":
            payload = database.request_cancel(args.job)
            dispatcher.cancel(args.job)
            _json(payload)
        else:
            payload = database.retry_job(args.job)
            dispatcher.submit(args.job)
            _json(payload)
        database.dispose()
        return 0
    database.dispose()
    if args.command == "serve":
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("serve requires the service optional dependency") from exc
        uvicorn.run(
            create_app(settings.workspace_root, settings=settings),
            host=settings.service.host,
            port=settings.service.port,
        )
        return 0
    from .tasks import configure_worker

    celery_app = configure_worker(settings)

    if args.command == "worker":
        celery_app.worker_main(
            [
                "worker",
                "--loglevel=INFO",
                f"--concurrency={settings.worker.concurrency}",
            ]
        )
    else:
        schedule = settings.workspace_root / ".operations" / "celerybeat-schedule"
        schedule.parent.mkdir(parents=True, exist_ok=True)
        celery_app.start(["beat", "--loglevel=INFO", f"--schedule={schedule}"])
    return 0


def _run_operations_command(args: argparse.Namespace, settings: Any, database: Any) -> int:
    from .maintenance import (
        create_backup,
        drill_backup,
        list_backups,
        restore_backup,
        verify_backup,
    )
    from .observability import (
        audit,
        check_alerts,
        export_diagnostics,
        operations_summary,
        prometheus_metrics,
    )
    from .policies import (
        apply_backup_retention,
        apply_retention,
        plan_backup_retention,
        plan_retention,
        purge_backup_trash,
        purge_trash,
    )
    from .reliability import maintenance_active, set_service_maintenance

    if args.command == "operations":
        name = args.operations_command
        if name == "metrics":
            _json({"metrics": prometheus_metrics(database)})
            return 0
        payload = (
            operations_summary(database)
            if name == "summary"
            else {"records": database.list_operations("audit")}
            if name == "audit"
            else {"alerts": check_alerts(database)}
            if name == "alerts"
            else {"output": str(export_diagnostics(database, args.job))}
        )
    elif args.command == "maintenance":
        name = args.maintenance_command
        payload = (
            {"maintenance": maintenance_active(settings.workspace_root)}
            if name == "status"
            else set_service_maintenance(database, name == "enter")
        )
        if name != "status":
            audit(database, f"maintenance.{name}", source="cli")
    elif args.command == "retention":
        workspace = load_workspace(args.workspace)
        record = next(
            (
                item
                for item in database.list_workspaces()
                if Path(item["storage_path"]).resolve() == workspace.root
            ),
            None,
        )
        workspace_id = record["workspace_id"] if record else workspace.root.name
        database.register_workspace(workspace_id, workspace.root)
        name = args.retention_command
        payload = (
            plan_retention(workspace, database)
            if name == "plan"
            else apply_retention(workspace, json.loads(Path(args.plan).read_text()), database)
            if name == "apply"
            else purge_trash(workspace, dry_run=not args.apply, db=database)
        )
        audit(database, f"retention.{name}", source="cli", workspace_id=workspace_id)
    else:
        name = args.backup_command
        try:
            if name == "create":
                payload = create_backup(args.workspace, settings, database, args.scope)
            elif name == "list":
                payload = {"backups": list_backups(settings)}
            elif name == "verify":
                payload = verify_backup(args.backup)
            elif name == "restore":
                payload = restore_backup(
                    args.backup,
                    args.target,
                    args.database_url,
                    max_restore_bytes=settings.backup.max_restore_bytes,
                )
            elif name == "drill":
                payload = drill_backup(args.backup, database_url=args.database_url)
            elif name == "retention-plan":
                payload = plan_backup_retention(settings, database)
            elif name == "retention-apply":
                payload = apply_backup_retention(
                    settings, json.loads(Path(args.plan).read_text()), database
                )
            else:
                payload = purge_backup_trash(settings, dry_run=not args.apply, db=database)
        except Exception:
            if name in {"verify", "drill"}:
                database.record_operation(
                    "backup_verify" if name == "verify" else "restore_drill", {"status": "failed"}
                )
            raise
        if name in {"verify", "drill"}:
            database.record_operation(
                "backup_verify" if name == "verify" else "restore_drill", {"status": "succeeded"}
            )
        audit(database, f"backup.{name}", source="cli")
    _json(payload)
    return 0


def run(args: argparse.Namespace) -> int:
    if args.command in {
        "config",
        "doctor",
        "database",
        "workspace",
        "job",
        "serve",
        "worker",
        "scheduler",
        "operations",
        "backup",
        "retention",
        "maintenance",
    }:
        return _run_service_command(args)
    if args.command == "init":
        _json({"workspace": str(init_workspace(args.workspace)), "config": "bipartitescope.toml"})
        return 0
    if args.command == "validate":
        return _validate(args)
    if args.command == "build":
        workspace = load_workspace(args.workspace)
        report = workspace.validate()
        if not report.valid:
            _json(report.to_dict())
            return 2
        with workspace_write_lock(workspace):
            snapshot = build_snapshot(workspace.load_graph(), workspace.build_config())
            store = SnapshotStore(workspace.artifacts)
            try:
                store.latest_id()
                activate = False
            except SnapshotIntegrityError:
                activate = True
            path = store.save(snapshot, activate=activate)
        _json({"snapshot_id": snapshot.snapshot_id, "path": str(path), "active": activate})
        return 0
    if args.command == "query":
        return _query(args)
    if args.command == "export":
        return _query(args, args.name)
    if args.command == "update":
        result = update_snapshot(load_workspace(args.workspace), args.events, args.features)
        _json(asdict(result))
        return 0
    if args.command == "recommend":
        workspace = load_workspace(args.workspace)
        store = SnapshotStore(workspace.artifacts)
        snapshot = store.load(args.snapshot or store.latest_id())
        result = recommend(
            snapshot,
            args.user,
            workspace.recommendation_config(args.top_n),
            events=all_events(workspace),
        )
        payload = asdict(result)
        if args.output:
            payload["output"] = _write_payload(args.output, payload)
        _json(payload)
        return 0
    if args.command == "feedback":
        event = normalize_event(
            {
                "event_id": args.event_id or str(uuid4()),
                "user_id": args.user,
                "item_id": args.item,
                "event_type": args.event_type,
                "event_value": args.value,
                "event_time": args.event_time or datetime.now(UTC).isoformat(),
            }
        )
        _json(record_feedback(load_workspace(args.workspace), event))
        return 0
    if args.command == "evaluate":
        workspace = load_workspace(args.workspace)
        _json(asdict(evaluate(workspace, args.snapshot, workspace.evaluation_config(args.k))))
        return 0
    if args.command == "snapshot":
        workspace = load_workspace(args.workspace)
        store = SnapshotStore(workspace.artifacts)
        if args.snapshot_command == "list":
            _json({"snapshots": store.list()})
        elif args.snapshot_command == "verify":
            _json(store.verify(args.snapshot))
        elif args.snapshot_command == "activate":
            _json({"snapshot_id": store.activate(args.snapshot), "active": True})
        elif args.snapshot_command in {"pin", "unpin"}:
            from .policies import pin_snapshot

            _json(pin_snapshot(workspace, args.snapshot, args.snapshot_command == "pin"))
        else:
            _json({"snapshot_id": store.rollback(args.snapshot), "active": True, "rollback": True})
        return 0
    if args.command == "benchmark":
        payload = _benchmark(args)
        if args.output:
            payload["output"] = _write_payload(args.output, payload)
        _json(payload)
        return 0
    raise AssertionError(f"unknown command: {args.command}")


def main(argv: list[str] | None = None) -> None:
    try:
        code = run(build_parser().parse_args(argv))
    except (
        FileNotFoundError,
        FileExistsError,
        KeyError,
        ValueError,
        InputValidationError,
        SnapshotIntegrityError,
        WorkspaceBusyError,
    ) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        code = 2
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        code = 1
    raise SystemExit(code)


def _api_config(raw: dict[str, Any], users: int) -> BuildConfig:
    encoder = dict(raw.get("encoder", {}))
    encoder.setdefault("latent_groups", min(8, users))
    return BuildConfig(
        AffinityConfig(**raw.get("affinity", {})),
        EncoderConfig(**encoder),
        raw.get("semantic_recall_budget", 100),
    )


def create_app(
    root: str | Path = "workspaces",
    *,
    settings: Any | None = None,
    database: Any | None = None,
    dispatcher: Any | None = None,
) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError("REST API requires the api optional dependency") from exc
    service_available = all(
        importlib.util.find_spec(name) is not None
        for name in ("sqlalchemy", "celery", "redis", "alembic")
    )
    if not service_available and (
        settings is not None or database is not None or dispatcher is not None
    ):
        raise RuntimeError("service settings require the service optional dependency")
    if settings is None:
        from .config import load_service_settings

        requested_root = Path(root).resolve()
        requested_root.mkdir(parents=True, exist_ok=True)
        settings = load_service_settings(
            overrides={
                "service": {"workspace_root": str(requested_root)},
                "database": {"url": f"sqlite+pysqlite:///{requested_root / 'service.sqlite3'}"},
                "worker": {"always_eager": True},
            }
        )
    workspace_root = settings.workspace_root
    workspace_root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(
        title="BipartiteScope",
        version="4.0.0",
        docs_url="/docs" if settings.api.enable_swagger else None,
        redoc_url="/redoc" if settings.api.enable_redoc else None,
    )
    if settings.api.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.api.cors_origins),
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def workspace_for(workspace_id: str, create: bool = False) -> Any:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", workspace_id):
            raise HTTPException(status_code=422, detail="invalid workspace identifier")
        path = (workspace_root / workspace_id).resolve()
        if workspace_root not in path.parents:
            raise HTTPException(
                status_code=422, detail="workspace path escapes the configured root"
            )
        if create and not path.exists():
            with workspace_write_lock(workspace_root):
                init_workspace(path)
        try:
            return load_workspace(path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "4.0.0"}

    def legacy_job(workspace: Any, kind: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not service_available:
            return None
        from .observability import audit
        from .tasks import InlineDispatcher, capture_inputs

        control = app.state.v3_database
        control.register_workspace(workspace.root.name, workspace.root)
        job, _ = control.create_job(
            workspace.root.name, kind, capture_inputs(workspace, kind, payload), source="api"
        )
        InlineDispatcher(settings, control).submit(job["job_id"])
        job = control.get_job(job["job_id"])
        audit(
            control,
            "job.submitted",
            source="api",
            workspace_id=workspace.root.name,
            job_id=job["job_id"],
            details={"kind": kind, "legacy": True},
        )
        if job["status"] != "succeeded":
            raise HTTPException(
                status_code=409 if job["status"] == "queued" else 422,
                detail=job.get("error_message")
                or "the synchronous operation is waiting for compute capacity",
            )
        return job["result"]

    @app.post("/workspaces/{workspace_id}/build")
    def api_build(workspace_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            workspace = workspace_for(workspace_id, create=True)
            managed = legacy_job(workspace, "build", payload)
            if managed is not None:
                return {
                    "workspace_id": workspace_id,
                    "snapshot_id": managed["snapshot_id"],
                    "active": managed["active"],
                }
            graph = CanonicalBipartiteGraph.from_edges_and_features(
                payload["edges"], payload["features"]
            )
            snapshot = build_snapshot(
                graph, _api_config(payload.get("config", {}), len(graph.u_ids))
            )
            store = SnapshotStore(workspace.artifacts)
            try:
                store.latest_id()
                activate = False
            except SnapshotIntegrityError:
                activate = True
            store.save(snapshot, activate=activate)
            return {
                "workspace_id": workspace_id,
                "snapshot_id": snapshot.snapshot_id,
                "active": activate,
            }
        except HTTPException:
            raise
        except WorkspaceBusyError:
            raise
        except (KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/workspaces/{workspace_id}/snapshots/{snapshot_id}/query/{entity_id}")
    def api_query(
        workspace_id: str,
        snapshot_id: str,
        entity_id: str,
        size_budget: int = 20,
    ) -> dict[str, Any]:
        try:
            workspace = workspace_for(workspace_id)
            snapshot = SnapshotStore(workspace.artifacts).load(snapshot_id)
            return asdict(
                QueryEngine(snapshot).search(entity_id, workspace.query_config(size_budget))
            )
        except HTTPException:
            raise
        except (KeyError, ValueError, SnapshotIntegrityError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/workspaces/{workspace_id}/update")
    def api_update(workspace_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            workspace = workspace_for(workspace_id)
            managed = legacy_job(workspace, "update", payload)
            if managed is not None:
                return managed
            events = tuple(normalize_event(record) for record in payload["events"])
            result = update_from_events(
                workspace,
                events,
                feature_names=payload.get("feature_names", ()),
                feature_updates=payload.get("features", {}),
            )
            return asdict(result)
        except HTTPException:
            raise
        except (KeyError, ValueError, SnapshotIntegrityError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/workspaces/{workspace_id}/recommend")
    def api_recommend(workspace_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            workspace = workspace_for(workspace_id)
            store = SnapshotStore(workspace.artifacts)
            snapshot = store.load(payload.get("snapshot_id") or store.latest_id())
            result = recommend(
                snapshot,
                str(payload["user_id"]),
                workspace.recommendation_config(payload.get("top_n")),
                events=all_events(workspace),
            )
            return asdict(result)
        except HTTPException:
            raise
        except (KeyError, ValueError, SnapshotIntegrityError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/workspaces/{workspace_id}/feedback")
    def api_feedback(workspace_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            value = {
                **payload,
                "event_id": payload.get("event_id") or str(uuid4()),
                "event_value": payload.get("event_value", 1.0),
                "event_time": payload.get("event_time") or datetime.now(UTC).isoformat(),
            }
            return record_feedback(workspace_for(workspace_id), normalize_event(value))
        except HTTPException:
            raise
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/workspaces/{workspace_id}/evaluate")
    def api_evaluate(workspace_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            workspace = workspace_for(workspace_id)
            managed = legacy_job(workspace, "evaluate", payload)
            if managed is not None:
                return managed
            return asdict(
                evaluate(
                    workspace,
                    payload.get("snapshot_id"),
                    workspace.evaluation_config(payload.get("k")),
                )
            )
        except HTTPException:
            raise
        except (ValueError, SnapshotIntegrityError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/workspaces/{workspace_id}/snapshots")
    def api_snapshots(workspace_id: str) -> dict[str, Any]:
        return {"snapshots": SnapshotStore(workspace_for(workspace_id).artifacts).list()}

    @app.post("/workspaces/{workspace_id}/snapshots/{snapshot_id}/activate")
    def api_activate(workspace_id: str, snapshot_id: str) -> dict[str, Any]:
        try:
            value = SnapshotStore(workspace_for(workspace_id).artifacts).activate(snapshot_id)
            return {"snapshot_id": value, "active": True}
        except HTTPException:
            raise
        except SnapshotIntegrityError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/workspaces/{workspace_id}/snapshots/{snapshot_id}/verify")
    def api_verify(workspace_id: str, snapshot_id: str) -> dict[str, Any]:
        try:
            return SnapshotStore(workspace_for(workspace_id).artifacts).verify(snapshot_id)
        except HTTPException:
            raise
        except SnapshotIntegrityError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not settings.api.enable_legacy_v2_routes:
        legacy_paths = {
            "/health",
            "/workspaces/{workspace_id}/build",
            "/workspaces/{workspace_id}/snapshots/{snapshot_id}/query/{entity_id}",
            "/workspaces/{workspace_id}/update",
            "/workspaces/{workspace_id}/recommend",
            "/workspaces/{workspace_id}/feedback",
            "/workspaces/{workspace_id}/evaluate",
            "/workspaces/{workspace_id}/snapshots",
            "/workspaces/{workspace_id}/snapshots/{snapshot_id}/activate",
            "/workspaces/{workspace_id}/snapshots/{snapshot_id}/verify",
        }
        app.router.routes = [
            route for route in app.router.routes if getattr(route, "path", None) not in legacy_paths
        ]

    if not service_available:
        return app

    from .api import install_v3_routes

    install_v3_routes(
        app,
        settings,
        database=database,
        dispatcher=dispatcher,
    )

    return app


if __name__ == "__main__":
    main()
