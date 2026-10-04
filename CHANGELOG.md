# Changelog

## 4.0.0 - 2026-10-04

- Added independent job heartbeats, execution generations, ownership checks, and fenced state changes to prevent a stale attempt from publishing results.
- Added correlated JSON logs, redacted operation audit, durable HTTP metrics, operation summaries, deduplicated alerts, and bounded diagnostic export.
- Added queued-job limits, heavy-job admission, free-disk checks, and optional workspace storage quotas.
- Added hash-verified retention plans, snapshot pinning, rollback protection, pending-task input protection, recoverable quarantine, and explicit validated trash purge.
- Added workspace and full-service directory backups with file allowlists, SHA-256 verification, event-index checks, and write coordination.
- Added restoration into new workspaces and an empty target control database, interrupted-task reconciliation, and recovery drills without automatic task replay.
- Added `/api/v4` operation, storage, retention, snapshot pinning, backup, and diagnostic routes plus matching management CLI commands.
- Added an additive V4 Alembic migration and expanded tests for reliability, operations, retention, and recovery.
- Retained the personal local deployment scope, existing workspaces and snapshots, and the unauthenticated API boundary.

This release also includes the service foundation developed under the V3 scope; it does not imply a separately published V3 release:

- Added immutable typed service configuration with TOML, `.env`, environment, and explicit-override precedence plus secret-safe public inspection.
- Added a SQLAlchemy and Alembic control plane for workspace registration, durable job state, attempts, progress events, idempotency, cancellation, retry, and workspace mutation leases.
- Added Redis-backed Celery workers and scheduling for asynchronous validation, build, incremental update, evaluation, benchmark, and snapshot verification jobs.
- Added the versioned `/api/v3` FastAPI interface with controlled uploads, asynchronous submission, job and artifact endpoints, request identifiers, structured errors, and dependency-aware health checks.
- Added resumable server-sent progress events backed by durable database replay and Redis notification.
- Retained optional local container configuration for PostgreSQL, Redis, migrations, API, worker, and scheduler; containers are not required for V4.
- Preserved the V2 Python API, CLI workflows, unversioned REST routes, workspace layout, and V1/V2 snapshot compatibility.
- Kept service imports optional and preserved the V2 standalone workflow.
- Documented the personal-service boundary: no authentication, no frontend, and no direct public deployment support.

## 2.0.0 - 2026-09-03

- Added idempotent CSV/JSONL event ingestion and append-only local event ledgers.
- Added incremental graph updates, affected-neighborhood affinity replacement, full fallback, and whole-graph warm-start training.
- Added explainable user-item recommendation, feedback collection, and chronological offline evaluation against two baselines.
- Added candidate snapshot lineage, manual verification, activation, rollback, and V1 snapshot compatibility.
- Expanded the CLI and localhost FastAPI interface for all V2 workflows.
- Consolidated the package into five Python files and all detailed guidance into `DOCUMENTATION.md`.
- Added repository checks for source layout, documentation layout, English-only tracked text, and absence of bundled datasets.

## 1.0.0 - 2026-08-25

- Renamed the application product to BipartiteScope.
- Added empty workspace initialization, TOML configuration, JSON validation reports, and query exports.
- Added atomic model snapshot writes, a latest pointer, and SHA-256 asset verification.
- Stabilized the normalized-cut training objective across supported PyTorch runtimes.
- Removed bundled CSV data; import and validation remain available for user-supplied data.
- Declared this validated engineering baseline as the final V1.0.0 release.

## 0.1.0 - 2026-08-21

- Created the public application-engine project structure.
- Added canonical graph validation and generic CSV import.
- Added sparse, stepwise Top-k structure-attribute affinity construction.
- Added offline dual-view learning, immutable snapshots, semantic recall, and BLC queries with support evidence.
- Added workspace CLI, REST API, recommendation and academic adapters, and the complete technical documentation set.
