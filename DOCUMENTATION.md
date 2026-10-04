# BipartiteScope Documentation

BipartiteScope 4.0.0 combines attributed bipartite graph analysis, incremental updates, community search, explainable recommendations, feedback, offline evaluation, and immutable snapshots with a durable local service. It adds operation management: worker heartbeats and execution fencing, logs and audit, persistent metrics and alerts, resource admission, safe retention, consistent backups, new-target recovery, and recovery drills.

V4 is intended for one trusted personal operator on one local host. It uses local workspaces and a shared filesystem, with no authentication, accounts, tenant isolation, or frontend. Bind the API to loopback. The release includes the service foundation developed under the V3 scope and supports upgrading directly from V2.

## Architecture

Service coordination is stored separately from graph and model files:

| Component | Responsibility |
| --- | --- |
| FastAPI | Versioned HTTP API, validation, controlled uploads, job submission, and SSE |
| SQLite or PostgreSQL | Workspace registry, jobs, attempts, heartbeats, execution generations, progress events, leases, audit, metrics, and alerts |
| Redis | Celery broker/result transport and low-latency progress notification |
| Celery worker | Numerical jobs plus backup, verification, retention, and diagnostics tasks |
| Celery beat | Stale-job recovery, durable dispatch recovery, and optional scheduled update dispatch |
| Workspace volume | User data, normalized events, reports, immutable snapshots, and exports |
| Local backup directory | Verified workspace or full-service bundles and recoverable backup quarantine |

The control database is not a graph database. Canonical graphs, weighted interactions, model states, and reports remain in workspace files. Progress is persisted in the control database; Redis accelerates notification but is not the event history. A workspace-local SQLite index tracks accepted events and applied batches separately from service jobs.

Build and update jobs are serialized per workspace. File locks coordinate local writers, while execution generations prevent stale attempts from changing job state or publishing assets. The default global heavy-job limit is one. Synchronous inference reads immutable snapshots.

## Installation

Python 3.11 and 3.12 are the tested release targets; the package accepts Python 3.11 or newer.

Install the full local development and service environment:

```bash
python -m pip install -e '.[train,service,dev]'
bipartite-scope --version
```

The extras have distinct purposes:

- `train` installs PyTorch for model training and warm starts.
- `api` installs only FastAPI, multipart upload support, and Uvicorn.
- `service` installs the API dependencies plus SQLAlchemy, Alembic, PostgreSQL support, Redis, and Celery.
- `dev` installs the test, lint, and package-build tools.

The standalone CLI can use local files and standard-library SQLite without PostgreSQL or Redis. Service commands need the `service` extra. Broker-backed asynchronous execution also needs a reachable Redis instance; SQLite is the default personal control database, and PostgreSQL is supported through its configured URL.

Local writer coordination uses POSIX file locks. The V4 operation layer targets Linux and macOS. Docker is not required. Existing container files are optional configuration, not a prerequisite for these workflows.

To start the service with the default local settings, use the same working directory and environment for all processes:

```bash
cp .env.example .env
bipartite-scope database upgrade
bipartite-scope init workspaces/demo
bipartite-scope workspace register --workspace ./workspaces/demo --id demo
bipartite-scope doctor
```

Supply your own input files, then launch each long-running process in a separate terminal:

```bash
bipartite-scope serve
bipartite-scope worker
bipartite-scope scheduler
```

`doctor` checks the configured dependencies; it does not start Redis or a database server. PostgreSQL backup and restore require the compatible `pg_dump` and `pg_restore` executables on the host.

## Workspace and data schemas

Create an empty workspace without sample data:

```bash
bipartite-scope init my-workspace
```

A workspace contains `data/`, `artifacts/`, `exports/`, `reports/`, and `bipartitescope.toml`. Add user-owned input files under `data/`.

The full-build edge schema is:

```text
u_id,v_id
```

The feature schema is:

```text
u_id,<one or more nonnegative feature columns>
```

Duplicate edges are merged. Every user referenced by an edge must have one finite, nonnegative feature row. All feature rows must have the same dimension. Business identifiers remain separate from internal matrix indexes.

Incremental input may be CSV, JSONL, or NDJSON. Every record uses:

```text
event_id,user_id,item_id,event_type,event_value,event_time
```

`event_id`, `user_id`, `item_id`, `event_type`, and `event_time` are required. `event_value` defaults to `1.0`. Supported types are `view`, `click`, `favorite`, `purchase`, `rating`, `dislike`, and `remove`. Timestamps must include a timezone and are normalized to UTC. Duplicate event IDs are reported and ignored within each workspace.

Positive events create or strengthen a relationship. `dislike` changes recommendation filtering without creating a positive edge. `remove` removes the current relationship. New items may be appended without features. New users require a feature row with the existing feature dimension. A feature-dimension or named-schema change requires a new full workspace build.

Accepted events are normalized into `data/events.jsonl`. Feedback is also appended to `data/feedback.jsonl`. A workspace-local `state.sqlite3` index enforces event-ID idempotency and records applied batches. These runtime files are not bundled with the project.

## Workspace algorithm configuration

The generated `bipartitescope.toml` controls graph and recommendation behavior:

- `[data]` selects edge and feature paths and their delimiter.
- `[core.affinity]` controls structure/attribute mixing, restart diffusion, steps, and sparse Top-k retention.
- `[core.encoder]` controls model dimensions, layers, latent groups, optimization, and loss weights.
- `[query]` controls community size, assignment/affinity balance, BLC balance, and the structural gate.
- `[incremental]` controls affected-set fallback and warm-start epochs.
- `[recommendation]` controls Top-N ranking weights, popularity penalty, and time decay.
- `[evaluation]` controls K and the minimum number of positive events.

The algorithm defaults are:

```toml
[incremental]
max_affected_ratio = 0.20
neighbor_hops = 2
warm_start_epochs = 20
fallback_to_full_build = true

[recommendation]
top_n = 20
community_weight = 0.40
affinity_weight = 0.30
behavior_weight = 0.20
recency_weight = 0.10
popularity_penalty = 0.05
half_life_days = 30

[evaluation]
k = 10
minimum_positive_events = 3
```

`latent_groups` is unsupervised model capacity and must not exceed the number of users.

## Service and operation configuration

Service configuration is validated by immutable typed models. Unknown fields inside recognized service sections and invalid ranges are rejected; unrelated top-level workspace algorithm sections are ignored. Values are resolved in this order, with later sources taking precedence:

1. `bipartite-scope.toml` or the path passed with `--config` or `--file`.
2. A `.env` file beside the selected TOML file.
3. Process environment variables.
4. Explicit Python overrides.

The service file uses these sections:

```toml
[service]
host = "127.0.0.1"
port = 8000
api_prefix = "/api/v3"
workspace_root = "workspaces"
request_max_bytes = 268435456
allow_insecure_remote = false

[database]
url = "sqlite+pysqlite:///bipartite_scope_service.sqlite3"
pool_size = 5
max_overflow = 5
pool_timeout_seconds = 30

[redis]
broker_url = "redis://127.0.0.1:6379/0"
result_url = "redis://127.0.0.1:6379/1"
progress_url = "redis://127.0.0.1:6379/2"

[worker]
concurrency = 2
prefetch_multiplier = 1
task_soft_time_limit_seconds = 3600
task_hard_time_limit_seconds = 3900
maximum_retries = 3
retry_backoff_seconds = 5
stale_job_seconds = 300
heartbeat_seconds = 10
always_eager = false

[scheduler]
enabled = true
incremental_update_enabled = false
incremental_update_seconds = 3600

[api]
enable_legacy_v2_routes = true
enable_swagger = true
enable_redoc = true
cors_origins = []

[resources]
max_queued_jobs = 100
max_workspace_queued_jobs = 10
max_running_heavy_jobs = 1
min_free_bytes = 268435456
workspace_max_bytes = 0

[retention]
candidate_snapshots = 10
rollback_snapshots = 2
reports_days = 30
uploads_days = 7
trash_days = 7
backups_keep = 7

[backup]
wait_seconds = 60
include_uploads = true
include_reports = true
max_restore_bytes = 1099511627776

[observability]
log_level = "INFO"
log_max_bytes = 5242880
log_backups = 3
queue_wait_seconds = 600
backup_max_age_seconds = 86400
alert_failure_count = 3
```

Common flat environment names are:

```text
BIPARTITE_SCOPE_WORKSPACE_ROOT
BIPARTITE_SCOPE_HOST
BIPARTITE_SCOPE_PORT
BIPARTITE_SCOPE_ALLOW_INSECURE_REMOTE
BIPARTITE_SCOPE_DATABASE_URL
BIPARTITE_SCOPE_REDIS_BROKER_URL
BIPARTITE_SCOPE_REDIS_RESULT_URL
BIPARTITE_SCOPE_REDIS_PROGRESS_URL
BIPARTITE_SCOPE_WORKER_CONCURRENCY
BIPARTITE_SCOPE_WORKER_ALWAYS_EAGER
BIPARTITE_SCOPE_SCHEDULER_ENABLED
BIPARTITE_SCOPE_INCREMENTAL_UPDATE_ENABLED
BIPARTITE_SCOPE_INCREMENTAL_UPDATE_SECONDS
```

Nested names such as `BIPARTITE_SCOPE_WORKER__HEARTBEAT_SECONDS`, `BIPARTITE_SCOPE_RESOURCES__MAX_RUNNING_HEAVY_JOBS`, and `BIPARTITE_SCOPE_RETENTION__BACKUPS_KEEP` are accepted. Configuration output redacts URL passwords. `config fingerprint` returns a deterministic SHA-256 value for the effective settings.

`backup.root` is optional. If omitted, bundles are stored in a sibling directory named `<workspace-root-name>-backups`. Keep this directory outside the workspace root. `workspace_max_bytes = 0` disables the workspace size quota; minimum free disk space still applies. Retention and alert thresholds do not start background deletion or external notifications.

The stale-job timeout must span at least three configured heartbeats. The hard task limit must exceed the soft limit. A non-loopback `service.host` is rejected unless `allow_insecure_remote` is explicitly set. That flag adds no authentication. CORS restrictions are also not an authorization boundary.

## Python API

Supported application imports are exposed from the package root:

```python
from bipartite_scope import (
    BuildConfig,
    CanonicalBipartiteGraph,
    EncoderConfig,
    EvaluationConfig,
    Event,
    IncrementalConfig,
    QueryConfig,
    QueryEngine,
    RecommendationConfig,
    ServiceDatabase,
    ServiceSettings,
    SnapshotStore,
    build_snapshot,
    create_app,
    create_service_app,
    evaluate,
    recommend,
    update_snapshot,
)
```

Service-only imports are lazy so the core package remains importable when service extras are not installed.

Build a graph and query a snapshot:

```python
graph = CanonicalBipartiteGraph.from_edges_and_features(
    [("u1", "i1"), ("u2", "i1")],
    {"u1": [1.0, 0.0], "u2": [1.0, 1.0]},
)
snapshot = build_snapshot(graph, BuildConfig(encoder=EncoderConfig(latent_groups=2)))
result = QueryEngine(snapshot).search("u1", QueryConfig(size_budget=10))
```

The package root is the supported compatibility boundary. Imports from removed V1 internal submodules remain unsupported.

## CLI

Commands returning structured results emit JSON. `--version` prints the version, and long-running service processes also write process logs. Validation and user-input failures return exit code `2`; unexpected internal failures return exit code `1`.

The standalone data and snapshot commands are unchanged:

```bash
bipartite-scope init my-workspace
bipartite-scope validate --workspace my-workspace
bipartite-scope build --workspace my-workspace
bipartite-scope query --workspace my-workspace --entity u1 --size 20
bipartite-scope export --workspace my-workspace --entity u1 --name community.json
bipartite-scope update --workspace my-workspace --events batch.jsonl
bipartite-scope recommend --workspace my-workspace --user u1 --top-n 20
bipartite-scope feedback --workspace my-workspace --user u1 --item i9 --event-type click
bipartite-scope evaluate --workspace my-workspace --k 10
bipartite-scope snapshot list --workspace my-workspace
bipartite-scope snapshot verify --workspace my-workspace --snapshot SNAPSHOT_ID
bipartite-scope snapshot activate --workspace my-workspace --snapshot SNAPSHOT_ID
bipartite-scope snapshot rollback --workspace my-workspace --snapshot SNAPSHOT_ID
bipartite-scope snapshot pin --workspace my-workspace --snapshot SNAPSHOT_ID
bipartite-scope snapshot unpin --workspace my-workspace --snapshot SNAPSHOT_ID
bipartite-scope benchmark --users 100 --items 200 --events 1000 --delta-ratio 0.05
```

Service configuration, dependency, database, workspace-registry, job, and process commands:

```bash
bipartite-scope config --file bipartite-scope.toml validate
bipartite-scope config --file bipartite-scope.toml show
bipartite-scope config --file bipartite-scope.toml fingerprint
bipartite-scope doctor --config bipartite-scope.toml
bipartite-scope database --config bipartite-scope.toml status
bipartite-scope database --config bipartite-scope.toml upgrade
bipartite-scope workspace --config bipartite-scope.toml register --workspace ./my-workspace --id demo
bipartite-scope workspace --config bipartite-scope.toml reconcile --workspace ./my-workspace --id demo
bipartite-scope job --config bipartite-scope.toml list --workspace demo
bipartite-scope job --config bipartite-scope.toml status --job JOB_ID
bipartite-scope job --config bipartite-scope.toml watch --job JOB_ID
bipartite-scope job --config bipartite-scope.toml cancel --job JOB_ID
bipartite-scope job --config bipartite-scope.toml retry --job JOB_ID
bipartite-scope serve --config bipartite-scope.toml
bipartite-scope worker --config bipartite-scope.toml
bipartite-scope scheduler --config bipartite-scope.toml
```

`bipartite-scope database --config bipartite-scope.toml upgrade` applies the packaged Alembic migrations and is the recommended command for installed distributions. From a source checkout, `alembic upgrade head` is an equivalent direct Alembic workflow when the same configuration is present.

Operation commands use `--config` before their subcommand:

```bash
bipartite-scope operations --config bipartite-scope.toml summary
bipartite-scope operations --config bipartite-scope.toml metrics
bipartite-scope operations --config bipartite-scope.toml audit
bipartite-scope operations --config bipartite-scope.toml alerts
bipartite-scope operations --config bipartite-scope.toml diagnostics --job JOB_ID
bipartite-scope maintenance --config bipartite-scope.toml status
bipartite-scope maintenance --config bipartite-scope.toml enter
bipartite-scope maintenance --config bipartite-scope.toml exit
```

`operations metrics` returns the Prometheus text inside a JSON `metrics` field. `operations diagnostics` returns a path to a private local JSON file; omit `--job` for a bounded overview of recent jobs. Manual maintenance can be entered only when there are no running jobs. Exit removes a manual marker owned by that workflow, not a marker held by an active backup.

Retention and backup commands are described in their workflow sections below.

## Service HTTP API

Start a configured service with:

```bash
bipartite-scope serve --config bipartite-scope.toml
```

OpenAPI is available at `/docs` and `/redoc` when enabled. This is generated API documentation, not a repository `docs/` directory. Responses include `X-Request-ID`. Domain errors use `application/problem+json` with a stable code and request identifier.

The existing service routes retain the default `/api/v3` prefix. V4 operation endpoints use `/api/v4`; clients do not need to change the recommendation or task URLs when upgrading.

System and registry endpoints:

```text
GET  /api/v3/version
GET  /api/v3/health/live
GET  /api/v3/health/ready
GET  /api/v3/config
POST /api/v3/workspaces
GET  /api/v3/workspaces
GET  /api/v3/workspaces/{workspace_id}
POST /api/v3/workspaces/{workspace_id}/reconcile
POST /api/v3/workspaces/{workspace_id}/uploads
```

The readiness endpoint checks the database, Redis broker, and writable workspace storage. Uploads accept only CSV, JSONL, or NDJSON, use generated storage identifiers, enforce the configured byte limit, and reject path separators in client filenames.

Asynchronous submissions return `202 Accepted` and a `Location` header:

```text
POST /api/v3/workspaces/{workspace_id}/validate
POST /api/v3/workspaces/{workspace_id}/build
POST /api/v3/workspaces/{workspace_id}/update
POST /api/v3/workspaces/{workspace_id}/evaluate
POST /api/v3/workspaces/{workspace_id}/benchmark
POST /api/v3/workspaces/{workspace_id}/snapshots/{snapshot_id}/verify
```

Job inspection and control:

```text
GET  /api/v3/jobs
GET  /api/v3/jobs/{job_id}
POST /api/v3/jobs/{job_id}/cancel
POST /api/v3/jobs/{job_id}/retry
GET  /api/v3/jobs/{job_id}/events
GET  /api/v3/jobs/{job_id}/artifacts
GET  /api/v3/jobs/{job_id}/artifacts/{artifact_name}
```

Low-latency inference and snapshot pointer changes remain synchronous:

```text
GET  /api/v3/workspaces/{workspace_id}/query/{entity_id}
POST /api/v3/workspaces/{workspace_id}/recommend
POST /api/v3/workspaces/{workspace_id}/feedback
GET  /api/v3/workspaces/{workspace_id}/snapshots
POST /api/v3/workspaces/{workspace_id}/snapshots/{snapshot_id}/activate
POST /api/v3/workspaces/{workspace_id}/snapshots/{snapshot_id}/rollback
```

Workspace IDs accept letters, digits, periods, underscores, and hyphens. Resolved paths must remain inside the configured service root.

## V4 operation HTTP API

```text
GET  /api/v4/version
GET  /api/v4/operations/summary
GET  /api/v4/operations/metrics
GET  /api/v4/operations/audit
GET  /api/v4/operations/alerts
POST /api/v4/operations/alerts/{alert_id}/acknowledge
GET  /api/v4/workspaces/{workspace_id}/storage
POST /api/v4/workspaces/{workspace_id}/retention/plan
POST /api/v4/workspaces/{workspace_id}/retention/apply
POST /api/v4/workspaces/{workspace_id}/snapshots/{snapshot_id}/pin
GET  /api/v4/backups
POST /api/v4/backups
POST /api/v4/backups/{backup_id}/verify
POST /api/v4/workspaces/{workspace_id}/diagnostics
```

`operations/metrics` returns Prometheus-compatible text. Audit supports `limit` and `offset` with deterministic ordering. The alert endpoint checks current conditions and persists state changes. Acknowledgement does not suppress checking or clear an active condition.

Retention apply accepts the complete plan returned by retention plan and submits an asynchronous job. Pinning defaults to `pinned=true`; use `?pinned=false` to unpin. Backup creation takes `workspace_id` and `scope` (`workspace` or `service`). Backup verification takes a JSON body containing `workspace_id`. Diagnostic submission accepts an optional `job_id`.

These expensive submissions return `202 Accepted` and use the existing `/api/v3/jobs/...` inspection and SSE routes. Restore, recovery drill, manual maintenance, backup retention, and permanent trash purge are CLI workflows in this release.

## Asynchronous jobs and idempotency

A submitted job moves through durable states including `queued`, `running`, `retrying`, `succeeded`, `failed`, `cancelling`, and `cancelled`. Attempts, independent heartbeats, execution generations, progress events, results, errors, cancellation requests, and worker leases are recorded in the control database.

Send an `Idempotency-Key` header with asynchronous submissions when a client may retry. Reusing the same key for the same workspace, job kind, and input returns the existing job. Reusing it with different input returns a conflict. Build and update jobs are serialized per workspace so concurrent mutations cannot race over snapshots or event ledgers.

Cancellation is cooperative. A cancellation request is persisted immediately, but a running numerical operation stops only when it reaches a cancellation check. Failed or cancelled jobs may be explicitly retried. Worker code retries selected transient infrastructure failures with bounded backoff; input and integrity failures are terminal.

When scheduling is enabled, the recovery task marks expired running jobs as failed, releases their workspace leases, and republishes queued jobs whose earlier broker dispatch did not complete. Optional periodic incremental updates are disabled by default and must be explicitly enabled.

After running jobs have drained or expired, recovery also clears orphaned backup-maintenance markers only when the corresponding process lock is free. Live operation locks and manual maintenance markers remain protected. Failed jobs still require an explicit retry; marker recovery does not replay them.

Heartbeat renewal is independent of numerical progress. A training phase may remain at the same progress value without being treated as a lost worker. Recovery invalidates the old execution generation; the old worker cannot publish results after losing ownership. A retry validates input and parent state and restarts the operation. Arbitrary optimizer-state checkpoint resume is not implemented.

The broker may redeliver a task. An atomic job claim prevents a duplicate delivery from starting a second attempt, while event IDs and batches protect interaction application. These safeguards do not promise universal exactly-once execution across arbitrary external side effects.

## Server-sent progress events

Subscribe to a job with:

```text
GET /api/v3/jobs/{job_id}/events
```

Each SSE record has a monotonically increasing numeric ID. Resume after a disconnect with either `Last-Event-ID` or the `after` query parameter. The server replays persisted database events before waiting for Redis notifications, so reconnect correctness does not depend on retaining a Redis pub/sub message. Idle streams emit heartbeat comments. A stream closes after the job reaches a terminal state and all persisted events have been delivered.

## V2 REST compatibility

The unversioned V2 routes remain enabled by default:

```text
GET  /health
POST /workspaces/{workspace_id}/build
GET  /workspaces/{workspace_id}/snapshots/{snapshot_id}/query/{entity_id}
POST /workspaces/{workspace_id}/update
POST /workspaces/{workspace_id}/recommend
POST /workspaces/{workspace_id}/feedback
POST /workspaces/{workspace_id}/evaluate
GET  /workspaces/{workspace_id}/snapshots
POST /workspaces/{workspace_id}/snapshots/{snapshot_id}/activate
POST /workspaces/{workspace_id}/snapshots/{snapshot_id}/verify
```

These routes execute in the API process and exist for compatibility, not for new long-running integrations. New clients should submit expensive operations through `/api/v3` and monitor their jobs.

## Logs, audit, metrics, and alerts

Correlated JSON logs are stored under `<workspace-root>/.operations/logs/`. Records contain UTC time, level, component, event, and available request, workspace, job, attempt, snapshot, and duration fields. Logging rotates by byte size and backup count. Known secret fields, URL passwords, and credential patterns are redacted.

Audit records preserve actions and outcomes in the control database. Sources identify the calling component (`cli`, `api`, `worker`, `scheduler`, or `system`), not an authenticated person. Job submission and control, snapshot pointer changes, maintenance, backup, diagnostics, and cleanup can be traced without inventing user identities.

The operation summary combines durable job counts, attempts, a recent timing sample, workspace sizes, disk availability, backup history, alerts, and aggregated HTTP requests. HTTP metrics use standard route templates and bounded labels. Worker and task IDs are available in logs and history rather than metric labels. Prometheus scraping is optional; Grafana and a hosted monitoring platform are not included.

Local alerts cover excessive queue wait, stale running-task heartbeats, repeated recent failures, low disk space, overdue backups, recorded backup verification failures, and recorded restore-drill failures. They preserve first/last seen times, acknowledgement, resolution, and recurrence. Acknowledged alerts stay acknowledged while their condition persists; cleared conditions resolve, and a later recurrence reopens them. Checks run through operation commands and endpoints; external message delivery is not implemented.

Diagnostics are bounded exports of redacted configuration, runtime version, operation summary, recent jobs, attempts, progress history, alerts, and relevant log excerpts. They do not export the full business dataset. Review a diagnostic file before sharing it because local paths and business identifiers may still be sensitive.

## Resource admission and storage

Submission checks global and per-workspace queue counts, available disk space, and the optional workspace size quota. Worker claims enforce the global heavy-job slot limit. Uploads retain request-size limits and check available storage. `workspace_max_bytes = 0` disables that quota, not the minimum-free-space check.

These limits are local admission controls. They are not CPU or memory isolation, nor a hard filesystem quota. Numerical output size is not predicted perfectly before execution. The storage endpoint reports real file totals by workspace category, free disk bytes, and quarantine size without following symlinks.

## Retention and recoverable cleanup

Generate a plan, inspect its eligible and protected paths, then apply that exact JSON plan:

```bash
bipartite-scope retention --config bipartite-scope.toml plan --workspace workspaces/demo > retention-plan.json
bipartite-scope retention --config bipartite-scope.toml apply --workspace workspaces/demo --plan retention-plan.json
bipartite-scope retention --config bipartite-scope.toml purge --workspace workspaces/demo
```

Plans include a hash, path fingerprints, reasons, protection, and estimated bytes. Apply rechecks current protection and file content. A changed file, a newly pinned snapshot, or a new pending-task reference makes the old plan invalid.

Active and pinned snapshots, the newest verified rollback snapshots, configured recent candidates, and queued/running task references are protected. Invalid active pointers, unreadable paths, symlinks, and unverifiable snapshots are handled conservatively. Event ledgers, feedback, primary datasets, configuration, and event indexes are not cleanup targets.

Apply moves eligible snapshot assets, old evaluation reports, and unused old uploads into `<workspace>/.trash/<operation-id>/`, with a tombstone inventory. Snapshot manifests are preserved in `artifacts/lineage/`. This move is recoverable but does not release disk space. Its result reports `quarantined_bytes` and `reclaimed_disk_bytes = 0`.

Trash purge defaults to a dry run. To permanently remove validated expired quarantine bundles, inspect the dry-run output and explicitly add `--apply`:

```bash
bipartite-scope retention --config bipartite-scope.toml purge --workspace workspaces/demo --apply
```

Purge checks age, completed tombstones, safe paths, current references, fingerprints, and exact inventories. Changed or unexpected contents remain protected. The purge result reports reclaimed file bytes; permanently removed files cannot be recovered by the application.

## Consistent backup and recovery

Create a workspace backup or a full-service backup:

```bash
bipartite-scope backup --config bipartite-scope.toml create --workspace workspaces/demo
bipartite-scope backup --config bipartite-scope.toml create --workspace workspaces --scope service
bipartite-scope backup --config bipartite-scope.toml list
bipartite-scope backup --config bipartite-scope.toml verify --backup BACKUP_DIRECTORY
```

A workspace bundle contains configuration, data, event ledgers, event index, snapshots, lineage, and the active pointer. Reports and staged uploads are included by default and can be excluded by configuration. A service bundle also contains every discovered/registered workspace and the exported control database, including task and operation history. Redis is not a backup fact source.

Backup coordinates a write pause and local writer locks. Full-service backup waits for running operations up to `backup.wait_seconds`; it fails if a safe pause cannot be reached. The bundle is staged and verified before publication. Its manifest records scope, version, workspace inventory, relative file paths, byte sizes, and SHA-256 digests. Verification checks exact inventories, snapshots, active state, event ledger/index agreement, and supported database integrity.

Restore into a new or empty target:

```bash
bipartite-scope backup --config bipartite-scope.toml restore --backup BACKUP_DIRECTORY --target restored-workspaces
bipartite-scope backup --config bipartite-scope.toml restore --backup SERVICE_BACKUP_DIRECTORY --target restored-workspaces --database-url sqlite+pysqlite:///restored_service.sqlite3
```

Full-service restoration requires an empty target control database; do not point it at the source database. Restoration verifies the bundle, restores files to a staging directory, reconciles stored paths and workspace references, checks state, and publishes the new target. Historical successful tasks remain historical. Previously queued, running, retrying, or cancelling tasks are interrupted, and old leases and delivery identifiers are cleared. They are not automatically replayed.

Run a recovery drill before relying on a bundle:

```bash
bipartite-scope backup --config bipartite-scope.toml drill --backup BACKUP_DIRECTORY
```

SQLite drills use a temporary target database. A PostgreSQL service drill requires an explicitly supplied empty `--database-url` and compatible PostgreSQL utilities. It uses that target database, so provide a dedicated empty target for each drill. The drill restores into temporary workspace files, checks event idempotency and pending state, and performs representative queries when a usable model is present. The result is a structured JSON report.

Backup retention keeps the newest verified `backups_keep` bundles per scope/source and protects unverifiable bundles. It uses the same plan, quarantine, and explicit-purge model:

```bash
bipartite-scope backup --config bipartite-scope.toml retention-plan > backup-retention-plan.json
bipartite-scope backup --config bipartite-scope.toml retention-apply --plan backup-retention-plan.json
bipartite-scope backup --config bipartite-scope.toml purge
bipartite-scope backup --config bipartite-scope.toml purge --apply
```

Backup bundles are not encrypted or signed. Keep separately protected copies when recovery from source-disk failure matters. A file hash detects changes; it does not authenticate a bundle from an untrusted party.

## Algorithm overview

The canonical input is `G=(U,V,E,X)`: a binary user-item incidence matrix `A` and a nonnegative user-feature matrix `X`. The build computes normalized structure and attribute transitions:

```text
P_s = D_U^-1 A D_V^-1 A^T
P_x = D_X^-1 X D_F^-1 X^T
```

The transitions are mixed before restart diffusion. Deterministic row-wise Top-k truncation is applied after every step, and the final zero-diagonal affinity `W` is symmetric.

The sparse dual-view encoder follows user-to-item-to-user propagation and affinity refinement. It produces reusable user embeddings `Z`, item embeddings, and soft latent assignments `S`. Training minimizes structure-attribute normalized cut, original bipartite support loss, and assignment orthogonality without labels. Finite-loss checks stop invalid builds.

Community search combines latent-assignment similarity, local affinity, shared supporting items, and the change in Bipartite-aware Local Conductance. The structure gate records accepted and rejected expansion evidence. Query-time search never retrains the model.

## Incremental updates

An update appends new user and item IDs after existing indexes so warm-start state remains aligned. The affected set includes changed users, users connected through changed items, feature-related users, graph-hop neighbors, and previous or new affinity neighbors.

The engine computes a candidate sparse affinity, replaces the closed affected region, restores symmetric Top-k structure, and keeps unrelated rows. It falls back to a full affinity build when the affected ratio exceeds the configured threshold or numerical integrity checks fail. The neural model is always trained on the complete sparse graph for the reduced `warm_start_epochs` budget. Compatible parameters and existing item embeddings are reused; new item embeddings use deterministic initialization.

Every update snapshot records its parent, build mode, schema version, batch hash, event counts, affected counts, fallback reason, hashes, diagnostics, and reused parameters. Event application remains explicit batch processing. Redis and Celery schedule batches; they do not turn the model into a per-event online learner.

## Recommendation and feedback

Recommendation first finds the query user's community, then draws candidate items from supporting users. Items already consumed, removed, or explicitly disliked by the query user are excluded.

The final score combines normalized community support, affinity-weighted peer support, weighted behavior, recency decay, and a popularity penalty. Results contain the snapshot ID, exact score components, supporting users, and a deterministic English explanation derived from those calculations.

Feedback validates and stores one event but does not retrain immediately. A later explicit or scheduled update consumes pending events and creates a candidate snapshot.

## Offline evaluation

Evaluation requires timestamped positive events. Legacy edges remain usable for building and recommendation but do not qualify as temporal evidence. Eligible users are split chronologically with the last positive event held out; users below `minimum_positive_events` are excluded and counted.

The report compares BipartiteScope with popularity and user-item co-occurrence baselines. It includes Precision@K, Recall@K, HitRate@K, NDCG@K, MRR, and catalog coverage. Outputs are written under:

```text
reports/evaluations/<evaluation-id>/metrics.json
reports/evaluations/<evaluation-id>/configuration.json
reports/evaluations/<evaluation-id>/report.md
```

Evaluation never activates a candidate. Promotion remains a manual decision.

## Snapshots and integrity

Snapshots are immutable directories containing a manifest plus sparse graph, weighted interaction, affinity, embedding, assignment, semantic-index, and model-state assets. Every asset has a recorded byte size and SHA-256 digest. Writes use a temporary directory and an atomic rename.

The first successful full build activates automatically only when no active pointer exists. Later builds are candidates. Activation verifies all assets before atomically changing `latest.json`. Rollback performs the same verified activation against an older snapshot and never deletes data.

V1 and V2 snapshots remain readable. Missing weighted interactions are interpreted as binary weights, and missing lineage fields as a full build. A separate asset schema version keeps application-version changes from invalidating compatible payloads. Pinning protects a verified snapshot from retention; unpinning does not delete it.

## Migration from V2 or the V3 development service

V4 keeps the existing workspace format and uses additive control-plane migrations. Upgrade an existing checkout as follows:

1. Stop service processes and keep an independent copy of the existing workspace and control database.
2. Install the V4 package with the extras required for the selected workflow.
3. Configure the same workspace root, a local SQLite or PostgreSQL control database, and Redis for asynchronous execution.
4. Apply migrations with `bipartite-scope database --config bipartite-scope.toml upgrade`. The schema advances through `0001_v3_control_plane` to `0002_v4_operations`.
5. Register or reconcile existing workspaces, then verify their active snapshots.
6. Restart the selected service processes with the same configuration and inspect readiness and operation summary.
7. Create a verified V4 backup and complete a recovery drill before enabling routine retention.

Graph assets do not move into the control database. Existing core imports, CLI workflows, `/api/v3` service routes, and default unversioned compatibility routes remain available. New operation interfaces use `/api/v4`. The V3 service scope is included in this release; a prior public V3 tag is not required for migration.

Development databases created without an Alembic revision require a migration review. V4 refuses to silently stamp or overwrite unversioned service tables. Preserve their independent backup and reconcile their schema explicitly before migration; do not delete a populated database to bypass this check.

## Testing and repository policy

```bash
python -m ruff check .
python -m compileall -q src migrations tests
python -m unittest discover -s tests -v
python -m build
```

CI targets Python 3.11 and 3.12. Tests cover core mathematical behavior, data validation, event idempotency, incremental/full graph equivalence, recommendation and evaluation, snapshot compatibility, typed settings, migrations, durable jobs, execution ownership, heartbeats, CLI, HTTP interfaces, SSE, logs, metrics, alerts, retention safety, backup verification, new-target recovery, and package installation.

Test data is generated under temporary directories. These checks do not establish production throughput or a live PostgreSQL/Redis deployment result; validate selected external services separately when using them.

Repository policy requires thirteen package files: `__init__.py`, `api.py`, `config.py`, `core.py`, `database.py`, `interface.py`, `maintenance.py`, `observability.py`, `policies.py`, `recommendation.py`, `reliability.py`, `storage.py`, and `tasks.py`. Detailed guidance is consolidated here, with no repository `docs/` tree, no committed CSV/JSONL/NDJSON datasets, and English-only tracked text.

## V4 limitations

- There is no authentication, authorization, account model, or tenant isolation.
- The API is not safe for direct public exposure.
- Operation coordination uses local POSIX locks on one host. Distributed filesystem coordination, Windows support for the operation layer, and high-availability failover are not implemented.
- Workspaces and backups use local filesystem storage. Object storage and off-host backup delivery are not included.
- Cancellation is cooperative and cannot interrupt every numerical kernel immediately.
- Scheduled updates batch pending events; they are not real-time stream processing.
- Snapshot activation remains a manual decision and is not driven by automatic metric thresholds.
- Resource limits are admission controls, not CPU/memory isolation or a hard filesystem quota.
- Backup uses a write pause, not a zero-downtime distributed checkpoint. PostgreSQL utility compatibility remains the operator's responsibility.
- Quarantine does not free disk space; explicit purge does. There is no automatic irreversible cleanup.
- Backups and diagnostic exports are not encrypted or cryptographically signed.
- Metrics and local alerts have no built-in external notification channel.
- There is no frontend, public benchmark dataset, managed monitoring dashboard, or autoscaling policy.

## Troubleshooting

- If configuration is rejected, run `bipartite-scope config --file PATH validate` and confirm that unknown keys are removed.
- If a remote bind is rejected, return to `127.0.0.1`; do not use `allow_insecure_remote` as a substitute for authentication.
- If readiness returns `503`, inspect its database, Redis, and storage checks, then run `bipartite-scope doctor` with the same configuration.
- If a job remains queued, inspect the worker process, broker URL, and job record; the durable job can be republished after transport recovery.
- If recovery marks a job as failed with `worker_lost`, verify that no worker is still executing it before requesting a retry.
- If a submission is refused, inspect the returned queue, disk, or storage code. Quarantining files does not increase free disk space; inspect validated trash-purge output before explicit deletion.
- If an operation reports a busy workspace, let its current writer finish. Manual maintenance cannot be entered while jobs are running.
- If a retention plan is stale, regenerate it and inspect the new protection. Do not edit its hash or target inventory to force application.
- If a backup is rejected, inspect its manifest, file hashes, active snapshot, and event-index consistency. Unverifiable bundles remain protected by retention.
- If a service restore rejects its database, provide a new empty target rather than the source database. Interrupted tasks require explicit resubmission.
- If a PostgreSQL drill is rejected, provide an explicit empty target database URL and compatible `pg_dump`/`pg_restore` utilities.
- If an alert remains acknowledged, the condition still exists. Acknowledgement records review; it does not repair the condition.
- If diagnostics are needed, export a job-specific file and review its paths and identifiers before sharing it.
- If an upload is rejected, use CSV, JSONL, or NDJSON and remain under `request_max_bytes`.
- If training is unavailable, install `bipartite-scope[train]` with a PyTorch build compatible with the active Python runtime.
- If validation fails, confirm all edge users have one finite nonnegative feature row and every row has the same feature columns.
- If an update rejects a user, provide its feature row with the unchanged feature dimension.
- If an event is ignored, inspect its event ID for a prior workspace registration.
- If an update performs a full fallback, inspect `affected_ratio` and `fallback_reason` in its result and snapshot manifest.
- If a query stops early, inspect its trace for an empty frontier or a failed BLC structure gate.
- If activation fails, run snapshot verification; modified or missing assets are never activated.
- If an SSE client reconnects, pass its last numeric event ID through `Last-Event-ID` or `after`.
- If evaluation has no eligible users, ingest at least the configured number of timestamped positive events per evaluated user.
