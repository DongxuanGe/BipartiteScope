# Security policy

## Reporting a vulnerability

Report suspected security issues privately to the repository maintainer or through a private GitHub security advisory when that feature is available. Do not open a public issue containing credentials, connection strings, private graph records, uploaded data, personally identifying information, or exploit details.

Include the affected version, deployment mode, reproduction steps, expected impact, and any suggested mitigation. Do not access data that is not yours or test against a public deployment without authorization.

## V4 deployment boundary

BipartiteScope 4.0.0 is intended for one trusted personal operator. It has no authentication, authorization, account model, or tenant isolation. Anyone who can reach the API can operate its workspaces and management endpoints. Do not expose it directly to the public internet.

The default service host is `127.0.0.1`. A non-loopback bind is rejected unless `allow_insecure_remote` is explicitly enabled. That setting only acknowledges the risk; it does not add security. If access beyond the local host is required, place the service behind a separately managed trusted gateway that provides TLS, authentication, authorization, request limits, and audit controls.

Restrict filesystem access to the workspace root, service database, Redis state, backups, logs, and diagnostics. The retained optional Compose configuration publishes only the API on loopback and does not publish PostgreSQL or Redis. Containers are not required for V4; the same access boundary applies to directly launched processes.

## Data and secret handling

- Use environment injection or a protected local `.env` file for connection secrets. Never commit `.env` or production connection strings.
- Configuration output redacts URL passwords, and operation logs and diagnostics redact recognized secret fields and credential patterns. Redaction is not a guarantee that arbitrary business data contains no sensitive information; review diagnostic files before sharing them.
- Treat workspace inputs, normalized events, feedback, reports, exports, model assets, control-plane rows, and backups as sensitive application data.
- Grant the API, worker, scheduler, PostgreSQL, and Redis only the filesystem and network access they require.
- Use encrypted storage and encrypted backups when local policy requires them; encryption at rest is not supplied by the application.
- Retention moves eligible files to recoverable quarantine. Explicit trash purge permanently deletes validated expired quarantine bundles and cannot be undone by the application.
- Backups are local integrity-checked directory bundles, not encrypted or signed archives. Store independent protected copies if the source disk is the only backup location.

## Implemented safeguards

Workspace identifiers and resolved paths are validated. Uploads use generated identifiers, restricted extensions, byte limits, and SHA-256 digests. Snapshots are verified before activation. Write locks coordinate local workspace changes; job heartbeats and execution generations reject stale attempts. Queue, heavy-job, disk, and storage admission limits reduce accidental overload.

Retention plans are hashed, checked against current protection, and rejected when targets change. Active, pinned, rollback, and pending-task assets are protected. Purge accepts only validated quarantine bundles. Backup verification checks allowed relative paths, exact inventories, hashes, snapshot state, and event-index consistency. Restoration requires a new target and does not replay interrupted tasks automatically.

These controls assume trusted local files and processes. They do not provide an authorization boundary, malware scanning, sandboxed numerical execution, or distributed isolation. Only accept data and configuration from the trusted operator.

## Dependency and release hygiene

Run the supported Python versions, update the selected database and Redis dependencies, and review migration guidance before upgrading. Verify backups and complete a recovery drill before relying on a backup. A passing readiness check confirms dependency availability, not authorization or public-exposure safety.
