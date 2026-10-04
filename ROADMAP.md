# Roadmap

- [x] M0: public project skeleton, Apache-2.0, package metadata, CI-ready layout.
- [x] M1: canonical graph and CSV validation (included in the first functional increment).
- [x] M2: sparse structure-attribute affinity builder.
- [x] M3: dual-view encoder and cut-guided offline trainer.
- [x] M4: immutable snapshots and BLC online query with explanations.
- [x] M5: CLI, REST API, adapters, and v0.1.0 core release.
- [x] M6: workspace-first CLI, TOML configuration, input quality reports, and verified snapshots (v1.0.0 final release).
- [x] M7: incremental updates, recommendation, feedback, evaluation, and snapshot promotion controls (v2.0.0).
- [x] Service foundation developed under V3: typed configuration, SQLite/PostgreSQL control plane, Redis and Celery asynchronous jobs, versioned API, and resumable progress events. Included in the V4 release.
- [x] V4: personal operation management, independent heartbeats and execution fencing, correlated logs and audit, persistent metrics and alerts, resource admission, safe retention, consistent backups, new-target recovery, and restore drills.
- [ ] V5: complete frontend, backend integration, large-scale performance, and closed-loop model operations.

The current scope uses local workspaces and a shared filesystem. Authentication, multi-user administration, tenant isolation, and object storage are not part of V4. Future roadmap items are development targets, not delivered capabilities.
