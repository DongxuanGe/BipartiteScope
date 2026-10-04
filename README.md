# BipartiteScope

BipartiteScope 4.0.0 is a local recommendation and community-search engine for attributed bipartite graphs. It combines incremental graph updates, explainable recommendations, offline evaluation, and immutable snapshots with durable asynchronous jobs and personal operation tools.

V4 adds independent worker heartbeats and execution fencing, correlated logs, operation audit, persistent metrics and alerts, queue and storage admission limits, safe retention, consistent backups, new-target restoration, and recovery drills. Graph and model files remain in local workspaces. SQLite or PostgreSQL stores service state; Redis and Celery support the asynchronous service.

The project is intended for one trusted personal operator. It has no authentication or frontend. The API binds to loopback by default, and the existing Python API, CLI workflows, `/api/v3` service routes, and V1/V2 snapshots remain supported. V4 operation routes use `/api/v4`.

## Quick start

Run the standalone workflow without external services:

```bash
python -m pip install -e '.[train]'
bipartite-scope init my-workspace
# Add your own edges.csv and features.csv under my-workspace/data/.
bipartite-scope validate --workspace my-workspace
bipartite-scope build --workspace my-workspace
bipartite-scope query --workspace my-workspace --entity YOUR_USER_ID
```

Install the asynchronous service and inspect its configuration:

```bash
python -m pip install -e '.[train,service]'
bipartite-scope config show
bipartite-scope database upgrade
bipartite-scope operations summary
```

Run the API, worker, and scheduler against the same configuration and local workspace root. Redis is required for broker-backed asynchronous execution. Docker is optional and is not required for the V4 workflow.

See [DOCUMENTATION.md](DOCUMENTATION.md) for installation, data schemas, configuration, Python API, CLI, HTTP API, algorithms, operation management, backup and recovery, migration, testing, and troubleshooting.

Source and releases: [DongxuanGe/BipartiteScope](https://github.com/DongxuanGe/BipartiteScope).

## License

Apache-2.0. See [LICENSE](LICENSE).
