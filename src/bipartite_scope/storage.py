from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from scipy import sparse

from .core import (
    AffinityBuilder,
    AffinityConfig,
    BuildConfig,
    CanonicalBipartiteGraph,
    EncoderConfig,
    EvaluationConfig,
    Event,
    IncrementalConfig,
    ModelSnapshot,
    QueryConfig,
    RecommendationConfig,
)
from .reliability import execution_commit, execution_mutation, snapshot_mutation, workspace_mutation

CONFIG_FILENAME = "bipartitescope.toml"
EVENT_TYPES = frozenset({"view", "click", "favorite", "purchase", "rating", "dislike", "remove"})
_DEFAULT_CONFIG = """[data]
edges = "data/edges.csv"
features = "data/features.csv"
delimiter = ","

[core.affinity]
beta = 0.5
restart = 0.3
steps = 3
top_k = 64

[core.encoder]
hidden_dim = 64
layers = 2
latent_groups = 8
epochs = 100
learning_rate = 0.01
lambda_b = 0.5
lambda_o = 0.1
seed = 7

[query]
size_budget = 20
rho = 0.5
eta_s = 0.65
delta_0 = 0.02
semantic_recall_budget = 100

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
"""


class SnapshotIntegrityError(RuntimeError):
    pass


class InputValidationError(ValueError):
    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__("; ".join(report.errors) or "input validation failed")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    edges_path: str
    features_path: str
    delimiter: str
    counts: Mapping[str, int]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "valid": self.valid}


def _read_csv(path: Path, delimiter: str = ",") -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return list(reader), tuple(reader.fieldnames or ())


def validate_csv_graph(
    edges_path: str | Path,
    features_path: str | Path,
    *,
    delimiter: str = ",",
) -> ValidationReport:
    edges_file, features_file = Path(edges_path), Path(features_path)
    errors: list[str] = []
    warnings: list[str] = []
    edge_rows: list[dict[str, str]] = []
    feature_rows: list[dict[str, str]] = []
    edge_columns: tuple[str, ...] = ()
    feature_columns: tuple[str, ...] = ()
    for path, label in ((edges_file, "edges"), (features_file, "features")):
        if not path.is_file():
            errors.append(f"{label} file does not exist: {path}")
    if not errors:
        edge_rows, edge_columns = _read_csv(edges_file, delimiter)
        feature_rows, feature_columns = _read_csv(features_file, delimiter)
    if not {"u_id", "v_id"}.issubset(edge_columns):
        errors.append("edge file requires u_id and v_id columns")
    feature_fields = tuple(column for column in feature_columns if column != "u_id")
    if "u_id" not in feature_columns or not feature_fields:
        errors.append("feature file requires u_id and one or more feature columns")
    edge_pairs = [(row.get("u_id", ""), row.get("v_id", "")) for row in edge_rows]
    feature_ids = [row.get("u_id", "") for row in feature_rows]
    if edge_rows and any(not user or not item for user, item in edge_pairs):
        errors.append("edge file contains blank identifiers")
    if feature_rows and any(not entity for entity in feature_ids):
        errors.append("feature file contains a blank u_id")
    duplicate_edges = len(edge_pairs) - len(set(edge_pairs))
    duplicate_features = len(feature_ids) - len(set(feature_ids))
    if duplicate_edges:
        warnings.append(f"{duplicate_edges} duplicate edge rows will be merged")
    if duplicate_features:
        errors.append(f"feature file contains {duplicate_features} duplicate u_id rows")
    invalid_values = 0
    negative_values = 0
    for row in feature_rows:
        for column in feature_fields:
            try:
                value = float(row.get(column, ""))
            except ValueError:
                invalid_values += 1
                continue
            if value < 0 or not np.isfinite(value):
                negative_values += 1
    if invalid_values:
        errors.append(f"feature file contains {invalid_values} non-numeric values")
    if negative_values:
        errors.append(f"feature file contains {negative_values} negative or non-finite values")
    missing = sorted({user for user, _ in edge_pairs} - set(feature_ids))
    if missing:
        errors.append(
            f"{len(missing)} U-side IDs used by edges have no feature row (for example: {missing[0]})"
        )
    unused = len(set(feature_ids) - {user for user, _ in edge_pairs})
    if unused:
        warnings.append(f"{unused} feature rows have no incident edge and remain isolated")
    if not edge_rows:
        errors.append("edge file contains no data rows")
    if not feature_rows:
        errors.append("feature file contains no data rows")
    return ValidationReport(
        str(edges_file),
        str(features_file),
        delimiter,
        {
            "edge_rows": len(edge_rows),
            "unique_edges": len(set(edge_pairs)),
            "feature_rows": len(feature_rows),
            "feature_dimensions": len(feature_fields),
            "duplicate_edges": duplicate_edges,
            "unused_feature_rows": unused,
        },
        tuple(errors),
        tuple(warnings),
    )


def load_csv_graph(
    edges_path: str | Path,
    features_path: str | Path,
    *,
    delimiter: str = ",",
) -> CanonicalBipartiteGraph:
    report = validate_csv_graph(edges_path, features_path, delimiter=delimiter)
    if not report.valid:
        raise InputValidationError(report)
    edge_rows, _ = _read_csv(Path(edges_path), delimiter)
    feature_rows, feature_columns = _read_csv(Path(features_path), delimiter)
    names = tuple(column for column in feature_columns if column != "u_id")
    features = {row["u_id"]: [float(row[name]) for name in names] for row in feature_rows}
    edges = [(row["u_id"], row["v_id"]) for row in edge_rows]
    return CanonicalBipartiteGraph.from_edges_and_features(edges, features, feature_names=names)


def load_feature_rows(
    path: str | Path,
    *,
    delimiter: str = ",",
) -> tuple[tuple[str, ...], dict[str, tuple[float, ...]]]:
    rows, columns = _read_csv(Path(path), delimiter)
    names = tuple(column for column in columns if column not in {"u_id", "user_id"})
    identity = "u_id" if "u_id" in columns else "user_id" if "user_id" in columns else ""
    if not identity or not names:
        raise ValueError("feature updates require u_id or user_id and feature columns")
    result: dict[str, tuple[float, ...]] = {}
    for row in rows:
        user = row[identity].strip()
        if not user or user in result:
            raise ValueError("feature update identifiers must be nonblank and unique")
        values = tuple(float(row[name]) for name in names)
        if not np.isfinite(values).all() or min(values, default=0) < 0:
            raise ValueError("feature update values must be finite and nonnegative")
        result[user] = values
    return names, result


def normalize_event(record: Mapping[str, Any]) -> Event:
    required = ("event_id", "user_id", "item_id", "event_type", "event_time")
    values = {name: str(record.get(name, "")).strip() for name in required}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"event is missing required fields: {', '.join(missing)}")
    event_type = values["event_type"].lower()
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unsupported event_type: {event_type}")
    try:
        event_value = float(record.get("event_value", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("event_value must be numeric") from exc
    if not np.isfinite(event_value) or event_value < 0:
        raise ValueError("event_value must be finite and nonnegative")
    try:
        timestamp = datetime.fromisoformat(values["event_time"])
    except ValueError as exc:
        raise ValueError("event_time must be a valid ISO 8601 timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("event_time must include a timezone")
    normalized_time = timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return Event(
        values["event_id"],
        values["user_id"],
        values["item_id"],
        event_type,
        event_value,
        normalized_time,
    )


def load_events(path: str | Path) -> tuple[Event, ...]:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        rows, _ = _read_csv(source)
    elif source.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"event on line {line_number} must be an object")  # noqa: TRY004
            rows.append(value)
    else:
        raise ValueError("event input must be CSV or JSONL")
    events: list[Event] = []
    for index, row in enumerate(rows, 2 if source.suffix.lower() == ".csv" else 1):
        try:
            events.append(normalize_event(row))
        except ValueError as exc:
            raise ValueError(f"invalid event record {index}: {exc}") from exc
    return tuple(events)


def _inside(root: Path, value: str | Path) -> Path:
    path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path escapes workspace root: {value}")
    return path


@dataclass(frozen=True, slots=True)
class Workspace:
    root: Path
    config: Mapping[str, Any]

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def state_database(self) -> Path:
        return self.root / "state.sqlite3"

    @property
    def data_paths(self) -> tuple[Path, Path, str]:
        data = self.config["data"]
        return (
            _inside(self.root, data["edges"]),
            _inside(self.root, data["features"]),
            data.get("delimiter", ","),
        )

    def build_config(self) -> BuildConfig:
        core = self.config["core"]
        return BuildConfig(
            AffinityConfig(**core["affinity"]),
            EncoderConfig(**core["encoder"]),
            self.config.get("query", {}).get("semantic_recall_budget", 100),
            2,
        )

    def query_config(self, size_budget: int | None = None) -> QueryConfig:
        values = dict(self.config.get("query", {}))
        if size_budget is not None:
            values["size_budget"] = size_budget
        return QueryConfig(**values)

    def incremental_config(self) -> IncrementalConfig:
        return IncrementalConfig(**self.config.get("incremental", {}))

    def recommendation_config(self, top_n: int | None = None) -> RecommendationConfig:
        values = dict(self.config.get("recommendation", {}))
        if top_n is not None:
            values["top_n"] = top_n
        return RecommendationConfig(**values)

    def evaluation_config(self, k: int | None = None) -> EvaluationConfig:
        values = dict(self.config.get("evaluation", {}))
        if k is not None:
            values["k"] = k
        return EvaluationConfig(**values)

    def validate(self) -> ValidationReport:
        edges, features, delimiter = self.data_paths
        return validate_csv_graph(edges, features, delimiter=delimiter)

    def load_graph(self) -> CanonicalBipartiteGraph:
        edges, features, delimiter = self.data_paths
        return load_csv_graph(edges, features, delimiter=delimiter)


@workspace_mutation
def init_workspace(root: str | Path) -> Path:
    target = Path(root).resolve()
    if target.exists() and any(path.name != ".write.lock" for path in target.iterdir()):
        raise FileExistsError(f"workspace is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for name in ("data", "artifacts", "exports", "reports"):
        (target / name).mkdir(exist_ok=True)
    (target / CONFIG_FILENAME).write_text(_DEFAULT_CONFIG, encoding="utf-8")
    return target


def load_workspace(root: str | Path) -> Workspace:
    target = Path(root).resolve()
    config_path = target / CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"workspace configuration not found: {config_path}")
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    for section in ("data", "core", "query"):
        if section not in config:
            raise ValueError(f"workspace configuration is missing [{section}]")
    if not {"affinity", "encoder"}.issubset(config["core"]):
        raise ValueError("workspace configuration requires affinity and encoder sections")
    return Workspace(target, config)


def _event_connection(workspace: Workspace) -> sqlite3.Connection:
    connection = sqlite3.connect(workspace.state_database)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS events ("
        "event_id TEXT PRIMARY KEY, payload TEXT NOT NULL, source TEXT NOT NULL, "
        "applied_snapshot TEXT, created_at TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS batches ("
        "batch_hash TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL, applied_at TEXT NOT NULL)"
    )
    return connection


def _event_payload(event: Event) -> dict[str, Any]:
    return asdict(event)


@workspace_mutation
@execution_mutation
def register_events(
    workspace: Workspace,
    events: Iterable[Event],
    *,
    source: str,
    ledger_name: str = "events.jsonl",
) -> tuple[tuple[Event, ...], tuple[str, ...]]:
    accepted: list[Event] = []
    duplicate_ids: list[str] = []
    connection = _event_connection(workspace)
    try:
        connection.execute("BEGIN")
        for event in events:
            payload = json.dumps(_event_payload(event), sort_keys=True, separators=(",", ":"))
            cursor = connection.execute(
                "INSERT OR IGNORE INTO events VALUES (?, ?, ?, NULL, ?)",
                (event.event_id, payload, source, datetime.now(UTC).isoformat()),
            )
            if cursor.rowcount:
                accepted.append(event)
            else:
                duplicate_ids.append(event.event_id)
        if accepted:
            for name in sorted({"events.jsonl", ledger_name}):
                with (workspace.data / name).open("a", encoding="utf-8") as handle:
                    for event in accepted:
                        handle.write(
                            json.dumps(_event_payload(event), sort_keys=True, separators=(",", ":"))
                            + "\n"
                        )
                    handle.flush()
                    os.fsync(handle.fileno())
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return tuple(accepted), tuple(duplicate_ids)


def pending_events(workspace: Workspace) -> tuple[Event, ...]:
    connection = _event_connection(workspace)
    try:
        rows = connection.execute(
            "SELECT payload FROM events WHERE applied_snapshot IS NULL ORDER BY created_at, event_id"
        ).fetchall()
    finally:
        connection.close()
    return tuple(normalize_event(json.loads(row[0])) for row in rows)


def all_events(workspace: Workspace) -> tuple[Event, ...]:
    connection = _event_connection(workspace)
    try:
        rows = connection.execute(
            "SELECT payload FROM events ORDER BY created_at, event_id"
        ).fetchall()
    finally:
        connection.close()
    return tuple(normalize_event(json.loads(row[0])) for row in rows)


@workspace_mutation
@execution_mutation
def mark_events_applied(
    workspace: Workspace,
    event_ids: Sequence[str],
    snapshot_id: str,
    batch_hash: str,
) -> None:
    connection = _event_connection(workspace)
    try:
        connection.executemany(
            "UPDATE events SET applied_snapshot = ? WHERE event_id = ? AND applied_snapshot IS NULL",
            [(snapshot_id, event_id) for event_id in event_ids],
        )
        connection.execute(
            "INSERT OR REPLACE INTO batches VALUES (?, ?, ?)",
            (batch_hash, snapshot_id, datetime.now(UTC).isoformat()),
        )
        connection.commit()
    finally:
        connection.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_graph(snapshot: ModelSnapshot) -> str:
    digest = hashlib.sha256()
    for value in (*snapshot.graph.u_ids, "|", *snapshot.graph.v_ids):
        digest.update(value.encode())
        digest.update(b"\0")
    for matrix in (
        snapshot.graph.incidence,
        snapshot.graph.u_features,
        snapshot.interaction_weights,
    ):
        value = matrix.tocsr()
        digest.update(value.data.tobytes())
        digest.update(value.indices.tobytes())
        digest.update(value.indptr.tobytes())
    return digest.hexdigest()


class SnapshotStore:
    _V1_ASSETS = (
        "A.npz",
        "X.npz",
        "W.npz",
        "Z.npy",
        "S.npy",
        "semantic_neighbors.npy",
        "semantic_scores.npy",
        "model.npz",
    )
    _V2_ASSETS = (
        "A.npz",
        "X.npz",
        "R.npz",
        "W.npz",
        "Z.npy",
        "V.npy",
        "S.npy",
        "semantic_neighbors.npy",
        "semantic_scores.npy",
        "model.npz",
    )

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _target(self, snapshot_id: str) -> Path:
        if (
            not snapshot_id
            or snapshot_id in {".", "..", "lineage"}
            or Path(snapshot_id).name != snapshot_id
            or "\\" in snapshot_id
        ):
            raise ValueError("invalid snapshot identifier")
        target = self.root / snapshot_id
        if target.is_symlink() or target.resolve().parent != self.root.resolve():
            raise SnapshotIntegrityError("snapshot path escapes the snapshot store")
        return target

    def _write_assets(self, directory: Path, snapshot: ModelSnapshot) -> None:
        sparse.save_npz(directory / "A.npz", snapshot.graph.incidence)
        sparse.save_npz(directory / "X.npz", snapshot.graph.u_features)
        sparse.save_npz(directory / "R.npz", snapshot.interaction_weights)
        sparse.save_npz(directory / "W.npz", snapshot.affinity)
        np.save(directory / "Z.npy", snapshot.embeddings, allow_pickle=False)
        np.save(directory / "V.npy", snapshot.item_embeddings, allow_pickle=False)
        np.save(directory / "S.npy", snapshot.assignments, allow_pickle=False)
        np.save(directory / "semantic_neighbors.npy", snapshot.neighbor_ids, allow_pickle=False)
        np.save(directory / "semantic_scores.npy", snapshot.neighbor_scores, allow_pickle=False)
        np.savez_compressed(directory / "model.npz", **snapshot.model_state)

    @snapshot_mutation
    def save(self, snapshot: ModelSnapshot, *, activate: bool = False) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self._target(snapshot.snapshot_id)
        if target.exists():
            raise FileExistsError(f"snapshot already exists: {snapshot.snapshot_id}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot.snapshot_id}.", dir=self.root))
        try:
            self._write_assets(temporary, snapshot)
            assets = {
                name: {
                    "sha256": _sha256(temporary / name),
                    "bytes": (temporary / name).stat().st_size,
                }
                for name in self._V2_ASSETS
            }
            manifest = {
                "snapshot_id": snapshot.snapshot_id,
                "created_at": snapshot.created_at,
                "core_version": "4.0.0",
                "asset_schema_version": 2,
                "schema_version": snapshot.config.schema_version,
                "complete": True,
                "input_hash": _hash_graph(snapshot),
                "u_ids": snapshot.graph.u_ids,
                "v_ids": snapshot.graph.v_ids,
                "feature_names": snapshot.graph.feature_names,
                "u_metadata": snapshot.graph.u_metadata,
                "v_metadata": snapshot.graph.v_metadata,
                "config": asdict(snapshot.config),
                "diagnostics": snapshot.diagnostics,
                "parent_snapshot_id": snapshot.parent_snapshot_id,
                "build_mode": snapshot.build_mode,
                "event_batch_hash": snapshot.event_batch_hash,
                "update_stats": snapshot.update_stats,
                "assets": assets,
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            with execution_commit():
                os.replace(temporary, target)
            if activate:
                self.activate(snapshot.snapshot_id)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return target

    def _manifest(self, snapshot_id: str) -> tuple[Path, dict[str, Any]]:
        target = self._target(snapshot_id)
        try:
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise SnapshotIntegrityError(
                f"snapshot manifest is missing or invalid: {snapshot_id}"
            ) from exc
        return target, manifest

    def verify(self, snapshot_id: str) -> dict[str, Any]:
        target, manifest = self._manifest(snapshot_id)
        if not manifest.get("complete") or manifest.get("snapshot_id") != snapshot_id:
            raise SnapshotIntegrityError("snapshot manifest is incomplete or mismatched")
        assets = manifest.get("assets")
        if not isinstance(assets, dict) or not assets:
            raise SnapshotIntegrityError("snapshot manifest has no asset integrity data")
        core_version = str(manifest.get("core_version", "1.0.0"))
        asset_schema = (
            1 if core_version.startswith("1.") else int(manifest.get("asset_schema_version", 2))
        )
        required = self._V2_ASSETS if asset_schema >= 2 else self._V1_ASSETS
        missing = set(required) - set(assets)
        if missing:
            raise SnapshotIntegrityError(
                f"snapshot manifest omits required assets: {sorted(missing)}"
            )
        for name, expected in assets.items():
            path = target / name
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or "\\" in name
                or path.is_symlink()
            ):
                raise SnapshotIntegrityError("snapshot manifest contains an unsafe asset path")
            if not isinstance(expected, dict) or not path.is_file():
                raise SnapshotIntegrityError(f"snapshot asset is missing: {name}")
            if path.stat().st_size != expected.get("bytes") or _sha256(path) != expected.get(
                "sha256"
            ):
                raise SnapshotIntegrityError(
                    f"snapshot asset failed integrity verification: {name}"
                )
        return {"snapshot_id": snapshot_id, "verified": True, "asset_count": len(assets)}

    @staticmethod
    def _config(data: Mapping[str, Any], schema_version: int) -> BuildConfig:
        return BuildConfig(
            AffinityConfig(**data.get("affinity", {})),
            EncoderConfig(**data.get("encoder", {})),
            int(data.get("semantic_recall_budget", 100)),
            schema_version,
        )

    def load(self, snapshot_id: str) -> ModelSnapshot:
        target, manifest = self._manifest(snapshot_id)
        self.verify(snapshot_id)
        return self._decode(target, manifest)

    def _decode(
        self,
        target: Path,
        manifest: Mapping[str, Any],
        affinity: sparse.csr_matrix | None = None,
    ) -> ModelSnapshot:
        graph = CanonicalBipartiteGraph(
            tuple(manifest["u_ids"]),
            tuple(manifest["v_ids"]),
            sparse.load_npz(target / "A.npz").tocsr(),
            sparse.load_npz(target / "X.npz").tocsr(),
            tuple(manifest.get("feature_names", ())),
            manifest.get("u_metadata", {}),
            manifest.get("v_metadata", {}),
        )
        schema_version = int(manifest.get("schema_version", 1))
        config = self._config(manifest.get("config", {}), schema_version)
        state = dict(np.load(target / "model.npz", allow_pickle=False))
        weights = (
            sparse.load_npz(target / "R.npz").tocsr()
            if (target / "R.npz").is_file()
            else graph.incidence.copy()
        )
        item_embeddings = (
            np.load(target / "V.npy", allow_pickle=False)
            if (target / "V.npy").is_file()
            else np.asarray(
                state.get("v_embedding", np.zeros((len(graph.v_ids), config.encoder.hidden_dim)))
            )
        )
        return ModelSnapshot(
            manifest["snapshot_id"],
            manifest["created_at"],
            graph,
            affinity if affinity is not None else sparse.load_npz(target / "W.npz").tocsr(),
            np.load(target / "Z.npy", allow_pickle=False),
            item_embeddings,
            np.load(target / "S.npy", allow_pickle=False),
            np.load(target / "semantic_neighbors.npy", allow_pickle=False),
            np.load(target / "semantic_scores.npy", allow_pickle=False),
            config,
            tuple(manifest.get("diagnostics", ())),
            state,
            weights,
            manifest.get("parent_snapshot_id"),
            manifest.get("build_mode", "full"),
            manifest.get("event_batch_hash"),
            manifest.get("update_stats", {}),
        )

    def recover(self, snapshot_id: str) -> ModelSnapshot:
        target, manifest = self._manifest(snapshot_id)
        assets = manifest.get("assets")
        core_version = str(manifest.get("core_version", "1.0.0"))
        asset_schema = (
            1 if core_version.startswith("1.") else int(manifest.get("asset_schema_version", 2))
        )
        required = self._V2_ASSETS if asset_schema >= 2 else self._V1_ASSETS
        if not isinstance(assets, dict):
            raise SnapshotIntegrityError("snapshot recovery requires asset integrity data")
        for name in set(required) - {"W.npz"}:
            expected = assets.get(name)
            path = target / name
            if not isinstance(expected, dict) or not path.is_file():
                raise SnapshotIntegrityError(f"snapshot recovery asset is missing: {name}")
            if path.stat().st_size != expected.get("bytes") or _sha256(path) != expected.get(
                "sha256"
            ):
                raise SnapshotIntegrityError(f"snapshot recovery asset failed verification: {name}")
        users = len(manifest["u_ids"])
        decoded = self._decode(target, manifest, sparse.csr_matrix((users, users)))
        affinity = AffinityBuilder(decoded.config.affinity).build(decoded.graph).affinity
        return ModelSnapshot(
            decoded.snapshot_id,
            decoded.created_at,
            decoded.graph,
            affinity,
            decoded.embeddings,
            decoded.item_embeddings,
            decoded.assignments,
            decoded.neighbor_ids,
            decoded.neighbor_scores,
            decoded.config,
            decoded.diagnostics,
            decoded.model_state,
            decoded.interaction_weights,
            decoded.parent_snapshot_id,
            decoded.build_mode,
            decoded.event_batch_hash,
            decoded.update_stats,
        )

    def pointer_id(self) -> str:
        try:
            snapshot_id = json.loads((self.root / "latest.json").read_text(encoding="utf-8"))[
                "snapshot_id"
            ]
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            raise SnapshotIntegrityError("no valid active snapshot pointer exists") from exc
        self._target(snapshot_id)
        return snapshot_id

    def latest_id(self) -> str:
        snapshot_id = self.pointer_id()
        self.verify(snapshot_id)
        return snapshot_id

    @snapshot_mutation
    def activate(self, snapshot_id: str) -> str:
        self.verify(snapshot_id)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / f".latest.{os.getpid()}.{uuid4().hex}.tmp"
        temporary.write_text(json.dumps({"snapshot_id": snapshot_id}), encoding="utf-8")
        with execution_commit():
            os.replace(temporary, self.root / "latest.json")
        return snapshot_id

    def rollback(self, snapshot_id: str) -> str:
        return self.activate(snapshot_id)

    def list(self) -> tuple[dict[str, Any], ...]:
        active = None
        try:
            active = self.pointer_id()
        except SnapshotIntegrityError:
            pass
        entries: list[dict[str, Any]] = []
        if not self.root.exists():
            return ()
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or path.name.startswith(".") or path.name == "lineage":
                continue
            try:
                _, manifest = self._manifest(path.name)
                verified = bool(self.verify(path.name)["verified"])
                entries.append(
                    {
                        "snapshot_id": path.name,
                        "created_at": manifest.get("created_at"),
                        "build_mode": manifest.get("build_mode", "full"),
                        "parent_snapshot_id": manifest.get("parent_snapshot_id"),
                        "active": path.name == active,
                        "verified": verified,
                    }
                )
            except SnapshotIntegrityError:
                entries.append(
                    {"snapshot_id": path.name, "active": path.name == active, "verified": False}
                )
        return tuple(entries)
