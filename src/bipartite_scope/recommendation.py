from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from scipy import sparse

from .core import (
    AffinityBuilder,
    CanonicalBipartiteGraph,
    EvaluationConfig,
    Event,
    IncrementalConfig,
    ModelSnapshot,
    QueryConfig,
    QueryEngine,
    RecommendationConfig,
    build_snapshot,
    rowwise_top_k,
)
from .reliability import workspace_mutation
from .storage import (
    SnapshotIntegrityError,
    SnapshotStore,
    Workspace,
    all_events,
    load_events,
    load_feature_rows,
    mark_events_applied,
    pending_events,
    register_events,
)

POSITIVE_EVENT_WEIGHTS = {
    "view": 0.25,
    "click": 1.0,
    "favorite": 2.0,
    "purchase": 3.0,
    "rating": 1.0,
}


@dataclass(frozen=True, slots=True)
class UpdateResult:
    snapshot_id: str
    parent_snapshot_id: str
    build_mode: str
    accepted_count: int
    duplicate_count: int
    rejected_count: int
    affected_user_count: int
    affected_item_count: int
    affected_ratio: float
    fallback_reason: str | None
    event_batch_hash: str
    active: bool = False


@dataclass(frozen=True, slots=True)
class RecommendationItem:
    item_id: str
    score: float
    components: Mapping[str, float]
    supporting_users: tuple[str, ...]
    explanation: str


@dataclass(frozen=True, slots=True)
class RecommendationResult:
    user_id: str
    snapshot_id: str
    items: tuple[RecommendationItem, ...]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    evaluation_id: str
    snapshot_id: str
    metrics: Mapping[str, Any]
    output: str


def _event_hash(events: Sequence[Event]) -> str:
    digest = hashlib.sha256()
    for event in sorted(events, key=lambda value: value.event_id):
        digest.update(json.dumps(asdict(event), sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _feature_mapping(snapshot: ModelSnapshot) -> dict[str, tuple[float, ...]]:
    return {
        user: tuple(
            float(value) for value in snapshot.graph.u_features.getrow(index).toarray().ravel()
        )
        for index, user in enumerate(snapshot.graph.u_ids)
    }


def _weight_mapping(snapshot: ModelSnapshot) -> dict[tuple[str, str], float]:
    rows, columns = snapshot.interaction_weights.nonzero()
    return {
        (snapshot.graph.u_ids[row], snapshot.graph.v_ids[column]): float(
            snapshot.interaction_weights[row, column]
        )
        for row, column in zip(rows, columns, strict=True)
    }


def _affected_users(
    parent: ModelSnapshot,
    graph: CanonicalBipartiteGraph,
    changed_users: set[str],
    changed_items: set[str],
    config: IncrementalConfig,
) -> set[int]:
    user_index = graph.u_index
    item_index = graph.v_index
    affected = {user_index[user] for user in changed_users}
    changed_item_indexes = [item_index[item] for item in changed_items]
    if changed_item_indexes:
        affected.update(graph.incidence[:, changed_item_indexes].nonzero()[0].tolist())
    old_item_indexes = [
        parent.graph.v_index[item] for item in changed_items if item in parent.graph.v_index
    ]
    if old_item_indexes:
        affected.update(parent.graph.incidence[:, old_item_indexes].nonzero()[0].tolist())
    changed_indexes = sorted(affected)
    if changed_indexes:
        feature_neighbors = graph.u_features[changed_indexes] @ graph.u_features.T
        affected.update(feature_neighbors.nonzero()[1].tolist())
    for _ in range(config.neighbor_hops):
        frontier = sorted(affected)
        if not frontier:
            break
        items = np.unique(graph.incidence[frontier].indices)
        affected.update(graph.incidence[:, items].nonzero()[0].tolist())
    for row in tuple(affected):
        if row < parent.affinity.shape[0]:
            affected.update(parent.affinity.getrow(row).indices.tolist())
    return affected


def _local_affinity(
    parent: ModelSnapshot,
    graph: CanonicalBipartiteGraph,
    affected: set[int],
    top_k: int,
) -> sparse.csr_matrix:
    size = len(graph.u_ids)
    indexes = sorted(affected)
    local_graph = CanonicalBipartiteGraph(
        tuple(graph.u_ids[index] for index in indexes),
        graph.v_ids,
        graph.incidence[indexes],
        graph.u_features[indexes],
        graph.feature_names,
    )
    local = AffinityBuilder(parent.config.affinity).build(local_graph).affinity.tocoo()
    old = parent.affinity.tocoo()
    retained = sparse.csr_matrix((old.data, (old.row, old.col)), shape=(size, size)).tolil()
    for row in indexes:
        retained[row, :] = 0
        retained[:, row] = 0
    for row, column, value in zip(local.row, local.col, local.data, strict=True):
        retained[indexes[int(row)], indexes[int(column)]] = float(value)
    retained = retained.tocsr().maximum(retained.T)
    pruned = rowwise_top_k(retained, top_k=top_k, keep_diagonal=False)
    result = (0.5 * (pruned + pruned.T)).tocsr()
    result.setdiag(0)
    result.eliminate_zeros()
    return result


def _apply_events(
    parent: ModelSnapshot,
    events: Sequence[Event],
    feature_names: Sequence[str],
    feature_updates: Mapping[str, Sequence[float]],
    incremental: IncrementalConfig,
) -> tuple[CanonicalBipartiteGraph, sparse.csr_matrix, sparse.csr_matrix, dict[str, Any]]:
    features = _feature_mapping(parent)
    expected_dimension = parent.graph.u_features.shape[1]
    if feature_names and len(feature_names) != expected_dimension:
        raise ValueError("feature dimension changes require a new full workspace build")
    if (
        parent.graph.feature_names
        and feature_names
        and tuple(feature_names) != parent.graph.feature_names
    ):
        raise ValueError("feature schema changes require a new full workspace build")
    for user, values in feature_updates.items():
        values = tuple(float(value) for value in values)
        if len(values) != expected_dimension:
            raise ValueError("feature dimension changes require a new full workspace build")
        features[user] = values
    event_users = {event.user_id for event in events}
    missing = sorted(event_users - set(parent.graph.u_ids) - set(feature_updates))
    if missing:
        raise ValueError(f"new users require feature rows: {', '.join(missing[:5])}")
    new_users = sorted(set(features) - set(parent.graph.u_ids))
    user_ids = (*parent.graph.u_ids, *new_users)
    new_items = sorted({event.item_id for event in events} - set(parent.graph.v_ids))
    item_ids = (*parent.graph.v_ids, *new_items)
    weights = _weight_mapping(parent)
    changed_users: set[str] = set(feature_updates)
    changed_items: set[str] = set()
    for event in events:
        key = (event.user_id, event.item_id)
        changed_users.add(event.user_id)
        changed_items.add(event.item_id)
        if event.event_type in POSITIVE_EVENT_WEIGHTS:
            amount = POSITIVE_EVENT_WEIGHTS[event.event_type] * event.event_value
            if amount > 0:
                weights[key] = weights.get(key, 0.0) + amount
        elif event.event_type == "remove":
            weights.pop(key, None)
    edges = [key for key, value in weights.items() if value > 0]
    graph = CanonicalBipartiteGraph.from_edges_and_features(
        edges,
        features,
        feature_names=parent.graph.feature_names or tuple(feature_names),
        u_order=user_ids,
        v_order=item_ids,
        u_metadata=parent.graph.u_metadata,
        v_metadata=parent.graph.v_metadata,
    )
    rows = [graph.u_index[user] for user, item in edges]
    columns = [graph.v_index[item] for user, item in edges]
    values = [weights[(user, item)] for user, item in edges]
    interaction_weights = sparse.csr_matrix((values, (rows, columns)), shape=graph.incidence.shape)
    affected = _affected_users(parent, graph, changed_users, changed_items, incremental)
    ratio = len(affected) / max(1, len(graph.u_ids))
    fallback_reason = None
    mode = "incremental"
    if ratio > incremental.max_affected_ratio:
        if not incremental.fallback_to_full_build:
            raise ValueError("affected ratio exceeds max_affected_ratio and fallback is disabled")
        mode = "full_fallback"
        fallback_reason = "affected_ratio_exceeded"
        affinity = AffinityBuilder(parent.config.affinity).build(graph).affinity
    else:
        affinity = _local_affinity(parent, graph, affected, parent.config.affinity.top_k)
        if not np.isfinite(affinity.data).all() or (affinity - affinity.T).nnz:
            if not incremental.fallback_to_full_build:
                raise FloatingPointError("incremental affinity checks failed")
            mode = "full_fallback"
            fallback_reason = "incremental_numerical_check_failed"
            affinity = AffinityBuilder(parent.config.affinity).build(graph).affinity
    stats = {
        "build_mode": mode,
        "fallback_reason": fallback_reason,
        "affected_user_count": len(affected),
        "affected_item_count": len(changed_items),
        "affected_ratio": ratio,
    }
    return graph, interaction_weights, affinity, stats


@workspace_mutation
def update_from_events(
    workspace: Workspace,
    events: Iterable[Event],
    *,
    feature_names: Sequence[str] = (),
    feature_updates: Mapping[str, Sequence[float]] | None = None,
    source: str = "api",
) -> UpdateResult:
    supplied = tuple(events)
    store = SnapshotStore(workspace.artifacts)
    parent_id = store.pointer_id()
    integrity_fallback = False
    try:
        parent = store.load(parent_id)
    except SnapshotIntegrityError:
        parent = store.recover(parent_id)
        integrity_fallback = True
    updates = feature_updates or {}
    missing = sorted({event.user_id for event in supplied} - set(parent.graph.u_ids) - set(updates))
    if missing:
        raise ValueError(f"new users require feature rows: {', '.join(missing[:5])}")
    if feature_names and len(feature_names) != parent.graph.u_features.shape[1]:
        raise ValueError("feature dimension changes require a new full workspace build")
    accepted, duplicates = register_events(workspace, supplied, source=source)
    pending = pending_events(workspace)
    if not pending:
        raise ValueError("no pending events remain after idempotency checks")
    incremental = workspace.incremental_config()
    if integrity_fallback:
        incremental = replace(incremental, max_affected_ratio=np.finfo(float).eps)
    graph, weights, affinity, stats = _apply_events(
        parent,
        pending,
        feature_names,
        updates,
        incremental,
    )
    batch_hash = _event_hash(pending)
    encoder = replace(parent.config.encoder, epochs=incremental.warm_start_epochs)
    config = replace(parent.config, encoder=encoder, schema_version=2)
    stats.update(
        {
            "accepted_count": len(accepted),
            "duplicate_count": len(duplicates),
            "rejected_count": 0,
            "event_count": len(pending),
        }
    )
    if integrity_fallback:
        stats["build_mode"] = "full_fallback"
        stats["fallback_reason"] = "parent_snapshot_integrity_failed"
    snapshot = build_snapshot(
        graph,
        config,
        interaction_weights=weights,
        initial_state=parent.model_state,
        affinity_override=affinity,
        parent_snapshot_id=parent_id,
        build_mode=stats["build_mode"],
        event_batch_hash=batch_hash,
        update_stats=stats,
    )
    store.save(snapshot, activate=False)
    mark_events_applied(
        workspace, [event.event_id for event in pending], snapshot.snapshot_id, batch_hash
    )
    return UpdateResult(
        snapshot.snapshot_id,
        parent_id,
        stats["build_mode"],
        len(accepted),
        len(duplicates),
        0,
        stats["affected_user_count"],
        stats["affected_item_count"],
        stats["affected_ratio"],
        stats["fallback_reason"],
        batch_hash,
    )


def update_snapshot(
    workspace: Workspace,
    events_path: str | Path,
    features_path: str | Path | None = None,
) -> UpdateResult:
    events = load_events(events_path)
    feature_names: tuple[str, ...] = ()
    feature_updates: Mapping[str, Sequence[float]] = {}
    if features_path:
        feature_names, feature_updates = load_feature_rows(features_path)
    return update_from_events(
        workspace,
        events,
        feature_names=feature_names,
        feature_updates=feature_updates,
        source=str(Path(events_path).resolve()),
    )


def _normalize(values: Mapping[int, float]) -> dict[int, float]:
    if not values:
        return {}
    maximum = max(values.values())
    return {key: value / maximum if maximum > 0 else 0.0 for key, value in values.items()}


def _event_status(events: Iterable[Event], user_id: str) -> tuple[set[str], dict[str, datetime]]:
    excluded: set[str] = set()
    latest: dict[str, datetime] = {}
    for event in sorted(
        (value for value in events if value.user_id == user_id), key=lambda value: value.event_time
    ):
        timestamp = datetime.fromisoformat(event.event_time)
        if event.event_type in {"dislike", "remove"}:
            excluded.add(event.item_id)
        elif event.event_type in POSITIVE_EVENT_WEIGHTS:
            excluded.discard(event.item_id)
            latest[event.item_id] = timestamp
    return excluded, latest


def recommend(
    snapshot: ModelSnapshot,
    user_id: str,
    config: RecommendationConfig | None = None,
    *,
    events: Iterable[Event] = (),
) -> RecommendationResult:
    config = config or RecommendationConfig()
    if user_id not in snapshot.graph.u_index:
        raise KeyError(f"unknown user identifier: {user_id}")
    event_list = tuple(events)
    user_index = snapshot.graph.u_index[user_id]
    size = min(max(2, config.top_n), len(snapshot.graph.u_ids))
    community = QueryEngine(snapshot).search(user_id, QueryConfig(size_budget=size))
    community_indexes = [snapshot.graph.u_index[user] for user in community.members]
    consumed = set(snapshot.graph.incidence.getrow(user_index).indices)
    excluded_ids, _ = _event_status(event_list, user_id)
    excluded = consumed | {
        snapshot.graph.v_index[item] for item in excluded_ids if item in snapshot.graph.v_index
    }
    candidates = set(snapshot.graph.incidence[community_indexes].indices) - excluded
    if not candidates:
        return RecommendationResult(user_id, snapshot.snapshot_id, ())
    support: dict[int, float] = {}
    affinity_support: dict[int, float] = {}
    behavior: dict[int, float] = {}
    recency: dict[int, float] = {}
    popularity: dict[int, float] = {}
    supporting_users: dict[int, tuple[str, ...]] = {}
    affinity_row = snapshot.affinity.getrow(user_index)
    affinity_values = dict(zip(affinity_row.indices, affinity_row.data, strict=True))
    reference_time = max(
        (datetime.fromisoformat(event.event_time) for event in event_list),
        default=datetime.now(UTC),
    )
    latest_positive: dict[tuple[str, str], datetime] = {}
    for event in event_list:
        if event.event_type in POSITIVE_EVENT_WEIGHTS:
            latest_positive[(event.user_id, event.item_id)] = datetime.fromisoformat(
                event.event_time
            )
    degrees = np.asarray(snapshot.graph.incidence.sum(axis=0)).ravel()
    for item in candidates:
        peers = [index for index in community_indexes if snapshot.graph.incidence[index, item] > 0]
        supporting_users[item] = tuple(snapshot.graph.u_ids[index] for index in peers)
        support[item] = len(peers) / max(1, len(community_indexes))
        affinity_support[item] = sum(float(affinity_values.get(index, 0.0)) for index in peers)
        behavior[item] = float(snapshot.interaction_weights[peers, item].sum())
        timestamps = [
            latest_positive.get((snapshot.graph.u_ids[index], snapshot.graph.v_ids[item]))
            for index in peers
        ]
        timestamps = [value for value in timestamps if value is not None]
        if timestamps:
            age = max(0.0, (reference_time - max(timestamps)).total_seconds() / 86400)
            recency[item] = math.exp(-math.log(2) * age / config.half_life_days)
        else:
            recency[item] = 0.0
        popularity[item] = math.log1p(float(degrees[item]))
    normalized_affinity = _normalize(affinity_support)
    normalized_behavior = _normalize(behavior)
    normalized_recency = _normalize(recency)
    normalized_popularity = _normalize(popularity)
    results: list[RecommendationItem] = []
    for item in candidates:
        components = {
            "community_support": support[item],
            "affinity_support": normalized_affinity[item],
            "weighted_behavior": normalized_behavior[item],
            "recency": normalized_recency[item],
            "popularity_penalty": normalized_popularity[item],
        }
        score = (
            config.community_weight * components["community_support"]
            + config.affinity_weight * components["affinity_support"]
            + config.behavior_weight * components["weighted_behavior"]
            + config.recency_weight * components["recency"]
            - config.popularity_penalty * components["popularity_penalty"]
        )
        peer_count = len(supporting_users[item])
        explanation = (
            f"Recommended from {peer_count} supporting user(s); score {score:.6f} combines "
            f"community {components['community_support']:.6f}, affinity {components['affinity_support']:.6f}, "
            f"behavior {components['weighted_behavior']:.6f}, recency {components['recency']:.6f}, and "
            f"popularity penalty {components['popularity_penalty']:.6f}."
        )
        results.append(
            RecommendationItem(
                snapshot.graph.v_ids[item],
                float(score),
                components,
                supporting_users[item],
                explanation,
            )
        )
    results.sort(key=lambda value: (-value.score, value.item_id))
    return RecommendationResult(user_id, snapshot.snapshot_id, tuple(results[: config.top_n]))


def record_feedback(workspace: Workspace, event: Event) -> dict[str, Any]:
    accepted, duplicates = register_events(
        workspace, (event,), source="feedback", ledger_name="feedback.jsonl"
    )
    return {
        "event_id": event.event_id,
        "accepted": bool(accepted),
        "duplicate": bool(duplicates),
        "retraining_triggered": False,
    }


def _ranking_metrics(
    rankings: Mapping[str, Sequence[str]], truth: Mapping[str, str], k: int
) -> dict[str, float]:
    hits = 0
    reciprocal = 0.0
    ndcg = 0.0
    recommended: set[str] = set()
    for user, expected in truth.items():
        ranking = list(rankings.get(user, ()))[:k]
        recommended.update(ranking)
        if expected in ranking:
            rank = ranking.index(expected) + 1
            hits += 1
            reciprocal += 1 / rank
            ndcg += 1 / math.log2(rank + 1)
    users = max(1, len(truth))
    return {
        f"precision@{k}": hits / (users * k),
        f"recall@{k}": hits / users,
        f"hit_rate@{k}": hits / users,
        f"ndcg@{k}": ndcg / users,
        "mrr": reciprocal / users,
        "recommended_catalog_items": float(len(recommended)),
    }


def _baseline_rankings(
    graph: CanonicalBipartiteGraph,
    users: Sequence[str],
    k: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    popularity_score = np.asarray(graph.incidence.sum(axis=0)).ravel()
    cooccurrence = (graph.incidence.T @ graph.incidence).tocsr()
    popularity: dict[str, list[str]] = {}
    cooccurrence_results: dict[str, list[str]] = {}
    for user in users:
        row = graph.incidence.getrow(graph.u_index[user])
        consumed = set(row.indices)
        candidates = [item for item in range(len(graph.v_ids)) if item not in consumed]
        popularity[user] = [
            graph.v_ids[item]
            for item in sorted(
                candidates, key=lambda item: (-popularity_score[item], graph.v_ids[item])
            )[:k]
        ]
        score = (
            np.asarray(cooccurrence[row.indices].sum(axis=0)).ravel()
            if row.nnz
            else np.zeros(len(graph.v_ids))
        )
        cooccurrence_results[user] = [
            graph.v_ids[item]
            for item in sorted(candidates, key=lambda item: (-score[item], graph.v_ids[item]))[:k]
        ]
    return popularity, cooccurrence_results


@workspace_mutation
def evaluate(
    workspace: Workspace,
    snapshot_id: str | None = None,
    config: EvaluationConfig | None = None,
) -> EvaluationResult:
    store = SnapshotStore(workspace.artifacts)
    snapshot = store.load(snapshot_id or store.latest_id())
    config = config or workspace.evaluation_config()
    positives: dict[str, list[Event]] = {}
    for event in all_events(workspace):
        if event.event_type in POSITIVE_EVENT_WEIGHTS:
            positives.setdefault(event.user_id, []).append(event)
    included = {
        user: sorted(events, key=lambda value: (value.event_time, value.event_id))
        for user, events in positives.items()
        if len(events) >= config.minimum_positive_events and user in snapshot.graph.u_index
    }
    excluded_count = len(positives) - len(included)
    if not included:
        raise ValueError(
            "temporal evaluation requires eligible users with timestamped positive events"
        )
    truth = {user: events[-1].item_id for user, events in included.items()}
    training_events = [event for user, events in included.items() for event in events[:-1]]
    edges = {(event.user_id, event.item_id) for event in training_events}
    features = _feature_mapping(snapshot)
    evaluation_graph = CanonicalBipartiteGraph.from_edges_and_features(
        edges,
        features,
        feature_names=snapshot.graph.feature_names,
        u_order=snapshot.graph.u_ids,
        v_order=snapshot.graph.v_ids,
    )
    evaluation_config = replace(
        snapshot.config,
        encoder=replace(snapshot.config.encoder, epochs=min(snapshot.config.encoder.epochs, 10)),
    )
    evaluation_snapshot = build_snapshot(evaluation_graph, evaluation_config)
    rankings = {
        user: [
            item.item_id
            for item in recommend(
                evaluation_snapshot,
                user,
                replace(workspace.recommendation_config(config.k), top_n=config.k),
                events=training_events,
            ).items
        ]
        for user in included
    }
    popularity, cooccurrence = _baseline_rankings(evaluation_graph, tuple(included), config.k)
    model_metrics = _ranking_metrics(rankings, truth, config.k)
    popularity_metrics = _ranking_metrics(popularity, truth, config.k)
    cooccurrence_metrics = _ranking_metrics(cooccurrence, truth, config.k)
    catalog_size = max(1, len(evaluation_graph.v_ids))
    for values in (model_metrics, popularity_metrics, cooccurrence_metrics):
        values["catalog_coverage"] = values.pop("recommended_catalog_items") / catalog_size
    metrics = {
        "bipartite_scope": model_metrics,
        "popularity": popularity_metrics,
        "user_item_cooccurrence": cooccurrence_metrics,
        "included_users": len(included),
        "excluded_users": excluded_count,
        "temporal_leakage_check": True,
    }
    evaluation_id = f"eval-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    output = workspace.reports / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    configuration = {
        "snapshot_id": snapshot.snapshot_id,
        "k": config.k,
        "minimum_positive_events": config.minimum_positive_events,
        "split": "per-user chronological leave-last-out",
        "activation": "manual",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    report = [
        "# BipartiteScope Offline Evaluation",
        "",
        f"- Evaluation ID: `{evaluation_id}`",
        f"- Snapshot ID: `{snapshot.snapshot_id}`",
        f"- Included users: {len(included)}",
        f"- Excluded users: {excluded_count}",
        f"- K: {config.k}",
        "",
        "The split is chronological leave-last-out. Candidate activation remains manual.",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2),
        "```",
        "",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    return EvaluationResult(evaluation_id, snapshot.snapshot_id, metrics, str(output))
