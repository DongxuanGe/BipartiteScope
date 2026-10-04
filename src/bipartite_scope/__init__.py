from .core import (
    AcademicExplorerAdapter,
    AffinityBuilder,
    AffinityConfig,
    BuildConfig,
    CanonicalBipartiteGraph,
    CommunityResult,
    EncoderConfig,
    EvaluationConfig,
    Event,
    IncrementalConfig,
    ModelSnapshot,
    QueryConfig,
    QueryEngine,
    RecommendationAdapter,
    RecommendationConfig,
    build_snapshot,
)
from .interface import create_app
from .recommendation import (
    EvaluationResult,
    RecommendationItem,
    RecommendationResult,
    UpdateResult,
    evaluate,
    recommend,
    record_feedback,
    update_snapshot,
)
from .storage import (
    InputValidationError,
    SnapshotIntegrityError,
    SnapshotStore,
    ValidationReport,
    Workspace,
    init_workspace,
    load_events,
    load_workspace,
    normalize_event,
)

__version__ = "4.0.0"

__all__ = [
    "AcademicExplorerAdapter",
    "AffinityBuilder",
    "AffinityConfig",
    "BuildConfig",
    "CanonicalBipartiteGraph",
    "CommunityResult",
    "EncoderConfig",
    "EvaluationConfig",
    "EvaluationResult",
    "Event",
    "IncrementalConfig",
    "InputValidationError",
    "ModelSnapshot",
    "QueryConfig",
    "QueryEngine",
    "RecommendationAdapter",
    "RecommendationConfig",
    "RecommendationItem",
    "RecommendationResult",
    "ServiceDatabase",
    "ServiceSettings",
    "SnapshotIntegrityError",
    "SnapshotStore",
    "UpdateResult",
    "ValidationReport",
    "Workspace",
    "build_snapshot",
    "create_app",
    "create_service_app",
    "evaluate",
    "init_workspace",
    "load_events",
    "load_workspace",
    "normalize_event",
    "recommend",
    "record_feedback",
    "update_snapshot",
]


def __getattr__(name: str):
    if name == "ServiceSettings":
        from .config import ServiceSettings

        return ServiceSettings
    if name == "ServiceDatabase":
        from .database import ServiceDatabase

        return ServiceDatabase
    if name == "create_service_app":
        from .api import create_service_app

        return create_service_app
    raise AttributeError(name)
