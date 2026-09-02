"""Pipeline package."""
from .manifest import (
    ProjectManifest,
    StageRecord,
    StageStatus,
    ArtifactRecord,
    ArtifactType,
    load_manifest,
    save_manifest,
    create_project_manifest,
    PIPELINE_STAGES,
    STAGE_DEPENDENCIES,
)
from .artifacts.registry import ArtifactRegistry, register_builtin_validators
from .stages.contracts import STAGE_CONTRACTS, PipelineStage, StageConfig, StageResult
from .storage import StorageRoot, get_storage_root, set_storage_root

__all__ = [
    "ProjectManifest",
    "StageRecord",
    "StageStatus",
    "ArtifactRecord",
    "ArtifactType",
    "load_manifest",
    "save_manifest",
    "create_project_manifest",
    "PIPELINE_STAGES",
    "STAGE_DEPENDENCIES",
    "ArtifactRegistry",
    "register_builtin_validators",
    "STAGE_CONTRACTS",
    "PipelineStage",
    "StageConfig",
    "StageResult",
    "StorageRoot",
    "get_storage_root",
    "set_storage_root",
]