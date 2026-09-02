"""Stage contracts — define inputs, outputs, config, and validation for each pipeline stage."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

from ..manifest import ArtifactType, StageRecord, ProjectManifest
from ..artifacts.registry import ArtifactRegistry


@dataclass
class StageConfig:
    """Configuration for a stage execution."""
    # Stage-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Model/provider settings
    model: Optional[str] = None
    device: Optional[str] = None
    provider: Optional[str] = None
    
    # Runtime settings
    timeout_seconds: int = 3600
    force_rerun: bool = False
    skip_validation: bool = False
    
    # GPU/CPU preferences
    require_gpu: bool = False
    gpu_memory_gb: Optional[float] = None


@dataclass
class StageResult:
    """Result of a stage execution."""
    success: bool
    stage_name: str
    duration_seconds: float = 0.0
    error: Optional[str] = None
    output_artifact_ids: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StageContract:
    """Defines the contract for a pipeline stage: inputs, outputs, config schema, validation."""
    
    def __init__(
        self,
        name: str,
        input_artifact_types: List[str],
        output_artifact_types: List[str],
        config_schema: Optional[Dict[str, Any]] = None,
        required_config_keys: Optional[List[str]] = None,
        optional_config_keys: Optional[List[str]] = None,
        required_artifacts: Optional[List[str]] = None,
        optional_artifacts: Optional[List[str]] = None,
    ):
        self.name = name
        self.input_artifact_types = input_artifact_types
        self.output_artifact_types = output_artifact_types
        self.config_schema = config_schema or {}
        self.required_config_keys = required_config_keys or []
        self.optional_config_keys = optional_config_keys or []
        self.required_artifacts = required_artifacts or []
        self.optional_artifacts = optional_artifacts or []
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate stage config against schema."""
        errors = []
        for key in self.required_config_keys:
            if key not in config:
                errors.append(f"Missing required config key: {key}")
        return errors
    
    def validate_inputs(self, registry: ArtifactRegistry) -> List[str]:
        """Check that required input artifacts exist."""
        errors = []
        for artifact_name in self.required_artifacts:
            # Check if artifact exists in registry
            found = False
            for artifact in registry.manifest.artifacts.values():
                if artifact.path.endswith(artifact_name) or artifact.artifact_id == artifact_name:
                    found = True
                    break
            if not found:
                errors.append(f"Required input artifact not found: {artifact_name}")
        return errors
    
    def get_input_artifacts(self, registry: ArtifactRegistry) -> Dict[str, Any]:
        """Get all input artifacts for this stage."""
        return {a.artifact_id: a for a in registry.manifest.artifacts.values()}


class PipelineStage(ABC):
    """Base class for pipeline stages."""
    
    def __init__(
        self,
        contract: StageContract,
        manifest: "ProjectManifest",
        artifact_root: Path,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.contract = contract
        self.manifest = manifest
        self.artifact_root = Path(artifact_root)
        self.config = config or {}
        self.registry = ArtifactRegistry(manifest, artifact_root)
        self.stage_record = manifest.get_stage(contract.name)
    
    @abstractmethod
    def run(self, stage_config: StageConfig) -> StageResult:
        """Execute the stage. Must be implemented by subclasses."""
        pass
    
    def validate_inputs(self) -> List[str]:
        """Validate that all required inputs are present."""
        return self.contract.validate_inputs(self.registry)
    
    def validate_config(self) -> List[str]:
        """Validate stage configuration."""
        return self.contract.validate_config(self.config)
    
    def get_input_artifact(self, artifact_type: str) -> Optional[Path]:
        """Get path to an input artifact by type."""
        artifacts = self.manifest.get_artifacts_by_type(artifact_type)
        if not artifacts:
            return None
        artifacts.sort(key=lambda a: a.created_at, reverse=True)
        return self.artifact_root / artifacts[0].path
    
    def register_output(
        self,
        artifact_type: str,
        relative_path: str,
        version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None,
        input_artifact_ids: Optional[List[str]] = None,
    ) -> str:
        """Register an output artifact."""
        record = self.registry.register_artifact(
            artifact_type=artifact_type,
            relative_path=relative_path,
            producer_stage=self.contract.name,
            version=version,
            metadata=metadata,
            input_artifact_ids=input_artifact_ids,
        )
        self.stage_record.output_artifact_ids.append(record.artifact_id)
        return record.artifact_id
    
    def mark_running(self) -> None:
        """Mark stage as running."""
        from ..manifest import StageStatus
        from datetime import datetime
        self.stage_record.status = StageStatus.RUNNING
        self.stage_record.started_at = datetime.utcnow().isoformat() + "Z"
        self.stage_record.config = self.config
        self.stage_record.config_hash = self.manifest.compute_config_hash()
    
    def mark_completed(self, duration: float) -> None:
        """Mark stage as completed."""
        from ..manifest import StageStatus
        from datetime import datetime
        self.stage_record.status = StageStatus.COMPLETED
        self.stage_record.completed_at = datetime.utcnow().isoformat() + "Z"
        self.stage_record.duration_seconds = duration
    
    def mark_failed(self, error: str, duration: float) -> None:
        """Mark stage as failed."""
        from ..manifest import StageStatus
        from datetime import datetime
        self.stage_record.status = StageStatus.FAILED
        self.stage_record.completed_at = datetime.utcnow().isoformat() + "Z"
        self.stage_record.duration_seconds = duration
        self.stage_record.error = error
    
    def mark_skipped(self, reason: str = "Already completed") -> None:
        """Mark stage as skipped."""
        from ..manifest import StageStatus
        from datetime import datetime
        self.stage_record.status = StageStatus.SKIPPED
        self.stage_record.completed_at = datetime.utcnow().isoformat() + "Z"
        self.stage_record.metadata["skip_reason"] = reason


# Stage contract definitions
STAGE_CONTRACTS = {
    "ingest": StageContract(
        name="ingest",
        input_artifact_types=[],
        output_artifact_types=["project_meta"],
        required_config_keys=["title"],
        optional_config_keys=["source_path"],
        required_artifacts=[],
    ),
    
    "transcription": StageContract(
        name="transcription",
        input_artifact_types=[],
        output_artifact_types=["transcript"],
        required_config_keys=[],
        optional_config_keys=["model", "language", "word_timestamps"],
        required_artifacts=["project_meta"],
    ),
    
    "scene_indexing": StageContract(
        name="scene_indexing",
        input_artifact_types=[],
        output_artifact_types=["scene_index"],
        required_config_keys=[],
        optional_config_keys=["threshold", "min_scene_len"],
        required_artifacts=["project_meta.source_path"],
    ),
    
    "movie_intelligence": StageContract(
        name="movie_intelligence",
        input_artifact_types=["transcript", "scene_index"],
        output_artifact_types=["movie_index", "semantic_index", "manifest"],
        required_config_keys=[],
        optional_config_keys=["enricher", "attach_keyframes", "max_frames", "group_max_scene_sec"],
        required_artifacts=["transcript", "scene_index"],
    ),
    
    "director": StageContract(
        name="director",
        input_artifact_types=["movie_index"],
        output_artifact_types=["director_plan", "grounded_script"],
        required_config_keys=[],
        optional_config_keys=["grounded", "num_concepts", "min_coverage", "target_sec", "provider"],
        required_artifacts=["movie_index"],
    ),
    
    "editorial": StageContract(
        name="editorial",
        input_artifact_types=["director_plan"],
        output_artifact_types=["editorial_plan", "editorial_timeline", "script", "editorial_decisions"],
        required_config_keys=[],
        optional_config_keys=["target_sec", "creative_task"],
        required_artifacts=["director_plan"],
    ),
    
    "scene_selection": StageContract(
        name="scene_selection",
        input_artifact_types=["director_plan"],
        output_artifact_types=["selected_scenes", "scene_ranking"],
        required_config_keys=[],
        optional_config_keys=["top_n"],
        required_artifacts=["director_plan"],
    ),
    
    "script": StageContract(
        name="script",
        input_artifact_types=["director_plan"],
        output_artifact_types=["script"],
        required_config_keys=[],
        optional_config_keys=["provider", "model", "target_sec"],
        required_artifacts=["director_plan"],
    ),
    
    "clip_extraction": StageContract(
        name="clip_extraction",
        input_artifact_types=["selected_scenes"],
        output_artifact_types=["extracted_clips"],
        required_config_keys=[],
        optional_config_keys=["codec", "crf"],
        required_artifacts=["selected_scenes", "project_meta.source_path"],
    ),
    
    "visual_generation": StageContract(
        name="visual_generation",
        input_artifact_types=["director_plan"],
        output_artifact_types=["generated_visuals", "asset_plan"],
        required_config_keys=[],
        optional_config_keys=["provider", "model", "prompts"],
        required_artifacts=["director_plan"],
    ),
    
    "tts": StageContract(
        name="tts",
        input_artifact_types=["script", "editorial_plan"],
        output_artifact_types=["tts_audio", "tts_meta"],
        required_config_keys=[],
        optional_config_keys=["provider", "model", "voice", "emotion", "pace"],
        required_artifacts=["script"],
    ),
    
    "render": StageContract(
        name="render",
        input_artifact_types=["extracted_clips", "tts_audio", "generated_visuals", "editorial_timeline"],
        output_artifact_types=["final_render", "render_job"],
        required_config_keys=[],
        optional_config_keys=["codec", "crf", "preset", "resolution"],
        required_artifacts=["tts_audio"],
    ),
    
    "qc": StageContract(
        name="qc",
        input_artifact_types=["final_render", "pipeline_status"],
        output_artifact_types=["qc_report", "pipeline_status"],
        required_config_keys=[],
        optional_config_keys=[],
        required_artifacts=["final_render"],
    ),
}