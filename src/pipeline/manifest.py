"""Project manifest — central record of project state and artifacts."""
from __future__ import annotations
import json
import uuid
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional, Dict, List
from enum import Enum


class StageStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ArtifactType(str, Enum):
    # Movie intelligence
    MOVIE_INDEX = "movie_index"
    SEMANTIC_INDEX = "semantic_index"
    CHARACTERS = "characters"
    EVENTS = "events"
    MANIFEST = "manifest"
    
    # Transcription
    TRANSCRIPT = "transcript"
    
    # Scene indexing
    SCENE_INDEX = "scene_index"
    SCENE_CARDS = "scene_cards"
    SCENE_RANKING = "scene_ranking"
    SELECTED_SCENES = "selected_scenes"
    
    # Director
    DIRECTOR_PLAN = "director_plan"
    GROUNDED_SCRIPT = "grounded_script"
    
    # Editorial
    EDITORIAL_PLAN = "editorial_plan"
    EDITORIAL_TIMELINE = "editorial_timeline"
    EDITORIAL_DECISIONS = "editorial_decisions"
    
    # Script
    SCRIPT = "script"
    
    # Assets
    EXTRACTED_CLIPS = "extracted_clips"
    GENERATED_VISUALS = "generated_visuals"
    ASSET_PLAN = "asset_plan"
    
    # Audio
    TTS_AUDIO = "tts_audio"
    TTS_META = "tts_meta"
    
    # Render
    FINAL_RENDER = "final_render"
    RENDER_JOB = "render_job"
    
    # Reports
    PROVIDER_MANIFEST = "provider_manifest"
    PIPELINE_STATUS = "pipeline_status"
    QC_REPORT = "qc_report"
    DIRECTOR_REASONING = "director_reasoning"
    DIRECTOR_VALIDATION = "director_validation"
    
    # Project
    PROJECT_META = "project_meta"
    PROJECT_MANIFEST = "project_manifest"


@dataclass
class ArtifactRecord:
    """Record of a single artifact produced by a stage."""
    artifact_id: str
    artifact_type: ArtifactType
    path: str
    producer_stage: str
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    content_hash: str = ""
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    input_artifact_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactRecord":
        # Handle legacy enum string
        if isinstance(data.get("artifact_type"), str):
            data["artifact_type"] = ArtifactType(data["artifact_type"])
        return cls(**data)


@dataclass
class StageRecord:
    """Record of a single pipeline stage execution."""
    name: str
    status: StageStatus = StageStatus.NOT_STARTED
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    config_hash: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    model: Optional[str] = None
    device: Optional[str] = None
    input_artifact_ids: List[str] = field(default_factory=list)
    output_artifact_ids: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageRecord":
        if isinstance(data.get("status"), str):
            data["status"] = StageStatus(data["status"])
        return cls(**data)


@dataclass
class ProjectManifest:
    """Complete project manifest — the source of truth for project state."""
    project_id: str
    title: str
    source_path: Optional[str] = None
    pipeline_version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    # Stage records
    stages: Dict[str, StageRecord] = field(default_factory=dict)
    
    # Artifact registry
    artifacts: Dict[str, ArtifactRecord] = field(default_factory=dict)
    
    # Global config used for this project
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Pipeline-wide metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "title": self.title,
            "source_path": self.source_path,
            "pipeline_version": self.pipeline_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "artifacts": {k: v.to_dict() for k, v in self.artifacts.items()},
            "config": self.config,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectManifest":
        manifest = cls(
            project_id=data["project_id"],
            title=data["title"],
            source_path=data.get("source_path"),
            pipeline_version=data.get("pipeline_version", "1.0"),
            created_at=data.get("created_at", datetime.utcnow().isoformat() + "Z"),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat() + "Z"),
            config=data.get("config", {}),
            metadata=data.get("metadata", {}),
        )
        manifest.stages = {k: StageRecord.from_dict(v) for k, v in data.get("stages", {}).items()}
        manifest.artifacts = {k: ArtifactRecord.from_dict(v) for k, v in data.get("artifacts", {}).items()}
        return manifest
    
    def get_stage(self, name: str) -> StageRecord:
        if name not in self.stages:
            self.stages[name] = StageRecord(name=name)
        return self.stages[name]
    
    def get_artifact(self, artifact_id: str) -> Optional[ArtifactRecord]:
        return self.artifacts.get(artifact_id)
    
    def register_artifact(self, artifact: ArtifactRecord) -> None:
        self.artifacts[artifact.artifact_id] = artifact
        self.updated_at = datetime.utcnow().isoformat() + "Z"
    
    def get_artifacts_by_type(self, artifact_type: ArtifactType) -> List[ArtifactRecord]:
        return [a for a in self.artifacts.values() if a.artifact_type == artifact_type]
    
    def get_artifacts_by_stage(self, stage_name: str) -> List[ArtifactRecord]:
        return [a for a in self.artifacts.values() if a.producer_stage == stage_name]
    
    def compute_config_hash(self) -> str:
        """Compute deterministic hash of project config."""
        config_str = json.dumps(self.config, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# Pipeline stage definitions (order matters for dependencies)
PIPELINE_STAGES = [
    "ingest",
    "transcription",
    "scene_indexing",
    "movie_intelligence",
    "director",
    "editorial",
    "scene_selection",
    "script",
    "clip_extraction",
    "visual_generation",
    "tts",
    "render",
    "qc",
]

# Stage dependencies (DAG)
STAGE_DEPENDENCIES = {
    "ingest": [],
    "transcription": ["ingest"],
    "scene_indexing": ["ingest"],
    "movie_intelligence": ["transcription", "scene_indexing"],
    "director": ["movie_intelligence"],
    "editorial": ["director"],
    "scene_selection": ["director"],
    "script": ["director"],
    "clip_extraction": ["scene_selection"],
    "visual_generation": ["director"],
    "tts": ["script", "editorial"],
    "render": ["clip_extraction", "tts", "visual_generation", "editorial"],
    "qc": ["render"],
}


def load_manifest(project_root: Path, project_id: str) -> ProjectManifest:
    """Load project manifest from disk."""
    manifest_path = project_root / project_id / "project_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            return ProjectManifest.from_dict(json.load(f))
    raise FileNotFoundError(f"Project manifest not found: {manifest_path}")


def save_manifest(manifest: ProjectManifest, project_root: Path) -> Path:
    """Save project manifest to disk."""
    project_dir = project_root / manifest.project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = project_dir / "project_manifest.json"
    manifest.updated_at = datetime.utcnow().isoformat() + "Z"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)
    return manifest_path


def create_project_manifest(
    project_root: Path,
    project_id: str,
    title: str,
    source_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> ProjectManifest:
    """Create a new project manifest."""
    manifest = ProjectManifest(
        project_id=project_id,
        title=title,
        source_path=source_path,
        config=config or {},
    )
    # Initialize all stages as NOT_STARTED
    for stage_name in PIPELINE_STAGES:
        manifest.stages[stage_name] = StageRecord(name=stage_name)
    save_manifest(manifest, project_root)
    return manifest