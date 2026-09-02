"""Artifact registry — track, validate, and query pipeline artifacts."""
from __future__ import annotations
import json
import hashlib
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

from ..manifest import ArtifactRecord, ArtifactType, ProjectManifest


@dataclass
class ArtifactValidationResult:
    """Result of artifact validation."""
    artifact_id: str
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    schema_version: Optional[str] = None


class ArtifactRegistry:
    """Registry for managing and validating pipeline artifacts."""
    
    def __init__(self, manifest: ProjectManifest, artifact_root: Path):
        self.manifest = manifest
        self.artifact_root = Path(artifact_root)
        self._schema_validators: Dict[ArtifactType, Callable[[Path], ArtifactValidationResult]] = {}
    
    def register_validator(
        self,
        artifact_type: ArtifactType,
        validator: Callable[[Path], ArtifactValidationResult],
    ) -> None:
        """Register a validation function for an artifact type."""
        self._schema_validators[artifact_type] = validator
    
    def register_artifact(
        self,
        artifact_type: ArtifactType,
        relative_path: str,
        producer_stage: str,
        version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None,
        input_artifact_ids: Optional[List[str]] = None,
    ) -> ArtifactRecord:
        """Register a new artifact in the manifest."""
        full_path = self.artifact_root / relative_path
        
        # Compute content hash and size
        content_hash = ""
        size_bytes = 0
        if full_path.exists():
            content_hash = self._compute_hash(full_path)
            size_bytes = full_path.stat().st_size
        
        artifact_id = f"{artifact_type.value}_{hashlib.sha256(relative_path.encode()).hexdigest()[:12]}"
        
        record = ArtifactRecord(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            path=relative_path,
            producer_stage=producer_stage,
            version=version,
            content_hash=content_hash,
            size_bytes=size_bytes,
            metadata=metadata or {},
            input_artifact_ids=input_artifact_ids or [],
        )
        
        self.manifest.register_artifact(record)
        return record
    
    def _compute_hash(self, path: Path) -> str:
        """Compute SHA256 hash of file content."""
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def validate_artifact(self, artifact_id: str) -> ArtifactValidationResult:
        """Validate an artifact against its schema."""
        record = self.manifest.get_artifact(artifact_id)
        if not record:
            return ArtifactValidationResult(
                artifact_id=artifact_id,
                valid=False,
                errors=[f"Artifact not found in registry: {artifact_id}"]
            )
        
        full_path = self.artifact_root / record.path
        if not full_path.exists():
            return ArtifactValidationResult(
                artifact_id=artifact_id,
                valid=False,
                errors=[f"Artifact file not found: {full_path}"]
            )
        
        # Check content hash
        current_hash = self._compute_hash(full_path)
        if record.content_hash and current_hash != record.content_hash:
            return ArtifactValidationResult(
                artifact_id=artifact_id,
                valid=False,
                errors=[f"Content hash mismatch: expected {record.content_hash[:16]}, got {current_hash[:16]}"]
            )
        
        # Run type-specific validator if registered
        validator = self._schema_validators.get(record.artifact_type)
        if validator:
            return validator(full_path)
        
        return ArtifactValidationResult(
            artifact_id=artifact_id,
            valid=True,
        )
    
    def validate_all(self) -> Dict[str, ArtifactValidationResult]:
        """Validate all registered artifacts."""
        results = {}
        for artifact_id in self.manifest.artifacts:
            results[artifact_id] = self.validate_artifact(artifact_id)
        return results
    
    def get_missing_dependencies(self, stage_name: str) -> List[str]:
        """Check if all required input artifacts for a stage exist and are valid."""
        from .manifest import STAGE_DEPENDENCIES
        
        missing = []
        for dep_stage in STAGE_DEPENDENCIES.get(stage_name, []):
            dep_artifacts = self.manifest.get_artifacts_by_stage(dep_stage)
            for artifact in dep_artifacts:
                if artifact.artifact_type in [ArtifactType.MOVIE_INDEX, ArtifactType.DIRECTOR_PLAN, 
                                              ArtifactType.SCENE_INDEX, ArtifactType.TRANSCRIPT,
                                              ArtifactType.SCRIPT, ArtifactType.EDITORIAL_PLAN,
                                              ArtifactType.EDITORIAL_TIMELINE, ArtifactType.TTS_AUDIO,
                                              ArtifactType.FINAL_RENDER]:
                    # Critical artifacts must exist
                    full_path = self.artifact_root / artifact.path
                    if not full_path.exists():
                        missing.append(f"{artifact.artifact_id} ({artifact.path})")
        return missing
    
    def can_run_stage(self, stage_name: str) -> tuple[bool, List[str]]:
        """Check if a stage can run (all dependencies satisfied)."""
        missing = self.get_missing_dependencies(stage_name)
        return len(missing) == 0, missing


def register_builtin_validators(registry: ArtifactRegistry) -> None:
    """Register built-in validators for standard artifact types."""
    
    def validate_json(path: Path) -> ArtifactValidationResult:
        try:
            with path.open("r", encoding="utf-8") as f:
                json.load(f)
            return ArtifactValidationResult(artifact_id=path.name, valid=True)
        except json.JSONDecodeError as e:
            return ArtifactValidationResult(
                artifact_id=path.name,
                valid=False,
                errors=[f"Invalid JSON: {e}"]
            )
    
    # JSON artifacts
    for at in [
        ArtifactType.MOVIE_INDEX, ArtifactType.SEMANTIC_INDEX,
        ArtifactType.CHARACTERS, ArtifactType.EVENTS, ArtifactType.MANIFEST,
        ArtifactType.TRANSCRIPT, ArtifactType.SCENE_INDEX, ArtifactType.SCENE_CARDS,
        ArtifactType.SCENE_RANKING, ArtifactType.SELECTED_SCENES,
        ArtifactType.DIRECTOR_PLAN, ArtifactType.GROUNDED_SCRIPT,
        ArtifactType.EDITORIAL_PLAN, ArtifactType.EDITORIAL_TIMELINE,
        ArtifactType.EDITORIAL_DECISIONS, ArtifactType.SCRIPT,
        ArtifactType.ASSET_PLAN, ArtifactType.TTS_META,
        ArtifactType.PROVIDER_MANIFEST, ArtifactType.PIPELINE_STATUS,
        ArtifactType.QC_REPORT, ArtifactType.DIRECTOR_VALIDATION,
        ArtifactType.PROJECT_META, ArtifactType.PROJECT_MANIFEST,
    ]:
        registry.register_validator(at, validate_json)
    
    def validate_wav(path: Path) -> ArtifactValidationResult:
        try:
            import wave
            with wave.open(str(path), "rb") as w:
                if w.getnchannels() == 0 or w.getframerate() == 0:
                    return ArtifactValidationResult(
                        artifact_id=path.name, valid=False,
                        errors=["Invalid WAV: no audio data"]
                    )
            return ArtifactValidationResult(artifact_id=path.name, valid=True)
        except Exception as e:
            return ArtifactValidationResult(
                artifact_id=path.name, valid=False,
                errors=[f"Invalid WAV: {e}"]
            )
    
    registry.register_validator(ArtifactType.TTS_AUDIO, validate_wav)
    
    def validate_video(path: Path) -> ArtifactValidationResult:
        try:
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,duration", "-of", "csv=p=0", str(path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return ArtifactValidationResult(
                    artifact_id=path.name, valid=False,
                    errors=[f"ffprobe failed: {result.stderr}"]
                )
            return ArtifactValidationResult(artifact_id=path.name, valid=True)
        except Exception as e:
            return ArtifactValidationResult(
                artifact_id=path.name, valid=False,
                errors=[f"Video validation failed: {e}"]
            )
    
    registry.register_validator(ArtifactType.FINAL_RENDER, validate_video)
    registry.register_validator(ArtifactType.EXTRACTED_CLIPS, validate_video)
    registry.register_validator(ArtifactType.GENERATED_VISUALS, validate_video)


def get_artifact_path(manifest: ProjectManifest, artifact_root: Path, artifact_type: ArtifactType) -> Optional[Path]:
    """Get the path to the primary artifact of a given type."""
    artifacts = manifest.get_artifacts_by_type(artifact_type)
    if not artifacts:
        return None
    # Return the most recently created
    artifacts.sort(key=lambda a: a.created_at, reverse=True)
    return artifact_root / artifacts[0].path