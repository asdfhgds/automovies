"""Transcription stage — run WhisperX on source video."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

from .contracts import PipelineStage, StageConfig, StageResult, STAGE_CONTRACTS
from ..manifest import ProjectManifest


class TranscriptionStage(PipelineStage):
    """Run transcription on source video."""
    
    def run(self, stage_config: StageConfig) -> StageResult:
        self.mark_running()
        
        try:
            project_id = self.manifest.project_id
            project_dir = self.artifact_root / project_id
            
            # Get source path from project meta
            meta_path = project_dir / "project_meta.json"
            if not meta_path.exists():
                return StageResult(
                    success=False,
                    stage_name=self.contract.name,
                    error="project_meta.json not found",
                )
            
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            
            source_path = meta.get("source_path")
            if not source_path:
                return StageResult(
                    success=False,
                    stage_name=self.contract.name,
                    error="No source_path in project_meta.json",
                )
            
            # Get config
            model = self.config.get("model", "large-v3")
            language = self.config.get("language")
            word_timestamps = self.config.get("word_timestamps", True)
            
            # Run transcription using existing adapter
            from transcription.adapter import transcribe
            
            transcribe(project_dir, source_path)
            
            # Verify output
            transcript_path = project_dir / "transcripts" / "transcript.json"
            if not transcript_path.exists():
                return StageResult(
                    success=False,
                    stage_name=self.contract.name,
                    error="Transcript not generated",
                )
            
            # Register artifact
            artifact_id = self.register_output(
                artifact_type="transcript",
                relative_path=f"{project_id}/transcripts/transcript.json",
                metadata={"model": model, "language": language, "word_timestamps": word_timestamps},
            )
            
            return StageResult(
                success=True,
                stage_name=self.contract.name,
                output_artifact_ids=[artifact_id],
                metrics={"model": model, "language": language},
            )
            
        except Exception as e:
            return StageResult(
                success=False,
                stage_name=self.contract.name,
                error=str(e),
            )


TranscriptionStage = TranscriptionStage