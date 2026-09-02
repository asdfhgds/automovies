"""Ingest stage — initialize project and register source video."""
from __future__ import annotations
import uuid
from pathlib import Path
from typing import Any, Dict

from .contracts import PipelineStage, StageConfig, StageResult, STAGE_CONTRACTS
from ..manifest import ProjectManifest, StageRecord


class IngestStage(PipelineStage):
    """Initialize project and register source video."""
    
    def run(self, stage_config: StageConfig) -> StageResult:
        self.mark_running()
        
        try:
            project_id = self.manifest.project_id
            title = self.config.get("title", "Untitled")
            source_path = self.config.get("source_path")
            
            if source_path:
                source_path = str(Path(source_path).expanduser().resolve())
                if not Path(source_path).exists():
                    return StageResult(
                        success=False,
                        stage_name=self.contract.name,
                        error=f"Source video not found: {source_path}",
                    )
            
            # Update project meta
            meta = {
                "project_id": project_id,
                "title": title,
                "source_path": source_path,
            }
            
            # Save project_meta.json
            meta_path = self.artifact_root / project_id / "project_meta.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            
            # Register artifact
            artifact_id = self.register_output(
                artifact_type="project_meta",
                relative_path=f"{project_id}/project_meta.json",
                metadata={"title": title, "source_path": source_path},
            )
            
            # Update manifest title
            self.manifest.title = title
            self.manifest.source_path = source_path
            
            return StageResult(
                success=True,
                stage_name=self.contract.name,
                output_artifact_ids=[artifact_id],
                metrics={"title": title, "has_source": bool(source_path)},
            )
            
        except Exception as e:
            return StageResult(
                success=False,
                stage_name=self.contract.name,
                error=str(e),
            )


# Export for dynamic import
IngestStage = IngestStage