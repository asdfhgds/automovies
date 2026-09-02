"""Scene indexing stage — run PySceneDetect on source video."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

from .contracts import PipelineStage, StageConfig, StageResult, STAGE_CONTRACTS
from ..manifest import ProjectManifest


class SceneIndexingStage(PipelineStage):
    """Run scene detection on source video."""
    
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
            threshold = self.config.get("threshold", 30.0)
            min_scene_len = self.config.get("min_scene_len", 1.0)
            
            # Run scene detection using existing adapter
            from scene_indexing.adapter import build_scene_cards
            
            build_scene_cards(project_dir, source_path)
            
            # Verify output
            scene_index_path = project_dir / "scenes" / "scene_index.json"
            if not scene_index_path.exists():
                return StageResult(
                    success=False,
                    stage_name=self.contract.name,
                    error="Scene index not generated",
                )
            
            # Register artifact
            artifact_id = self.register_output(
                artifact_type="scene_index",
                relative_path=f"{project_id}/scenes/scene_index.json",
                metadata={"threshold": threshold, "min_scene_len": min_scene_len},
            )
            
            return StageResult(
                success=True,
                stage_name=self.contract.name,
                output_artifact_ids=[artifact_id],
                metrics={"threshold": threshold, "min_scene_len": min_scene_len},
            )
            
        except Exception as e:
            return StageResult(
                success=False,
                stage_name=self.contract.name,
                error=str(e),
            )


SceneIndexingStage = SceneIndexingStage