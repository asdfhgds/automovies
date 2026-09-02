"""Movie intelligence stage — build enriched movie index from transcript and scenes."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

from .contracts import PipelineStage, StageConfig, StageResult, STAGE_CONTRACTS
from ..manifest import ProjectManifest


class MovieIntelligenceStage(PipelineStage):
    """Build movie intelligence from transcript and scene index."""
    
    def run(self, stage_config: StageConfig) -> StageResult:
        self.mark_running()
        
        try:
            project_id = self.manifest.project_id
            project_dir = self.artifact_root / project_id
            
            # Get config
            enricher = self.config.get("enricher", "heuristic")
            attach_keyframes = self.config.get("attach_keyframes", False)
            max_frames = self.config.get("max_frames", 1)
            group_max_scene_sec = self.config.get("group_max_scene_sec", 30.0)
            
            # Run movie intelligence using existing analyzer
            from movie_understanding.analyzer import build_movie_index
            from movie_understanding.enrich_factory import create_scene_enricher_from_env
            
            scene_enricher = create_scene_enricher_from_env()
            scene_enricher.name = enricher
            
            movie_index = build_movie_index(
                project_dir=Path(self.artifact_root) / project_id,
                scene_enricher=scene_enricher,
                attach_keyframes=attach_keyframes,
                max_frames=max_frames,
                group_max_scene_sec=group_max_scene_sec,
            )
            
            # Verify outputs
            movie_index_path = self.artifact_root / project_id / "movie_index.json"
            semantic_index_path = self.artifact_root / project_id / "semantic_index.json"
            
            if not movie_index_path.exists():
                return StageResult(
                    success=False,
                    stage_name=self.contract.name,
                    error="movie_index.json not generated",
                )
            
            # Register artifacts
            artifact_ids = []
            
            artifact_id = self.register_output(
                artifact_type="movie_index",
                relative_path=f"{project_id}/movie_index.json",
                metadata={
                    "enricher": enricher,
                    "attach_keyframes": attach_keyframes,
                    "group_max_scene_sec": group_max_scene_sec,
                    "num_scenes": len(movie_index.get("scenes", [])),
                },
            )
            artifact_ids.append(artifact_id)
            
            if semantic_index_path.exists():
                artifact_id = self.register_output(
                    artifact_type="semantic_index",
                    relative_path=f"{project_id}/semantic_index.json",
                )
                artifact_ids.append(artifact_id)
            
            manifest_path = self.artifact_root / project_id / "manifest.json"
            if manifest_path.exists():
                artifact_id = self.register_output(
                    artifact_type="manifest",
                    relative_path=f"{project_id}/manifest.json",
                )
                artifact_ids.append(artifact_id)
            
            return StageResult(
                success=True,
                stage_name=self.contract.name,
                output_artifact_ids=artifact_ids,
                metrics={
                    "enricher": enricher,
                    "num_scenes": len(movie_index.get("scenes", [])),
                    "duration_sec": movie_index.get("movie", {}).get("duration_sec", 0),
                },
            )
            
        except Exception as e:
            return StageResult(
                success=False,
                stage_name=self.contract.name,
                error=str(e),
            )


MovieIntelligenceStage = MovieIntelligenceStage