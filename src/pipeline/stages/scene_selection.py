"""Scene selection stage — rank and select scenes for extraction."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

from .contracts import PipelineStage, StageConfig, StageResult, STAGE_CONTRACTS
from ..manifest import ProjectManifest


class SceneSelectionStage(PipelineStage):
    """Rank and select scenes based on director thesis."""
    
    def run(self, stage_config: StageConfig) -> StageResult:
        self.mark_running()
        
        try:
            project_id = self.manifest.project_id
            project_dir = self.artifact_root / project_id
            
            # Get config
            top_n = self.config.get("top_n", 3)
            
            # Load director plan for thesis
            director_plan_path = project_dir / "director_plan.json"
            if not director_plan_path.exists():
                return StageResult(
                    success=False,
                    stage_name=self.contract.name,
                    error="director_plan.json not found",
                )
            
            with director_plan_path.open("r", encoding="utf-8") as f:
                director_plan = json.load(f)
            
            thesis = director_plan.get("thesis")
            if not thesis:
                return StageResult(
                    success=False,
                    stage_name=self.contract.name,
                    error="No thesis in director_plan.json",
                )
            
            # Run scene ranking
            from scene_selection.ranker import rank_scenes
            from scene_selection.selector import select_scenes
            
            print(f"Ranking scenes for thesis: {thesis}")
            rank_scenes(project_dir, thesis, top_k=20)
            
            # Select scenes
            entries = select_scenes(project_dir, top_n=top_n)
            sel_path = project_dir / "scenes" / "selected_scenes.json"
            
            print(f"Selected {len(entries)} scene(s)")
            
            # Verify output
            if not sel_path.exists():
                return StageResult(
                    success=False,
                    stage_name=self.contract.name,
                    error="selected_scenes.json not generated",
                )
            
            # Register artifacts
            artifact_ids = []
            
            artifact_id = self.register_output(
                artifact_type="selected_scenes",
                relative_path=f"{project_id}/scenes/selected_scenes.json",
            )
            artifact_ids.append(artifact_id)
            
            ranking_path = project_dir / "scene_ranking.json"
            if ranking_path.exists():
                artifact_id = self.register_output(
                    artifact_type="scene_ranking",
                    relative_path=f"{project_id}/scene_ranking.json",
                )
                artifact_ids.append(artifact_id)
            
            return StageResult(
                success=True,
                stage_name=self.contract.name,
                output_artifact_ids=artifact_ids,
                metrics={
                    "thesis": thesis[:80],
                    "top_n": top_n,
                    "selected_count": len(entries),
                },
            )
            
        except Exception as e:
            return StageResult(
                success=False,
                stage_name=self.contract.name,
                error=str(e),
            )


SceneSelectionStage = SceneSelectionStage