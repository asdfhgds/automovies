"""Editorial stage — build editorial plan, timeline, and evidence-aligned script."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

from .contracts import PipelineStage, StageConfig, StageResult, STAGE_CONTRACTS
from ..manifest import ProjectManifest


class EditorialStage(PipelineStage):
    """Build editorial plan, timeline, and script."""
    
    def run(self, stage_config: StageConfig) -> StageResult:
        self.mark_running()
        
        try:
            project_id = self.manifest.project_id
            project_dir = self.artifact_root / project_id
            
            # Get config
            target_sec = self.config.get("target_sec", 90)
            creative_task = self.config.get("creative_task", "")
            
            # Load director plan
            director_plan_path = project_dir / "director_plan.json"
            if not director_plan_path.exists():
                return StageResult(
                    success=False,
                    stage_name=self.contract.name,
                    error="director_plan.json not found",
                )
            
            with director_plan_path.open("r", encoding="utf-8") as f:
                director_plan = json.load(f)
            
            # Determine if grounded
            grounded = director_plan.get("grounded", False)
            
            # Run editorial pipeline
            from editorial.director import create_editorial_plan
            from editorial.grounded import GroundedEditorialPlanner
            from editorial.script import build_editorial_script
            from editorial.timeline import EditorialTimelineBuilder
            from movie_understanding import movie_memory
            from script.grounded import load_grounded_script
            
            if grounded:
                grounded_script = load_grounded_script(project_dir)
                if grounded_script and grounded_script.get("sections"):
                    planner = GroundedEditorialPlanner(script=grounded_script)
                    plan = planner.create_plan(
                        movie_index=movie_memory.load_movie_index(project_dir),
                        director_plan=director_plan,
                        creative_task=creative_task,
                        target_sec=target_sec,
                    )
                    movie_memory.save_json(project_dir, "editorial_plan.json", plan.to_dict())
                else:
                    plan = create_editorial_plan(
                        project_dir,
                        creative_task=creative_task,
                        target_sec=target_sec,
                    )
            else:
                plan = create_editorial_plan(
                    project_dir,
                    creative_task=creative_task,
                    target_sec=target_sec,
                )
            
            # Build editorial script
            movie_index = movie_memory.load_movie_index(project_dir)
            script = build_editorial_script(project_dir, plan, movie_index)
            
            # Build timeline
            meta_path = project_dir / "project_meta.json"
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            
            source_path = meta.get("source_path")
            builder = EditorialTimelineBuilder(source_path=source_path)
            timeline = builder.build(project_dir, plan, script)
            
            # Verify outputs
            artifact_ids = []
            
            editorial_plan_path = project_dir / "editorial_plan.json"
            if editorial_plan_path.exists():
                artifact_id = self.register_output(
                    artifact_type="editorial_plan",
                    relative_path=f"{project_id}/editorial_plan.json",
                )
                artifact_ids.append(artifact_id)
            
            editorial_timeline_path = project_dir / "editorial_timeline.json"
            if editorial_timeline_path.exists():
                artifact_id = self.register_output(
                    artifact_type="editorial_timeline",
                    relative_path=f"{project_id}/editorial_timeline.json",
                )
                artifact_ids.append(artifact_id)
            
            script_path = project_dir / "script.json"
            if script_path.exists():
                artifact_id = self.register_output(
                    artifact_type="script",
                    relative_path=f"{project_id}/script.json",
                )
                artifact_ids.append(artifact_id)
            
            editorial_decisions_path = project_dir / "editorial_decisions.json"
            if editorial_decisions_path.exists():
                artifact_id = self.register_output(
                    artifact_type="editorial_decisions",
                    relative_path=f"{project_id}/editorial_decisions.json",
                )
                artifact_ids.append(artifact_id)
            
            return StageResult(
                success=True,
                stage_name=self.contract.name,
                output_artifact_ids=artifact_ids,
                metrics={
                    "grounded": grounded,
                    "target_sec": target_sec,
                    "timeline_segments": len(timeline.get("segments", [])),
                },
            )
            
        except Exception as e:
            return StageResult(
                success=False,
                stage_name=self.contract.name,
                error=str(e),
            )


EditorialStage = EditorialStage