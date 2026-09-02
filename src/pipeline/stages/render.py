"""Render stage — assemble final video using FFmpeg."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

from .contracts import PipelineStage, StageConfig, StageResult, STAGE_CONTRACTS
from ..manifest import ProjectManifest


class RenderStage(PipelineStage):
    """Assemble final video using FFmpeg."""
    
    def run(self, stage_config: StageConfig) -> StageResult:
        self.mark_running()
        
        try:
            project_id = self.manifest.project_id
            project_dir = self.artifact_root / project_id
            
            # Get config
            codec = self.config.get("codec", "libx264")
            crf = self.config.get("crf", 18)
            preset = self.config.get("preset", "medium")
            resolution = self.config.get("resolution")
            
            # Determine render mode
            editorial_timeline_path = project_dir / "editorial_timeline.json"
            if editorial_timeline_path.exists():
                # Editorial render
                from editorial.render import assemble_editorial
                assemble_editorial(project_dir)
                print("Editorial: assembled renders/final_render.mp4")
            else:
                # Standard render
                from editor.ffmpeg_editor import assemble
                assemble(project_dir)
                print("Assembled final_render.mp4 via standard editor")
            
            # Check output
            render_path = project_dir / "renders" / "final_render.mp4"
            if not render_path.exists():
                return StageResult(
                    success=False,
                    stage_name=self.contract.name,
                    error="Final render not generated",
                )
            
            # Register artifacts
            artifact_ids = []
            
            artifact_id = self.register_output(
                artifact_type="final_render",
                relative_path=f"{project_id}/renders/final_render.mp4",
                metadata={
                    "codec": codec,
                    "crf": crf,
                    "preset": preset,
                    "resolution": resolution,
                },
            )
            artifact_ids.append(artifact_id)
            
            render_job_path = project_dir / "render_job.json"
            if render_job_path.exists():
                artifact_id = self.register_output(
                    artifact_type="render_job",
                    relative_path=f"{project_id}/render_job.json",
                )
                artifact_ids.append(artifact_id)
            
            return StageResult(
                success=True,
                stage_name=self.contract.name,
                output_artifact_ids=artifact_ids,
                metrics={
                    "codec": codec,
                    "crf": crf,
                    "preset": preset,
                    "resolution": resolution,
                    "size_bytes": render_path.stat().st_size,
                },
            )
            
        except Exception as e:
            return StageResult(
                success=False,
                stage_name=self.contract.name,
                error=str(e),
            )


RenderStage = RenderStage