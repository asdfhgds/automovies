"""Clip extraction stage — extract video clips for selected scenes."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

from .contracts import PipelineStage, StageConfig, StageResult, STAGE_CONTRACTS
from ..manifest import ProjectManifest


class ClipExtractionStage(PipelineStage):
    """Extract video clips for selected scenes."""
    
    def run(self, stage_config: StageConfig) -> StageResult:
        self.mark_running()
        
        try:
            project_id = self.manifest.project_id
            project_dir = self.artifact_root / project_id
            
            # Get config
            codec = self.config.get("codec", "libx264")
            crf = self.config.get("crf", 18)
            
            # Load selected scenes
            sel_path = project_dir / "scenes" / "selected_scenes.json"
            if not sel_path.exists():
                return StageResult(
                    success=False,
                    stage_name=self.contract.name,
                    error="selected_scenes.json not found",
                )
            
            with sel_path.open("r", encoding="utf-8") as f:
                selections = json.load(f)
            
            if not selections:
                return StageResult(
                    success=False,
                    stage_name=self.contract.name,
                    error="No scenes selected",
                )
            
            # Get source video
            meta_path = project_dir / "project_meta.json"
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            
            source = meta.get("source_path")
            if not source:
                return StageResult(
                    success=False,
                    stage_name=self.contract.name,
                    error="No source video registered",
                )
            
            # Extract clips
            from editor.clip_extractor import extract_clip
            
            out_dir = project_dir / "assets" / "scenes"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            extracted = []
            for sel in selections:
                out_file = out_dir / f"{sel.get('scene_id')}.mp4"
                extract_clip(source, sel.get("start_sec"), sel.get("end_sec"), str(out_file))
                extracted.append(str(out_file))
                print(f"Extracted scene clip -> {out_file}")
            
            if not extracted:
                return StageResult(
                    success=False,
                    stage_name=self.contract.name,
                    error="No scene clips were extracted",
                )
            
            # Register artifact
            artifact_id = self.register_output(
                artifact_type="extracted_clips",
                relative_path=f"{project_id}/assets/scenes",
                metadata={
                    "codec": codec,
                    "crf": crf,
                    "count": len(extracted),
                    "scenes": [s.get("scene_id") for s in selections],
                },
            )
            
            return StageResult(
                success=True,
                stage_name=self.contract.name,
                output_artifact_ids=[artifact_id],
                metrics={
                    "extracted_count": len(extracted),
                    "codec": codec,
                    "crf": crf,
                },
            )
            
        except Exception as e:
            return StageResult(
                success=False,
                stage_name=self.contract.name,
                error=str(e),
            )


ClipExtractionStage = ClipExtractionStage