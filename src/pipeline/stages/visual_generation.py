"""Visual generation stage — generate visual assets via ComfyUI or other providers."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

from .contracts import PipelineStage, StageConfig, StageResult, STAGE_CONTRACTS
from ..manifest import ProjectManifest


class VisualGenerationStage(PipelineStage):
    """Generate visual assets via ComfyUI or other providers."""
    
    def run(self, stage_config: StageConfig) -> StageResult:
        self.mark_running()
        
        try:
            project_id = self.manifest.project_id
            project_dir = self.artifact_root / project_id
            
            # Get config
            provider = self.config.get("provider", "comfyui")
            model = self.config.get("model")
            prompts = self.config.get("prompts", [])
            
            # Load asset plan from director
            asset_plan_path = project_dir / "asset_plan.json"
            if not asset_plan_path.exists():
                # Try to create from director plan
                director_plan_path = project_dir / "director_plan.json"
                if director_plan_path.exists():
                    with director_plan_path.open("r", encoding="utf-8") as f:
                        director_plan = json.load(f)
                    # Create basic asset plan from director visual strategy
                    asset_plan = {"assets": []}
                    visual_strategy = director_plan.get("visual_strategy", [])
                    if isinstance(visual_strategy, list):
                        for i, vs in enumerate(visual_strategy):
                            asset_plan["assets"].append({
                                "id": f"asset_{i}",
                                "prompt": vs,
                                "type": "image",
                            })
                    with asset_plan_path.open("w", encoding="utf-8") as f:
                        json.dump(asset_plan, f, ensure_ascii=False, indent=2)
            
            if not asset_plan_path.exists():
                return StageResult(
                    success=False,
                    stage_name=self.contract.name,
                    error="No asset plan available for visual generation",
                )
            
            # Run visual generation
            if provider == "comfyui":
                from visual_generation.comfyui_client import generate_assets
                generate_assets(project_dir)
            else:
                return StageResult(
                    success=False,
                    stage_name=self.contract.name,
                    error=f"Unknown visual generation provider: {provider}",
                )
            
            # Check for generated assets
            assets_dir = project_dir / "assets" / "generated"
            generated_files = list(assets_dir.glob("*.png")) + list(assets_dir.glob("*.jpg"))
            
            artifact_ids = []
            
            if generated_files:
                artifact_id = self.register_output(
                    artifact_type="generated_visuals",
                    relative_path=f"{project_id}/assets/generated",
                    metadata={
                        "provider": provider,
                        "model": model,
                        "count": len(generated_files),
                    },
                )
                artifact_ids.append(artifact_id)
            
            if asset_plan_path.exists():
                artifact_id = self.register_output(
                    artifact_type="asset_plan",
                    relative_path=f"{project_id}/asset_plan.json",
                )
                artifact_ids.append(artifact_id)
            
            return StageResult(
                success=True,
                stage_name=self.contract.name,
                output_artifact_ids=artifact_ids,
                metrics={
                    "provider": provider,
                    "generated_count": len(generated_files),
                },
            )
            
        except Exception as e:
            return StageResult(
                success=False,
                stage_name=self.contract.name,
                error=str(e),
            )


VisualGenerationStage = VisualGenerationStage