"""Script generation stage — generate script from director plan."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict

from .contracts import PipelineStage, StageConfig, StageResult, STAGE_CONTRACTS
from ..manifest import ProjectManifest


class ScriptStage(PipelineStage):
    """Generate script from director plan."""
    
    def run(self, stage_config: StageConfig) -> StageResult:
        self.mark_running()
        
        try:
            project_id = self.manifest.project_id
            project_dir = self.artifact_root / project_id
            
            # Get config
            provider = self.config.get("provider")
            model = self.config.get("model")
            target_sec = self.config.get("target_sec", 90)
            
            # Check if editorial mode already produced script
            script_path = project_dir / "script.json"
            if script_path.exists():
                # Script already exists from editorial stage
                artifact_id = self.register_output(
                    artifact_type="script",
                    relative_path=f"{project_id}/script.json",
                    metadata={"source": "editorial"},
                )
                return StageResult(
                    success=True,
                    stage_name=self.contract.name,
                    output_artifact_ids=[artifact_id],
                    metrics={"source": "editorial"},
                )
            
            # Determine provider
            use_qwen = provider == "qwen" or os.getenv("SCRIPT_PROVIDER", "mock").lower() == "qwen"
            
            if use_qwen:
                # Qwen script generation
                from script.qwen_writer import generate_script_qwen
                
                model = model or os.getenv("SCRIPT_MODEL") or "Qwen/Qwen3-4B-Instruct-2507"
                device = os.getenv("SCRIPT_DEVICE", "auto")
                
                # Set environment
                os.environ["SCRIPT_PROVIDER"] = "qwen"
                os.environ["SCRIPT_MODEL"] = model
                os.environ["SCRIPT_DEVICE"] = device
                
                result = generate_script_qwen(project_dir, model=model, device=device)
                
                artifact_id = self.register_output(
                    artifact_type="script",
                    relative_path=f"{project_id}/script.json",
                    metadata={
                        "provider": "qwen",
                        "model": result.get("script_model"),
                        "device": result.get("script_device"),
                    },
                )
                
                return StageResult(
                    success=True,
                    stage_name=self.contract.name,
                    output_artifact_ids=[artifact_id],
                    metrics={
                        "provider": "qwen",
                        "model": result.get("script_model"),
                        "device": result.get("script_device"),
                    },
                )
            else:
                # Deterministic script
                from script.writer import generate_script
                
                generate_script(project_dir)
                
                if not script_path.exists():
                    return StageResult(
                        success=False,
                        stage_name=self.contract.name,
                        error="Script not generated",
                    )
                
                artifact_id = self.register_output(
                    artifact_type="script",
                    relative_path=f"{project_id}/script.json",
                    metadata={"provider": "deterministic"},
                )
                
                return StageResult(
                    success=True,
                    stage_name=self.contract.name,
                    output_artifact_ids=[artifact_id],
                    metrics={"provider": "deterministic"},
                )
            
        except Exception as e:
            return StageResult(
                success=False,
                stage_name=self.contract.name,
                error=str(e),
            )


ScriptStage = ScriptStage