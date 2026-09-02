"""Director stage — run grounded or creative director."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict

from .contracts import PipelineStage, StageConfig, StageResult, STAGE_CONTRACTS
from ..manifest import ProjectManifest


class DirectorStage(PipelineStage):
    """Run grounded or creative director."""
    
    def run(self, stage_config: StageConfig) -> StageResult:
        self.mark_running()
        
        try:
            project_id = self.manifest.project_id
            project_dir = self.artifact_root / project_id
            
            # Get config
            grounded = self.config.get("grounded", "false").lower() == "true"
            num_concepts = self.config.get("num_concepts", 5)
            min_coverage = self.config.get("min_coverage", 0.4)
            target_sec = self.config.get("target_sec", 90)
            provider = self.config.get("provider")
            
            if grounded:
                # Use grounded director via orchestrator's _run_grounded_director
                from app.orchestrator import _run_grounded_director
                from director.provider_factory import get_director_config_from_env, get_llm_provider_from_config
                from utils.strict import strict_mode_enabled
                
                # Set up environment for grounded director
                os.environ["GROUNDED_DIRECTOR"] = "true"
                os.environ["DIRECTOR_NUM_CONCEPTS"] = str(num_concepts)
                os.environ["DIRECTOR_MIN_COVERAGE"] = str(min_coverage)
                os.environ["EDITORIAL_TARGET_SEC"] = str(target_sec)
                if provider:
                    os.environ["DIRECTOR_PROVIDER"] = provider
                
                strict = strict_mode_enabled()
                if strict:
                    os.environ["REQUIRE_REAL_LLM"] = "true"
                
                meta_path = project_dir / "project_meta.json"
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                
                movie_metadata = {
                    "title": meta.get("title", "Untitled"),
                    "duration_sec": meta.get("duration_sec", 0),
                    "source": meta.get("source_path"),
                }
                
                api = _run_grounded_director(
                    project_dir, meta_path, movie_metadata, strict, target_sec
                )
                
                plan_path = api.get("plan_path")
                if not plan_path or not plan_path.exists():
                    return StageResult(
                        success=False,
                        stage_name=self.contract.name,
                        error=api.get("error", "Grounded director failed"),
                    )
                
                # Register artifacts
                artifact_ids = []
                
                artifact_id = self.register_output(
                    artifact_type="director_plan",
                    relative_path=f"{project_id}/director_plan.json",
                    metadata={
                        "grounded": True,
                        "provider": api.get("director_provider"),
                        "model": api.get("director_model"),
                    },
                )
                artifact_ids.append(artifact_id)
                
                grounded_script_path = project_dir / "grounded_script.json"
                if grounded_script_path.exists():
                    artifact_id = self.register_output(
                        artifact_type="grounded_script",
                        relative_path=f"{project_id}/grounded_script.json",
                    )
                    artifact_ids.append(artifact_id)
                
                return StageResult(
                    success=True,
                    stage_name=self.contract.name,
                    output_artifact_ids=artifact_ids,
                    metrics={
                        "grounded": True,
                        "provider": api.get("director_provider"),
                        "model": api.get("director_model"),
                    },
                )
            else:
                # Use creative director (legacy)
                from director.creative_director import CreativeDirector
                from director.provider_factory import get_director_config_from_env, get_llm_provider_from_config
                
                director_config = get_director_config_from_env()
                if provider:
                    director_config["provider"] = provider
                
                llm_provider = get_llm_provider_from_config(director_config)
                if not llm_provider:
                    return StageResult(
                        success=False,
                        stage_name=self.contract.name,
                        error="No LLM provider available for creative director",
                    )
                
                # Load inputs
                scenes_path = project_dir / "scenes" / "scene_index.json"
                transcript_path = project_dir / "transcripts" / "transcript.json"
                
                scene_index = []
                if scenes_path.exists():
                    with scenes_path.open("r", encoding="utf-8") as f:
                        scene_index = json.load(f)
                        if isinstance(scene_index, dict) and "scenes" in scene_index:
                            scene_index = scene_index["scenes"]
                
                transcript = {"segments": []}
                if transcript_path.exists():
                    with transcript_path.open("r", encoding="utf-8") as f:
                        transcript = json.load(f)
                
                meta_path = project_dir / "project_meta.json"
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                
                movie_metadata = {
                    "title": meta.get("title", "Untitled"),
                    "duration_sec": meta.get("duration_sec", 0),
                    "source": meta.get("source_path"),
                }
                
                director = CreativeDirector(provider=llm_provider, memory_dir=project_dir / "memory")
                result = director.develop_production_plan(
                    movie_metadata=movie_metadata,
                    scene_index=scene_index,
                    transcript=transcript,
                )
                
                production_plan = result.get("production_plan", {})
                selected_concept = result.get("selected_concept", {})
                
                director_plan = {
                    "thesis": selected_concept.get("thesis", ""),
                    "hook": selected_concept.get("hook", ""),
                    "title": selected_concept.get("title", movie_metadata["title"]),
                    "tone": selected_concept.get("tone", ""),
                    "structure": production_plan.get("structure", []),
                    "scenes_to_extract": production_plan.get("scenes", []),
                    "creative_generation": True,
                    "concept": selected_concept,
                    "production_plan": production_plan,
                    "all_concepts": result.get("generated_concepts", []),
                }
                
                plan_path = project_dir / "director_plan.json"
                with plan_path.open("w", encoding="utf-8") as f:
                    json.dump(director_plan, f, ensure_ascii=False, indent=2)
                
                artifact_id = self.register_output(
                    artifact_type="director_plan",
                    relative_path=f"{project_id}/director_plan.json",
                    metadata={"grounded": False, "provider": director_config.get("provider")},
                )
                
                return StageResult(
                    success=True,
                    stage_name=self.contract.name,
                    output_artifact_ids=[artifact_id],
                    metrics={
                        "grounded": False,
                        "provider": director_config.get("provider"),
                        "concept": selected_concept.get("title"),
                    },
                )
            
        except Exception as e:
            return StageResult(
                success=False,
                stage_name=self.contract.name,
                error=str(e),
            )


DirectorStage = DirectorStage