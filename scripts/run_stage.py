"""Stage runner — execute individual pipeline stages with resume/skip logic."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline import (
    ProjectManifest, StageRecord, StageStatus, load_manifest, save_manifest,
    create_project_manifest, PIPELINE_STAGES, STAGE_DEPENDENCIES,
    ArtifactRegistry, register_builtin_validators, get_storage_root,
)
from pipeline.stages.contracts import STAGE_CONTRACTS, StageConfig, StageResult, PipelineStage


# Import stage implementations
def import_stage_module(stage_name: str) -> Optional[type]:
    """Dynamically import stage implementation module."""
    try:
        if stage_name == "ingest":
            from pipeline.stages import ingest
            return ingest.IngestStage
        elif stage_name == "transcription":
            from pipeline.stages import transcription
            return transcription.TranscriptionStage
        elif stage_name == "scene_indexing":
            from pipeline.stages import scene_indexing
            return scene_indexing.SceneIndexingStage
        elif stage_name == "movie_intelligence":
            from pipeline.stages import movie_intelligence
            return movie_intelligence.MovieIntelligenceStage
        elif stage_name == "director":
            from pipeline.stages import director
            return director.DirectorStage
        elif stage_name == "editorial":
            from pipeline.stages import editorial
            return editorial.EditorialStage
        elif stage_name == "scene_selection":
            from pipeline.stages import scene_selection
            return scene_selection.SceneSelectionStage
        elif stage_name == "script":
            from pipeline.stages import script
            return script.ScriptStage
        elif stage_name == "clip_extraction":
            from pipeline.stages import clip_extraction
            return clip_extraction.ClipExtractionStage
        elif stage_name == "visual_generation":
            from pipeline.stages import visual_generation
            return visual_generation.VisualGenerationStage
        elif stage_name == "tts":
            from pipeline.stages import tts
            return tts.TTSStage
        elif stage_name == "render":
            from pipeline.stages import render
            return render.RenderStage
        elif stage_name == "qc":
            from pipeline.stages import qc
            return qc.QCStage
    except ImportError as e:
        print(f"Warning: Could not import stage module {stage_name}: {e}")
    return None


@dataclass
class RunOptions:
    project_id: str
    stage_name: str
    force: bool = False
    skip_validation: bool = False
    config: Dict[str, Any] = None
    storage_root: Optional[Path] = None
    dry_run: bool = False
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}
        self.storage = get_storage_root(self.storage_root)
    
    @property
    def project_root(self) -> Path:
        return self.storage.root
    
    @property
    def artifact_root(self) -> Path:
        return self.storage.root


def load_project_manifest(opts: RunOptions) -> ProjectManifest:
    """Load or create project manifest."""
    project_dir = opts.storage.project_path(opts.project_id)
    manifest_path = project_dir / "project_manifest.json"
    
    if manifest_path.exists():
        manifest = load_manifest(opts.storage.root, opts.project_id)
        # Ensure all stages are initialized
        for stage_name in PIPELINE_STAGES:
            if stage_name not in manifest.stages:
                manifest.stages[stage_name] = StageRecord(name=stage_name)
        return manifest
    else:
        # Try to load from legacy project_meta.json
        meta_path = project_dir / "project_meta.json"
        if meta_path.exists():
            import json
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            manifest = create_project_manifest(
                opts.storage.root,
                opts.project_id,
                title=meta.get("title", "Untitled"),
                source_path=meta.get("source_path"),
            )
            return manifest
        raise FileNotFoundError(f"Project not found: {project_dir}")


def check_dependencies(manifest: ProjectManifest, stage_name: str, storage: "StorageRoot") -> tuple[bool, List[str]]:
    """Check if all stage dependencies are satisfied."""
    missing = []
    for dep in STAGE_DEPENDENCIES.get(stage_name, []):
        dep_stage = manifest.get_stage(dep)
        if dep_stage.status != StageStatus.COMPLETED:
            missing.append(f"Stage '{dep}' not completed (status: {dep_stage.status.value})")
        else:
            # Check that dependency produced required artifacts
            contract = STAGE_CONTRACTS.get(dep)
            if contract:
                for artifact_name in contract.required_artifacts:
                    # Check if artifact exists
                    found = False
                    for artifact in manifest.artifacts.values():
                        if artifact.producer_stage == dep and artifact.path.endswith(artifact_name):
                            found = True
                            break
                    if not found:
                        missing.append(f"Missing artifact from '{dep}': {artifact_name}")
    return len(missing) == 0, missing


def can_skip_stage(manifest: ProjectManifest, stage_name: str, force: bool) -> tuple[bool, str]:
    """Determine if stage can be skipped."""
    if force:
        return False, "Force rerun requested"
    
    stage = manifest.get_stage(stage_name)
    if stage.status == StageStatus.COMPLETED:
        # Check if outputs still exist and are valid
        contract = STAGE_CONTRACTS.get(stage_name)
        if contract:
            for artifact_name in contract.output_artifact_types:
                found = False
                for artifact in manifest.artifacts.values():
                    if artifact.producer_stage == stage_name and artifact.artifact_type.value == artifact_name:
                        found = True
                        break
                if not found:
                    return False, f"Output artifact missing: {artifact_name}"
        return True, f"Stage already completed"
    
    return False, ""


def run_stage(opts: RunOptions) -> int:
    """Execute a single pipeline stage."""
    # Load manifest
    manifest = load_project_manifest(opts)
    
    # Setup artifact registry
    registry = ArtifactRegistry(manifest, opts.storage.root)
    register_builtin_validators(registry)
    
    # Get stage contract
    contract = STAGE_CONTRACTS.get(opts.stage_name)
    if not contract:
        print(f"Error: Unknown stage: {opts.stage_name}")
        return 1
    
    # Check if we can skip
    if not opts.force:
        can_skip, reason = can_skip_stage(manifest, opts.stage_name, opts.force)
        if can_skip:
            print(f"[SKIP]  Skipping {opts.stage_name}: {reason}")
            return 0
    
    # Check dependencies
    deps_ok, missing = check_dependencies(manifest, opts.stage_name, opts.storage)
    if not deps_ok:
        print(f"[FAIL] Cannot run {opts.stage_name}: missing dependencies")
        for m in missing:
            print(f"  - {m}")
        return 1
    
    # Import stage implementation
    stage_class = import_stage_module(opts.stage_name)
    if not stage_class:
        print(f"[FAIL] Stage implementation not found: {opts.stage_name}")
        print(f"   Available stages: {list(STAGE_CONTRACTS.keys())}")
        return 1
    
    # Create stage instance
    stage_config = StageConfig(
        parameters=opts.config,
        force_rerun=opts.force,
        skip_validation=opts.skip_validation,
    )
    
    stage = stage_class(contract, manifest, opts.storage.root, opts.config)
    
    # Validate inputs
    input_errors = stage.validate_inputs()
    if input_errors and not opts.skip_validation:
        print(f"[FAIL] Input validation failed for {opts.stage_name}:")
        for e in input_errors:
            print(f"  - {e}")
        return 1
    
    # Validate config
    config_errors = stage.validate_config()
    if config_errors:
        print(f"[FAIL] Config validation failed for {opts.stage_name}:")
        for e in config_errors:
            print(f"  - {e}")
        return 1
    
    if opts.dry_run:
        print(f"[DRY] Dry run: would execute {opts.stage_name}")
        print(f"   Inputs: {contract.required_artifacts}")
        print(f"   Outputs: {contract.output_artifact_types}")
        print(f"   Config: {opts.config}")
        return 0
    
    # Execute stage
    print(f"[RUN]  Running {opts.stage_name}...")
    stage.mark_running()
    save_manifest(manifest, opts.storage.root)
    
    start_time = time.monotonic()
    try:
        result = stage.run(stage_config)
        duration = time.monotonic() - start_time
        
        if result.success:
            stage.mark_completed(duration)
            print(f"[OK] {opts.stage_name} completed in {duration:.1f}s")
            if result.output_artifact_ids:
                print(f"   Produced {len(result.output_artifact_ids)} artifacts")
            if result.metrics:
                for k, v in result.metrics.items():
                    print(f"   {k}: {v}")
        else:
            stage.mark_failed(result.error or "Unknown error", duration)
            print(f"[FAIL] {opts.stage_name} failed after {duration:.1f}s: {result.error}")
        
        save_manifest(manifest, opts.storage.root)
        return 0 if result.success else 1
        
    except Exception as e:
        duration = time.monotonic() - start_time
        stage.mark_failed(str(e), duration)
        save_manifest(manifest, opts.storage.root)
        print(f"[FAIL] {opts.stage_name} failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_all_stages(opts: RunOptions) -> int:
    """Run all pipeline stages in order."""
    manifest = load_project_manifest(opts)
    registry = ArtifactRegistry(manifest, opts.storage.root)
    register_builtin_validators(registry)
    
    # Determine which stages to run
    stages_to_run = []
    for stage_name in PIPELINE_STAGES:
        can_skip, reason = can_skip_stage(manifest, stage_name, opts.force)
        if not can_skip:
            deps_ok, missing = check_dependencies(manifest, stage_name, opts.storage)
            if deps_ok:
                stages_to_run.append(stage_name)
            else:
                print(f"⏸  {stage_name}: waiting for dependencies")
                for m in missing:
                    print(f"   - {m}")
        else:
            print(f"[SKIP]  {stage_name}: {reason}")
    
    if not stages_to_run:
        print("[OK] All stages up to date")
        return 0
    
    print(f"📋 Will run {len(stages_to_run)} stages: {', '.join(stages_to_run)}")
    
    # Run each stage
    for stage_name in stages_to_run:
        stage_opts = RunOptions(
            project_id=opts.project_id,
            stage_name=stage_name,
            force=opts.force,
            skip_validation=opts.skip_validation,
            config=opts.config,
            storage_root=opts.storage_root,
            dry_run=opts.dry_run,
        )
        result = run_stage(stage_opts)
        if result != 0:
            print(f"[FAIL] Pipeline stopped at {stage_name}")
            return result
    
    print("[OK] All stages completed successfully")
    return 0


def show_status(opts: RunOptions) -> int:
    """Show project status."""
    manifest = load_project_manifest(opts)
    
    print(f"\n[PROJECT] {manifest.title} ({manifest.project_id})")
    print(f"   Source: {manifest.source_path or 'Not set'}")
    print(f"   Pipeline version: {manifest.pipeline_version}")
    print(f"   Created: {manifest.created_at}")
    print(f"   Updated: {manifest.updated_at}")
    print(f"   Storage root: {opts.storage.root}")
    print(f"   Config: {json.dumps(manifest.config, indent=2) if manifest.config else '{}'}")
    print()
    
    print("Stage Status:")
    print(f"  {'Stage':<25} {'Status':<15} {'Duration':<10} {'Artifacts'}")
    print(f"  {'-'*65}")
    
    for stage_name in PIPELINE_STAGES:
        stage = manifest.stages.get(stage_name)
        if not stage:
            continue
        status_icon = {
            StageStatus.NOT_STARTED: "[ ]",
            StageStatus.RUNNING: "[>]",
            StageStatus.COMPLETED: "[OK]",
            StageStatus.FAILED: "[FAIL]",
            StageStatus.SKIPPED: "[SKIP]",
        }.get(stage.status, "[?]")
        
        artifact_count = len(stage.output_artifact_ids)
        duration = f"{stage.duration_seconds:.1f}s" if stage.duration_seconds > 0 else "-"
        
        print(f"  {stage_name:<25} {status_icon} {stage.status.value:<12} {duration:<10} {artifact_count}")
        
        if stage.error:
            print(f"    Error: {stage.error}")
    
    print()
    print(f"Artifacts: {len(manifest.artifacts)} total")
    
    # Show artifacts by type
    from collections import Counter
    type_counts = Counter(a.artifact_type.value for a in manifest.artifacts.values())
    for atype, count in sorted(type_counts.items()):
        print(f"  {atype}: {count}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="AutoMovies Pipeline Stage Runner")
    parser.add_argument("--storage-root", default=None, help="Root directory for project storage (default: auto-detect)")
    parser.add_argument("--project-id", required=True, help="Project ID")
    parser.add_argument("--stage", help="Stage to run (or 'all' for full pipeline)")
    parser.add_argument("--force", action="store_true", help="Force rerun even if completed")
    parser.add_argument("--skip-validation", action="store_true", help="Skip input validation")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    parser.add_argument("--config", action="append", help="Config key=value pairs (e.g. --config grounded=true)")
    parser.add_argument("--status", action="store_true", help="Show project status and exit")
    
    args = parser.parse_args()
    
    storage_root = Path(args.storage_root).resolve() if args.storage_root else None
    
    # Parse config
    config = {}
    for c in args.config or []:
        if "=" in c:
            k, v = c.split("=", 1)
            # Try to parse value
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
            config[k] = v
    
    opts = RunOptions(
        project_id=args.project_id,
        stage_name=args.stage or "all",
        force=args.force,
        skip_validation=args.skip_validation,
        config=config,
        storage_root=storage_root,
        dry_run=args.dry_run,
    )
    
    if args.status:
        return show_status(opts)
    
    if not args.stage:
        print("Error: --stage required (use 'all' for full pipeline)")
        parser.print_help()
        return 1
    
    if args.stage == "all":
        return run_all_stages(opts)
    else:
        return run_stage(opts)


if __name__ == "__main__":
    sys.exit(main())