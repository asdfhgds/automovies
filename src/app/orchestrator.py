"""Orchestrator: run MVP pipeline using local stubs."""
from pathlib import Path
import time
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / 'src'

# Ensure src directory is in Python path for imports
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def start_pipeline(project_id: str):
    project_dir = ROOT / 'data' / project_id
    print(f"Starting pipeline for project {project_id} -> {project_dir}")

    # Phase: Transcription (adapter)
    try:
        from transcription.adapter import transcribe
        # read project meta if available to pass source path
        meta_path = project_dir / 'project_meta.json'
        source = None
        if meta_path.exists():
            try:
                import json
                m = json.loads(meta_path.read_text(encoding='utf-8'))
                source = m.get('source_path')
            except Exception:
                source = None
        transcribe(project_dir, source)
    except Exception as e:
        print(f"Transcription failed: {e}")

    # Phase: Scene indexing (adapter: PySceneDetect preferred)
    try:
        from scene_indexing.adapter import build_scene_cards
        build_scene_cards(project_dir, source)
    except Exception as e:
        print(f"Scene indexing failed: {e}")

    # Phase: Director planning (creative or deterministic)
    plan_path = None
    try:
        import json
        import os
        
        # Load scene index and transcript for director
        scenes_file = project_dir / 'scenes' / 'scene_index.json'
        transcript_file = project_dir / 'transcripts' / 'transcript.json'
        
        scene_index = []
        transcript = {"segments": []}
        
        if scenes_file.exists():
            with scenes_file.open('r', encoding='utf-8') as f:
                scene_index = json.load(f)
                # Handle both list and dict formats
                if isinstance(scene_index, dict) and "scenes" in scene_index:
                    scene_index = scene_index["scenes"]
        
        if transcript_file.exists():
            with transcript_file.open('r', encoding='utf-8') as f:
                transcript = json.load(f)
        
        # Get movie metadata
        meta_path = project_dir / 'project_meta.json'
        movie_metadata = {
            "title": "Untitled",
            "duration_sec": 0,
            "source": source,
        }
        if meta_path.exists():
            with meta_path.open('r', encoding='utf-8') as f:
                meta = json.load(f)
                movie_metadata["title"] = meta.get("title", "Untitled")
        
        # Try creative director first (if enabled)
        use_creative = os.getenv('CREATIVE_DIRECTOR_ENABLED', 'false').lower() == 'true'
        
        if use_creative and scene_index and transcript:
            try:
                from director.creative_director import CreativeDirector
                from director.provider_factory import get_director_config_from_env, get_llm_provider_from_config
                
                print("Using creative director (LLM-backed)...")
                
                # Load provider from configuration
                director_config = get_director_config_from_env()
                provider = get_llm_provider_from_config(director_config)
                
                if not provider:
                    print("Failed to load LLM provider. Falling back to deterministic planner.")
                    use_creative = False
                else:
                    memory_dir = project_dir / 'memory'
                    director = CreativeDirector(provider=provider, memory_dir=memory_dir)
                    
                    result = director.develop_production_plan(
                        movie_metadata=movie_metadata,
                        scene_index=scene_index,
                        transcript=transcript,
                    )
                    
                    # Extract production plan and selected concept
                    production_plan = result.get("production_plan", {})
                    selected_concept = result.get("selected_concept", {})
                    
                    # Build director plan output compatible with existing downstream
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
                    
                    # Write director plan to file
                    plan_path = project_dir / "director_plan.json"
                    with plan_path.open('w', encoding='utf-8') as f:
                        json.dump(director_plan, f, ensure_ascii=False, indent=2)
                    
                    print(f"Creative director thesis: {director_plan['thesis'][:80]}...")
                
            except Exception as e:
                print(f"Creative director failed: {e}. Falling back to deterministic planner.")
                use_creative = False
        
        # Fallback to deterministic planner
        if not use_creative:
            from director.planner import plan_director
            plan_path = plan_director(project_dir)
            print("Using deterministic director (fallback)")
            
    except Exception as e:
        print(f"Director planning failed: {e}")
        plan_path = None

    # Phase: Script generation
    try:
        from script.writer import generate_script
        generate_script(project_dir)
    except Exception as e:
        print(f"Script generation failed: {e}")

    # Phase: Scene ranking (use director thesis if available)
    try:
        from scene_selection.ranker import rank_scenes
        thesis = None
        if plan_path and plan_path.exists():
            try:
                import json
                p = json.loads(plan_path.read_text(encoding='utf-8'))
                thesis = p.get('thesis')
            except Exception:
                thesis = None
        if thesis:
            print(f"Ranking scenes for thesis: {thesis}")
            rank_scenes(project_dir, thesis, top_k=20)
        else:
            print("No thesis found; skipping scene ranking")
    except Exception as e:
        print(f"Scene ranking failed: {e}")

    # Phase: Scene selection
    try:
        from scene_selection.selector import select_best_scene
        sel_path = select_best_scene(project_dir)
        print(f"Selected scene -> {sel_path}")
    except Exception as e:
        print(f"Scene selection failed: {e}")
        raise

    # Phase: Clip extraction
    try:
        from editor.clip_extractor import extract_clip
        # read selected scene and project_meta for source
        import json
        sel = json.loads(sel_path.read_text(encoding='utf-8'))
        meta_path = project_dir / 'project_meta.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        source = meta.get('source_path')
        if not source:
            raise RuntimeError('No source video registered for clip extraction')
        out_dir = project_dir / 'assets' / 'scenes'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{sel.get('scene_id')}.mp4"
        extract_clip(source, sel.get('start_sec'), sel.get('end_sec'), str(out_file))
        print(f"Extracted scene clip -> {out_file}")
    except Exception as e:
        print(f"Clip extraction failed: {e}")
        raise

    # Phase: Visual generation
    try:
        from visual_generation.comfyui_client import generate_assets
        generate_assets(project_dir)
    except Exception as e:
        print(f"Visual generation failed: {e}")

    # Phase: Audio (TTS)
    try:
        from audio.tts_adapter import synthesize_voice
        synthesize_voice(project_dir)
    except Exception as e:
        print(f"TTS failed: {e}")

    # Phase: Assembly
    try:
        from editor.ffmpeg_editor import assemble
        assemble(project_dir)
    except Exception as e:
        print(f"Assembly failed: {e}")

    # Phase: QC
    try:
        from qc.critic import run_qc
        report = run_qc(project_dir)
        print(f"QC checks: {report.get('checks')}")
    except Exception as e:
        print(f"QC failed: {e}")

    print("Pipeline completed (MVP). Check data/<project_id>/renders for output and data/<project_id>/reports for QC.")
