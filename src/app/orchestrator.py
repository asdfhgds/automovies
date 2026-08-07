"""Orchestrator: run MVP pipeline using local stubs."""
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent.parent


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

    # Phase: Director planning
    try:
        from director.planner import plan_director
        plan_path = plan_director(project_dir)
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
