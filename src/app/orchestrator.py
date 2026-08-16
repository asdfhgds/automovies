"""Orchestrator: run the MVP pipeline using local stubs or real providers.

Honors strict GPU validation mode:

    STUDIO_PROFILE=colab-gpu
    REQUIRE_REAL_LLM=true

When strict mode is active the pipeline MUST use real Qwen (director + script)
on CUDA. Any attempt to use a mock provider or the deterministic director/script
raises a clear error, and a provider_manifest.json records exactly what ran.
"""
from pathlib import Path
import time
import sys
import os
import json

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / 'src'

# Ensure src directory is in Python path for imports
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.strict import strict_mode_enabled, require_cuda, require_real_provider


def start_pipeline(project_id: str):
    project_dir = ROOT / 'data' / project_id
    strict = strict_mode_enabled()
    print(f"Starting pipeline for project {project_id} -> {project_dir}")
    print(f"Strict GPU mode (REQUIRE_REAL_LLM): {'ON' if strict else 'OFF'}")

    pipeline_t0 = time.monotonic()
    manifest = {
        "project_id": project_id,
        "strict_mode": strict,
        "profile": os.getenv('STUDIO_PROFILE', 'local'),
    }

    # Read source path
    source = None
    meta_path = project_dir / 'project_meta.json'
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            source = meta.get('source_path')
        except Exception:
            source = None

    if strict:
        gpu = require_cuda()
        print(f"[STRICT] CUDA available: {gpu}")
        dc = _director_config()
        if dc.get('provider', 'mock') != 'qwen':
            raise RuntimeError(
                "REQUIRE_REAL_LLM=true but DIRECTOR_PROVIDER != qwen. "
                "GPU validation refuses the deterministic/mock director."
            )

    # Phase: Transcription (adapter)
    transcription_real = False
    t0 = time.monotonic()
    try:
        from transcription.adapter import transcribe
        transcribe(project_dir, source)
        tf = project_dir / 'transcripts' / 'transcript.json'
        if tf.exists():
            try:
                td = json.loads(tf.read_text(encoding='utf-8'))
                transcription_real = td.get('provider') not in (None, 'none')
            except Exception:
                transcription_real = False
    except Exception as e:
        print(f"Transcription failed: {e}")
    manifest['transcription_real'] = transcription_real
    manifest['transcription_seconds'] = round(time.monotonic() - t0, 2)

    # Phase: Scene indexing (adapter: PySceneDetect preferred)
    try:
        from scene_indexing.adapter import build_scene_cards
        build_scene_cards(project_dir, source)
    except Exception as e:
        print(f"Scene indexing failed: {e}")

    # Phase: Movie intelligence (transcripts + scenes -> movie_index.json).
    # Must run BEFORE the director when the grounded director is enabled: the
    # director reasons over the enriched movie intelligence (Qwen3-VL story
    # cards when VISION_ENRICHER=qwen3vl), not raw shots.
    grounded_director = os.getenv('GROUNDED_DIRECTOR', 'false').lower() == 'true'
    if grounded_director:
        vt0 = time.monotonic()
        manifest['vision_enricher'] = _build_movie_index_if_needed(project_dir, strict)
        manifest['vision_seconds'] = round(time.monotonic() - vt0, 2)

    # Phase: Director planning (grounded | creative Qwen | deterministic fallback)
    # GROUNDED_DIRECTOR=true routes through the movie-grounded Creative Director,
    # making its selected concept + evidence the source of truth for the script
    # and editorial stages below.
    plan_path = None
    use_creative = ((strict or os.getenv('CREATIVE_DIRECTOR_ENABLED', 'false').lower() == 'true')
                    and not grounded_director)
    director_provider = 'deterministic'
    director_model = None
    director_device = None
    director_real = False
    t0 = time.monotonic()
    try:
        scenes_file = project_dir / 'scenes' / 'scene_index.json'
        transcript_file = project_dir / 'transcripts' / 'transcript.json'

        scene_index = []
        transcript = {"segments": []}

        if scenes_file.exists():
            with scenes_file.open('r', encoding='utf-8') as f:
                scene_index = json.load(f)
                if isinstance(scene_index, dict) and "scenes" in scene_index:
                    scene_index = scene_index["scenes"]

        if transcript_file.exists():
            with transcript_file.open('r', encoding='utf-8') as f:
                transcript = json.load(f)

        movie_metadata = {"title": "Untitled", "duration_sec": 0, "source": source}
        if meta_path.exists():
            with meta_path.open('r', encoding='utf-8') as f:
                meta = json.load(f)
                movie_metadata["title"] = meta.get("title", "Untitled")

        if grounded_director:
            api = _run_grounded_director(project_dir, meta_path, movie_metadata,
                                         strict, target_sec=float(os.getenv('EDITORIAL_TARGET_SEC', '90')))
            plan_path = api.get('plan_path')
            director_provider = api.get('director_provider', director_provider)
            director_model = api.get('director_model')
            director_device = api.get('director_device')
            director_real = api.get('director_real', False)
            if api.get('error'):
                if strict:
                    raise RuntimeError(f"Grounded director failed in strict mode: {api['error']}")
                print(f"Grounded director failed: {api['error']}")
        elif use_creative and scene_index and transcript:
            from director.creative_director import CreativeDirector
            from director.provider_factory import get_director_config_from_env, get_llm_provider_from_config

            print("Using creative director (LLM-backed)...")
            director_config = get_director_config_from_env()
            provider = get_llm_provider_from_config(director_config)

            if strict:
                provider = require_real_provider(provider, 'Director')

            if provider:
                director_provider = director_config.get('provider', 'qwen')
                director_model = director_config.get('model')
                director_device = director_config.get('device')

                memory_dir = project_dir / 'memory'
                director = CreativeDirector(provider=provider, memory_dir=memory_dir)
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
                    "director_provider": director_provider,
                    "director_model": director_model,
                    "director_device": provider.device_resolved or director_device,
                }
                plan_path = project_dir / "director_plan.json"
                with plan_path.open('w', encoding='utf-8') as f:
                    json.dump(director_plan, f, ensure_ascii=False, indent=2)

                director_real = True
                print(f"Creative director thesis: {director_plan['thesis'][:80]}...")

                manifest['director_qwen_load_time_sec'] = getattr(provider, 'model_load_time_sec', None)
                manifest['director_qwen_generation_times'] = getattr(provider, 'generation_times', [])
            else:
                if strict:
                    raise RuntimeError("Strict mode: Qwen director provider could not be created.")
                print("Provider unavailable for creative director; using deterministic planner.")

        if not director_real:
            if strict:
                raise RuntimeError(
                    "Strict GPU validation requires the creative Qwen director, but it "
                    "did not execute. Refusing deterministic fallback."
                )
            from director.planner import plan_director
            plan_path = plan_director(project_dir)
            print("Using deterministic director (fallback)")
    except Exception as e:
        if strict:
            raise RuntimeError(f"Director planning failed in strict mode: {e}") from e
        print(f"Director planning failed: {e}")
        plan_path = None

    manifest['director_provider'] = director_provider if director_real else ('deterministic' if not use_creative else director_provider)
    manifest['director_model'] = director_model
    manifest['director_device'] = director_device
    manifest['director_real_generation'] = director_real
    manifest['director_seconds'] = round(time.monotonic() - t0, 2)

    # Editorial mode: the AI "editor" turns the thesis into an evidence-driven
    # cut (editorial_plan.json -> editorial script -> editorial timeline ->
    # excerpt extraction -> editorial render). Enabled with EDITORIAL_MODE=true.
    editorial_mode = os.getenv('EDITORIAL_MODE', 'false').lower() == 'true'
    editorial_plan_built = False
    editorial_timeline_built = False

    # Free GPU memory between heavy stages. The Qwen model stays cached (class-level
    # cache), so the script stage can reuse it when the model name matches and we
    # don't load two model copies into a 16GB T4.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Phase: EDITORIAL planning + evidence-aligned script + timeline.
    # Requires the movie index (transcripts + scenes already built above).
    if editorial_mode:
        try:
            manifest['vision_enricher'] = _build_movie_index_if_needed(project_dir, strict)
            print("Editorial: movie_index.json + semantic_index.json built")
        except Exception as e:
            raise RuntimeError(f"Editorial movie analysis failed: {e}") from e

        try:
            sed_t0 = time.monotonic()
            from editorial.director import create_editorial_plan
            from editorial.grounded import GroundedEditorialPlanner
            from editorial.script import build_editorial_script
            from editorial.timeline import EditorialTimelineBuilder
            from movie_understanding import movie_memory

            # Grounded path: the editor consumes the grounded script's sections
            # + real excerpt windows (no fresh retrieval). Otherwise (default)
            # use the heuristic planner as before.
            grounded_plan = movie_memory.load_json(project_dir, "director_plan.json", {})
            use_grounded_editorial = bool(grounded_plan.get("grounded"))
            if use_grounded_editorial:
                from script.grounded import load_grounded_script
                grounded_script = load_grounded_script(project_dir)
                if grounded_script and grounded_script.get("sections"):
                    planner = GroundedEditorialPlanner(script=grounded_script)
                    plan = planner.create_plan(
                        movie_index=movie_memory.load_movie_index(project_dir),
                        director_plan=grounded_plan,
                        creative_task=os.getenv('EDITORIAL_CREATIVE_TASK', ''),
                        target_sec=float(os.getenv('EDITORIAL_TARGET_SEC', '90')),
                    )
                    movie_memory.save_json(project_dir, "editorial_plan.json", plan.to_dict())
                else:
                    plan = create_editorial_plan(
                        project_dir,
                        creative_task=os.getenv('EDITORIAL_CREATIVE_TASK', ''),
                        target_sec=float(os.getenv('EDITORIAL_TARGET_SEC', '90')))
            else:
                plan = create_editorial_plan(
                    project_dir,
                    creative_task=os.getenv('EDITORIAL_CREATIVE_TASK', ''),
                    target_sec=float(os.getenv('EDITORIAL_TARGET_SEC', '90')))

            movie_index = movie_memory.load_movie_index(project_dir)
            script = build_editorial_script(project_dir, plan, movie_index)
            editorial_plan_built = True
            print("Editorial: editorial_plan.json + script.json (evidence-aligned) written")

            meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
            src = meta.get('source_path') or None
            builder = EditorialTimelineBuilder(source_path=src)
            timeline = builder.build(project_dir, plan, script)
            editorial_timeline_built = True
            editorial_timeline_built = True
            n_video = sum(
                len(seg.get('video', []))
                for seg in timeline.get('segments', []))
            print(f"Editorial: timeline wrote {n_video} excerpt clip(s)")
            manifest['script_seconds'] = round(time.monotonic() - sed_t0, 2)
            manifest['script_provider'] = 'editorial'
            manifest['script_real_generation'] = bool(director_real)
            manifest['script_model'] = director_model
        except Exception as e:
            if strict:
                raise RuntimeError(f"Editorial planning failed in strict mode: {e}") from e
            print(f"Editorial planning failed: {e}")
    manifest['editorial_mode'] = editorial_mode
    manifest['editorial_plan_built'] = editorial_plan_built
    manifest['editorial_timeline_built'] = editorial_timeline_built

    # Phase: Scene ranking (use director thesis if available). Runs BEFORE script
    # generation because both the Qwen and deterministic writers read the
    # selected-scene evidence produced here.
    try:
        from scene_selection.ranker import rank_scenes
        thesis = None
        if plan_path and plan_path.exists():
            try:
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

    # Phase: Scene selection (multi-scene, evidence-driven)
    sel_path = None
    try:
        from scene_selection.selector import select_scenes
        top_n = 3
        if plan_path and plan_path.exists():
            try:
                p = json.loads(plan_path.read_text(encoding='utf-8'))
                requested = p.get('scenes_to_extract')
                if isinstance(requested, list) and requested:
                    top_n = len(requested)
                elif isinstance(requested, int) and requested > 0:
                    top_n = requested
            except Exception:
                top_n = 3
        entries = select_scenes(project_dir, top_n=top_n)
        sel_path = project_dir / 'scenes' / 'selected_scenes.json'
        print(f"Selected {len(entries)} scene(s) -> {sel_path}")
    except Exception as e:
        print(f"Scene selection failed: {e}")
        raise

    # Phase: Script generation (Qwen when strict or configured, else deterministic).
    # In editorial mode the evidence-aligned script was already written by the
    # editorial stage above -> keep it and record provenance.
    if editorial_plan_built:
        script_real = bool(director_real)
        script_provider = 'editorial'
        script_model = director_model
        script_device = director_device
        print("Editorial: script.json from editorial planner (thesis from creative director)")
    else:
        _run_script_stage(project_dir, strict, director_real, director_model, manifest)

    # Phase: Clip extraction (one per selected scene). The editorial timeline
    # extracts its own excerpt clips, so this is only needed for the plain cut.
    if not editorial_timeline_built:
        try:
            from editor.clip_extractor import extract_clip
            selections = json.loads(sel_path.read_text(encoding='utf-8'))
            meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
            source = meta.get('source_path')
            if not source:
                raise RuntimeError('No source video registered for clip extraction')
            out_dir = project_dir / 'assets' / 'scenes'
            out_dir.mkdir(parents=True, exist_ok=True)
            extracted = []
            for sel in selections:
                out_file = out_dir / f"{sel.get('scene_id')}.mp4"
                extract_clip(source, sel.get('start_sec'), sel.get('end_sec'), str(out_file))
                extracted.append(str(out_file))
                print(f"Extracted scene clip -> {out_file}")
            if not extracted:
                raise RuntimeError('No scene clips were extracted')
        except Exception as e:
            print(f"Clip extraction failed: {e}")
            raise

    # Phase: Visual generation
    try:
        from visual_generation.comfyui_client import generate_assets
        generate_assets(project_dir)
    except Exception as e:
        print(f"Visual generation failed: {e}")

    # Phase: Audio (TTS) — real provider or mock; strict REQUIRES real TTS.
    tts_real = False
    t0 = time.monotonic()
    try:
        from audio.tts_adapter import synthesize_voice, _tts_config
        from generation.provider_factory import get_tts_provider
        from utils.strict import require_real_tts, tts_strict_mode_enabled

        tts_config = _tts_config()
        tts_provider_obj = get_tts_provider(tts_config)
        if tts_strict_mode_enabled():
            tts_provider_obj = require_real_tts(tts_provider_obj, 'TTS')
        synthesize_voice(project_dir)
        pname = getattr(tts_provider_obj, 'name', 'unknown')
        tts_real = bool(pname not in ('mock', 'unknown')) and not bool(getattr(tts_provider_obj, 'mock', False))
        if tts_strict_mode_enabled() and not tts_real:
            raise RuntimeError(
                "REQUIRE_REAL_TTS=true but narration was not produced by a real "
                "TTS model. Refusing mock audio in the real-movie run."
            )
        manifest['tts_provider'] = getattr(tts_provider_obj, 'name', 'unknown')
        manifest['tts_model'] = getattr(tts_provider_obj, 'model', None)
        manifest['tts_device'] = getattr(tts_provider_obj, 'device', None)
        manifest['tts_real'] = tts_real
        manifest['tts_seconds'] = round(time.monotonic() - t0, 2)
    except Exception as e:
        if os.getenv('REQUIRE_REAL_TTS', 'false').lower() == 'true':
            raise RuntimeError(f"TTS failed in strict real-TTS mode: {e}") from e
        print(f"TTS failed: {e}")
        manifest['tts_error'] = str(e)

    # Phase: TTS benchmark (optional; records provider/device/time/duration/sr/status)
    try:
        if os.getenv('RUN_TTS_BENCHMARK', 'false').lower() == 'true':
            from generation.tts_benchmark import benchmark_tts
            from utils.io import read_json
            script_data = read_json(project_dir / 'script.json') or {}
            bench = benchmark_tts(
                text=script_data.get('voiceover_text', ''),
                output_dir=project_dir / 'reports',
                narration=script_data.get('narration_properties'),
            )
            manifest['tts_benchmark'] = bench.get('results', [])
    except Exception as e:
        print(f"TTS benchmark failed: {e}")

    # Phase: Assembly
    try:
        if editorial_timeline_built:
            from editorial.render import assemble_editorial
            assemble_editorial(project_dir)
            print("Editorial: assembled renders/final_render.mp4 via editorial renderer")
        else:
            from editor.ffmpeg_editor import assemble
            assemble(project_dir)
    except Exception as e:
        print(f"Assembly failed: {e}")
        if strict:
            raise RuntimeError(f"Assembly failed in strict mode: {e}") from e

    # Phase: QC
    try:
        from qc.critic import run_qc
        report = run_qc(project_dir)
        print(f"QC checks: {report.get('checks')}")
    except Exception as e:
        print(f"QC failed: {e}")

    manifest['pipeline_total_seconds'] = round(time.monotonic() - pipeline_t0, 2)
    manifest_path = project_dir / 'provider_manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f"Provider manifest written -> {manifest_path}")
    print("Pipeline completed. Check data/<project_id>/renders and /reports.")


def _director_config():
    from director.provider_factory import get_director_config_from_env
    return get_director_config_from_env()


def _build_movie_index_if_needed(project_dir, strict: bool) -> str:
    """Build movie_index.json + semantic_index.json only when missing or stale.

    The Grounded director reasons over the enriched movie intelligence, so this
    runs BEFORE the director phase (not inside the editorial block). Idempotent:
    a fresh Colab run never rebuilds a movie index that already matches the
    requested VISION_ENRICHER. Returns the scene enricher name actually used.
    """
    from movie_understanding.analyzer import MovieAnalyzer
    from movie_understanding import movie_memory

    vision_requested = os.getenv('VISION_ENRICHER', 'heuristic').lower() == 'qwen3vl'
    want = 'qwen3vl' if vision_requested else 'heuristic'
    existing = movie_memory.load_movie_index(project_dir)
    if existing is not None:
        if (existing.get("provenance") or {}).get("scene_enricher") == want:
            print(f"Movie intelligence already current (enricher={want}); skipping build")
            return want
    try:
        MovieAnalyzer(
            attach_keyframes=vision_requested,
            max_frames=int(os.getenv('VISION_MAX_FRAMES', '1')),
        ).analyze(project_dir)
    except Exception as e:
        if strict:
            raise RuntimeError(f"Movie intelligence build failed in strict mode: {e}") from e
        print(f"Movie intelligence build failed: {e}")
    return want


def _run_script_stage(project_dir: Path, strict: bool, director_real: bool,
                      director_model: str, manifest: dict) -> None:
    """Run the legacy script generation stage (non-editorial path)."""
    import time as _time

    script_provider = 'deterministic'
    script_model = None
    script_device = None
    script_real = False
    t0 = _time.monotonic()
    try:
        sp = os.getenv('SCRIPT_PROVIDER', 'mock').lower()
        use_qwen_script = strict or sp == 'qwen'
        if use_qwen_script:
            from script.qwen_writer import generate_script_qwen
            model = os.getenv('SCRIPT_MODEL') or 'Qwen/Qwen3-4B-Instruct-2507'
            device = os.getenv('SCRIPT_DEVICE', 'auto')
            if strict and device in ('auto', 'cpu'):
                require_cuda()
                device = 'cuda'
            # If the script wants a different model than the one cached from the
            # director stage, drop the cache first to avoid holding both in VRAM.
            if director_real and model != director_model:
                try:
                    from director.providers.qwen import QwenProvider
                    QwenProvider.release_model()
                    print(f"[SCRIPT] Dropping cached director model; loading {model}")
                except Exception as e:
                    print(f"[SCRIPT] Could not release director model: {e}")
            result = generate_script_qwen(project_dir, model=model, device=device)
            script_provider = 'qwen'
            script_model = result.get('script_model', model)
            script_device = result.get('script_device', device)
            script_real = True
            manifest['script_qwen_load_time_sec'] = result.get('qwen_load_time_sec')
            manifest['script_qwen_generation_time_sec'] = result.get('qwen_generation_time_sec')
        else:
            from script.writer import generate_script
            generate_script(project_dir)
            script_provider = 'deterministic'
        if strict and not script_real:
            raise RuntimeError(
                "Strict GPU validation requires a Qwen-generated script, but the "
                "script stage did not use Qwen. Refusing deterministic script fallback."
            )
    except Exception as e:
        if strict:
            raise RuntimeError(f"Script generation failed in strict mode: {e}") from e
        print(f"Script generation failed: {e}")

    manifest['script_provider'] = script_provider
    manifest['script_model'] = script_model
    manifest['script_device'] = script_device
    manifest['script_real_generation'] = script_real
    manifest['script_seconds'] = round(_time.monotonic() - t0, 2)


def _run_grounded_director(project_dir, meta_path, movie_metadata, strict,
                           target_sec: float = 90.0) -> dict:
    """Run the movie-grounded Creative Director + grounded script + report.

    Writes ``director_plan.json`` (grounding contract keys), ``grounded_script.json``
    and ``reports/script_grounding_report.md``. Returns a status dict for the
    orchestrator's manifest.
    """
    import time as _time
    from director.provider_factory import get_director_config_from_env, get_llm_provider_from_config
    from director.grounded import MovieGroundedDirector
    from director.scene_facts import SceneFacts
    from director.grounding_contract import (
        build_grounding_contract,
        save_grounding_contract,
    )
    from director.providers.mock_llm import GroundedMockLLM
    from script.grounded import GroundedScriptGenerator, save_grounded_script
    from script.grounding_report import write_script_grounding_report

    status = {
        "plan_path": None,
        "director_provider": "grounded",
        "director_model": None,
        "director_device": None,
        "director_real": False,
        "error": None,
    }
    t0 = _time.monotonic()
    project_dir = Path(project_dir)

    # The grounded director reasons over the movie intelligence. Prefer the
    # enriched movie_index.json (with story cards); fall back to scene_index.
    import json as _json
    movie_index_path = project_dir / "movie_index.json"
    if movie_index_path.exists():
        movie_index = _json.loads(movie_index_path.read_text(encoding="utf-8"))
    else:
        scenes_file = project_dir / "scenes" / "scene_index.json"
        scenes = []
        if scenes_file.exists():
            raw = _json.loads(scenes_file.read_text(encoding="utf-8"))
            scenes = raw.get("scenes", raw) if isinstance(raw, dict) else raw
        if not scenes:
            status["error"] = "no movie intelligence (movie_index or scenes) to ground concepts on"
            return status
        movie_index = {"movie": {"title": movie_metadata.get("title", "Untitled")},
                       "scenes": scenes}

    facts = SceneFacts.from_movie_intelligence(movie_index=movie_index)
    if not len(facts):
        status["error"] = "movie intelligence contained no usable scenes"
        return status

    director_config = get_director_config_from_env()
    provider = get_llm_provider_from_config(director_config)
    if strict:
        from utils.strict import require_real_provider
        provider = require_real_provider(provider, 'Grounded Director')
    if provider is None:
        status["error"] = "no director provider configured for grounded director"
        return status

    llm = getattr(provider, "generate_text", None)
    if llm is None:
        llm = GroundedMockLLM(scene_index=movie_index.get("scenes", []))
    memory_dir = project_dir / "memory"
    director = MovieGroundedDirector(llm=llm, memory_dir=memory_dir)

    metadata = {
        "title": movie_index.get("movie", {}).get("title") or "Untitled",
        "duration_sec": movie_index.get("movie", {}).get("duration_sec", 0),
        "source": movie_index.get("source_path") or movie_metadata.get("source"),
    }
    try:
        result = director.develop(
            movie_metadata=metadata,
            scale_facts=facts,
            num_concepts=int(os.getenv('DIRECTOR_NUM_CONCEPTS', '5')),
            min_coverage=float(os.getenv('DIRECTOR_MIN_COVERAGE', '0.4')),
            duration_sec=int(target_sec),
        )
    except Exception as e:
        status["error"] = str(e)
        return status

    selected = result.get("selected_concept")
    if selected is None:
        status["error"] = "grounded director rejected every concept (no usable evidence)"
        return status

    # Write the director reasoning report + normalized grounding contract.
    director.write_report(project_dir, result)
    contract = build_grounding_contract(result, facts, movie_index)
    errors = _grounding_contract_errors(contract)
    if errors:
        status["error"] = "grounded contract invalid: " + "; ".join(errors)
        return status
    save_grounding_contract(project_dir, contract)

    # Build the grounded script from the contract + movie intelligence.
    try:
        script = GroundedScriptGenerator(target_sec=target_sec).generate(
            contract, movie_index, project_id=str(project_dir.name))
    except Exception as e:
        status["error"] = f"grounded script failed: {e}"
        return status
    save_grounded_script(project_dir, script)
    try:
        write_script_grounding_report(project_dir, contract, script)
    except Exception as e:
        print(f"[GROUNDED] script grounding report failed: {e}")

    plan = result.get("plan") or {}
    director_plan = {
        "thesis": contract["concept"].get("thesis", ""),
        "hook": contract["concept"].get("hook", ""),
        "title": contract["concept"].get("title", metadata["title"]),
        "tone": contract.get("editorial_intent", {}).get("tone", "analytical"),
        "structure": [
            {"id": s.get("type"), "goal": s.get("type"),
             "target_seconds": int(s.get("target_seconds", 10))}
            for s in script.get("sections", [])
        ],
        "scenes_to_extract": [s["scene_id"] for s in contract.get("supporting_scenes", [])],
        "creative_generation": True,
        "grounded": True,
        "concept": contract["concept"],
        "production_plan": plan,
        "evidence_strategy": plan.get("evidence_strategy", {}),
        "editorial_intent": contract.get("editorial_intent", {}),
        "supporting_scenes": contract.get("supporting_scenes", []),
        "all_concepts": result.get("generated_concepts", []),
        "director_provider": director_config.get("provider", "grounded"),
        "director_model": director_config.get("model"),
        "director_device": getattr(provider, "device_resolved", None) or director_config.get("device"),
    }
    plan_path = project_dir / "director_plan.json"
    plan_path.write_text(
        _json.dumps(director_plan, ensure_ascii=False, indent=2), encoding="utf-8")

    status.update({
        "plan_path": plan_path,
        "director_provider": director_config.get("provider", "grounded"),
        "director_model": director_config.get("model"),
        "director_device": getattr(provider, "device_resolved", None) or director_config.get("device"),
        "director_real": llm is not None and not isinstance(llm, GroundedMockLLM),
        "grounded_time_sec": round(_time.monotonic() - t0, 2),
    })
    return status


def _grounding_contract_errors(contract: dict) -> list:
    from director.grounding_contract import contract_is_valid
    return contract_is_valid(contract)