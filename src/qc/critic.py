"""Quality control: perform basic checks on produced artifacts.

Adds audio-grade checks on top of the artifact checks:

- render exists, non-empty, and has a playable duration (ffprobe)
- the render carries an audio stream
- the narration was produced by a real TTS provider (from tts_meta.json)
- the audio does not clip (volumedetect max_volume <= 0 dBFS)
- the TTS benchmark report exists when it was requested
"""
import json
import subprocess
from pathlib import Path


def _ffprobe_json(path: Path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration,size",
            "-show_entries", "stream=index,codec_type,codec_name",
            "-of", "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except Exception:
        return None


def _max_volume_db(path: Path):
    """Return the max_volume in dBFS for the first audio stream (None on failure)."""
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-i", str(path),
            "-af", "volumedetect", "-f", "null", "-",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    for line in (result.stderr or "").splitlines():
        if "max_volume" in line:
            try:
                return float(line.split("max_volume:")[1].split(" dB")[0])
            except Exception:
                return None
    return None


def run_qc(project_dir: Path):
    project_dir = Path(project_dir)
    checks = {}
    # check director plan
    checks['director_plan'] = (project_dir / 'director_plan.json').exists()
    checks['script'] = (project_dir / 'script.json').exists()
    # Grounded/editorial runs produce PySceneDetect's scene_index.json and the
    # enriched movie_index.json; the legacy mock path wrote scene_cards.json.
    # Any real scene index is sufficient for QC.
    checks['scene_cards'] = any(p.exists() for p in (
        project_dir / 'scenes' / 'scene_cards.json',
        project_dir / 'scenes' / 'scene_index.json',
        project_dir / 'movie_index.json',
    ))
    checks['assets'] = any((project_dir / 'assets').glob('*')) if (project_dir / 'assets').exists() else False
    checks['render'] = (project_dir / 'renders' / 'final_render.mp4').exists()

    # Editorial cut: require plan + timeline + excerpts instead of the legacy
    # single-clip-per-scene artifacts.
    editorial = False
    editorial_timeline = project_dir / 'timeline' / 'editorial_timeline.json'
    editorial_plan = project_dir / 'editorial_plan.json'
    if editorial_timeline.exists() and editorial_plan.exists():
        tl = json.loads(editorial_timeline.read_text(encoding='utf-8'))
        if tl.get('mode') == 'editorial':
            editorial = True
            excerpts = [
                clip['content_path']
                for seg in tl.get('segments', [])
                for clip in seg.get('video', [])
            ]
            checks['editorial_plan'] = True
            checks['editorial_timeline'] = True
            checks['editorial_excerpts'] = (
                len(excerpts) > 0
                and all(Path(p).exists() and Path(p).stat().st_size > 0 for p in excerpts)
            )

    # multi-scene cut: require a selection file and a clip per selected scene
    selected_scenes_path = project_dir / 'scenes' / 'selected_scenes.json'
    selected_scene_path = project_dir / 'scenes' / 'selected_scene.json'
    if editorial:
        checks['selected_scenes'] = (selected_scenes_path.exists()
                                     or selected_scene_path.exists())
        checks['scene_clips'] = True
    elif selected_scenes_path.exists():
        selections = json.loads(selected_scenes_path.read_text(encoding='utf-8'))
        checks['selected_scenes'] = isinstance(selections, list) and len(selections) > 0
        checks['scene_clips'] = all(
            (project_dir / 'assets' / 'scenes' / f"{s.get('scene_id')}.mp4").exists()
            for s in selections
        ) if isinstance(selections, list) else False
    elif selected_scene_path.exists():
        checks['selected_scenes'] = True
        sel = json.loads(selected_scene_path.read_text(encoding='utf-8'))
        checks['scene_clips'] = (project_dir / 'assets' / 'scenes' / f"{sel.get('scene_id')}.mp4").exists()
    else:
        checks['selected_scenes'] = False
        checks['scene_clips'] = False

    # --- render probe (playability) ---
    render = project_dir / 'renders' / 'final_render.mp4'
    checks['render_duration_sec'] = None
    checks['render_has_video'] = False
    checks['render_has_audio'] = False
    if render.exists() and render.stat().st_size > 0:
        probe = _ffprobe_json(render)
        if probe:
            try:
                checks['render_duration_sec'] = round(float(probe.get('format', {}).get('duration', 0)), 3)
            except (TypeError, ValueError):
                checks['render_duration_sec'] = None
            streams = probe.get('streams', [])
            checks['render_has_video'] = any(s.get('codec_type') == 'video' for s in streams)
            checks['render_has_audio'] = any(s.get('codec_type') == 'audio' for s in streams)
        checks['render_non_empty'] = True
    else:
        checks['render_non_empty'] = False

    # --- audio quality: real TTS + no clipping ---
    tts_meta_path = project_dir / 'audio' / 'tts_meta.json'
    if tts_meta_path.exists():
        meta = json.loads(tts_meta_path.read_text(encoding='utf-8'))
        checks['narration_real_tts'] = bool(not meta.get('mock', True))
        checks['narration_provider'] = meta.get('voice_provider')
        checks['narration_model'] = meta.get('voice_model')
        checks['narration_device'] = meta.get('voice_device')
        checks['narration_duration_sec'] = meta.get('duration_sec')
        checks['narration_sample_rate'] = meta.get('sample_rate')
    else:
        checks['narration_real_tts'] = False
        checks['narration_provider'] = None

    if render.exists() and render.stat().st_size > 0:
        peak = _max_volume_db(render)
        checks['render_max_volume_db'] = peak
        checks['no_clipping'] = peak is not None and peak <= 0.0
    else:
        checks['render_max_volume_db'] = None
        checks['no_clipping'] = False

    checks['tts_benchmark'] = (project_dir / 'reports' / 'tts_benchmark.json').exists()

    # --- P0 render-stability checks (fail closed) ---
    # 1. TTS input contract: narration_inputs.json proves the provider received
    #    only sanitized narration (source == narration_extractor).
    from audio.narration_contract import (
        NarrationSanitizationError,
        build_tts_inputs,
    )
    tts_input_ok = True
    narration_inputs_path = project_dir / 'audio' / 'narration_inputs.json'
    if narration_inputs_path.exists():
        try:
            ni = json.loads(narration_inputs_path.read_text(encoding='utf-8'))
            contract_ok = bool(ni.get('schema') == 'tts_input_contract_v1')
            count = int(ni.get('count', 0))
            checks['tts_input_count'] = count
            tts_input_ok = contract_ok and count > 0
        except Exception:
            tts_input_ok = False
    else:
        # Reject silently leaked scripts even if the artifact is absent.
        script_file = project_dir / 'script.json'
        if script_file.exists():
            try:
                build_tts_inputs(json.loads(script_file.read_text(encoding='utf-8')))
                checks['tts_input_count'] = 0
            except (NarrationSanitizationError, ValueError):
                tts_input_ok = False
        else:
            tts_input_ok = False
    checks['tts_input'] = tts_input_ok

    # 2. Post-render black-frame + duration + stream validation.
    from render.validate import (
        RenderValidationError,
        qc_black_threshold,
        validate_render_file,
    )
    render_ok = True
    try:
        post = validate_render_file(
            render, require_audio=True,
            max_black_sec=qc_black_threshold(),
        )
        checks['render_black'] = True
        checks['render_frames'] = post.frames
        checks['black_segments_sec'] = [round(b, 3) for b in post.black_segments_sec]
        checks['black_frame_check'] = True
    except RenderValidationError:
        checks['render_black'] = False
        checks['black_frame_check'] = False
        render_ok = False

    # 3. Timeline coverage + multi-scene + script->timeline contract.
    editorial_timeline_path = project_dir / 'timeline' / 'editorial_timeline.json'
    coverage_ok = True
    if editorial_timeline_path.exists():
        try:
            from render.validate import (
                validate_multi_scene,
                validate_script_timeline_mapping,
                validate_timeline_coverage,
            )
            tl = json.loads(editorial_timeline_path.read_text(encoding='utf-8'))
            cov = validate_timeline_coverage(tl)
            checks['timeline_coverage_seconds'] = round(cov.visual_coverage, 3)
            checks['timeline_uncovered_seconds'] = round(cov.uncovered_seconds, 3)
            checks['timeline_coverage'] = True
            if (project_dir / 'script.json').exists():
                validate_script_timeline_mapping(
                    json.loads((project_dir / 'script.json').read_text(encoding='utf-8')),
                    tl,
                )
                checks['script_timeline'] = True
            try:
                validate_multi_scene(tl)
                checks['multi_scene'] = True
            except RenderValidationError:
                checks['multi_scene'] = False
                coverage_ok = False
        except RenderValidationError:
            checks['timeline_coverage'] = False
            coverage_ok = False
    else:
        checks['timeline_coverage'] = None
        checks['multi_scene'] = None
        checks['script_timeline'] = None

    report = {
        'project': str(project_dir.name),
        'checks': checks,
        'passed': all(
            v is True for k, v in checks.items()
            if k.startswith(('director', 'script', 'scene', 'assets', 'render', 'selected',
                             'narration_real_tts', 'no_clipping', 'tts_benchmark', 'editorial',
                             'tts_input', 'black_frame', 'timeline_coverage', 'multi_scene',
                             'script_timeline'))
        ),
    }
    out = project_dir / 'reports'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'qc_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"QC report written -> {out / 'qc_report.json'}")
    return report
