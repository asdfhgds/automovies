Autonomous Movie & TV Video Studio

This repository contains an MVP scaffold for an autonomous movie/TV commentary studio. The system is a pipeline of specialized modules: director, transcription, scene indexing, scene selection, script, visual generation (ComfyUI), TTS, editor (FFmpeg), and QC.

See AUTONOMOUS_MOVIE_STUDIO_SPEC.md for the full specification.

## Local MVP workflow

The local profile runs without heavyweight AI models. It uses the repository's
deterministic transcription, director, script, and TTS fallbacks, then renders
a valid MP4 with FFmpeg:

```bash
PYTHONPATH=src python src/main.py doctor
PYTHONPATH=src python src/main.py init \
  --title "My commentary" \
  --source tests/fixtures/test_speech.mp4
PYTHONPATH=src python src/main.py run --project-id <project-id>
```

The completed project contains:

- `script.json` — generated narration sections
- `audio/voice.wav` — valid local mock-TTS audio
- `timeline/timeline.json` — persisted video, voice, and subtitle tracks
- `renders/final_render.mp4` — FFmpeg-rendered H.264/AAC video
- `renders/render_job.json` — render inputs and export metadata
- `reports/qc_report.json` — basic artifact checks

The real WhisperX, PySceneDetect, Qwen, TTS, image, and video providers remain
optional. When they are not installed, the pipeline falls back to local
implementations so the full artifact flow can still be exercised.

## Multi-scene evidence selection

The pipeline now selects up to three distinct, ranked scenes for a production
plan. It writes `scenes/selected_scenes.json` as the canonical artifact and
also keeps `scenes/selected_scene.json` pointing to the first scene for older
tools. The selected clips are extracted and placed sequentially on the local
FFmpeg timeline.
