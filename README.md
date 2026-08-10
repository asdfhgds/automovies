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

- `script.json` — generated narration sections (+ `narration_properties`)
- `audio/voice.wav` + `audio/tts_meta.json` — narration audio and provider metadata
- `timeline/timeline.json` — persisted video, voice, and subtitle tracks
- `renders/final_render.mp4` — FFmpeg-rendered H.264/AAC video
- `renders/subtitles.srt` — burned captions (when libass is available)
- `renders/render_job.json` — render inputs, ducking/normalization metadata
- `reports/qc_report.json` — artifact checks incl. audio quality and no-clipping

## Real TTS (open-source, on-device)

Three real TTS providers live behind the common `TTSProvider` interface and are
switched via `TTS_PROVIDER` (or the `configs/profiles.yaml` `tts` block):

| Provider     | Env value     | Model                  | Sample rate | Notes                        |
|--------------|---------------|------------------------|-------------|------------------------------|
| Kokoro       | `kokoro`      | `hexgrad/Kokoro-82M`   | 24 kHz      | default; tone→voice, pace→speed |
| Chatterbox   | `chatterbox`  | `resembleai/chatterbox`| 16 kHz      | zero-shot voice cloning (`TTS_VOICE_PATH`) |
| Qwen3-TTS    | `qwen3_tts`   | `Qwen/Qwen3-TTS`       | 24 kHz      | built-in voices Chelsie/George/Koren |

Director-controlled delivery (`tone`, `emotion`, `pace`, `energy`,
`dramatic_intensity`) is recorded in `script.json` as `narration_properties` and
passed to the provider; each provider reports which subset it honors in
`tts_meta.json` / the benchmark.

### Strict production TTS

`REQUIRE_REAL_TTS=true` (with `TTS_DEVICE=cuda`) makes the real-movie run refuse
mock/pyttsx3 audio:

```bash
STUDIO_PROFILE=colab-gpu REQUIRE_REAL_LLM=true \
REQUIRE_REAL_TTS=true TTS_PROVIDER=kokoro TTS_DEVICE=cuda \
python src/main.py run --project-id <project-id>
```

### TTS benchmark

Synthesizes the same narration with every installed provider and records
provider/model/device/generation time/duration/sample rate/status:

```bash
TTS_DEVICE=cuda python src/main.py benchmark-tts --output-dir reports
# or inside the pipeline run:
RUN_TTS_BENCHMARK=true python src/main.py run --project-id <project-id>
```

Real TTS is intentionally disabled on CPU (`status: cpu_skipped`) so local runs
stay fast — synthesize on GPU only.

## Real-movie run (GPU / Colab)

The movie file is **never committed**; only its path is stored in
`project_meta.json` (the `data/` dir is git-ignored). Use the notebook:

- `notebooks/colab_real_movie_tts.ipynb` — mounts Google Drive, installs deps,
  runs the real Qwen + real TTS pipeline on a T4/A100, benchmarks TTS, and
  validates the MP4 with QC + ffprobe.

The real WhisperX, PySceneDetect, Qwen, TTS, image, and video providers remain
optional. When they are not installed, the pipeline falls back to local
implementations so the full artifact flow can still be exercised.
