PROJECT STATUS — Autonomous Movie Studio

Current state (after scaffold and MVP enhancements):

- CLI: implemented (src/main.py). Supports `init` and `run`.
  - `init` now accepts `--title` and `--source` (absolute path stored in project_meta.json).
- Orchestrator: implemented and runs pipeline stages sequentially using local stubs (src/app/orchestrator.py).
- Modules with stubs implemented:
  - transcription (src/transcription/whisper_stub.py) — whisperx_adapter implemented and unit-tested
  - scene_indexing (src/scene_indexing/scene_detector.py)
  - director (src/director/planner.py) — deterministic planner implemented to produce thesis from scene_index.json
  - script (src/script/writer.py)
  - visual_generation (src/visual_generation/comfyui_client.py)
  - audio (src/audio/tts_adapter.py)
  - editor (src/editor/ffmpeg_editor.py)
  - qc (src/qc/critic.py)
- JSON schemas added (configs/schemas).
- AUTONOMOUS_MOVIE_STUDIO_SPEC.md added.
- Smoke test executed: created project a607061d-5224-42fd-a9cb-0aa74ab54a9b and pipeline produced artifacts under data/<project_id>/.

What works (real):
- Project initialization and metadata storage (including source path).
- Pipeline orchestration and stub stages producing placeholder artifacts.

What remains a stub / next tasks (priority order):
1. Real Movie Input: ensure pipeline reads and uses local source video (CLI registers path). -- IMPLEMENTED (metadata stored) — next: use source in stages that require it.
2. Real Transcription: replace whisper_stub with WhisperX adapter (produce word-level timestamps).
3. Real Scene Detection: integrate PySceneDetect to produce scene_index.json.
4. Scene Ranking: implement lexical/keyword-based ranking; deterministic and testable.
5. Director: replace planner stub with model-backed structured planner that validates against director_plan.schema.json.
6. Script generation: generate interpretive narration preserving thesis and timing estimates.
7. TTS: add adapters for Qwen3 and Chatterbox; make configurable.
8. Scene Clip Extraction: FFmpeg-based safe extractor producing data/<project>/assets/scenes/*.mp4.
9. First Real Render: assemble real MP4 using extracted clips, TTS audio, subtitles, basic cuts.

Tests to add:
- project init
- transcript parsing
- scene detection
- scene-card creation
- scene ranking
- director validation
- script validation
- TTS adapter selection
- clip extraction
- pipeline orchestration

Commands used during development:
- python src/main.py init --title "Smoke Test"
- python src/main.py run --project-id <id>

Known problems / considerations:
- Heavy models not integrated; imports must be lazy.
- Sample video fixture not yet added; add a small public-domain clip for tests.
- Windows vs POSIX path handling: pathlib used, but ensure cross-platform tests.

Next immediate step (this session):
- Use registered source_path if present for scene extraction stage (when implemented). Update orchestrator to skip scene extraction if source missing. Begin implementing WhisperX adapter in src/transcription/whisperx_adapter.py as a next PR.

Updated after scene ranking integration and clip extraction implementation.

What was implemented in this session:

1. Scene Ranking Integration
   - The scene ranker (src/scene_selection/ranker.py) is integrated into the orchestrator (src/app/orchestrator.py).
   - The orchestrator reads the director's produced director_plan.json and passes the thesis to the ranker.
   - The ranker writes data/<project_id>/scenes/scene_ranking.json with ranking entries.

2. Scene Selection
   - Implemented selector (src/scene_selection/selector.py) that reads scene_ranking.json and the scene index (scene_index.json or scene_cards.json).
   - Selection rules: pick highest-scoring scene with valid start/end, positive duration.
   - Writes data/<project_id>/scenes/selected_scene.json.

3. FFmpeg Clip Extraction
   - Added src/editor/clip_extractor.py with extract_clip(source_path, start_sec, end_sec, output_path).
   - Validates inputs, checks ffmpeg/ffprobe availability, creates output directory, runs ffmpeg re-encoding with accurate cut, captures errors.
   - Added probe_duration(path) to measure duration via ffprobe.

4. Orchestrator Integration
   - Orchestrator now runs ranking, selection, and extraction (writes assets/scenes/<scene_id>.mp4).
   - Orchestrator raises clear errors when selection or extraction fails.

5. Tests
   - tests/test_scene_indexing.py (existing) passed.
   - tests/test_scene_ranking.py passed.
   - Added tests/test_selector_and_extractor.py which creates a tiny test video using ffmpeg, runs ranking/selection/extraction, and verifies the extracted clip duration.
   - Ran full test suite: 3 passed, 1 skipped.

Files changed/added (high level):
- Modified: src/app/orchestrator.py (integrated ranking, selection, extraction)
- Added: src/scene_selection/selector.py
- Added: src/editor/clip_extractor.py
- Added tests: tests/test_selector_and_extractor.py
- Updated PROJECT_STATUS.md

Remaining limitations / notes:
- Transcription uses a placeholder/whisper fallback; WhisperX integration remains to be implemented.
- PySceneDetect adapter defers to stub when scenedetect is not installed; full PySceneDetect integration requires the package and may be slow.
- The director is still a simple planner stub; replacing with a model-backed director is next priority after reliable pipeline.
- Visual generation and advanced assets remain stubs.
- FFmpeg must be available on PATH for extraction tests and real extraction.

Next recommended task:
- Implement WhisperX adapter (replace stub) and ensure word-level timestamps or fall back to OpenAI Whisper for CPU.
- Then integrate PySceneDetect for robust scene detection when available.

Commands used for verification:
- pytest -q (with PYTHONPATH=src)
- python src/main.py init --title "Smoke Test"
- python src/main.py run --project-id <id>

Status: This task (Integrate scene ranking and implement clip extraction) is complete and tested.

Additional updates:

- Added optional WhisperX integration test at tests/test_whisperx_integration.py (marker: integration). The test is optional and runs only when explicitly requested (pytest -m integration).
- Added an end-to-end GPU integration test at tests/integration/test_e2e_integration.py (marker: integration). This runs the full pipeline from WhisperX through PySceneDetect, selection and FFmpeg extraction; it requires a small test video fixture at tests/fixtures/test_speech.mp4 and an environment with whisperx, pyscenedetect, and ffmpeg available.
- Committed the integration tests locally; attempted git push but remote push failed due to permission (HTTP 403). Local commits exist; see GIT_PUSH_INSTRUCTIONS.md for pushing from a machine with credentials.
- Unit test suite (fast tests): 6 passed, 1 skipped in this environment.
- Added a doctor command: python src/main.py doctor — reports Python version, torch/CUDA/GPU info, ffmpeg/ffprobe, whisperx, pyscenedetect, and nvidia-smi availability.
- Added GPU validation helper script: scripts/gpu_validate.sh and Colab instructions in COLAB_INSTRUCTIONS.md to reproduce the GPU validation environment and run the pipeline automatically in Colab or a VM.
- WhisperX adapter implementation exists and is unit-tested, but real-model GPU integration is pending and requires an environment with whisperx/models and a CUDA-enabled GPU.
- A CPU path is available: the repository includes a Whisper (openai/whisper) fallback which can run on CPU for small test videos. Running the full GPU validation is still required to satisfy the task definition.
