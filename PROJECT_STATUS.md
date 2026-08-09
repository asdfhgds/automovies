# PROJECT STATUS — Autonomous Movie Studio

**Last Updated**: After Multi-Scene Selection Session

## Executive Summary

The Autonomous Movie Studio project now has a **complete, modular architecture** supporting:

- **Multi-profile execution** (local laptop development vs GPU-accelerated Colab)
- **Provider-based adapter pattern** for all generation capabilities
- **Mock implementations** for local testing without downloading models
- **Configuration-driven provider selection** (no code changes needed)
- **Quality control and validation** systems
- **Real WhisperX, PySceneDetect, and Qwen LLM** integration
- **40+ passing tests** with zero regressions

## Current Architecture

### Core Pipeline

```
Video File
  ↓
WhisperX Transcription (real or mock)
  ↓
PySceneDetect Scene Detection (real or mock)
  ↓
Scene Indexing with transcript association
  ↓
Creative Director (Qwen LLM or deterministic)
  ↓
Thesis-based Scene Ranking (deterministic)
  ↓
Scene Selection (multi-scene, non-overlapping top-K)
  ↓
FFmpeg Clip Extraction (real)
  ↓
Script Generation (mock)
  ↓
TTS Synthesis (mock)
  ↓
Timeline Assembly
  ↓
QC Validation
  ↓
Final Render
```

### Project Structure

```
src/
├── app/
│   ├── config.py
│   ├── project.py
│   └── orchestrator.py
├── director/
│   ├── creative_director.py
│   ├── providers/
│   │   ├── base.py
│   │   ├── mock.py
│   │   └── qwen.py
│   ├── prompts/
│   │   ├── base.py
│   │   ├── concept_generation.py
│   │   ├── concept_critique.py
│   │   ├── production_plan.py
│   │   ├── context_builder.py
│   │   └── json_utils.py
│   └── provider_factory.py
├── understanding/
│   ├── transcription/
│   │   ├── adapter.py
│   │   └── whisperx_adapter.py
│   └── scenes/
│       ├── detector.py
│       └── indexer.py
├── generation/
│   ├── base.py (Provider interfaces)
│   ├── mock.py (Mock implementations)
│   ├── provider_factory.py
│   └── __init__.py
├── editing/
│   ├── clip_extractor.py
│   ├── timeline.py (Timeline data structures)
│   └── __init__.py
├── quality/
│   ├── validator.py (QC validator)
│   └── __init__.py
├── scene_selection/
│   ├── ranker.py
│   └── selector.py
├── script/
│   └── writer.py
├── audio/
│   └── tts_adapter.py
├── visual_generation/
│   └── comfyui_client.py
├── qc/
│   └── critic.py
└── utils/
    ├── doctor.py (Enhanced)
    └── io.py

configs/
├── app.yaml
├── profiles.yaml (NEW: Multi-profile configuration)
└── schemas/
    └── *.json
```

## What's Implemented

### ✅ Core Infrastructure

- **CLI** (`src/main.py`): `init`, `run`, `doctor` commands
- **Project management**: Project initialization, metadata storage, directory structure
- **Orchestrator**: Pipeline stage execution with clear error handling
- **Environment doctor**: Enhanced with profile detection, provider availability, GPU info

### ✅ Real Implementations (Working with Tests)

- **WhisperX Transcription** (src/transcription/whisperx_adapter.py)
  - Real speech-to-text with word-level timestamps
  - Lazy model loading
  - GPU/CPU auto-detection
  - Unit tests passing
  
- **PySceneDetect Integration** (src/scene_indexing/adapter.py)
  - Real scene/shot boundary detection
  - Transcript-to-scene association
  - Scene card generation
  - Unit tests passing

- **Creative Director Framework** (src/director/)
  - Qwen LLM provider with lazy loading
  - Structured prompts (concept generation, critique, production planning)
  - Robust JSON parsing
  - Context limiting for long videos
  - MockLLMProvider for local development
  - Provider factory for dynamic selection
  - 26 unit tests + 8 integration tests passing

- **Scene Ranking** (src/scene_selection/ranker.py)
  - Deterministic lexical/keyword scoring
  - Thesis-based ranking
  - Unit tests passing

- **Scene Selection** (src/scene_selection/selector.py)
  - Multi-scene selection (`select_scenes` → `selected_scenes.json`)
  - Non-overlapping, minimum-duration, timestamp-validated picks
  - Backward-compatible single scene (`selected_scene.json`)
  - Writes selection order for the editing stage

- **FFmpeg Clip Extraction** (src/editor/clip_extractor.py)
  - Real video clip extraction
  - Re-encoding for accurate frame boundaries
  - Input validation
  - Error handling

### ✅ Architecture & Interfaces

- **Provider Interfaces** (src/generation/base.py)
  - ScriptProvider: Narration/script generation
  - TTSProvider: Text-to-speech
  - ImageProvider: Image generation
  - VideoProvider: Video generation

- **Mock Providers** (src/generation/mock.py)
  - MockScriptProvider: Deterministic placeholder scripts
  - MockTTSProvider: Silent WAV file generation
  - MockImageProvider: PNG placeholder generation
  - MockVideoProvider: MP4 placeholder generation

- **Provider Factory** (src/generation/provider_factory.py)
  - Dynamic provider loading based on configuration
  - Environment variable overrides
  - Graceful fallback to mocks

- **Timeline System** (src/editing/timeline.py)
  - Timeline, TimelineTrack, TimelineItem classes
  - Support for voice, video, music, SFX, text tracks
  - Timeline validation
  - TimelineBuilder for easy construction

- **Quality Control** (src/quality/validator.py)
  - QCValidator class for comprehensive output validation
  - File existence checking
  - JSON schema validation
  - Transcript validation
  - Scene index validation
  - Video/audio file verification
  - Report generation

### ✅ Configuration System

- **Multi-profile support** (configs/profiles.yaml)
  - `local`: Laptop development (all mocks, FFmpeg only)
  - `colab-gpu`: GPU-accelerated (real models where available)
  - Environment-based profile selection
  - Provider configuration cascading

- **Environment variables** for all provider overrides:
  - STUDIO_PROFILE (local|colab-gpu)
  - DIRECTOR_PROVIDER (mock|qwen)
  - DIRECTOR_MODEL
  - TTS_PROVIDER
  - etc.

### ✅ Testing

- **40+ tests passing** with zero regressions
  - 26 Qwen provider unit tests
  - 10 Creative director tests
  - 4 Scene ranking tests
  - Integration tests marked with `@pytest.mark.llm_integration`
  - Fast local test suite (<30 seconds)
  - Optional GPU integration tests

- **Pytest configuration** (pyproject.toml)
  - PYTHONPATH set to `src`
  - Test markers: llm_integration, slow, integration
  - Proper test discovery

### ✅ Enhanced Doctor Command

```bash
python src/main.py doctor
```

Outputs:
- Platform, Python version
- FFmpeg, ffprobe availability
- GPU/CUDA status with VRAM
- PyTorch version and device info
- Installed models (WhisperX, Whisper, PySceneDetect)
- **Active profile** detection
- **Provider availability** for all capabilities
- Recommendations for profile switching
- JSON summary for automation

## What's Mocked (For Local Development)

- **Script generation** (MockScriptProvider)
- **TTS** (MockTTSProvider - silent WAV files)
- **Image generation** (MockImageProvider - PNG placeholders)
- **Video generation** (MockVideoProvider - MP4 placeholders)

These mocks allow:
- Full pipeline execution on weak laptops
- Fast testing without GPU
- Architecture validation
- Data flow verification

## Profiles Explained

### Local Profile (`STUDIO_PROFILE=local`)
```yaml
providers:
  llm: mock
  transcription: mock
  script: mock
  tts: mock
  image: mock
  video: mock
```

- Runs completely on CPU
- All mocks - no real model downloads
- Fast iteration for development
- Tests complete in <30 seconds
- Validates architecture and orchestration

### Colab-GPU Profile (`STUDIO_PROFILE=colab-gpu`)
```yaml
providers:
  llm: qwen
  transcription: whisperx
  script: mock
  tts: mock
  image: mock
  video: mock
```

- Real WhisperX, Qwen LLM on GPU
- Mock generation (to be replaced later)
- Requires CUDA/GPU
- For actual AI execution and validation

## Key Design Principles

1. **Adapter Pattern**: All heavyweight capabilities use provider interfaces
2. **Lazy Loading**: Models don't load until first use
3. **Configuration-Driven**: No code changes to switch providers
4. **Graceful Fallback**: Mocks always available
5. **Separation of Concerns**: Provider logic decoupled from business logic
6. **Test Isolation**: Unit tests use mocks, integration tests optional
7. **Environment-Aware**: Auto-detects GPU and recommends profile

## Known Limitations

1. **No quantization support** (int8/int4) - limits which GPUs work with which models
2. **Keyword-based ranking** - scene ranking is deterministic lexical scoring; typed evidence (visual/emotional/dialogue) from the director is not yet used in selection
3. **Deterministic director** - only fallback when Qwen unavailable
4. **Mock generation providers** - real TTS/image/video integration deferred
5. **No human feedback loop** - one-shot generation only
6. **No cost tracking** - no visibility into token usage

## Test Commands

```bash
# Local development (all fast tests, ~15 seconds)
pytest

# Real-model tests (WhisperX/Qwen/E2E) are skipped by default.
# Opt in explicitly:
STUDIO_RUN_REAL_TESTS=1 pytest -m "slow or integration"

# GPU integration tests (requires GPU + models)
pytest -m llm_integration

# Specific test
pytest tests/test_qwen_provider.py -v

# Environment check
python src/main.py doctor

# Run pipeline (local mocks)
python src/main.py init --title "Test" --source video.mp4
python src/main.py run --project-id <id>
```

## Next Steps (Priority Order)

### 1. Timeline-Based Rendering Integration
- ✅ Connect existing orchestrator to timeline system
- ✅ Implement renderer that consumes Timeline objects
- ✅ Add FFmpeg command generation from timeline
- ✅ Persist `timeline/timeline.json` and `renders/render_job.json`
- ✅ Produce a valid H.264/AAC MP4 in the local profile

### 2. Multi-Scene Selection (Evidence-Driven Cut)
- ✅ Multi-scene selector (`select_scenes` → `selected_scenes.json`)
- ✅ Non-overlapping, timestamp-validated, minimum-duration scene picks
- ✅ Backward-compatible `selected_scene.json` (no clobbering)
- ✅ Multi-clip FFmpeg extraction (one clip per selected scene)
- ✅ Multi-clip timeline + concatenated render (scene clips + voiceover + subtitles)
- ✅ Script narration references all selected scenes in order
- ✅ QC validates the multi-scene cut
- ⏳ Typed evidence (visual/emotional/dialogue) from the director to drive selection

### 3. Script Generation Integration
- ✅ Integrate deterministic script generation into orchestrator
- ✅ Generate narration sections from the director thesis and scene index
- ✅ Update the timeline with voiceover and subtitle sections
- ⏳ Create a real script provider when an LLM model is available

### 4. Real GPU Validation (Colab)
- Clone repo into Colab with GPU
- Set STUDIO_PROFILE=colab-gpu
- Run full pipeline with real Qwen + WhisperX
- Validate output artifacts

### 5. TTS Integration
- Implement Kokoro or Qwen3-TTS provider
- Replace mock with real synthesis
- Add voice/emotion parameters

### 6. Image/Video Generation
- Add ComfyUI provider
- Integrate with timeline
- Replace mocks with real generation

## Commands for Next Developer

```bash
# Environment setup
python src/main.py doctor

# Local development workflow
python -m pytest                  # Run fast tests
python src/main.py init --title "MyProject" --source video.mp4
python src/main.py run --project-id <id>

# GPU/Colab workflow
export STUDIO_PROFILE=colab-gpu
python src/main.py doctor        # Verify GPU available
python -m pytest -m llm_integration
python src/main.py run --project-id <id>

# Check active configuration
grep -A 20 "profiles:" configs/profiles.yaml
```

## Files Changed This Session (Multi-Scene Selection)

- **Created**:
  - `tests/test_multi_scene_selection.py` - Multi-scene selector unit tests

- **Modified**:
  - `src/scene_selection/selector.py` - Added `select_scenes` (multi-scene), made `select_best_scene` non-destructive
  - `src/editor/ffmpeg_editor.py` - Multi-clip timeline + concatenated FFmpeg render
  - `src/script/writer.py` - Narration references all selected scenes
  - `src/app/orchestrator.py` - Multi-scene selection + per-scene clip extraction
  - `src/qc/critic.py` - QC checks for selected_scenes.json and scene clips
  - `tests/test_timeline_rendering.py` - Added multi-clip render test
  - `tests/test_whisperx_integration.py` - Gated behind `STUDIO_RUN_REAL_TESTS`
  - `tests/integration/test_e2e_integration.py` - Gated behind `STUDIO_RUN_REAL_TESTS`
  - `tests/test_transcription_adapter.py` - Forces stub path (no model load)
  - `pyproject.toml` - Default `pytest` excludes `slow`/`llm_integration`

## Test Results

```
52 fast tests ............................. PASS
  - 26 Qwen provider tests
  - 10 Creative director tests
  - 5 Multi-scene selection tests
  - Multi-clip rendering test (FFmpeg)
  - Existing ranking / selection / extraction / timeline tests
2 skipped (env-gated real-model tests)
8 deselected (slow / llm_integration)
─────────────────────────────────────────
TOTAL: 52 passing, 0 failures, ~16s runtime
```

## Status for Handoff

✅ **Architecture**: Complete and validated
✅ **Local Development**: Ready on weak laptops
✅ **Testing**: Comprehensive, fast
✅ **Multi-Scene Cut**: Selection, extraction, timeline, render, and QC wired end-to-end
⏳ **Typed Evidence Selection**: Director-driven evidence typing not yet used
✅ **Documentation**: Profiles, provider system, configuration
⏳ **GPU Validation**: Ready (needs Colab execution)
✅ **Timeline Rendering**: Local FFmpeg rendering integrated and validated
✅ **Multi-Scene Selection**: Implemented and wired through extraction, render, and QC
⏳ **Real TTS/Image/Video**: Mocks complete, integration pending

## Continuation Instructions

The project is now a complete, modular architecture that can be:

1. **Developed locally** with mocks on a weak laptop
2. **Executed with real models** on Google Colab GPU
3. **Extended easily** by adding new providers

To continue:

1. Pick next feature from "Next Steps" above
2. Implement using the provider/adapter pattern established
3. Add mock implementation first
4. Write unit tests with mocks
5. Add real implementation with GPU tests
6. Test on Colab
7. Update this status document

No blockers. Ready for GPU validation or next feature.
