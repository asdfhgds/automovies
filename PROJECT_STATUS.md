# PROJECT STATUS — Autonomous Movie Studio

**Last Updated**: Movie-grounded Creative Director milestone **built and unit-tested**
(real-Qwen clip validation pending execution). The Director now reads the existing
Movie Intelligence (`movie_index.json` → `SceneFacts`), builds a compact
fact-grounded context, asks real Qwen for 5 diverse concepts, **rejects** any
concept whose `required_evidence` is not actually present in the scenes, selects
the strongest grounded concept, and emits a scene-aware director plan
(`reports/director_reasoning.md`). Evidence verifier, new retrieval system, and
script wiring are intentionally **not** implemented. 221 fast tests pass (21 new
grounded-director tests).

Prior milestone (kept): Movie Intelligence validation **executed on a real movie**
+ **dense-embedding retrieval layer added and measured**. The real run
(`notebooks/colab_vision_gpu.ipynb`, Colab T4, `Qwen/Qwen2.5-VL-3B-Instruct`
bf16) covered project `5398e39c-d35b-481a-b580-42d7224732eb`, 120.078s window
(opening credits → opening monologue — **Portuguese-dubbed clip this time**;
English is the norm for normal runs), 33 scenes, **33/33 scenes vision-enriched**
(`provenance=qwen3vl`), ~3.6s/scene, no OOM. TF-IDF retrieval verdicts: **1 GOOD /
2 PARTIAL / 5 WRONG**; temporal probe OOM-free but mostly anchored at 0.0s.
**New**: `SemanticIndex` dense-embedding path (`--method embedding`,
sentence-transformers; lazy, env-driven, honest failure — never silently falls
back to TF-IDF). Measured locally on the real corpus with MiniLM: scores lift
0.02→0.20 range and re-rank (tension query now surfaces the gun-to-mouth
scene-30 at #2; emphasized-object stays GOOD at #1), but a monolingual-English
embedder on a Portuguese-dubbed clip only partially bridges narrative queries —
use `paraphrase-multilingual-MiniLM-L12-v2` for non-English. Full artifacts
preserved under `data/5398e39c-.../` (gitignored by design — scene cards carry
the movie's dialogue; the human-verdict record is tracked under `reports/`).
191 tests pass locally.

## Executive Summary

The Autonomous Movie Studio project now has a **complete, modular architecture** supporting:

- **Multi-profile execution** (local laptop development vs GPU-accelerated Colab)
- **Provider-based adapter pattern** for all generation capabilities
- **Mock implementations** for local testing without downloading models
- **Configuration-driven provider selection** (no code changes needed)
- **Quality control and validation** systems
- **Real WhisperX, PySceneDetect, and Qwen LLM** integration
- **Strict GPU mode (`REQUIRE_REAL_LLM=true`)**: refuses mock/deterministic fallback on GPU boxes so a real Qwen run is provable
- **Strict real-TTS mode (`REQUIRE_REAL_TTS=true`)**: refuses mock/pyttsx3 audio in production runs
- **Real open-source TTS providers**: Kokoro (default), Chatterbox, Qwen3-TTS behind one `TTSProvider` interface
- **Director-controlled narration properties**: tone/emotion/pace/energy/dramatic-intensity, honored per provider capability
- **TTS benchmarking**: same narration across providers, recording model/device/gen-time/duration/sample-rate/status
- **Cinematic audio mix**: film-ducking, music-ducking, EBU R128 normalization, true-peak limiter, burned subtitles
- **Real Qwen script writer** (`script/qwen_writer.py`) integrated into the orchestrator
- **Provider manifest** (`provider_manifest.json`) recording exactly which providers/models executed
- **191+ passing tests** (fast suite ~3-4 min) with zero regressions; real-model tests explicitly gated

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
Creative Director (Qwen LLM or deterministic; strict mode requires Qwen)
  ↓
Thesis-based Scene Ranking (deterministic)
  ↓
Scene Selection (multi-scene, non-overlapping top-K)
  ↓
FFmpeg Clip Extraction (real)
  ↓
Script Generation (real Qwen or deterministic; strict mode requires Qwen)
  ↓
TTS Synthesis (real Kokoro/Chatterbox/Qwen3-TTS or mock; strict mode requires real)
  ↓
Cinematic Audio Mix (ducking + loudnorm + true-peak limiter + burned subtitles)
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
├── movie_understanding/          (Movie Intelligence Layer)
│   ├── analyzer.py               MovieAnalyzer -> movie_index.json
│   ├── scene_analyzer.py         SceneEnricher interface + heuristic
│   ├── vision_enricher.py        Qwen3VLEnricher (Qwen2.5-VL/Qwen3-VL)
│   ├── keyframes.py              FFmpeg per-scene keyframe extraction
│   ├── enrich_factory.py         VISION_ENRICHER env selection + strict guard
│   ├── semantic_index.py         TF-IDF evidence retrieval index
│   ├── character_analyzer.py / event_index.py / text_utils.py
│   └── movie_memory.py           persistence
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

- **Strict GPU Validation Mode** (src/utils/strict.py, app/orchestrator.py)
  - `REQUIRE_REAL_LLM=true` → `require_cuda()` fails hard without CUDA; director
    and script stages refuse mock/deterministic providers (no silent fallback)
  - `provider_manifest.json` records provider, model, device, and load/generation timings
  - Doctor reports `strict_gpu_ok` and per-stage provider/model resolution
  - Unit tests in `tests/test_strict_mode.py` + `tests/test_qwen_script_writer.py`

- **Real Qwen Script Writer** (src/script/qwen_writer.py)
  - Loads director plan + selected scenes, prompts Qwen for narration sections
  - Parses/validates JSON into the canonical `script.json` schema (no fallback)
  - Records `script_model`, `script_device`, `script_dtype`, load + generation timings
  - Supports `SCRIPT_DTYPE=4bit` for VRAM-constrained GPUs

- **OOM-safe Qwen loading** (src/director/providers/qwen.py)
  - Class-level model cache: director + script stages share ONE loaded model
    (two copies would exceed a 16GB T4)
  - `low_cpu_mem_usage` + `device_map="auto"` + `QWEN_VRAM_RESERVE_GB` headroom
  - SDPA attention; `release_model()` + `empty_cache()` between stages
  - `DIRECTOR_DTYPE=4bit` / `SCRIPT_DTYPE=4bit` NF4 quantized loading (~4GB)

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

- **Real TTS Providers** (src/generation/kokoro.py, chatterbox.py, qwen_tts.py)
  - KokoroProvider: Kokoro-82M (default; lazy-loaded, CPU-skip on no CUDA)
  - ChatterboxProvider: Voice-cloning TTS (instruct/mimic modes)
  - QwenTTSProvider: Qwen3-TTS (voice-key-driven)
  - TTSProvider interface (src/generation/tts_adapter.py): meta/properties/`supported`
  - MockTTSProvider: Silent WAV fallback (non-strict mode only)

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

- **Script generation** (MockScriptProvider) — real Qwen writer used in strict/colab profiles
- **TTS** (MockTTSProvider - silent WAV files) — real Kokoro/Chatterbox/Qwen3-TTS in colab profile; mock rejected when `REQUIRE_REAL_TTS=true`
- **Image generation** (MockImageProvider - PNG placeholders) — real providers pending
- **Video generation** (MockVideoProvider - MP4 placeholders) — real providers pending

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
  script: qwen
  tts: kokoro
  image: mock
  video: mock
```

- Real WhisperX, Qwen LLM, Qwen script generation, and Kokoro TTS on GPU
- Mock image/video generation (to be replaced later)
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

1. **Vision fields need GPU + model** - `location`/`actions`/`objects`/
   `visual_description`/`visual_events`/`emotional_cues`/`themes`/`mood`/
   `cinematography`/`confidence` are only filled by `VISION_ENRICHER=qwen3vl` on
   CUDA; the heuristic enricher leaves them `None` + provenance-flagged by
   design. The semantic index *does* consume those fields for evidence
   retrieval (TF-IDF over the vision words), but retrieval is word-match only —
   not true embeddings.
2. **Keyword-based ranking** - the base ranker is deterministic lexical scoring; evidence tags are now used by selection, but the base ranker still needs an LLM-aware pass
3. **Small-model fidelity** - Qwen3-4B occasionally echoes prompt examples or produces loose JSON; guarded by placeholder detection, retry, and a plain-text fallback, but a 7B+ model would be more reliable
4. **TTS emotion is approximated** - Kokoro maps tone/emotion to fixed-personality voices and pace/energy to speed; Chatterbox/Qwen3-TTS report emotion/pace as unsupported in the released packages. True expressive control is not yet available
5. **Real TTS requires GPU** - providers skip CPU synthesis by design (`cpu_skipped`); a CUDA box is required for the benchmark/pipeline runs
6. **No human feedback loop** - one-shot generation only
7. **No cost tracking** - no visibility into token usage
8. **T4 VRAM budget (QLM/Qwen3-VL)** - `device_map="auto"` on a 16GB T4 offloads
   part of a 7B VL model to CPU (slower per-scene generation). The validated real
   run used **3B bf16** (`Qwen/Qwen2.5-VL-3B-Instruct`, ~6s/scene, no OOM);
   `VISION_DTYPE=4bit` / `VISION_ATTN=eager` / `VISION_MAX_IMAGE_PX=560` are the
   band-aids that made it fit and avoid the Qwen2.5-VL device-side-assert.
9. **Temporal localization is still weak (real-run verified)** - `probe_temporal`
   runs OOM-free per-frame (single-image sampling), but events are mostly anchored
   at `time_sec: 0.0` (only scene-32 got a real anchor at 116.43s). Treat any
   timestamp as "within this scene, unplaced".
10. **OCR / transcript misreads + hallucinated on-screen text (real-run)** - the
    model read "Tyler Durden" as "Talier Durden" in several scenes, "Fight Club"
    as "THE JUST BROTHERS"/"CLUB", a title card as "Davty Flatcher", and
    hallucinated a "gun-in-mouth" confusion in places. **Confidence ≠
    correctness** (scene-29 scored 0.88 while including a hallucinated title).
11. **Retrieval was TF-IDF word-overlap only — now has a dense-embedding path
    (measured)**: the TF-IDF 8-query eval scored **1 GOOD / 2 PARTIAL / 5
    WRONG** (good when query vocabulary literally appears in vision fields).
    The new `--method embedding` path (sentence-transformers, lazy/env-driven,
    honest failure) lifts scores ~0.02→0.20 and re-ranks meaningfully — on the
    real corpus with MiniLM the tension query surfaced the gun-to-mouth scene-30
    at #2 and the emphasized-object query stayed GOOD at #1 — but still does not
    fix narrative/thematic queries (fate, contradiction, choice-vs-control) and
    **MiniLM is monolingual-English**: use
    `RETRIEVAL_EMBEDDER_MODEL=sentence-transformers/paraphrase-multilingual-
    MiniLM-L12-v2` for non-English clips. LLM-based retrieval is the next lever.

## Test Commands

```bash
# Local development (all fast tests, ~50 seconds)
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

### 0. Editorial Pipeline (Evidence-Driven Cut) ✅ Local / ⏳ GPU
- ✅ Movie intelligence layer (`src/movie_understanding/`): analyzer, scene
  enricher (summary/topics/dialogue/characters/tone), character index, event
  index, semantic index (TF-IDF), movie memory persistence
- ✅ **Vision scene enrichment (Qwen3-VL)**: `keyframes.py` (FFmpeg per-scene
  frames), `Qwen3VLEnricher` (lazy shared Qwen2.5-VL/Qwen3-VL, 4-bit option,
  fills location/actions/objects/visual_description/visual_events/emotional_cues/
  themes/mood/cinematography/confidence with per-field `provenance=qwen3vl`),
  `enrich_factory.py` env selection, `REQUIRE_REAL_VISION` strict mode, wired
  into `MovieAnalyzer(attach_keyframes=True)` + orchestrator
- ✅ **Device-dispatch fix (real-T4 proven)**: never call `.model.to("cuda")`
  after `device_map="auto"` dispatches the model — fixes "You can't move a
  model that has some modules offloaded to cpu or disk"; regression-tested
- ✅ **Vision-aware retrieval + director artifacts**: `SemanticIndex` consumes
  vision fields; analyzer writes `scene_index_v2.json`, `movie_memory/` bundle,
  and `reports/movie_understanding_report.md`
- ✅ **Retrieval eval + temporal probe**: `scripts/evaluate_retrieval.py`
  (queries → `reports/retrieval_evaluation.json`/`.md`, blank
  GOOD/PARTIAL/WRONG fields) and `Qwen3VLEnricher.probe_temporal()` (ordered,
  approx-timestamped visual events) — both covered by tests
- ✅ Editorial planning (`src/editorial/`): EditorialPlan models,
  heuristic planner (hook/thesis/evidence/close), evidence retrieval
  (semantic + lexical + dialogue), evidence-aligned script with short captions,
  editorial timeline builder (excerpt extraction), editorial FFmpeg renderer
  (per-clip speed/crop/hold, xfade transitions, edge fades, narration-dominant
  audio mix, burned subtitles), QC wiring
- ✅ Orchestrator `EDITORIAL_MODE=true` path: movie analysis → editorial plan →
  editorial script → editorial timeline → editorial render
- ✅ Fixed: editorial timeline was wired to `movie_index` instead of the script
  (narration windows/segments now use the real planned timing)
- ✅ Fixed: editorial renderer now applies `fps=` to every clip so `xfade`
  inputs share a timebase (mixed frame-rate sources no longer fail with
  "First input link timebase ... do not match ... xfade timebase")
- ✅ **Real Qwen3-VL movie-understanding run EXECUTED (Colab T4)**: project
  `5398e39c-d35b-481a-b580-42d7224732eb` — `Qwen/Qwen2.5-VL-3B-Instruct` bf16,
  120.078s window (Portuguese-dubbed clip; normally English), 33/33 scenes enriched (`provenance=qwen3vl`), ~3.6s/scene,
  no OOM, all artifacts produced (`scene_index_v2.json`, `semantic_index.json`,
  `movie_understanding_report.md`, `retrieval_evaluation.json`/`.md`,
  `temporal_probe.json`). Human verdicts: retrieval **1 GOOD / 2 PARTIAL /
  5 WRONG** (TF-IDF has no semantics); temporal probe OOM-free but mostly
  unanchored (0.0s). Full artifacts preserved under
  `data/5398e39c-.../` + `reports/` (gitignored by design); the human-verdict
  retrieval record is tracked at `data/5398e39c-.../reports/retrieval_evaluation.json`.
- ✅ **Dense-embedding retrieval layer**: `SemanticIndex` now supports
  `build(..., embedder=...)` — corpus dense vectors computed at build time,
  cosine search (TF-IDF stays the default/fallback). `embedding_retriever.py`
  (`SentenceEmbedder` lazy sentence-transformers + env-driven factory
  `RETRIEVAL_EMBEDDER`/`RETRIEVAL_EMBEDDER_MODEL`/`RETRIEVAL_DEVICE`);
  `scripts/evaluate_retrieval.py --method embedding [--embedder module:attr]`
  refuses to silently fall back to TF-IDF. **Measured on the real corpus
  (MiniLM)**: scores 0.02→0.20, tension→scene-30 gun-to-mouth at #2, object→
  stays #1; narrative queries still sub-GOOD (MiniLM is English-only — use the
  multilingual model for non-English clips). 8 new tests.
- ✅ Local E2E: 191 tests pass (vision, artifacts, retrieval, editorial
  orchestrator, movie-understanding, semantic-embedding suites); editorial orchestrator test proves all artifacts

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
- ✅ Typed evidence (visual/emotional/dialogue) from the director drives scene selection
  (select scenes by evidence tags, not just keyword score)

### 3. Script Generation Integration
- ✅ Integrate deterministic script generation into orchestrator
- ✅ Generate narration sections from the director thesis and scene index
- ✅ Update the timeline with voiceover and subtitle sections
- ✅ Real Qwen script provider (`script/qwen_writer.py`) wired into the orchestrator
  (used when strict mode or `SCRIPT_PROVIDER=qwen`)

### 4. Real GPU Validation (Colab)
- ✅ Strict GPU mode implemented (`REQUIRE_REAL_LLM=true`) with hard failures
- ✅ Doctor reports strict prerequisites + per-stage provider/model
- ✅ Provider manifest written after every pipeline run
- ✅ `notebooks/colab_qwen_validation.ipynb` (14 cells) drives the real director
  + script Qwen path on a T4 through the same code the orchestrator uses
- ✅ `scripts/colab_setup.sh` idempotent Colab dependency setup
- ✅ **Executed on a real T4 and PASSED** (`provider_manifest.json` attached to the
  validation ticket): `director_real_generation: true`, `script_real_generation: true`,
  `transcript_real: true`, model `Qwen/Qwen3-4B-Instruct-2507`, device `cuda`,
  director stage ~119s / script generation ~57s. The model loaded once (shared
  class-level cache; script `qwen_load_time_sec: 0.0`) so no OOM on a 16GB T4.
  Output was genuine: a specific, evidence-grounded thesis, a real 5-section
  production plan, and a coherent multi-paragraph narration referencing the scene.
- ✅ Robustness fixes proven in the successful run:
  - Placeholder-echo guard (`src/utils/json_guard.py`): a 4B model that copies the
    prompt's example JSON is detected and retried instead of silently accepted
  - Prompt examples replaced with ALL-CAPS markers + "replace, don't copy"
  - Plain-text fallback parses `Title: / Hook: / Thesis:` lines when JSON fails
  - Chat-template wrapping + decode-only-new-tokens for instruct models
  - Notebook guards against stale kernels (re-imports from disk if the loaded
    `qwen.py` is pre-fix) and against reusing a stale validation clip

### 5. TTS Integration ✅ (Real TTS + Cinematic Audio Mix)
- ✅ Real open-source TTS providers behind one `TTSProvider` interface:
  `src/generation/kokoro.py` (Kokoro-82M, default), `src/generation/chatterbox.py`
  (voice cloning), `src/generation/qwen_tts.py` (Qwen3-TTS)
- ✅ Provider factory + `available_tts_providers()`; switched via `TTS_PROVIDER` or profiles
- ✅ Strict production mode `REQUIRE_REAL_TTS=true` refuses mock/pyttsx3 audio
- ✅ Director-controlled narration properties (tone/emotion/pace/energy/intensity)
  → `script.json["narration_properties"]`, honored per provider capability
- ✅ TTS benchmark: same narration across providers → `reports/tts_benchmark.json`
  (model/device/gen-time/duration/sample-rate/status; `cpu_skipped` when no CUDA)
- ✅ Audio pipeline: film ducking (sidechaincompress), music ducking, EBU R128
  loudnorm, true-peak `alimiter` (no clipping), burned subtitles (SRT/libass)
- ✅ GPU notebook `notebooks/colab_real_movie_tts.ipynb` (real movie + real TTS)
  + `scripts/colab_tts_setup.sh`
- ⏳ Real-TTS GPU run on a user-supplied movie (pending)

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

## Movie-Grounded Creative Director Milestone (NEW)

**Objective**: make the Creative Director actually consume the richer Movie
Intelligence representation that already exists, and reason only from what is
actually present in the movie.

### How Movie Intelligence feeds the Director
1. `SceneFacts.from_movie_intelligence(movie_index)` / `from_project_dir()`
   normalizes any of the existing on-disk representations
   (`movie_index.json` with `scenes[].story`, a bare scene list, or the
   `movie_memory/` / `scene_index_v2.json` bundle) into a uniform list of
   `SceneFact` records carrying only vision/transcript facts.
2. `DirectorContextBuilder.build_concept_generation_context()` renders a
   compact, **token-limited** per-scene summary (SCENE id / Time / Characters /
   Actions / Objects / Visual / Mood / Themes / Dialogue) plus a
   "what actually exists" vocabulary (known characters / locations / objects),
   and the creative-memory summary. It deliberately does **not** dump the raw
   scene JSON into the prompt.
3. `EvidenceAnalyzer` lexically grounds each concept's `required_evidence`
   claims to real scenes (no new retrieval system — it inspects the existing
   index) and returns coverage HIGH / MED / LOW.
4. `MovieGroundedDirector` orchestrates: generate 5 diverse concepts → evidence
   gate (reject generic or un-evidenced concepts, regenerate substitutes) →
   `ConceptCritic` feasibility scoring → select strongest → scene-aware plan →
   `CreativeMemory` store → `reports/director_reasoning.md`.

### Design decisions (per milestone)
- **Concepts carry `required_evidence`** (concrete claims checked against real
  scenes) instead of only `supporting_scene_types`; the critic now scores an
  actually-grounded evidence coverage.
- **The Director may reject its own idea**: a concept whose `required_evidence`
  does not match at least `min_coverage` of scenes, or whose thesis is a generic
  platitude, is rejected and replaced (up to 2 regeneration rounds); rejected
  concepts are shown in the report.
- **Hallucination prevention**: unknown characters are rendered as
  `unknown_character_01 (low confidence)`; the context lists only names/objects/
  locations that exist, and `SceneFacts.is_grounded()` lets callers verify a
  claim is real.
- **No script wiring yet**: the pipeline stops after the selected concept +
  plan (§11). Evidence verifier, new embedding model, TTS replacement, subtitle
  redesign, generative video/image, and YouTube automation are intentionally NOT
  implemented (§15).

### Files added
- `src/director/scene_facts.py` — normalizer + fact access + hallucination guard.
- `src/director/context_builder.py` — compact, token-limited director context.
- `src/director/evidence.py` — EvidenceAnalyzer (coverage, scene mapping,
  evidence strategy, visual motifs).
- `src/director/concepts.py` — prompt builders + tolerant JSON parsing for the
  concept / plan / rejection / diversity schemas.
- `src/director/report.py` — renders `reports/director_reasoning.md`.
- `src/director/grounded.py` — `MovieGroundedDirector` orchestration.
- `scripts/run_director_validation.py` — gated real-Qwen Colab validation
  (5 concepts → select → plan → `director_reasoning.md`).
- `tests/test_grounded_director.py` (21 tests: context builder, truncation,
  hallucination guard, evidence availability, diversity, rejection, memory,
  plan schema).
- `tests/test_grounded_director_real_qwen.py` — gated (`llm_integration`).
- `notebooks/colab_vision_gpu.ipynb` — added Cell 7d (grounded director); retitled
  Cell 7b doc to make `embedding` the default and flipped `METHOD="embedding"`.

### Context size / model / evidence record
- Context builder caps at `max_tokens` minus `reserve_for_output` (default
  4096−2048); scene summaries are truncated to fit, recorded in `context_meta`.
- Model: real Qwen via the existing `QwenProvider` (gated, `generate_text`).
- Per milestone §16, the real-run metrics (concepts generated, selected concept,
  evidence coverage, failures, known hallucination cases, next bottleneck) are to
  be recorded in `reports/director_validation.json` by
  `scripts/run_director_validation.py` when executed on the validated movie
  (`bc6384be-...`).

## Files Changed This Session (Real Qwen GPU Validation)

- **Created**:
  - `src/utils/strict.py` - Strict GPU mode guards (`require_cuda`, `require_real_provider`)
  - `src/utils/json_guard.py` - Placeholder-echo detection for small LLMs
  - `src/script/qwen_writer.py` - Real Qwen narration script writer
  - `src/script/__init__.py` - Package export
  - `scripts/colab_setup.sh` - Idempotent Colab dependency setup
  - `notebooks/colab_qwen_validation.ipynb` - 14-cell real-Qwen validation notebook
  - `tests/test_strict_mode.py` - Strict-mode unit tests
  - `tests/test_qwen_script_writer.py` - Qwen script helper unit tests
  - `tests/test_multi_scene_selection.py` - Multi-scene selector unit tests

- **Modified**:
  - `src/app/orchestrator.py` - Strict guard phase, real Qwen script stage, provider manifest
  - `src/director/provider_factory.py` - Fixed `src.`-prefix import bug; added strict provider checks
  - `src/director/planner.py` - Fixed `src.`-prefix import bug
  - `src/director/providers/qwen.py` - Default model 4B-Instruct-2507,
    `generate_text`, load/generation timing, shared model cache, 4-bit loading,
    chat-template prompts, hardened JSON extraction + repair, placeholder guard,
    seed-jitter retry, plain-text concept fallback
  - `src/script/qwen_writer.py` - Placeholder-free script prompt + narration echo guard
  - `src/director/providers/transport_base.py` / `local.py` - Context manager strict flag
  - `src/director/providers/api.py` - Single-device config for real LLM
  - `src/director/creative_director.py` - Multi-scene evidence-driven selection
  - `src/director/prompts/context_builder.py` (or as needed) - transcript in director context
  - `src/understanding/transcription/` - Multi-scene association support
  - `src/scene_selection/selector.py` - Evidence-tag-driven multi-scene selection
  - `src/utils/doctor.py` - STRICT GPU MODE section, transformers/accelerate/model info, JSON keys
  - `configs/app.yaml` - Script config; default model Qwen/Qwen3-4B-Instruct-2507
  - `configs/profiles.yaml` - colab-gpu script→qwen; qwen provider block + script provider block
  - Documentation: `COLAB_INSTRUCTIONS.md`, `PROJECT_STATUS.md`, `DEVELOPMENT_ROADMAP.md`

## Files Changed This Session (Editorial Milestone)

- **Created**:
  - `src/movie_understanding/` — analyzer, scene/character/event analyzers,
    semantic index, text utils, movie memory persistence
  - `src/editorial/` — plan models, director (heuristic + Qwen stub),
    retrieval, script builder, timeline builder, subtitle chunking, renderer
  - `tests/` — `test_editorial.py`, `test_editorial_orchestrator.py`,
    `test_editorial_render.py`, `test_movie_understanding.py`,
    `editorial_fixtures.py`, `__init__.py`
  - `notebooks/colab_editorial_gpu.ipynb` — 18-cell GPU validation notebook
    for the editorial pipeline (real Qwen + real TTS + editorial render)
  - `NEXT_MILESTONE.md` — handoff: PROVEN/EXPERIMENTAL/FAILED/KNOWN
    LIMITATIONS/CURRENT BOTTLENECK/NEXT ACTION

- **Modified**:
  - `src/app/orchestrator.py` — `EDITORIAL_MODE=true` phase (movie analysis,
    editorial plan/script/timeline, editorial assembly); editorial timeline
    wired to the script (was movie_index); `_run_script_stage` extraction
  - `src/editorial/render.py` — per-clip timebase normalization for xfade
  - `src/qc/critic.py` — editorial cut checks
  - `notebooks/colab_real_movie_tts.ipynb` — replaced by editorial GPU notebook
  - Documentation: `PROJECT_STATUS.md`, `DEVELOPMENT_ROADMAP.md`,

## Files Changed This Session (Vision Scene Enrichment Milestone)

- **Created**:
  - `src/movie_understanding/keyframes.py` — FFmpeg keyframe extraction per
    scene window (`extract_scene_keyframes`, `extract_all_scene_keyframes`,
    `snapshot_frame`), no OpenCV dependency
  - `src/movie_understanding/vision_enricher.py` — `Qwen3VLEnricher`:
    lazy shared Qwen2.5-VL/Qwen3-VL load (AutoProcessor+AutoModel, SDPA,
    device_map=auto, 4-bit NF4 via `VISION_DTYPE`), chat-template image
    handling with processor/tokenizer/decoder API fallback, JSON repair,
    fills location/actions/visual_description/themes/mood with per-field
    `provenance=qwen3vl`; degrades to heuristic or raises under strict
  - `src/movie_understanding/enrich_factory.py` — env-driven
    `VISION_ENRICHER`/`VISION_MODEL`/`VISION_DEVICE`/`VISION_DTYPE`/
    `VISION_MAX_FRAMES` + `require_real_vision` strict guard
  - `scripts/colab_vision_setup.sh` — idempotent Qwen-VL transformers setup
  - `notebooks/colab_vision_gpu.ipynb` — 10-cell GPU validation notebook for
    the vision layer (keyframes → real Qwen3-VL → vision scene cards + QA)
  - `tests/test_vision_enrichment.py` — 33 tests (keyframes, JSON repair,
    fake-VL enrich/degrade/strict, analyzer integration, factory/env, device-map
    dispatch regression, temporal probe)

- **Modified**:
  - `src/movie_understanding/analyzer.py` — `attach_keyframes=` option,
    `create_scene_enricher_from_env()` default enricher, provenance
    `keyframes` + `scene_enricher` name
  - `src/movie_understanding/__init__.py` — export enricher factory
  - `src/movie_understanding/scene_analyzer.py` — `mood` field in story schema
  - `src/utils/strict.py` — `REQUIRE_REAL_VISION` + `require_real_vision`
  - `src/app/orchestrator.py` — vision enricher + keyframe attachment in the
    editorial movie-analysis stage
  - `configs/app.yaml`, `configs/profiles.yaml` — `vision:` provider block
  - Documentation: `PROJECT_STATUS.md`, `DEVELOPMENT_ROADMAP.md`, `NEXT_MILESTONE.md`

## Files Changed This Session (Movie Intelligence Validation Prep)

- **Created**:
  - `src/movie_understanding/artifacts.py` — `scene_index_v2.json` (versioned
    enriched scene index), `movie_memory/` bundle (index + scene v2 + semantic +
    characters + events + manifest), `reports/movie_understanding_report.md`
  - `scripts/evaluate_retrieval.py` — retrieval-evaluation harness: milestone
    queries → `reports/retrieval_evaluation.json` + `.md` (blank
    `human_assessment` GOOD/PARTIAL/WRONG fields)
  - `tests/test_artifacts.py` — 10 tests (scene_index_v2, movie_memory bundle,
    understanding report, analyzer emits artifacts, vision-field retrieval,
    eval harness reports)

- **Modified**:
  - `src/movie_understanding/vision_enricher.py` — extended schema: `objects`,
    `visual_events` (approx-timestamped), `emotional_cues`, `cinematography`,
    `confidence` + provenance; added `probe_temporal()` (ordered events across N
    keyframes, honest failure); device-map `.to()` guard fix
  - `src/movie_understanding/scene_analyzer.py` — heuristic enricher now emits
    the extended story schema (new fields `None` + `unavailable (vision/LLM)`)
  - `src/movie_understanding/semantic_index.py` — corpus + `to_dict` now include
    the vision fields so on-screen content is queryable
  - `src/movie_understanding/analyzer.py` — writes `scene_index_v2.json` +
    `movie_memory/` after every analyze
  - `tests/test_vision_enrichment.py` — 28 tests (new-field assertions,
    device-map regression, temporal probe)
  - `notebooks/colab_vision_gpu.ipynb` — cells 7/7b/7c produce all artifacts,
    run retrieval eval, run temporal probe; cell 8 shows new fields
  - Documentation: `PROJECT_STATUS.md`, `DEVELOPMENT_ROADMAP.md`, `NEXT_MILESTONE.md`

## Files Changed This Session (Real Movie Understanding Run — Finalized)

- **Created** (real-run artifacts for project `5398e39c-d35b-481a-b580-42d7224732eb`):
  - `data/5398e39c-.../movie_index.json`, `scene_index_v2.json`,
    `semantic_index.json`, `events.json`, `characters.json`, `manifest.json`
  - `data/5398e39c-.../reports/movie_understanding_report.md`,
    `retrieval_evaluation.json`/`.md` (human assessments filled),
    `temporal_probe.json`
  - Note: `data/` is gitignored by design (scene cards echo the movie's
    dialogue transcript). Full artifacts are preserved locally; only the
    retrieval human-verdict record is tracked (force-added).

- **Modified**:
  - Documentation: `PROJECT_STATUS.md` (real-run summary, honest verdicts,
    real-known-limitations, 191-test count), `DEVELOPMENT_ROADMAP.md`,
    `NEXT_MILESTONE.md` (milestone completed with measurements; next bottleneck
    is retrieval semantics + temporal localization)

## Files Changed This Session (Dense-Embedding Retrieval)

- **Created**:
  - `src/movie_understanding/embedding_retriever.py` — `SentenceEmbedder`
    (lazy sentence-transformers, env-driven model/device) +
    `create_embedder_from_env()` (`RETRIEVAL_EMBEDDER` = `sentence-transformers`
    or `module:attr` factory; never loads a model at creation)
  - `tests/test_semantic_embeddings.py` — 8 tests: synonym-query embedding vs
    TF-IDF (embedding finds the diner scene TF-IDF misses), method tags/dense
    cosine, empty corpus, eval-harness embedding wiring via a stub
    `module:attr` embedder, honest failure when the embedder is missing, env
    parsing
- **Modified**:
  - `src/movie_understanding/semantic_index.py` — `build(..., embedder=)`
    computes dense doc vectors at build time; `search()` cosine (+ transcript /
    dialogue overlap); TF-IDF stays the default; removed the legacy
    `_cosine_dict` helper
  - `scripts/evaluate_retrieval.py` — `--method {tfidf,embedding}` +
    `--embedder module:attr`; JSON/MD record the method; **exits non-zero with
    an actionable message instead of silently falling back to TF-IDF**
  - `scripts/colab_vision_setup.sh` — installs `sentence-transformers`
  - `notebooks/colab_vision_gpu.ipynb` — Cell 7b has a `METHOD`
    `tfidf`/`embedding` form; markdown explains the embedding pass
  - Documentation: `PROJECT_STATUS.md`, `NEXT_MILESTONE.md`,
    `DEVELOPMENT_ROADMAP.md` (measurement recorded)

## Files Changed This Session (Movie-Grounded Creative Director)

- **Created**:
  - `src/director/scene_facts.py` — Movie Intelligence normalizer → `SceneFact`
    records; hallucination-guard vocabulary.
  - `src/director/context_builder.py` — compact, token-limited director context
    (scene summaries + known-character/object/location vocabulary + memory).
  - `src/director/evidence.py` — `EvidenceAnalyzer`: grounds concepts to real
    scenes, coverage HIGH/MED/LOW, evidence strategy, visual motifs.
  - `src/director/concepts.py` — prompt builders (generation / rejection /
    plan), tolerant JSON parsing, diversity metric, generic-thesis detection.
  - `src/director/report.py` — `reports/director_reasoning.md` renderer.
  - `src/director/grounded.py` — `MovieGroundedDirector` orchestration.
  - `scripts/run_director_validation.py` — gated real-Qwen Colab validation.
  - `tests/test_grounded_director.py` (21 tests) and
    `tests/test_grounded_director_real_qwen.py` (gated `llm_integration`).
- **Modified**:
  - `src/director/__init__.py` — export the new director modules.
  - `notebooks/colab_vision_gpu.ipynb` — Cell 7d (grounded director); Cell 7b
    default `METHOD="embedding"` + doc.
  - Documentation: `PROJECT_STATUS.md`, `DEVELOPMENT_ROADMAP.md`,
    `NEXT_MILESTONE.md`.

## Test Results

```
221 fast tests ............................ PASS (incl. 21 new grounded-director tests)
```
  - 26 Qwen provider tests (incl. placeholder-guard + plain-text fallback)
  - 10 Creative director tests
  - 17 Strict-mode + Qwen script writer tests
  - 5 Multi-scene selection tests
  - 12 Real-TTS provider unit tests (no model loading; find_spec availability)
  - 5 TTS strict-mode (REQUIRE_REAL_TTS) tests
  - 3 TTS adapter tests (meta + narration props + strict rejection)
  - 2 TTS benchmark tests (mock baseline + unavailable-provider reporting)
  - 5 Audio-mix render-command tests (ducking/loudnorm/limiter/srt/silent clips)
  - 33 Vision enrichment tests (keyframes, JSON repair, fake-VL enrich/degrade/strict,
    analyzer integration, factory/env selection, device-map dispatch regression,
    temporal probe)
  - 10 Movie-intelligence artifact tests (scene_index_v2, movie_memory bundle,
    understanding report, analyzer emits artifacts, vision-field retrieval,
    eval harness reports)
  - Multi-clip rendering test (FFmpeg) — now with film ducking + subtitles
  - Existing ranking / selection / extraction / timeline tests
3 skipped (real-TTS GPU tests + real-TTS benchmark — gated behind STUDIO_RUN_REAL_TESTS=1 + CUDA)
11 deselected (slow / llm_integration)
─────────────────────────────────────────
TOTAL: 191 passing, 0 failures

## Status for Handoff

✅ **Architecture**: Complete and validated
✅ **Local Development**: Ready on weak laptops
✅ **Testing**: Comprehensive, fast (191 passing, ~3-4 min)
✅ **Multi-Scene Cut**: Selection, extraction, timeline, render, and QC wired end-to-end
✅ **Typed Evidence Selection**: Director-driven evidence typing used by selection
✅ **Real Script Provider**: Qwen narration writer integrated into the orchestrator
✅ **Strict GPU Mode**: Hard-fail validation with provider manifest; doctor reporting
✅ **Strict Real-TTS Mode**: `REQUIRE_REAL_TTS=true` refuses mock audio; doctor reports `tts_ok`
✅ **Real TTS Providers**: Kokoro (default), Chatterbox, Qwen3-TTS behind one interface;
   lazy-loaded, shared class-level model cache, per-provider `supported` reporting
✅ **Narration Properties**: tone/emotion/pace/energy/intensity in `script.json`, applied per provider
✅ **TTS Benchmark**: `benchmark-tts` CLI + `RUN_TTS_BENCHMARK=true` hook → `tts_benchmark.json`
✅ **Cinematic Audio Mix**: film + music ducking, loudnorm, true-peak limiter, burned subtitles
✅ **Local E2E Render Validated**: mock-profile run produced a playable 29.6s MP4
   (H.264 1280x720 + AAC, subtitles burned, `no_clipping: true`, peak −2.9 dB)
✅ **Qwen3-VL Vision Scene Enrichment**: `keyframes.py` + `Qwen3VLEnricher` +
   `enrich_factory.py` + `REQUIRE_REAL_VISION` strict mode; fills
   location/actions/objects/visual_description/visual_events/emotional_cues/
   themes/mood/cinematography/confidence with per-field provenance; device-map
   dispatch fix proven on a real T4; 44 vision/artifact/retrieval tests
✅ **Colab GPU Validation (LLM)**: PASSED on a real T4 with Qwen/Qwen3-4B-Instruct-2507
   (real concepts + narration, no OOM, no placeholder echo)
✅ **Documentation**: Profiles, provider system, configuration, Colab flow, TTS, vision
✅ **Real Qwen3-VL movie-understanding run**: EXECUTED on a real T4 with
   `Qwen/Qwen2.5-VL-3B-Instruct` (bf16) — 33/33 scenes vision-enriched, no OOM,
   all artifacts in `data/5398e39c-d35b-481a-b580-42d7224732eb/`. Retrieval eval:
   1 GOOD / 2 PARTIAL / 5 WRONG (TF-IDF limits). Temporal probe: runs, mostly
   unanchored. Project earned its honest verdict — further improvement needs a
   semantic (embedder/LLM) retrieval layer + stronger temporal localization.
✅ **Movie-grounded Creative Director**: reads the existing Movie Intelligence,
   generates 5 diverse concepts with `required_evidence`, rejects unsupported /
   generic ideas, selects the strongest grounded concept, emits a scene-aware
   plan + `director_reasoning.md`. 221 fast tests pass. Real-Qwen clip
   validation (Colab) is gated/pending execution.
⏳ **Real-Movie + Real-TTS GPU run**: notebook `colab_real_movie_tts.ipynb` ready;
   needs a user-supplied legally-owned movie to execute on a T4/A100
⏳ **Image/Video Generation**: Mocks complete, integration pending

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

No blockers. Real TTS providers + cinematic audio pipeline are implemented and
tested; the remaining proof is executing `notebooks/colab_real_movie_tts.ipynb`
on a GPU with a user-supplied movie, then evaluating several real videos before
deciding the next milestone (semantic understanding / director intelligence /
script quality / scene retrieval / TTS-emotion / montage / image gen / cloud video).
