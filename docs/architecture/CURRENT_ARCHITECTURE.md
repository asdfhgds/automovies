# Current Architecture Audit

> Documented: 2026-09-02
> Based on: commit `f1930b1`

---

## 1. High-Level Overview

The AutoMovies project is an autonomous movie-analysis/video-production pipeline. The current architecture is **monolithic** with a single orchestrator that runs all stages sequentially in one process.

```
Movie File
    │
    ▼
┌─────────────────────────────────────────┐
│           ORCHESTRATOR                  │
│  (src/app/orchestrator.py)              │
│                                         │
│  1. Transcription                       │
│  2. Scene Indexing                      │
│  3. Movie Intelligence                  │
│  4. Director (Grounded/Creative)        │
│  5. Editorial Mode                      │
│  6. Scene Ranking/Selection             │
│  7. Script Generation                   │
│  8. Clip Extraction                     │
│  9. Visual Generation (ComfyUI)         │
│  10. TTS                                │
│  11. Assembly (FFmpeg)                  │
│  12. QC                                 │
└─────────────────────────────────────────┘
    │
    ▼
Final Render + Manifests
```

---

## 2. Entry Points

| File | Purpose |
|------|---------|
| `src/main.py` | CLI: `init`, `run`, `benchmark-tts`, `doctor` |
| `scripts/run_pipeline.py` | Thin wrapper calling `src.main:main()` |
| `scripts/run_director_validation.py` | Standalone V4 director validation |
| `generate_notebook.py` | Generates Colab notebook |

---

## 3. Pipeline Stages (Current)

| Phase | Module | Input Artifacts | Output Artifacts | GPU? |
|-------|--------|-----------------|------------------|------|
| Transcription | `transcription.adapter` | `project_meta.json.source_path` | `transcripts/transcript.json` | Yes (WhisperX) |
| Scene Indexing | `scene_indexing.adapter` | `source_path` | `scenes/scene_index.json` | CPU (PySceneDetect) |
| Movie Intelligence | `movie_understanding.analyzer` | `transcript.json`, `scene_index.json` | `movie_index.json`, `semantic_index.json` | Optional (Qwen3-VL) |
| Director (Grounded) | `director.grounded` | `movie_index.json` | `director_plan.json`, `grounded_script.json` | Yes (Qwen) |
| Director (Creative) | `director.creative_director` | `scene_index.json`, `transcript.json` | `director_plan.json` | Yes (Qwen) |
| Editorial Plan | `editorial.director`/`editorial.grounded` | `director_plan.json`, `grounded_script.json` | `editorial_plan.json`, `script.json` | No |
| Timeline | `editorial.timeline` | `editorial_plan.json`, `script.json` | `editorial_timeline.json` | No |
| Scene Ranking | `scene_selection.ranker` | `director_plan.json.thesis` | `scene_ranking.json` | No |
| Scene Selection | `scene_selection.selector` | `scene_ranking.json` | `scenes/selected_scenes.json` | No |
| Script Gen (Qwen) | `script.qwen_writer` | `director_plan.json` | `script.json` | Yes (Qwen) |
| Script Gen (Det) | `script.writer` | `director_plan.json` | `script.json` | No |
| Clip Extraction | `editor.clip_extractor` | `selected_scenes.json`, `source_path` | `assets/scenes/*.mp4` | CPU (FFmpeg) |
| Visual Generation | `visual_generation.comfyui_client` | `asset_plan.json` | `assets/generated/*.png` | Optional (ComfyUI) |
| TTS | `audio.tts_adapter` | `script.json` | `audio/voice.wav` | Yes (Kokoro/Chatterbox) |
| Assembly | `editorial.render`/`editor.ffmpeg_editor` | timeline + clips + audio | `renders/final_render.mp4` | CPU (FFmpeg) |
| QC | `qc.critic` | all artifacts | `reports/qc_report.json` | No |
| Pipeline Status | `quality.pipeline_status` | all artifacts | `reports/pipeline_status.json` | No |

---

## 4. Artifact Directory Structure

```
data/<project_id>/
├── project_meta.json              # Project metadata (id, title, source_path)
├── movie_index.json               # Enriched movie intelligence
├── semantic_index.json            # Semantic search index
├── characters.json                # Character index
├── events.json                    # Event index
├── manifest.json                  # Movie intelligence provenance
├── director_plan.json             # Director output (thesis, structure, etc.)
├── grounded_script.json           # Grounded script (if grounded director)
├── editorial_plan.json            # Editorial decisions
├── editorial_timeline.json        # Timeline with excerpt windows
├── script.json                    # Final script for rendering
├── scenes/
│   ├── scene_index.json           # Raw PySceneDetect shots
│   ├── scene_cards.json           # Alternative format
│   ├── selected_scenes.json       # Selected excerpt scenes
│   ├── scene_ranking.json         # Ranked scenes by thesis relevance
│   ├── keyframes/                 # Extracted keyframes
│   └── assets/
│       ├── scenes/                # Extracted clip files
│       └── generated/             # ComfyUI generated images
├── transcripts/
│   ├── transcript.json            # WhisperX output with word timestamps
│   └── transcript.txt             # Plain text transcript
├── audio/
│   ├── voice.wav                  # TTS narration
│   └── tts_meta.json              # TTS metadata
├── renders/
│   └── final_render.mp4           # Final video output
├── reports/
│   ├── provider_manifest.json     # Pipeline execution manifest
│   ├── pipeline_status.json       # QC verdict (PASS/REVISE/FAIL)
│   ├── qc_report.json             # Detailed QC checks
│   ├── director_reasoning.md      # Director reasoning report
│   ├── director_validation.json   # Director validation result
│   └── retrieval_evaluation_*.json # Retrieval evaluation
├── asset_plan.json                # Visual generation plan
├── timeline.json                  # Alternative timeline format
└── render_job.json                # Render job metadata
```

---

## 5. Component Coupling Analysis

### A. Monolithic Orchestrator (`src/app/orchestrator.py`)
- **753 lines** — single function `start_pipeline()` runs everything
- Direct imports of all stage modules
- No abstraction between stages
- Environment variables control behavior (`GROUNDED_DIRECTOR`, `EDITORIAL_MODE`, `REQUIRE_REAL_LLM`, etc.)
- Hard-coded path: `ROOT / 'data' / project_id`

### B. Tight Coupling Examples
1. **Director runs inside orchestrator** — `_run_grounded_director()` called directly
2. **Editorial calls director internals** — loads `director_plan.json` and inspects `grounded` field
3. **Scene ranking reads director thesis** — parses `director_plan.json` for thesis
4. **Script generation reads director output** — expects specific `director_plan.json` structure
5. **Visual generation reads `asset_plan.json`** — produced by director
6. **TTS reads `script.json`** — expects specific structure
7. **Assembly reads timeline/editorial output** — expects specific JSON structure

### C. Python Object Coupling
- `MovieAnalyzer` receives `SceneEnricher` instance (GPU model)
- `CreativeDirector` receives `provider` instance (LLM)
- `GroundedEditorialPlanner` receives `GroundedScript` object
- Models cached in class variables (shared across stages)

---

## 6. Existing Persistent Artifacts

| Artifact | Producer | Format | Schema? |
|----------|----------|--------|---------|
| `project_meta.json` | `main.py:init_project` | JSON | Implicit |
| `movie_index.json` | `MovieAnalyzer` | JSON | Implicit (docstring) |
| `semantic_index.json` | `MovieAnalyzer` | JSON | Implicit |
| `director_plan.json` | Orchestrator/Director | JSON | Implicit |
| `grounded_script.json` | `GroundedScriptGenerator` | JSON | `grounding_contract.py` |
| `editorial_plan.json` | `EditorialDirector`/`GroundedEditorialPlanner` | JSON | Implicit |
| `editorial_timeline.json` | `EditorialTimelineBuilder` | JSON | Implicit |
| `script.json` | Script generators | JSON | Implicit |
| `transcripts/transcript.json` | `transcription.adapter` | JSON | Implicit |
| `scenes/scene_index.json` | `scene_indexing.adapter` | JSON | Implicit |
| `reports/provider_manifest.json` | Orchestrator | JSON | Implicit |
| `reports/pipeline_status.json` | `quality.pipeline_status` | JSON | `pipeline_status.py` |
| `audio/voice.wav` | TTS adapter | WAV | N/A |
| `renders/final_render.mp4` | FFmpeg editor | MP4 | N/A |

**Gap**: No formal schema validation for most artifacts. Only `pipeline_status.py` and `grounding_contract.py` have validation.

---

## 7. Stage Independence Assessment

| Stage | Can Run Independently? | Blockers |
|-------|------------------------|----------|
| Transcription | ⚠️ Partial | Needs `project_meta.json.source_path` |
| Scene Indexing | ⚠️ Partial | Needs `source_path` |
| Movie Intelligence | ⚠️ Partial | Needs `transcript.json` + `scene_index.json` |
| Director (Grounded) | ❌ No | Called from orchestrator; needs `movie_index.json`; GPU model init inside |
| Director (Creative) | ❌ No | Called from orchestrator; needs `scene_index.json` + `transcript.json` |
| Editorial | ❌ No | Called from orchestrator; reads `director_plan.json` internals |
| Scene Ranking | ❌ No | Called from orchestrator; reads `director_plan.json` |
| Script Generation | ❌ No | Called from orchestrator; reads `director_plan.json` |
| Clip Extraction | ⚠️ Partial | Needs `selected_scenes.json` + `source_path` |
| TTS | ⚠️ Partial | Needs `script.json` + GPU model init |
| Visual Generation | ⚠️ Partial | Needs `asset_plan.json` + ComfyUI |
| Assembly | ❌ No | Called from orchestrator; reads timeline |
| QC | ⚠️ Partial | Called from orchestrator; reads all artifacts |

**Key Finding**: No stage can be run independently via CLI. All are embedded in `orchestrator.py`.

---

## 8. Configuration & Environment Variables

Current behavior controlled by environment variables (not config files):

| Variable | Purpose | Default |
|----------|---------|---------|
| `STUDIO_PROFILE` | Profile name (`local`, `colab-gpu`) | `local` |
| `REQUIRE_REAL_LLM` | Strict GPU mode for director | `false` |
| `REQUIRE_REAL_TTS` | Strict GPU mode for TTS | `false` |
| `GROUNDED_DIRECTOR` | Use grounded director | `false` |
| `EDITORIAL_MODE` | Enable editorial pipeline | `false` |
| `VISION_ENRICHER` | Scene enricher (`heuristic`, `qwen3vl`) | `heuristic` |
| `CREATIVE_DIRECTOR_ENABLED` | Use creative director | `false` |
| `DIRECTOR_NUM_CONCEPTS` | Number of concepts | `5` |
| `DIRECTOR_MIN_COVERAGE` | Min evidence coverage | `0.4` |
| `EDITORIAL_TARGET_SEC` | Target video duration | `90` |
| `SCRIPT_PROVIDER` | Script provider (`mock`, `qwen`) | `mock` |
| `RUN_TTS_BENCHMARK` | Run TTS benchmark | `false` |

**Gap**: No central config file; all env vars.

---

## 9. Storage Model

- **Hard-coded**: `ROOT / 'data' / project_id` (in `orchestrator.py:29`)
- **Colab notebooks**: Hard-code `/content/automovies/data/...`
- **No abstraction** for artifact root
- **No support** for Google Drive / external storage

---

## 10. Resume / Skip Logic

| Stage | Idempotent? | Skip Logic |
|-------|-------------|------------|
| Movie Intelligence | ✅ Yes | Checks `provenance.scene_enricher` matches requested |
| Others | ❌ No | Always re-runs |

---

## 11. Testing

- 348 tests collected (pytest)
- Integration test: `tests/integration/test_e2e_integration.py`
- Unit tests for each component
- Mock providers for GPU-less testing
- Strict mode tests (`REQUIRE_REAL_LLM=true`)

---

## 12. Identified Architectural Problems

| # | Problem | Impact |
|---|---------|--------|
| 1 | Monolithic orchestrator | Cannot run stages independently; all-or-nothing |
| 2 | No project manifest | No central record of stage status, artifact locations, versions |
| 3 | No artifact registry | Cannot query what exists, who produced it, if valid |
| 4 | No stage contracts | Stages assume specific JSON structures; no validation |
| 5 | No dependency graph | Stages run in fixed order; no explicit dependencies |
| 6 | No resume/skip | Re-runs completed stages (except movie intelligence) |
| 7 | Hard-coded paths | Cannot move artifacts to Google Drive / different machine |
| 8 | Env var config | No versioned, auditable configuration |
| 9 | No schema validation | Artifacts can drift; silent failures |
| 10 | GPU model caching | Class-level caches prevent multi-GPU / multi-process |
| 11 | No CLI for individual stages | Must run full pipeline |
| 12 | Colab notebooks hard-code paths | Cannot persist to Google Drive |

---

## 13. Migration Strategy (Planned)

See `FUTURE_DISTRIBUTED_ARCHITECTURE.md` for target state.

Key principles:
1. **Wrap existing stages** — don't rewrite logic, add thin wrappers
2. **Add manifest + registry** — central source of truth
3. **Create stage runner CLI** — `run_stage.py --project X --stage Y`
4. **Define stage dependencies** — explicit DAG
5. **Add schema validation** — JSON Schema for each artifact
6. **Abstract storage** — configurable artifact root
7. **Keep existing orchestrator working** — gradual migration

---

## 14. Data Samples

### `project_meta.json`
```json
{
  "project_id": "03cc7905-8507-4264-aa4d-9d45e7843508",
  "title": "Integration Test",
  "source_path": "C:\\Users\\hp\\Documents\\Default Project\\automovies\\tests\\fixtures\\test_speech.mp4"
}
```

### `manifest.json` (Movie Intelligence)
```json
{
  "scene_index_version": 3,
  "scene_enricher": "qwen3vl",
  "grouping": { "method": "deterministic_greedy", "max_scene_sec": 30.0, ... },
  "keyframes": true
}
```

### `director_plan.json` (Creative)
```json
{
  "project_id": "01c2b0a9-5c7b-4130-aa48-933ee72b87b4",
  "content_type": "scene_analysis",
  "topic": "Generated Director Plan",
  "thesis": "Explore how hello / world / very shape the emotional arc of scene-1.",
  "hook": "An engaging opening...",
  "tone": "analytical",
  "structure": [...],
  "visual_strategy": ["use_scene_clip"],
  "music_mood": "subtle_tension",
  "length_target_sec": 90
}
```

### `provider_manifest.json` (Pipeline Run)
```json
{
  "project_id": "...",
  "strict_mode": true,
  "profile": "colab-gpu",
  "transcription_real": true,
  "director_provider": "qwen",
  "director_real_generation": true,
  "editorial_mode": true,
  "editorial_plan_built": true,
  "tts_provider": "kokoro",
  "tts_real": true,
  "pipeline_status": "PASS",
  "pipeline_total_seconds": 451.23
}
```

---

## 15. Conclusion

The current architecture works for single-machine, single-GPU runs but cannot:
- Run stages on different GPUs/machines
- Resume after interruption
- Share artifacts across repositories
- Persist to Google Drive / cloud storage
- Validate artifact contracts
- Provide per-stage CLI

The refactoring must introduce:
1. **Project Manifest** — central project state
2. **Stage Registry** — registered stages with contracts
3. **Artifact Registry** — tracked artifacts with metadata
4. **Stage Runner** — CLI for independent stage execution
5. **Dependency Graph** — explicit DAG with validation
6. **Storage Abstraction** — configurable artifact root
7. **Schema Validation** — JSON Schema for each artifact