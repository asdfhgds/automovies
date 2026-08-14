# Development Roadmap — Autonomous Movie Studio

## Current Status (Completed) ✅

### Editorial Pipeline (Evidence-Driven Cut)
- ✅ Movie intelligence layer (`src/movie_understanding/`): analyzer, scene
  enrichment (summary/topics/dialogue/characters/tone), character/event
  indexes, semantic index, movie memory
- ✅ Editorial planning (`src/editorial/`): EditorialPlan models, heuristic
  planner, evidence retrieval, evidence-aligned script (short captions),
  editorial timeline (excerpts), editorial FFmpeg renderer
- ✅ Orchestrator `EDITORIAL_MODE=true` wired end-to-end; QC covers the
  editorial cut
- ✅ 183 tests pass locally (vision + editorial + movie-understanding + artifacts + retrieval suites added)
- ✅ GPU validation notebooks: `colab_editorial_gpu.ipynb`, `colab_vision_gpu.ipynb` —
  `colab_vision_gpu.ipynb` **executed on a real T4** (project `5398e39c...`,
  `Qwen/Qwen2.5-VL-3B-Instruct` bf16, 33/33 scenes enriched, no OOM)

### Vision Scene Enrichment (Qwen3-VL) — Built + Tested, Real-Movie Run Pending
- ✅ `src/movie_understanding/keyframes.py` — FFmpeg keyframe extraction per
  scene window (no OpenCV dependency; window validation, missing-source errors)
- ✅ `src/movie_understanding/vision_enricher.py` — `Qwen3VLEnricher`
  implementing the `SceneEnricher` interface: lazy Qwen2.5-VL/Qwen3-VL load
  (shared class-level cache), GPU-optional, 4-bit NF4, SDPA, chat-template
  image handling with processor API fallback; fills `location` / `actions` /
  `objects` / `visual_description` / `visual_events` / `emotional_cues` /
  `themes` / `mood` / `cinematography` / `confidence` with per-field
  `provenance=qwen3vl`; degrades to heuristic (fields `None` +
  `unavailable(<reason>)`) without GPU or keyframes, or raises under
  `REQUIRE_REAL_VISION=true`
- ✅ **Device-dispatch fix**: `.model.to("cuda")` is never called after
  `device_map="auto"` dispatches the model (fixes "You can't move a model that
  has some modules offloaded to cpu or disk" seen on a real T4); regression
  tests for both the dispatched and un-dispatched load paths
- ✅ `src/movie_understanding/enrich_factory.py` — env-driven selection
  (`VISION_ENRICHER`, `VISION_MODEL`, `VISION_DEVICE`, `VISION_DTYPE`,
  `VISION_MAX_FRAMES`) + `require_real_vision` strict guard
- ✅ `MovieAnalyzer(attach_keyframes=True)` + orchestrator editorial path now
  run the vision enricher (`VISION_ENRICHER=qwen3vl`)
- ✅ **Vision-aware semantic retrieval**: `SemanticIndex` corpus includes
  location / actions / objects / visual_description / visual_events /
  emotional_cues / themes / mood / cinematography, so on-screen content is
  queryable
- ✅ **Director-facing artifacts**: `scene_index_v2.json` (versioned enriched
  scene index), `movie_memory/` bundle (index + scene v2 + semantic +
  characters + events + manifest), and `reports/movie_understanding_report.md`
  are written by the analyzer
- ✅ **Retrieval evaluation harness** (`scripts/evaluate_retrieval.py`):
  milestone queries → `reports/retrieval_evaluation.json` + `.md` with blank
  `human_assessment` fields (GOOD/PARTIAL/WRONG)
- ✅ **Temporal probe** (`Qwen3VLEnricher.probe_temporal`): orders visual
  events with approximate timestamps across N keyframes; honest when it cannot
  localize
- ✅ 44 vision/artifact/retrieval tests; **183 total passing**
- ✅ `scripts/colab_vision_setup.sh` + `notebooks/colab_vision_gpu.ipynb`
  (cells 7/7b/7c produce all artifacts, retrieval eval, temporal probe)
- ✅ **Real Qwen3-VL movie-understanding run EXECUTED** on a Colab T4 with a
  user-supplied movie (project `5398e39c-d35b-481a-b580-42d7224732eb`):
  `Qwen/Qwen2.5-VL-3B-Instruct` bf16, 120.078s window, 33/33 scenes enriched
  (`provenance=qwen3vl`), ~3.6s/scene, no OOM. Human verdicts: retrieval
  **1 GOOD / 2 PARTIAL / 5 WRONG** (TF-IDF word-overlap — the measured next
  weakness); temporal probe OOM-free but mostly unanchored. Full artifacts
  preserved under `data/5398e39c-.../` + `reports/` (gitignored by design).

### Foundation: Movie Understanding (Real & Tested)
- ✅ **Transcription**: Whisper/WhisperX with word-level timestamps
- ✅ **Scene Detection**: PySceneDetect for shot/scene boundaries
- ✅ **Transcript-Scene Mapping**: Associates dialogue with scenes
- ✅ **Deterministic Director**: Generates thesis from scene index
- ✅ **Scene Ranking**: Lexical/keyword-based scoring
- ✅ **Scene Selection**: Picks best-ranked valid scene
- ✅ **Clip Extraction**: FFmpeg-based safe extraction
- ✅ **Pipeline Orchestration**: Full end-to-end workflow

### Creative Generation (Framework Complete, Mock LLM)
- ✅ **CreativeDirector Framework**: Memory, Critic, Provider interface
- ✅ **ConceptCritic**: 6-dimensional scoring heuristics
- ✅ **MockLLMProvider**: Deterministic testing without API calls
- ✅ **Orchestrator Integration**: Creative mode routing and fallback
- ✅ **Tests**: 21 passing (10 unit + 4 E2E + 7 existing)
- ✅ **Documentation**: Full CREATIVE_DIRECTOR_GUIDE.md

### Real LLM (Qwen, On-Device / GPU) — Completed
- ✅ **QwenProvider** (director/providers/qwen.py): real Transformers generation on CUDA,
  lazy loading, `generate_text`, load + generation timing
- ✅ **CreativeDirector on real Qwen**: multi-scene, evidence-driven production planning
- ✅ **Real Qwen script writer** (`src/script/qwen_writer.py`): narration sections from
  director plan + selected scenes, canonical `script.json` schema, no silent fallback
- ✅ **Strict GPU mode** (`REQUIRE_REAL_LLM=true`): `require_cuda()` + provider
  hard-fail guards; deterministic/mock providers are refused on GPU boxes
- ✅ **Provider manifest**: `provider_manifest.json` records provider, model, device,
  and load/generation timings each run
- ✅ **Default model**: `Qwen/Qwen3-4B-Instruct-2507` (fits T4 16GB); `30B-A3B` for A100
- ✅ **Colab artifacts**: `notebooks/colab_qwen_validation.ipynb` (14 cells) +
  `scripts/colab_setup.sh` (idempotent setup)
- ✅ **Tests**: 17 new (strict mode + Qwen script writer), 69 total passing

### Quality Metrics
- **Test Coverage**: 69 tests, ~16s runtime
- **No Regressions**: All existing functionality preserved
- **Architecture**: Modular, pluggable, fault-tolerant

---

## Phase 1: Real LLM Integration (Recommended Next)

### Goal
Replace MockLLMProvider with real LLM to generate actual creative concepts.

### Effort: ~20-40 hours

### Tasks

#### 1.1 Choose & Configure LLM Provider (4 hours)
- [ ] Choose provider: **Anthropic Claude** (recommended) OR OpenAI OR Replicate OR Ollama
- [ ] Acquire API credentials
- [ ] Set up environment configuration
- [ ] Document setup in README

**Recommendation: Anthropic Claude**
- Best reasoning capabilities (ideal for analysis)
- Available via AWS Bedrock or direct API
- ~$3 per 1M input tokens, ~$15 per 1M output tokens
- Test with $5-10 free credits

#### 1.2 Implement Real Provider (12-16 hours)
- [ ] Create `src/director/providers/anthropic_provider.py` (or chosen provider)
- [ ] Implement `generate_concepts()` with real LLM calls
- [ ] Implement `generate_production_plan()` with real LLM
- [ ] Add error handling, retry logic, timeouts
- [ ] Add request/response logging for debugging
- [ ] Write comprehensive docstrings

**Deliverable**: Provider that generates 3-5 real concepts in <30s

#### 1.3 Integration Testing (4-6 hours)
- [ ] Create `tests/test_anthropic_provider_integration.py`
- [ ] Test concept generation with real LLM
- [ ] Test production plan generation
- [ ] Test error handling (invalid API key, timeout, rate limit)
- [ ] Test fallback to mock provider on LLM failure
- [ ] Mark tests with `@pytest.mark.requires_anthropic` to skip by default

#### 1.4 End-to-End Validation (2-4 hours)
- [ ] Run full pipeline with real LLM
- [ ] Verify director_plan.json with real LLM thesis
- [ ] Compare mock vs real output (quality, specificity)
- [ ] Document example outputs (3-5 real concepts)
- [ ] Update PROJECT_STATUS.md with results

#### 1.5 Documentation (2-3 hours)
- [ ] Update CREATIVE_DIRECTOR_GUIDE.md with real provider section
- [ ] Add cost estimation guide (how to monitor API usage)
- [ ] Add troubleshooting section for common errors
- [ ] Add performance benchmarks (latency, cost per concept)

### Success Criteria
- [ ] Pipeline runs with real LLM (CREATIVE_DIRECTOR_ENABLED=true)
- [ ] Generated thesis is specific (not generic)
- [ ] Scene ranking works with real thesis
- [ ] Full pipeline produces valid artifacts
- [ ] Tests pass (integration marked appropriately)
- [ ] Documentation complete and clear

### Risk Mitigation
- Start with cheap/free tier API credits
- Set strict token limits to control costs
- Implement caching to avoid duplicate calls
- Fallback to mock provider if LLM unavailable

---

## Phase 2: Voice Generation & Scriptwriting (Parallel or Following Phase 1)

### Goal
Generate narration script and synthesize voice for video essay.

### Effort: ~30-50 hours

### Current State
- ✅ Real Qwen script generation (`src/script/qwen_writer.py`) integrated into the orchestrator
- ✅ TTS adapter with real providers: Kokoro (default), Chatterbox, Qwen3-TTS
- ✅ Director-controlled narration properties (tone/emotion/pace/energy/intensity)
- ✅ Cinematic audio mix: film/music ducking, loudnorm, true-peak limiter, burned subtitles
- ✅ TTS benchmark CLI + GPU notebook ready
- ⏳ Real-movie + real-TTS GPU run pending (user-supplied movie)

### Tasks

#### 2.1 Script Generation (16-20 hours) ✅ COMPLETE
- [x] Design script format (segments, timing, speaker notes, subtitles)
- [x] Implement script writer using Qwen (`src/script/qwen_writer.py`)
- [x] Scene-matched narration (sync with visual elements) + subtitle generation
- [x] Tests (mock LLM testing, structure validation)

**Example Output**:
```json
{
  "segments": [
    {
      "start_sec": 0,
      "end_sec": 5,
      "narration": "Every story is architecture...",
      "subtitle": "Every story is architecture...",
      "scene_references": ["scene_001"]
    }
  ]
}
```

#### 2.2 TTS Integration (12-20 hours) ✅ COMPLETE
- [x] Evaluate TTS options → Kokoro-82M (default), Chatterbox, Qwen3-TTS
- [x] Implement TTS provider interface + factory (`TTS_PROVIDER` switch)
- [x] Voice selection/customization (provider-specific: Kokoro voices, Chatterbox clone, Qwen ref speech)
- [x] Narration-property handling (emotion/pace/energy per provider capability)
- [x] Tests (provider unit tests, strict-mode tests, adapter tests)

#### 2.3 Script-TTS Sync (4-8 hours) ✅ COMPLETE
- [x] Narration pacing honored (speed per provider)
- [x] Silence padding in render pipeline
- [x] Sync verified via local E2E render (narration fits timeline, no clipping)

#### 2.4 Testing & Validation (2-4 hours) ⏳ PARTIAL
- [x] TTS provider unit tests + strict-mode tests (no model load)
- [x] Benchmark CLI across providers → `reports/tts_benchmark.json`
- [ ] Test real TTS output quality on GPU (real-movie notebook run)
- [ ] Integration test: full pipeline with real script + real TTS (pending GPU run)

### Success Criteria
- [x] Script generation produces coherent narration matching director thesis
- [x] TTS providers integrated behind one interface; mock rejected in strict mode
- [x] Narration duration fits production plan timing (local E2E render OK)
- [x] Full pipeline: video → director → script → TTS → rendered output (validated locally with mocks; real-TTS GPU run pending)

---

## Phase 3: Visual Generation & Composition (Following Phase 2)

### Goal
Generate or find visual assets that match the creative concept.

### Effort: ~40-60 hours

### Current State
- ✅ Visual generation stub exists (`src/visual_generation/comfyui_client.py`)
- ❌ No actual image/video generation
- ❌ No asset sourcing logic

### Tasks

#### 3.1 Asset Strategy (4-8 hours)
- [ ] Decide approach:
  - **Option A**: Use only extracted video clips (cheapest, limited)
  - **Option B**: Generate synthetic images (Stable Diffusion, Midjourney)
  - **Option C**: Source licensed stock footage (Getty, Envato)
  - **Option D**: Mix of above (recommended)
- [ ] Design visual asset format and pipeline

#### 3.2 Stock Footage Integration (12-16 hours)
- [ ] Integrate with stock footage API (Envato, Getty, Pexels)
- [ ] Implement asset matching (theme, mood, duration)
- [ ] Add license checking and attribution
- [ ] Write tests

#### 3.3 Image Generation (16-24 hours)
- [ ] Integrate with image gen model:
  - Stable Diffusion (local or API)
  - Midjourney (API)
  - DALL-E 3 (OpenAI API)
- [ ] Implement prompt generation from director thesis
- [ ] Handle batch generation (multiple variants)
- [ ] Add quality filtering
- [ ] Write tests

#### 3.4 Video Composition (8-12 hours)
- [ ] Arrange assets in sequence matching script timing
- [ ] Add transitions and effects
- [ ] Overlay text (titles, quotes, captions)
- [ ] Write tests

### Success Criteria
- [ ] Visuals match director thesis/tone
- [ ] Timing synchronized with narration
- [ ] Asset licensing valid
- [ ] Full pipeline produces visually coherent output

---

## Phase 4: YouTube Integration & Publishing

### Goal
Publish generated content to YouTube (optional, if desired).

### Effort: ~20-30 hours

### Tasks
- [ ] YouTube API integration
- [ ] Metadata generation (title, description, tags, thumbnail)
- [ ] Upload with proper licensing/credits
- [ ] Analytics tracking
- [ ] Scheduling (publish at optimal times)

---

## Phase 5: Feedback Loops & Optimization

### Goal
Improve quality and reduce costs through iteration.

### Tasks
- [ ] Human feedback collection (curator reviews concepts)
- [ ] Metrics tracking (engagement, retention, view count)
- [ ] A/B testing (different concepts, scripts, visuals)
- [ ] Cost optimization (caching, batching, model selection)
- [ ] Performance optimization (latency, throughput)

---

## Technology Stack Reference

### Current (Implemented)
- **Transcription**: Whisper (openai), WhisperX
- **Scene Detection**: PySceneDetect
- **Director (Deterministic)**: Lexical keyword matching
- **Director (Creative)**: Qwen3-4B-Instruct-2507 via Transformers on CUDA (or mock locally)
- **Script Generation**: Qwen3-4B-Instruct-2507 (`script/qwen_writer.py`) or deterministic
- **Clip Extraction**: FFmpeg
- **Video Assembly**: FFmpeg
- **Testing**: pytest, mock LLM provider

### To Be Integrated
- **Image Generation**: Stable Diffusion, ComfyUI, Midjourney, DALL-E 3
- **Stock Footage**: Envato, Getty, Pexels
- **Video Composition**: advanced montage/visual effects beyond FFmpeg+libass
- **YouTube**: youtube-dl, PyYouTube, selenium

---

## Estimated Timeline

| Phase | Effort | Timeline | Dependencies |
|-------|--------|----------|--------------|
| 1. Real LLM | 20-40h | 1-2 weeks | None |
| 2. Script & TTS | 30-50h | 2-3 weeks | Phase 1 | ✅ done (GPU real-TTS run pending) |
| 3. Visual Gen | 40-60h | 3-4 weeks | Phase 2 |
| 4. YouTube | 20-30h | 1-2 weeks | Phase 3 (optional) |
| 5. Optimization | 20-30h | 2-3 weeks | All phases |
| **Total** | **130-210h** | **8-14 weeks** | - |

---

## Priority Queue

### High Priority (Do First)
1. ✅ Real Qwen LLM provider (Phase 1 — DONE, tested on GPU path)
2. ✅ Script generation via LLM (Qwen) — subsumed by the real script writer
3. ✅ Real TTS (Phase 2) — Kokoro / Chatterbox / Qwen3-TTS + audio mix + benchmark
4. ✅ Editorial pipeline (movie intelligence + evidence-driven editorial plan/script/timeline/render) — local E2E proven
5. ✅ **Validated Movie Intelligence on the real movie** — `colab_vision_gpu.ipynb`
   normal run on a T4: 33/33 scenes vision-enriched, ~3.6s/scene, no OOM;
   retrieval eval 1 GOOD / 2 PARTIAL / 5 WRONG, temporal probe OOM-free/weak
   anchors (project `5398e39c...`). Next: semantic retrieval layer + feed the
   validated scene knowledge into the Creative Director
6. ⏳ Evaluate several real videos; pick next milestone from the largest visible weakness

### Medium Priority (Do After)
4. Integrate stock footage API
5. Add cost optimization (caching, batching)
6. Implement metrics tracking

### Low Priority (Do Last)
7. YouTube publishing
8. Advanced feedback loops
9. Performance optimization for scale

---

## Definition of Done

### Phase 1 (DONE — Real LLM)
- [x] Real LLM provider integrated and tested (Qwen, on-device Transformers)
- [x] Mock provider available as fallback (non-strict mode)
- [x] Documentation complete (CREATIVE_DIRECTOR_GUIDE.md, COLAB_INSTRUCTIONS.md updated)
- [x] No regressions (69 tests passing)
- [x] End-to-end path exercised locally (strict mode fails safely without CUDA)
- [x] Colab execution of the validation notebook — PASSED on real T4 with Qwen/Qwen3-4B-Instruct-2507 (real concepts + narration, no OOM)
- [x] PROJECT_STATUS.md updated

### Phase 2 (Voice Generation & Scriptwriting) — TTS COMPLETE, GPU run pending
- [x] Real script generation w/ Qwen (`src/script/qwen_writer.py`)
- [x] TTS integration: Kokoro (default), Chatterbox, Qwen3-TTS behind one interface
- [x] Director-controlled narration properties (tone/emotion/pace/energy/intensity)
- [x] TTS benchmark CLI + `RUN_TTS_BENCHMARK=true` hook (`reports/tts_benchmark.json`)
- [x] Strict real-TTS mode (`REQUIRE_REAL_TTS=true`, no mock audio)
- [x] Cinematic audio mix: film/music ducking, loudnorm, true-peak limiter, burned subtitles
- [x] GPU notebook `notebooks/colab_real_movie_tts.ipynb` + `scripts/colab_tts_setup.sh`
- [ ] **Run the real-movie GPU pipeline** (user-supplied movie, T4/A100) → real MP4
- [ ] Evaluate several real videos; document weaknesses → decide next milestone

### Handoff Notes for Next Developer

**What's Ready:**
- Complete mock framework (memory, critic, provider interface)
- Full test suite (21 tests)
- Orchestrator integration (just swap provider)
- Clear documentation (CREATIVE_DIRECTOR_GUIDE.md)

**What to Do:**
1. Follow CREATIVE_DIRECTOR_GUIDE.md "Integrating a Real LLM Provider" section
2. Choose LLM (Anthropic recommended)
3. Implement provider class (example code in guide)
4. Test with end-to-end pipeline
5. Update documentation with results

**Environment Setup:**
```bash
# Clone repo
git clone <repo>
cd automovies

# Install dependencies
pip install -r requirements.txt

# Set LLM credentials
export ANTHROPIC_API_KEY="..."  # or OPENAI_API_KEY, etc.

# Run with creative director
export CREATIVE_DIRECTOR_ENABLED=true
python src/main.py init --title "Test" --source tests/fixtures/test_speech.mp4
python src/main.py run --project-id <id>
```

---

## Questions? See Also

- `PROJECT_STATUS.md` — Architecture and current implementation details
- `CREATIVE_DIRECTOR_GUIDE.md` — How to integrate real LLM providers
- `tests/test_creative_director.py` — Unit test examples
- `tests/test_creative_director_e2e.py` — Integration test examples
- `src/director/` — Framework source code (well-commented)
