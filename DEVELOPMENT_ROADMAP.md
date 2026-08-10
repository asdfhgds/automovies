# Development Roadmap — Autonomous Movie Studio

## Current Status (Completed) ✅

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
- ✅ Script generation stub exists (`src/script/writer.py`)
- ✅ TTS adapter stub exists (`src/audio/tts_adapter.py`)
- ❌ No real script generation logic
- ❌ No real TTS provider integrated

### Tasks

#### 2.1 Script Generation (16-20 hours)
- [ ] Design script format (segments, timing, speaker notes, subtitles)
- [ ] Implement script writer using LLM (Claude/GPT-4 recommended)
  - Input: director_plan, scene_index, transcript, tone
  - Output: narration script with timing
- [ ] Add scene-matched narration (sync with visual elements)
- [ ] Generate subtitles/captions
- [ ] Write tests (mock LLM testing, structure validation)

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

#### 2.2 TTS Integration (12-20 hours)
- [ ] Evaluate TTS options:
  - Qwen3-TTS (low latency, good quality)
  - Chatterbox (multilingual, expressive)
  - ElevenLabs (natural voices)
  - Google Cloud TTS (reliable)
- [ ] Implement TTS adapter
- [ ] Add voice selection/customization
- [ ] Handle multilingual narration
- [ ] Write tests

#### 2.3 Script-TTS Sync (4-8 hours)
- [ ] Ensure narration timing matches intended duration
- [ ] Handle narration pacing (fast, normal, slow)
- [ ] Add silence/pauses between sections
- [ ] Test sync with video timeline

#### 2.4 Testing & Validation (2-4 hours)
- [ ] Test script generation with real LLM
- [ ] Test TTS output quality
- [ ] Test sync with visual elements
- [ ] Integration test: full pipeline with script + TTS

### Success Criteria
- [ ] Script generation produces coherent narration matching director thesis
- [ ] TTS produces natural-sounding voice
- [ ] Narration duration matches production plan timing
- [ ] Full pipeline: video → director → script → TTS → rendered output

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
- **TTS**: Qwen3-TTS, Chatterbox, ElevenLabs
- **Image Generation**: Stable Diffusion, Midjourney, DALL-E 3
- **Stock Footage**: Envato, Getty, Pexels
- **Video Composition**: FFmpeg, moviepy, OpenCV
- **YouTube**: youtube-dl, PyYouTube, selenium

---

## Estimated Timeline

| Phase | Effort | Timeline | Dependencies |
|-------|--------|----------|--------------|
| 1. Real LLM | 20-40h | 1-2 weeks | None |
| 2. Script & TTS | 30-50h | 2-3 weeks | Phase 1 |
| 3. Visual Gen | 40-60h | 3-4 weeks | Phase 2 |
| 4. YouTube | 20-30h | 1-2 weeks | Phase 3 (optional) |
| 5. Optimization | 20-30h | 2-3 weeks | All phases |
| **Total** | **130-210h** | **8-14 weeks** | - |

---

## Priority Queue

### High Priority (Do First)
1. ✅ Real Qwen LLM provider (Phase 1 — DONE, tested on GPU path; Colab run pending)
2. ✅ Script generation via LLM (Qwen) — subsumed by the real script writer
3. ⏳ Execute `notebooks/colab_qwen_validation.ipynb` on T4 and attach the manifest
4. ⏳ Real TTS (Phase 2) — **NEXT TASK**: Qwen3-TTS / Chatterbox / Kokoro

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
- [ ] Colab execution of the validation notebook (final proof on T4/A100)
- [x] PROJECT_STATUS.md updated

### Phase 2 (NEXT — Voice Generation & Scriptwriting)
- [ ] Real script generation w/ Qwen — **DONE** (see `src/script/qwen_writer.py`)
- [ ] TTS integration (Qwen3-TTS / Chatterbox / Kokoro)
- [ ] Script-TTS sync and pacing
- [ ] Full pipeline: video → director (Qwen) → script (Qwen) → TTS → rendered output

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
