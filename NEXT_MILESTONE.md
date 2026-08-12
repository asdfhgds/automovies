# NEXT MILESTONE

**Last Updated**: Qwen3-VL vision scene enrichment built + tested locally;
GPU validation notebook ready — `notebooks/colab_vision_gpu.ipynb`.

## Status Tags

- **PROVEN**
  - Editorial planning subsystem (`src/editorial/`) — plan → evidence-aligned script → timeline → FFmpeg render → QC.
  - Movie intelligence layer (`src/movie_understanding/`) — analyzer, scene/character/event/semantic indexing (heuristic-first).
  - **Qwen3-VL vision scene enrichment** (`src/movie_understanding/`) —
    `keyframes.py` (FFmpeg per-scene frames, no OpenCV), `vision_enricher.py`
    (`Qwen3VLEnricher` implementing the `SceneEnricher` interface: lazy shared
    Qwen2.5-VL/Qwen3-VL load, GPU-optional, 4-bit option, per-field
    `provenance=qwen3vl`), `enrich_factory.py` (env-driven
    `VISION_ENRICHER`/`VISION_MODEL`/`VISION_DEVICE`/`VISION_DTYPE`/`VISION_MAX_FRAMES`),
    `REQUIRE_REAL_VISION` strict mode (`utils/strict.py`), wired into
    `MovieAnalyzer` (`attach_keyframes=`) and the orchestrator editorial path.
  - Local E2E: **164 tests pass** (26 new vision tests); vision layer degrades
    to heuristic with `unavailable(<reason>)` provenance when no GPU/keyframes,
    or hard-errors under strict mode.
  - Renderer fix: per-clip `fps=` normalization so `xfade` sees uniform timebases
    (fixes "First input link timebase … do not match … xfade timebase" on mixed-rate sources).

- **EXPERIMENTAL**
  - Qwen3-VL real-model path — code follows the proven Qwen provider patterns
    (class-level cache, `device_map="auto"`, SDPA, 4-bit NF4, chat-template
    image handling with processor API fallback) but has **never run on a real GPU**.
  - Evidence retrieval (semantic TF-IDF + lexical + dialogue overlap) — heuristics only, no embedder/LLM.
  - Editorial narration is deterministic around the real (Qwen) thesis — not yet an LLM editorial writer.
  - Real TTS (Kokoro priority) — quality/performance needs a real-GPU eval.

- **FAILED (baseline to beat)**
  - The first real-movie output (pre-editorial) was: clips in sequence + robotic TTS + paragraph subtitles.

- **KNOWN LIMITATIONS**
  - Vision fields (`location`, `actions`, `visual_description`, `themes`, `mood`)
    are only filled when `VISION_ENRICHER=qwen3vl` runs on CUDA; the heuristic
    enricher still leaves them `None` + provenance-flagged (honest, by design).
  - Semantic retrieval still uses TF-IDF only — it does not yet query the vision
    fields (e.g. "find the scene where they sit in a diner at night") despite
    those fields now existing.
  - Captions are chunked short (≤3 words) but timed by even distribution unless real word timestamps exist.
  - TTS emotion/pace control is approximate per provider (Kokoro voice+speed; not true expressive control).

- **CURRENT BOTTLENECK**: real-model proof. Local tests prove the vision layer's
  interface/fallback/strict contracts, but Qwen3-VL on a real GPU (keyframe →
  visual story card quality) is unverified. Second: retrieval doesn't consume
  the vision fields yet.

- **NEXT ACTION**
  1. Run `notebooks/colab_vision_gpu.ipynb` on a GPU runtime with a user-supplied
     movie (T4, `REQUIRE_REAL_VISION=true`, trim 60–180s first).
  2. Score 10+ scene cards: location / actions / visual_description / themes /
     mood grounded in each frame; log drift cases (crowds, fast motion, title cards).
  3. If vision QA passes → wire the vision fields into `EvidenceRetriever` /
     `SemanticIndex` so the editorial director can query *what's on screen*.
  4. Then run the full editorial GPU notebook with `VISION_ENRICHER=qwen3vl`.