# NEXT MILESTONE

**Last Updated**: Movie Intelligence validation *prepared* — all tooling to run
the real Qwen3-VL scene analysis + retrieval eval on a Colab T4 is built and
tested (178 fast tests pass). The real-movie run itself is **pending** (needs a
user-supplied movie on a GPU runtime).

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
  - **Device-dispatch fix**: never call `.model.to("cuda")` after
    `device_map="auto"` has dispatched the model (accelerate raises "You can't
    move a model that has some modules offloaded to cpu or disk"). Proven on a
    real T4 run; regression-tested in `test_vision_enrichment.py`.
  - **Extended scene schema**: story cards now carry `objects`,
    `visual_events` (approx-timestamped), `emotional_cues`, `cinematography`,
    and `confidence` (all vision/LLM-gated with honest provenance; heuristic
    leaves them `None`).
  - **Vision-aware retrieval**: `SemanticIndex` corpus now consumes location /
    actions / objects / visual_description / visual_events / emotional_cues /
    themes / mood / cinematography, so "find the scene where they sit in a diner
    at night" actually matches on-screen content.
  - **Director-facing artifacts**: `scene_index_v2.json` (versioned enriched
    index), `movie_memory/` bundle (index + scene v2 + semantic + characters +
    events + manifest), and `reports/movie_understanding_report.md` are written
    by the analyzer (regression-tested).
  - **Retrieval evaluation harness**: `scripts/evaluate_retrieval.py` runs the
    milestone's natural-language queries and writes
    `reports/retrieval_evaluation.json` + `.md` with blank
    `human_assessment` (GOOD/PARTIAL/WRONG) fields for a human reviewer.
  - **Temporal probe**: `Qwen3VLEnricher.probe_temporal()` orders visual events
    with *approximate* timestamps from N keyframes spread across the scene; it
    reports honestly when it cannot localize an event to a time.
  - Local E2E: **178 fast tests pass** (vision, artifacts, retrieval, editorial,
    movie-understanding suites).

- **EXPERIMENTAL**
  - Qwen3-VL real-model path — the `.to()` fix is proven on a real T4 (the
    previous run loaded the 7B model with offload to CPU without crashing), but
    the *full* vision scene-analysis + retrieval-eval run has **not yet been
    executed end-to-end against the real movie**.
  - Temporal event localization — `probe_temporal` exists and is unit-tested,
    but real-frame quality is unverified (this milestone's job).
  - Evidence retrieval (semantic TF-IDF + lexical + dialogue overlap) — now
    includes vision fields; no embedder/LLM yet.
  - Editorial narration is deterministic around the real (Qwen) thesis — not yet an LLM editorial writer.
  - Real TTS (Kokoro priority) — quality/performance needs a real-GPU eval.

- **FAILED (baseline to beat)**
  - The first real-movie output (pre-editorial) was: clips in sequence + robotic TTS + paragraph subtitles.

- **KNOWN LIMITATIONS**
  - Vision fields (`location`, `actions`, `objects`, `visual_description`,
    `visual_events`, `emotional_cues`, `themes`, `mood`, `cinematography`,
    `confidence`) are only filled when `VISION_ENRICHER=qwen3vl` runs on CUDA;
    the heuristic enricher still leaves them `None` + provenance-flagged
    (honest, by design).
  - Semantic retrieval is TF-IDF only — it matches the *words* of the vision
    fields, not true semantic embeddings. Queries that rephrase without any
    shared vocabulary will miss (this is exactly what the retrieval eval
    measures).
  - `device_map="auto"` on a 16GB T4 offloads some 7B-VL weights to CPU —
    slower per-scene generation, and peak VRAM < full model. Documented, not
    hidden.
  - Temporal timestamps are model-approximate and frame-sampling-dependent
    (evenly spaced keyframes, not full-video event detection).
  - Captions are chunked short (≤3 words) but timed by even distribution unless real word timestamps exist.
  - TTS emotion/pace control is approximate per provider (Kokoro voice+speed; not true expressive control).

- **CURRENT BOTTLENECK**: real-model proof of *understanding quality*. Local
  tests prove the interfaces/fallback/strict contracts and that the retrieval
  index consumes vision fields, but Qwen3-VL on a real GPU (keyframe → visual
  story card quality, retrieval GOOD/PARTIAL/WRONG, temporal localization) is
  unverified against the real movie.

- **NEXT ACTION** (validation milestone, in order)
  1. Run `notebooks/colab_vision_gpu.ipynb` on a GPU runtime with a user-supplied
     movie (T4, `REQUIRE_REAL_VISION=true`; trim 60–180s first; keep
     `device_map=auto`; drop `VISION_DTYPE=4bit` if generation OOMs).
  2. Cells 7–9: inspect 10+ scene cards (location/actions/objects/visual
     description/events/cues/themes/mood/cinematography/confidence grounded in
     each frame), log drift cases.
  3. Cell 7b: run retrieval eval → fill `human_assessment`
     (GOOD/PARTIAL/WRONG) in `reports/retrieval_evaluation.json`; inspect top
     hits manually.
  4. Cell 7c: run the temporal probe on 3 scenes; record where exact timing
     holds and where it is weak (document, don't pretend).
  5. Compare the three scenes the *old* pipeline selected: does the new system
     know more (visual analysis, characters, actions, events, dialogue, emotion,
     visual details) than the old transcript/generic-summary output?
  6. Record performance (GPU, VRAM, model, dtype, device_map, load time, scene
     count, frames, analysis time, avg time/scene, offload note).
  7. Record the milestone verdict honestly (PROJECT_STATUS.md): provider
     COMPLETE; real scene analysis PASS/FAIL; retrieval GOOD/PARTIAL/WRONG;
     temporal GOOD/PARTIAL/WRONG; T4 VRAM; model; runtime; limitations.
  8. Do NOT start TTS/script/image/video/editing work yet. If intelligence is
     useful, the next task is Movie Intelligence → Creative Director → evidence
     retrieval → editorial plan using the validated scene knowledge.
