# NEXT MILESTONE

**Last Updated**: Movie Intelligence validation milestone **completed on a real
movie**. `notebooks/colab_vision_gpu.ipynb` ran end-to-end on a Colab T4 with
`Qwen/Qwen2.5-VL-3B-Instruct` (bf16): project `5398e39c-d35b-481a-b580-42d7224732eb`,
120.078s window (Portuguese-dubbed clip; English is the norm), 33 scenes, 33/33 enriched (`provenance=qwen3vl`), ~3.6s/scene,
no OOM. Milestone verdicts (human): retrieval **1 GOOD / 2 PARTIAL / 5 WRONG**
(TF-IDF has no semantic reasoning) and temporal probe works but mostly unanchored.
183 fast tests pass. **Next bottleneck: retrieval semantics + temporal localization.**
Full artifacts preserved under `data/5398e39c-.../` (gitignored by design); the
retrieval human-verdict record is tracked in `reports/`.

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
  - Local E2E: **183 fast tests pass** (vision, artifacts, retrieval, editorial,
    movie-understanding suites).
  - **Real-movie run executed (Colab T4)**: project `5398e39c-d35b-481a-b580-
    42d7224732eb` — `Qwen/Qwen2.5-VL-3B-Instruct` (bf16), 120.078s window
    (Portuguese-dubbed clip; English is the normal case),
    33/33 scenes vision-enriched, ~3.6s/scene, no OOM; all artifacts (scene
    index v2, semantic index, understanding report, retrieval eval, temporal
    probe) preserved under `data/5398e39c-.../` + `reports/` (gitignored by
    design).

- **EXPERIMENTAL**
  - Qwen3-VL real-model path — **executed end-to-end on a real T4** (33 scenes,
    33/33 enriched, no OOM, ~3.6s/scene). Remaining gaps measured, not assumed:
    OCR/transcript misreads ("Talier Durden" for "Tyler Durden", hallucinated
    title cards, "THE JUST BROTHERS" for "Fight Club") and confidence-≠-correctness.
  - Temporal event localization — the probe runs OOM-free (per-frame single-image
    sampling), but events mostly anchor to `time_sec: 0.0` (only scene-32 got a
    real anchor, 116.43s). Real-frame quality is **weak**; not full-video event
    detection.
  - Evidence retrieval (semantic TF-IDF + lexical + dialogue overlap) — includes
    vision fields, but the 8-query human eval measured **1 GOOD / 2 PARTIAL /
    5 WRONG**: works for literal vocabulary (emphasized object), fails every
    narrative/thematic query. No embedder/LLM yet.
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
  - Semantic retrieval is TF-IDF only — measured on the real movie: **1 GOOD /
    2 PARTIAL / 5 WRONG** across the 8 milestone queries. It matches the *words*
    of the vision fields, not meaning; queries that rephrase without shared
    vocabulary miss (fate, contradiction, choice-vs-control all FAIL).
  - `device_map="auto"` on a 16GB T4 offloads some 7B-VL weights to CPU; the
    validated run used 3B bf16 (no offload, ~6s/scene). `VISION_DTYPE=4bit`,
    `VISION_ATTN=eager`, `VISION_MAX_IMAGE_PX=560` are the proven band-aids.
  - Temporal timestamps are model-approximate and frame-sampling-dependent —
    real-run evidence: most events anchored at `time_sec: 0.0`; treat any
    timestamp as "within this scene, unplaced".
  - OCR / transcript slips (real-run): "Talier Durden" for "Tyler Durden",
    hallucinated title cards ("Davty Flatcher", "THE JUST BROTHERS"); model
    confidence (0.88 on scene-29) does **not** imply correctness.
  - Captions are chunked short (≤3 words) but timed by even distribution unless real word timestamps exist.
  - TTS emotion/pace control is approximate per provider (Kokoro voice+speed; not true expressive control).

- **CURRENT BOTTLENECK**: retrieval *semantics*. The real run proved keyframes →
  Qwen3-VL → rich, honest story cards at ~3.6s/scene on a T4, but the retrieval
  step reads them by word overlap: query quality is 1/8 GOOD. Until retrieval
  has an embedding (or LLM) layer, the scene knowledge can't be reliably turned
  into an evidence-driven editorial plan find. Secondary bottleneck: temporal
  event localization (currently scene-scoped, not time-scoped).

- **NEXT ACTION** (after the completed validation run)
  1. **Add a semantic retrieval layer**: embed the story-card corpus (vision +
     transcript fields) with the same Qwen3-VL/LLM (or a small embedder) and
     re-run `scripts/evaluate_retrieval.py` — target flipping the measured
     1/8 GOOD upward and turning PARTIALs into GOODs without losing Q4.
  2. **Strengthen temporal localization**: extend `probe_temporal()` to emit a
     short "when does X happen" answer per event cluster rather than scene-scoped
     0.0 anchors; consider denser keyframes on the loudest segment only.
  3. **Add OCR/transcript guardrails**: spell-check pass over vision-extracted
     text/names against the transcript+title list ("Tyler Durden" ≠ "Talier
     Durden"); suppress clearly-hallucinated title cards.
  4. Feed the *validated* scene knowledge into the Creative Director / editorial
     plan: select scenes by the semantic + object evidence now proven to be real.
  5. Then the next GPU proof: real-movie + real-TTS editorial render
     (`colab_real_movie_tts.ipynb`), then evaluate several real videos and pick
     the next milestone from the largest visible weakness.
