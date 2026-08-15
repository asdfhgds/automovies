# NEXT MILESTONE

**Last Updated**: Movie-grounded Creative Director milestone **built and
unit-tested; real-Qwen clip validation gated/pending**.

What this milestone does (see `PROJECT_STATUS.md` for details): the Director now
reads the existing Movie Intelligence (`movie_index.json` → `SceneFacts`), builds
a compact, token-limited, fact-grounded context, asks real Qwen for **5 genuinely
different concepts** (each with `required_evidence`), **rejects** any concept
whose evidence is not actually present in the scenes (plus generic-thesis
rejection, with regeneration), scores the survivors with `ConceptCritic` +
evidence coverage, selects the strongest, and emits a scene-aware plan and an
inspectable `reports/director_reasoning.md`.

**Explicitly NOT implemented** (per milestone): evidence verifier, new embedding/
retrieval model, TTS replacement, subtitle redesign, generative video/image,
YouTube automation, and **script wiring** — the pipeline stops at the selected
concept + director plan so the Director can be validated on its own.
**221 fast tests pass** (21 new grounded-director tests).

Prior milestone (kept): Movie Intelligence validation **completed on a real
movie**. `notebooks/colab_vision_gpu.ipynb` ran end-to-end on a Colab T4 with
`Qwen/Qwen2.5-VL-3B-Instruct` (bf16): project `5398e39c-d35b-481a-b580-42d7224732eb`,
120.078s window (Portuguese-dubbed clip; English is the norm), 33 scenes, 33/33 enriched (`provenance=qwen3vl`), ~3.6s/scene,
no OOM. Milestone verdicts (human): retrieval **1 GOOD / 2 PARTIAL / 5 WRONG**
(TF-IDF has no semantic reasoning) and temporal probe works but mostly unanchored.
**Dense-embedding retrieval is now built and measured** (MiniLM on the real
corpus: scores ~0.02→0.20, tension surface scene-30 at #2, object stays #1; the
English-only embedder only partially helps the PT clip). 191 fast tests pass.
**Next bottleneck: stronger retrieval semantics (multilingual/LLM) + temporal
localization.**
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
  - **Dense-embedding retrieval (new, measured)**: `SemanticIndex.build(
    ..., embedder=)` dense path (cosine; TF-IDF stays default),
    `embedding_retriever.py` (lazy sentence-transformers, env-driven factory,
    honest failure), `evaluate_retrieval.py --method embedding` (never silently
    falls back). Local measurement on the real corpus (MiniLM): scores
    ~0.02→0.20 across the 8 queries, tension surfaces scene-30 (gun-to-mouth)
    at #2, emphasized-object stays #1 — but the English-only embedder only
    partially bridges the Portuguese-dubbed clip's narrative queries (use
    `paraphrase-multilingual-MiniLM-L12-v2` for non-English).
  - **Movie-grounded Creative Director (new)**: `SceneFacts` normalizes the
    existing Movie Intelligence; `DirectorContextBuilder` makes a compact,
    token-limited, fact-grounded context; `EvidenceAnalyzer` grounds
    `required_evidence` to real scenes (coverage HIGH/MED/LOW, no new retrieval
    layer); `MovieGroundedDirector` generates 5 diverse concepts → rejects
    generic / un-evidenced ideas (with regeneration) → `ConceptCritic` +
    coverage → select → scene-aware plan → `CreativeMemory` →
    `reports/director_reasoning.md`. Hallucination-safe (unknown characters
    marked `unknown_character_01 (low confidence)`; vocab limited to facts that
    exist). Stops at the plan — NOT wired to the script stage.
- Local E2E: **221 fast tests pass** (vision, artifacts, retrieval, editorial,
    movie-understanding, semantic-embedding, grounded-director suites).
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
  - Evidence retrieval — TF-IDF (word-overlap) and now dense embeddings; the
    TF-IDF human eval was **1 GOOD / 2 PARTIAL / 5 WRONG**, and the MiniLM
    embedding pass improved/reranked (tension→scene-30 #2, object stays #1)
    but still misses narrative/thematic queries on this clip. No LLM-based
    retrieval yet.
  - Editorial narration is deterministic around the real (Qwen) thesis — not yet an LLM editorial writer.
  - Real TTS (Kokoro priority) — quality/performance needs a real-GPU eval.
  - **Movie-grounded Creative Director — real-Qwen clip validation pending**: the
    director + evidence pipeline + reasoning report are built and unit-tested
    (mock LLM), but the real-Qwen run on the validated movie
    (`bc6384be-...`, via `scripts/run_director_validation.py` → Cell 7d) is
    gated and has NOT yet been executed/measured on a GPU. Until then, whether
    real concepts are specific, non-generic, and grounded is unproven.

- **FAILED (baseline to beat)**
  - The first real-movie output (pre-editorial) was: clips in sequence + robotic TTS + paragraph subtitles.

- **KNOWN LIMITATIONS**
  - Vision fields (`location`, `actions`, `objects`, `visual_description`,
    `visual_events`, `emotional_cues`, `themes`, `mood`, `cinematography`,
    `confidence`) are only filled when `VISION_ENRICHER=qwen3vl` runs on CUDA;
    the heuristic enricher still leaves them `None` + provenance-flagged
    (honest, by design).
  - Semantic retrieval is TF-IDF by default with a dense path now available
    (`--method embedding`, sentence-transformers). **Measured on the real
    movie**: TF-IDF was **1 GOOD / 2 PARTIAL / 5 WRONG**; MiniLM embeddings
    lifted scores ~0.02→0.20 and re-ranked (tension→gun-to-mouth at #2; object
    stays #1) but English-only embeddings only partially help PT clips — set
    `RETRIEVAL_EMBEDDER_MODEL` to a multilingual model (or add LLM retrieval)
    for non-English; narrative/thematic queries still miss.
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
  - The Director's evidence matching is **lexical only** (phrase/token overlap
    against the scene facts — deliberately no new retrieval system). On the
    validated western clip (`bc6384be`) no characters were identified and the
    dialogue is garbled, so concepts must lean on location/objects/mood/themes;
    a concept whose claims don't literally appear is rejected (this is a feature,
    but it means semantically-paraphrased-but-true evidence can be under-counted).

- **CURRENT BOTTLENECK**: **validating the Movie-grounded Creative Director on the
  real movie with real Qwen.** The infrastructure is built and unit-tested
  (221 fast tests). The unproven link is that real Qwen produces 5 genuinely
  different, grounded, non-generic concepts on the real `bc6384be` intelligence,
  and that the evidence gate rejects only the right things. Secondary bottleneck
  (prior milestone, kept): retrieval *semantics* — TF-IDF scored 1/8 GOOD and
  MiniLM helped but narrative/thematic queries still miss on non-English clips.

- **NEXT ACTION** (validate the grounded director, then wire it to the script)
  1. **Execute the real-Qwen grounded-director run** on the validated movie:
     `python scripts/run_director_validation.py --project data/bc6384be-...` on
     a Colab T4 (or Notebook Cell 7d). Inspect `reports/director_reasoning.md`:
     are the 5 concepts specific, non-generic, grounded in actual scenes, and
     meaningfully different (diversity metric)? Which concepts were rejected and
     why? Fill/record the per-milestone fields (§16): context size, model,
     concepts generated, selected concept, evidence coverage, failures, known
     hallucination cases, next bottleneck.
  2. **Iterate on grounding quality** if the real run under-delivers: relax/refine
     `required_evidence` matching (stemming / synonym-aware lexical matching, not a
     new retrieval system), or improve the compact context so Qwen cites real scenes.
  3. **Do NOT wire the script stage yet** (§11). Once the director is validated
     as genuinely specific + grounded, the *next* milestone connects the selected
     concept + evidence strategy to narrative/editorial generation.
  4. (Background, prior milestone) stronger retrieval semantics + temporal
     localization remain open research items.
