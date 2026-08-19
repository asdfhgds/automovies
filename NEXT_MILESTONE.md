# NEXT MILESTONE

**Last Updated**: Movie-grounded Creative Director **real-Qwen run EXECUTED on a
Colab T4 — VERDICT: FAIL**. All 18 generated concepts were rejected
(`LOW 0/3 matched`); the generator **hallucinated a different film** (father/son
family drama — waiting room, broken clock, dinner table, photograph, kitchen —
none grounded in the facts), while the rejection gate behaved correctly.
Fix required on the demonstrated problem: anchor generation to the verbatim
grounded vocabulary (see below).

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
**272 fast tests pass** (23 new grounded-director tests).

**Real-Qwen run (Colab T4, executed)**: `Qwen/Qwen3-4B-Instruct-2507` (4bit,
cuda), model load 49.03s, 3 LLM calls, 12 substitutes, wall clock 427.05s.
**Generated 0 / rejected 18 / selected NONE, diversity 0.000.** Every one of the
18 rejected concepts shows `coverage LOW (0/3 matched)`. Human verdict **FAIL**:
- **Hallucinated film**: the concepts describe a father/son family drama
  (waiting room, broken clock, dinner table, red dress, photograph, kitchen,
  plate, book) that does NOT exist in the scene facts (desert, riverbank,
  bus/subway interior, convenience store, cash register, mirror, toothbrush,
  horse, cowboys, sheriff uniforms, burning car). Invented terms with 0 scenes:
  clock, kitchen, apartment, plate, bottle, book, photograph, drawer, dinner,
  father, mother, family.
- **Formulaic**: every thesis uses the same "The X is not just Y — it is Z /
  film uses X ... symbol ... emotional ..." AI-essay pattern.
- **The rejection gate worked correctly** — it rejected all 18. That is the
  milestone's strongest component.
- **Matcher is brittle for partly-real evidence** (`mirror` 1, `counter` 1,
  `rain` 2, `table` 1, `red` 2, `dress` 5 all exist in the facts yet scored
  0/3), and `is_grounded()` substring semantics are misleading (`son`/`door`
  True from garbled tokens like "solution"/"person"). But loosening matching
  alone would ADMIT hallucinations — not a valid fix.

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
    token-limited, fact-grounded context; `EvidenceAnalyzer` grounds concepts to
    real scenes; `MovieGroundedDirector` generates 5 diverse concepts → rejects
    generic / un-evidenced ideas (bounded regeneration: initial batch + one
    corrective retry, then FAIL rather than force-through) → `ConceptCritic` +
    coverage → select → scene-aware plan → `CreativeMemory` →
    `reports/director_reasoning.md`. Hallucination-safe (unknown characters
    marked `unknown_character_01 (low confidence)`; vocab limited to facts that
    exist). Stops at the plan — NOT wired to the script stage.
  - **Structured evidence contract (`evidence_refs`)**: the milestone schema is
    extended so every concept separates its *creative claim* from *evidence
    references* — each ref names a canonical identifier the movie actually
    contains (scene id, character, object, location, action, event, theme,
    mood, dialogue). `required_evidence` is derived from the refs (one source of
    truth, no duplicate schema) and forwarded, with the refs, into the grounding
    contract for the script stage. Grounding is exact-ID-first, then
    canonical/alias vocabulary matching, then exact token containment — never
    arbitrary substring (`is_grounded("son")` is no longer True just because
    "person" appears). `concept_evidence` reports `requested_refs` /
    `matched_refs` / `missing_refs` and `matched_scenes`; the reasoning report
    shows every ref's per-scene status.
- Local E2E: **288 fast tests pass** (vision, artifacts, retrieval, editorial,
    movie-understanding, semantic-embedding, grounded-director, evidence-contract
    suites).
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

- **CURRENT BOTTLENECK**: **the generator hallucinates a different film.** The
  real-Qwen run (Colab T4) produced 0/18 grounded concepts — all 18 were
  correctly rejected because they were almost entirely invented (family-drama
  scenes absent from the facts) and formulaic. The rejection gate is proven;
  the unproven/weak link is the generation prompt+context: real Qwen does not
  reuse the provided known-objects/locations vocabulary. Secondary kept:
  matcher brittleness for partly-real evidence (loosening alone would admit
  hallucinations, so it is NOT a standalone fix).

- **NEXT ACTION** (fix the demonstrated generator grounding problem, re-validate, then wire to script)
  1. **Fix the generator (demonstrated problem, director scope only)**: strengthen
     the generation prompt + context so `required_evidence` MUST be verbatim,
     single-object/single-location claims drawn from the provided known
     objects/locations/tokens, and explicitly forbid inventing scenes,
     characters, or objects. Keep the strict evidence gate. Recalibrate
     matching only for genuinely-present evidence (single-noun checks; never
     admit ungrounded claims). Add focused unit tests.
  2. **Re-run the real-Qwen validation** on the T4
     (`notebooks/colab_grounded_director_validation.ipynb` →
     `scripts/run_director_validation.py --project data/bc6384be-...`) and
     inspect `director_reasoning.md` + `director_validation.json`: do several
     concepts survive with coverage ≥ MED? Are they specific, non-generic,
     grounded in real scenes, different from each other? Update
     `reports/director_validation.md` / `.json` human-eval fields.
  3. **Do NOT wire the script stage yet** (§11). Once the director is validated
     as genuinely specific + grounded, the *next* milestone connects the selected
     concept + evidence strategy to narrative/editorial generation.
  4. (Background, prior milestone) stronger retrieval semantics + temporal
     localization remain open research items.
