# Director Validation — Human Evaluation

**Status**: REAL-QWEN RUN EXECUTED (Colab T4) — **Verdict: FAIL**.
Project: `bc6384be-47a5-4ee8-8674-7ff861472026`.

---

## 1. Runtime record (measured on the run)

- GPU: Tesla T4
- VRAM total: see run `director_validation.json` (not in console summary)
- VRAM peak allocated: see run `director_validation.json`
- Model: Qwen/Qwen3-4B-Instruct-2507
- Device: cuda (strict `REQUIRE_REAL_LLM=true`)
- Dtype: 4bit (configured in notebook)
- Model load time (sec): 49.03
- Total generation time (sec): 378.0
- LLM calls: 3
- Substitutes generated: 12
- Wall clock (sec): 427.05
- Concepts generated (surviving): 0
- Concepts rejected: 18 (6 initial + 12 substitutes) — **every one `LOW (0/3 matched)`**
- Selected concept: NONE
- Diversity metric: 0.000

## 2. Decision gate

- **FAIL** — the director produced **zero usable concepts**. The 18 generated
  concepts are **mostly hallucinated / unrelated to this movie** and formulaic;
  while the evidence gate behaved correctly (rejected all 18), the generator is
  not producing actionable concepts on this intelligence.

## 3. Per-concept human evaluation

All 18 rejected concepts share the same pathology (verified against the same
SceneFacts the run used — `data/bc6384be-.../movie_index.json`):

- **Specific to this movie**: BAD (all) — the concepts describe a father/son
  family drama (waiting room, dinner table, red dress, broken clock, photograph,
  kitchen, plates, books) that does NOT exist in the scene facts
  (desert, riverbank, bus/subway interior, convenience store, cash register,
  mirror, toothbrush, horse, cowboys, sheriff uniforms, burning car).
- **Generic/AI feel**: HIGH (all) — same repeated formula
  "The X is not just Y — it is Z / film uses X ... symbol ... emotional ...".
- **Evidence actually exists**: BAD for ~half (clock, kitchen, apartment,
  plate, bottle, book, photograph, drawer, dinner, father, mother, family all
  have 0 scenes); PARTIAL for the rest — individual real terms *exist*
  (`mirror` 1, `counter` 1, `window` 3, `rain` 2, `table` 1, `red` 2, `dress` 5)
  but the multi-word evidence claims were rejected because the matcher requires
  EVERY token of the phrase in the same scene.
- **Selected/supporting scenes relevant**: BAD — no concept had any supporting
  scenes (`coverage LOW (0/3 matched)` for all 18).

Representative examples:
- Rejected 2 "The Broken Clock ... Time as a Symbol" — `clock` has **0 scenes**
- Rejected 3 "The Unwilling Mentor: Why the Father Doesn't Teach the Son" —
  `father` **0 scenes**; family drama entirely invented
- Rejected 7 "The Empty Chair at the Dinner Table" — `chair`/`dinner` 0 scenes
- Rejected 4 "The Mirror That Lies" — `mirror` DOES exist (1 scene) but the
  claim phrasing meant 0/3 matched — the matcher rejected a partly-real idea

## 4. Grounding-quality audit (EvidenceAnalyzer)

- Hallucination guard + rejection gate: **work as designed** — rejected all 18
  un-evidenced concepts; that is the milestone's strongest part.
- Lexical matcher is **too strict AND too loose simultaneously**:
  - too strict: all-token phrase match rejects real evidence
    (`rain` exists but "rain that falls only on the window" needs every token);
    `fire`→0 while facts have `burning car`.
  - too loose at the token level: `son`/`door` "grounded=True" come from
    substring hits inside garbled transcript tokens (e.g. "solution",
    "person"), i.e. `is_grounded()` substring semantics can be misleading.
- Relaxing matching alone would NOT fix the milestone — it would admit
  hallucinated concepts (e.g. "clock" would still have no supported scenes).

## 5. Hallucination examples observed

The generator invented a whole different film. Concrete invented elements that
are NOT grounded anywhere in the facts: waiting room, broken clock, father/son
relationship, red dress (single terms `red`/`dress` occur but the compound
`red dress` pairing is not evidenced), photograph, note, drawer, book, kitchen,
apartment key, doormat, shoes, dinner table, plate, bottle, rain-window
pairing. The rejection path correctly rejected all of these — documented proof
the gate works.

## 6. Main problems discovered (demonstrated)

1. **Generator hallucination (primary)**: real Qwen, from the grounded
   context, produced concepts for a film that does not exist in the facts
   (family drama) and cites evidence with 0% ground truth (all 18 × 0/3).
2. **Evidence phrasing not anchored to the vocabulary**: the model never
   reused the provided known-objects/locations terms as verbatim claims.
3. **Matcher brittleness (secondary)**: all-token phrase match + substring
   `is_grounded` distort coverage for partly-real concepts.

## 7. Fix required (per task: ONLY the demonstrated problems)

- Strengthen the **generation prompt + context** so evidence references MUST be
  drawn verbatim from the provided known objects/locations/tokens (as
  single claims), and explicitly forbid inventing scenes/characters/objects.
- Keep the strict evidence gate (it is proven correct).
- Re-calibrate matching only to help genuinely-present evidence
  (single-token noun evidence + scene-internal token check), never to admit
  ungrounded claims.
- Re-run the Colab notebook; iterate until grounded concepts survive.

## 8. Tests passed

288 fast tests (after this fix), including the new
`tests/test_grounded_evidence_contract.py` (exact-scene-id refs, canonical
vocabulary aliases, no-substring matching, structured `concept_evidence`
fields, coverage labels, bounded regeneration). Real-Qwen re-run is gated on a
GPU machine (`python -m pytest tests/test_grounded_director_real_qwen.py -m
llm_integration -v`).

## 9. Contract change implemented (this fix)

- **`evidence_refs`** replaces free-form evidence claims as the authoritative
  grounding contract: `{"kind": "scene", "scene_id": "scene-1"}` /
  `{"kind": "object", "value": "revolver"}` etc. (`required_evidence` is now a
  derived field — one rendered line per ref — so no duplicate schema exists).
- Grounding order: exact scene id → canonical vocabulary identifier / alias →
  exact token containment. Arbitrary substring matching is no longer used
  (fixes the demonstrated "son"→"person" false positive).
- Bounded regeneration: initial batch of 5, then at most one corrective retry;
  if nothing grounds, the run FAILS with `selected_concept=None` /
  `plan=None` instead of forcing an ungrounded concept through.
- `concept_evidence` now exposes `requested_refs`, `matched_refs`,
  `missing_refs` and `matched_scenes`; the reasoning report lists every ref's
  matched-scene status and the missing evidence.