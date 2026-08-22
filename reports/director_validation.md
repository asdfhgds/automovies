# Director Validation — Human Evaluation (Run 2)

**Status**: REAL-QWEN RUN EXECUTED (Colab T4) — **Verdict: PLAN_REJECTED**.
Project: `bc6384be-47a5-4ee8-8674-7ff861472026`.

Run 2 used the current code (after significant-token matching, thesis claim
grounding, and `_is_location_confident`). Run 1 (same movie) was **VERDICT FAIL —
0/18 concepts grounded**; this run is a dramatic improvement but the milestone is
**NOT complete**: concepts now survive and are grounded, yet **no plan was
emitted** (strict plan gate rejected it, coverage 26% < 55%).

---

## 1. Runtime record (measured on the run)

- GPU: Tesla T4 (15.64 GB total; peak allocated 4022.5 MB)
- Model: Qwen/Qwen3-4B-Instruct-2507, device cuda, dtype 4bit
- Model load time (sec): 174.01
- Generation times (sec): 146.62, 72.04, 35.16, 23.84 (total 277.66)
- LLM calls: 4 | regeneration rounds: 1 | substitutes generated: 3
- Wall clock (sec): 451.83
- Concepts generated (surviving): 6 | Rejected: 3 | Selected: index 0
- Selected coverage: HIGH (6/6 refs, 3/3 claim refs), feasibility 0.60
- Diversity metric: **0.929** (run 1: 0.000)
- Plan: **REJECTED** (coverage 25.7%, min 55%) → `plan: null`

## 2. Decision gate

- **PLAN_REJECTED** — the evidence gate is now admitting genuinely grounded
  concepts (6 survivors, all `coverage HIGH`), and the **strict plan gate**
  correctly refused the selected concept's ungrounded plan. No invalid plan was
  emitted (the gate works).

## 3. Per-concept human evaluation (verified against the same facts the run used)

**A — "The Clock That Never Ticks" (SELECTED)**: ⚠️ **central premise invented**.
The thesis rests on "the 12:00 mark" on a clock — but `clock` and `12:00`
appear **0 times** in the movie facts. Every declared evidence_ref is REAL
(scene-1/3/6, "another person partially visible" → scene-12, "woman's face" →
scene-13, "looking around" → scene-6/7/9/…) so coverage is HIGH — the model
"gamed" the ref list with real, peripheral vocabulary while its star object does
not exist.

**B — "The Revolver as a Symbol of Unresolved Violence"**: ⚠️ same leak.
`revolver`/`kitchen`/`son's room` have 0 occurrences; the refs that matched are
real but unrelated ("looking around", "confrontation" scene-17, scene-6).

**C — "The Light in the Window"**: boundary. Window/snowy/subway-car refs are
real (scene-10/13), but the narrative ("emotional descent of the father") is
ungrounded (`father` 0 occurrences).

**D — "The Car That Never Moved"**: ✅ fully grounded — car (many scenes),
garage interior (11), subway car window (13), car interior (15/16), burning car
(23) all real. Specific to this movie, no invented nouns.

**E — "The Bench and the Dusk"**: ✅ fully grounded — dog (2), bench (3), arm
(14), riverbank (1/2), rural road (3). Specific, non-generic, grounded.

**F — "Anxiety in the Bathroom"**: ✅ refs grounded (mirror 14, snowy window 10,
woman's face 13, convenience store 6); minor thesis overreach ("father").

**Rejected (3)** — all correctly rejected with `LOW` claim coverage: "Weight of
Silence" (kitchen/revolver), "Son's Choice" (father/son/leaving), "Father's
Silence" (father/approaches). The run-1 failure mode (family-drama hallucination
hour-cited with real scene ids) is now handled: the claim gate rejects it.

## 4. Plan-gate audit (what was rejected)

`plan_rejection.audit`: coverage **0.257** (min 0.55), sufficient **false**.
- Invented (not in the movie): deliberate, sense, accumulation, allowing, weight,
  emerge, duration, rather, action, ups, surfaces, away, eyes, empty, objects,
  shown, environmental, occasional, hum, breath, dominates, unexplained, occur.
- Elsewhere (in the movie but outside the evidence scenes): hands, resting, gaze.

The 4B model wrote generic cinema prose in `editorial_direction` instead of
staying inside the evidence-scene vocabulary + `PLAN_EDITORIAL_TERMS` whitelist.
The gate is NOT too strict for grounded editing vocabulary — the mock grounded
plan (whitelist-vocabulary) passes at 100% coverage.

## 5. Main problems (demonstrated this run)

1. **Plan prompt (new primary)**: with concepts surviving, the plan model no
   longer reuses the evidence scene's vocabulary in `editorial_direction`; the
   strict plan gate (correctly) rejects. Next fix: constrain the plan prompt the
   same way concepts were constrained (worked example + verbatim evidence-scene
   vocabulary + forbidden generic film-speak).
2. **Thesis-noun leak (secondary)**: `derive_refs` currently lets a concept pass
   on *peripheral* real refs while the thesis's central noun (clock/revolver/
   father) is invented. The gate should require the thesis's salient object/
   character to appear among the resolved refs — or derive refs strictly from
   prose so peripheral declared refs cannot carry an invented premise.
3. **Fixed since run 1**: generator reuses the grounded vocabulary; invented
   family-drama concepts are now rejected correctly; diversity is 0.929.

## 6. Milestone status

**NOT COMPLETE.** Concepts: substantially validated (specific, diverse,
grounded). Plan: not yet emitted — the strict plan gate did its job and refused
an ungrounded editorial plan. Next action: fix the plan prompt for
evidence-scene-vocabulary anchoring (mirror of the concept-gate fix), close the
thesis-noun leak, re-run on the T4, and only then wire to the script stage.