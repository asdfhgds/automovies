# Director Validation — Human Evaluation

**Status**: AWAITING REAL-QWEN RUN — these fields are intentionally BLANK until a
human inspects the real-Qwen output for project
`bc6384be-47a5-4ee8-8674-7ff861472026`.

Do NOT fill these from unit tests. Fill them only after running
`scripts/run_director_validation.py` (or
`notebooks/colab_grounded_director_validation.ipynb`) on a Colab T4 and reading
`director_reasoning.md` + `director_validation.json`.

---

## 1. Runtime record (from `director_validation.json` → `runtime`)

- GPU: _not measured_
- VRAM total: _not measured_
- VRAM peak allocated: _not measured_
- Model: _not measured_
- Device: _not measured_
- Dtype: _not measured_
- Model load time (sec): _not measured_
- Total generation time (sec): _not measured_
- LLM calls: _not measured_
- Regeneration rounds: _not measured_
- Substitutes generated: _not measured_
- Wall clock (sec): _not measured_
- Concepts generated: _not measured_
- Concepts rejected: _not measured_
- Selected concept: _not measured_
- Diversity metric: _not measured_

## 2. Decision gate

- **PASS** / **PARTIAL** / **FAIL**: _awaiting run_
- Justification: _awaiting run_

## 3. Per-concept human evaluation

For each generated concept (copy the block per concept):

```
### Concept <label>
- title: ...
- thesis: ...
- hook: ...
- required_evidence: ...
- evidence_coverage (from report): ...
- Specific (to this movie): GOOD | PARTIAL | BAD
- Generic/AI feel: LOW | MEDIUM | HIGH
- Evidence actually exists in the scenes: GOOD | PARTIAL | BAD
- Selected/supporting scenes relevant: GOOD | PARTIAL | BAD
- Human notes: ...
```

### Concept A
_awaiting run_

### Concept B
_awaiting run_

### Concept C
_awaiting run_

### Concept D
_awaiting run_

### Concept E
_awaiting run_

## 4. Grounding-quality audit (EvidenceAnalyzer)

Check whether lexical matching missed any real evidence due to:
synonyms (`weapon` vs `revolver`), paraphrases, character aliases, implicit
references, related objects/locations. Record concrete examples.

- _awaiting run_

## 5. Hallucination examples observed

Did any concept invent characters / objects / locations / events? Was the
rejection path triggered? Record concrete examples.

- _awaiting run_

## 6. Main problems discovered

- _awaiting run_

## 7. Code changes made as a result

- _awaiting run_

## 8. Tests passed

- _awaiting run_