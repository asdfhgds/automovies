"""Concept generation and final-plan prompting for the grounded director.

These builders talk to an ``llm`` callable of the form ``str -> str`` (the raw
model output). The director layer is responsible for wiring either the Qwen
provider (``provider.generate_text``) or a mock. Everything here is pure string
generation + tolerant JSON parsing; no model-side assumptions leak in.

The concept schema is the milestone's, extended with a structured evidence
contract::

    {
      "title", "hook", "thesis", "why_interesting",
      "evidence_refs": [ {"kind": "scene", "scene_id": "scene-1"},
                         {"kind": "object", "value": "revolver"} ],
      "visual_opportunity", "format"
    }

``evidence_refs`` is the *authoritative* grounding contract: every ref must name
an identifier (scene id / character / object / location / action / event /
theme / mood / dialogue) that literally exists in the movie intelligence.
``required_evidence`` is kept as a *derived* convenience field (one line per
ref) so every existing downstream consumer stays on a single source of truth —
no duplicate schema.

plus an optional ``diversity_angle`` tag so five concepts are measured as
meaningfully different across the requested dimensions.
"""
import json
import logging
import re
from typing import Callable, Dict, Any, List, Optional

logger = logging.getLogger(__name__)

CONCEPT_REQUIRED_FIELDS = ("title", "hook", "thesis", "why_interesting",
                           "evidence_refs", "visual_opportunity", "format")

#: The evidence-ref kinds the grounding matcher understands. Anything else is
#: treated as a generic ``text`` ref (matched by exact token presence only).
EVIDENCE_KINDS = frozenset({
    "scene", "character", "object", "location", "action", "event",
    "theme", "mood", "dialogue", "text",
})

DIVERSITY_DIMENSIONS = [
    "philosophy", "psychology", "character", "symbolism", "cinematography",
    "narrative_structure", "irony", "thematic_interpretation",
]

# Claim types for claim-level grounding (Phase 2/3).
CLAIM_TYPES = frozenset({
    "ACTION", "CHARACTER", "OBJECT", "LOCATION", "DIALOGUE", "VISUAL",
    "TEMPORAL", "COMPARISON", "RELATIONSHIP", "CAUSAL", "EMOTIONAL",
})

# Generic thesis patterns the critic must reject as "not a real idea".
GENERIC_THESIS_PATTERNS = (
    "explores violence",
    "movie explores",
    "film explores",
    "this movie",
    "this film",
    "teaches us",
    "focused analysis",
    "key moment",
    "important scene",
    "characters show emotions",
    "good versus evil",
)


# -- Structured Editorial Plan Schema (V4) ------------------------------------

# Allowed enum values for structured editorial plan fields.
PLAN_TRANSITIONS = frozenset({
    "cut", "crossfade", "fade", "dissolve", "jump_cut", "match_cut",
    "smash_cut", "wipe", "iris", "none",
})

PLAN_PACING = frozenset({
    "slow", "measured", "moderate", "gradual", "steady", "rhythmic",
    "rapid", "fast", "accelerating", "decelerating", "variable",
})

PLAN_RHYTHM = frozenset({
    "slow", "steady", "measured", "syncopated", "driving", "pulsing",
    "irregular", "free",
})

PLAN_EMPHASIS = frozenset({
    "character", "action", "object", "location", "emotion", "dialogue",
    "visual", "sound", "silence", "contrast", "repetition", "detail",
})

PLAN_REPETITION = frozenset({
    "none", "motif", "callback", "echo", "parallel", "mirror", "loop",
})

PLAN_PURPOSE = frozenset({
    "contrast", "parallel", "progression", "reveal", "emphasis",
    "transition", "pacing", "mood", "character", "theme", "tension",
    "resolution", "setup", "payoff",
})

PLAN_AUDIO_MOVIE = frozenset({
    "retain", "mute", "filter", "duck",
})

PLAN_AUDIO_NARRATION = frozenset({
    "none", "minimal", "moderate", "dominant", "continuous", "sparse",
})

PLAN_AUDIO_MUSIC = frozenset({
    "none", "low", "moderate", "high", "diegetic_only", "score_only",
})

# Valid keys for structured editorial plan (V4)
STRUCTURED_PLAN_KEYS = frozenset({
    "visual", "editing", "audio",
})

VISUAL_PLAN_KEYS = frozenset({
    "scene_id", "start_sec", "end_sec", "source_fact_refs",
})

EDITING_PLAN_KEYS = frozenset({
    "transition", "pacing", "rhythm", "emphasis", "repetition", "purpose",
})

AUDIO_PLAN_KEYS = frozenset({
    "movie_audio", "narration", "music",
})


# -- Evidence references (the structured grounding contract) -----------------

def _normalize_ref(raw: Any) -> Optional[Dict[str, Any]]:
    """Coerce one evidence reference into ``{kind, scene_id|value}`` or None.

    Accepts the structured dict form (``{"kind": ..., "value": ...}`` /
    ``{"kind": "scene", "scene_id": ...}``) as well as legacy strings such as
    ``"revolver"`` or ``"scene: scene-1"``.
    """
    if isinstance(raw, dict):
        kind = str(raw.get("kind") or "text").strip().lower()
        if kind not in EVIDENCE_KINDS:
            kind = "text"
        if kind == "scene":
            sid = str(raw.get("scene_id") or raw.get("value") or "").strip()
            if not sid:
                return None
            return {"kind": "scene", "scene_id": sid}
        value = str(raw.get("value") or raw.get("scene_id") or "").strip()
        if not value:
            return None
        return {"kind": kind, "value": value}
    if isinstance(raw, str):
        item = raw.strip()
        if not item:
            return None
        m = re.match(r"^scene\s*[:#]\s*(.+)$", item, re.IGNORECASE)
        if m:
            return {"kind": "scene", "scene_id": m.group(1).strip()}
        return {"kind": "text", "value": item}
    return None


def concept_refs(concept: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The authoritative evidence refs of a concept.

    Reads ``evidence_refs`` when present; otherwise derives refs from the legacy
    ``required_evidence`` strings (so older outputs stay analysable).
    """
    raw = concept.get("evidence_refs")
    if isinstance(raw, list) and raw:
        out = []
        for r in raw:
            ref = _normalize_ref(r)
            if ref:
                out.append(ref)
        if out:
            return out
    legacy = concept.get("required_evidence") or []
    if isinstance(legacy, str):
        legacy = [legacy]
    out = []
    for item in legacy:
        ref = _normalize_ref(str(item))
        if ref:
            out.append(ref)
    return out


def render_ref(ref: Dict[str, Any]) -> str:
    """One-line render of a ref (used to derive ``required_evidence``)."""
    if ref.get("kind") == "scene":
        return str(ref.get("scene_id") or "")
    return str(ref.get("value") or "")


def _refs_line(ref: Dict[str, Any]) -> str:
    """Reprint a ref as a stable schema line (for prompts/reports)."""
    if ref.get("kind") == "scene":
        return f'{{"kind": "scene", "scene_id": "{ref.get("scene_id", "")}"}}'
    return f'{{"kind": "{ref.get("kind", "text")}", "value": "{ref.get("value", "")}"}}'


# -- Prompt builders ---------------------------------------------------------


def build_generation_prompt(
    context: str,
    num_concepts: int = 5,
    user_topic: Optional[str] = None,
) -> str:
    """Prompt to generate ``num_concepts`` genuinely different, grounded concepts."""
    topics = "\n".join(
        f"  {i + 1}. {d}" for i, d in enumerate(DIVERSITY_DIMENSIONS)
    )
    user_part = f"\n\n## USER FOCUS\n{user_topic}" if user_topic else ""
    return f"""
You are a creative director turning a real movie into 5-60 original, evidence-based
video-essay concepts. You will be shown the ONLY facts that exist about the movie.

TASK: Generate {num_concepts} genuinely different concepts for a 60-120 second
movie-analysis video. Select each concept along a DIFFERENT dimension.

Available divergence dimensions (pick a distinct one per concept, at least 6 of
these must appear across the set):
{topics}

FACT vs INTERPRETATION (CRITICAL):
- FACTS are directly in the Movie Intelligence: scene IDs, characters, objects,
  locations, actions, events, dialogue, visual events, timestamps.
  Example: "scene-1 contains a revolver" | "the car appears in scene-7"
- INTERPRETATIONS are creative conclusions you draw FROM facts:
  themes, emotional readings, symbolic meanings, causal chains, comparisons.
  Example: "the revolver symbolizes entrapment" | "the loop structure mirrors grief"
- The system WILL REJECT concepts that present INTERPRETATIONS as FACTS.
- You MUST ground every FACTUAL CLAIM in your thesis/hook in the inventory.
- INTERPRETATIONS are encouraged — but label them as such in why_interesting.

MANDATORY GROUNDING (from the context above):
1. Separate your CREATIVE CLAIM from your EVIDENCE REFERENCES.
   title / hook / thesis / why_interesting are interpretation; the system
   derives your evidence_refs by scanning that prose for the movie's verbs,
   objects, locations and characters — so WRITE your thesis/hook IN TERMS OF
   the identifiers that actually appear in WHAT ACTUALLY EXISTS.
2. Name concrete, catalogued items in your prose: an object, a location, an
   action, or a scene id shown in WHAT ACTUALLY EXISTS. If you write only
   abstract words (memory, gloom, free will) or invented nouns (kitchen,
   notebook, father), you will be REJECTED because nothing of yours is grounded.
3. You MAY also fill evidence_refs yourself — they only help if they are exact
   canonical identifiers from WHAT ACTUALLY EXISTS / the scene cards, copied
   VERBATIM (character for character). Wrong guesses are simply discarded and
   replaced by refs derived from your prose.
4. The matcher is exact and token-based: "son" will NOT match just because
   "person" also appears; an object you cite must literally be listed.
5. If the movie lacks material for a thesis, do NOT force it. Pick a thesis the
   available scenes actually support.
6. A WORKED EXAMPLE is included in the context (## WORKED EXAMPLE). It shows
   the ONLY way evidence_refs can pass: values copied verbatim from the scene
   cards. Mirror its ref style; write your own wording.
7. ZERO TOLERANCE FOR HALLUCINATION: Every noun in your thesis/hook that names
   a character, object, location, action, or scene MUST appear VERBATIM in the
   WHAT ACTUALLY EXISTS section above. If you invent a single noun — a
   character name, object, location, or scene ID not in the provided vocabulary —
   your concept WILL BE REJECTED. Check every noun against the vocabulary before
   writing.

DO NOT produce five versions of "the movie explores violence/problem X". Each
concept must have its own hook, thesis, and a distinct set of grounded
evidence_refs.

Return ONLY valid JSON (no markdown, no code fences) with this structure:
{{
  "concepts": [
    {{
      "title": "A SPECIFIC TITLE",
      "hook": "An engaging opening that draws the viewer in",
      "thesis": "A specific, defensible, evidence-based argument about THIS movie",
      "why_interesting": "Why this angle is surprising / worth watching (label interpretations vs facts)",
      "evidence_refs": [
        {{"kind": "scene", "scene_id": "scene-1"}},
        {{"kind": "object", "value": "revolver"}}
      ],
      "visual_opportunity": "Concrete visual/editing treatment you would shoot or find in scenes",
      "format": "short_video_essay",
      "diversity_angle": "the divergence dimension this concept explores"
    }}
  ]
}}

evidence_refs kinds: scene | character | object | location | action | event |
theme | mood | dialogue. Use 3-6 grounded refs per concept; at least one of them
must be a scene id. Do NOT put interpretation inside evidence_refs — only
catalogued identifiers.
{user_part}

Generate now:
"""


def build_rejection_prompt(
    context: str,
    rejected: List[Dict[str, Any]],
    substitutes_needed: int,
    ref_failures: Optional[List[List[str]]] = None,
    ref_feedback: Optional[List[List[Dict[str, Any]]]] = None,
) -> str:
    """Prompt to replace concepts that failed evidence grounding.

    ``ref_failures`` (optional, parallel to ``rejected``) is the per-concept
    list of rendered refs that were NOT FOUND in the movie (e.g. from
    ``concept_evidence(concept)["unmatched_claims"]``). Showing the exact
    failed refs gives the model corrective, deterministic feedback instead of a
    vague "your evidence was rejected".

    ``ref_feedback`` (optional, parallel to ``rejected``) is the richer,
    structured form — per-concept list of ``{"kind", "value", "found",
    "scenes", "suggestions"}`` records from ``EvidenceAnalyzer.ref_feedback``.
    When present it takes precedence; each NOT-FOUND ref lists verbatim
    candidate identifiers so the model replaces the hallucination with a real
    fact instead of guessing again.
    """
    rejected_lines = []
    for i, c in enumerate(rejected):
        line = (f"- [{c.get('title', '?')}] thesis={c.get('thesis', '')} "
                f"evidence_refs=[{', '.join(_refs_line(r) for r in concept_refs(c))}]")
        if ref_feedback:
            records = (ref_feedback or [None] * len(rejected))[i]
            if records:
                line += "\n    Failed refs (NOT FOUND in the movie; do NOT reuse):"
                for record in records:
                    kind = record.get("kind", "text")
                    value = record.get("value", "")
                    if record.get("found"):
                        continue
                    line += (f"\n      - kind={kind} value={value!r} "
                             f"[NOT FOUND]")
                    suggestions = record.get("suggestions") or []
                    if suggestions:
                        listed = "; ".join(suggestions[:6])
                        line += (
                            f"\n        VERBATIM {kind} candidates in this "
                            f"movie: {listed}"
                        )
                    else:
                        line += (
                            f"\n        (no {kind} identifiers exist in this "
                            "movie — drop this ref entirely)"
                        )
        elif ref_failures:
            failures = (ref_failures or [None] * len(rejected))[i]
            if failures:
                line += ("\n    These refs were NOT FOUND in the movie, do NOT "
                         "reuse them: " + ", ".join(failures))
        rejected_lines.append(line)
    rejected_blob = "\n".join(rejected_lines)
    return f"""
You are re-running a concept brainstorm. {substitutes_needed} of the previous
concepts were REJECTED because the movie's scenes did not actually contain the
evidence they claimed. Below each rejected concept we list the exact refs that
do NOT exist in this movie — replace them with identifiers copied VERBATIM from
the scene cards / WHAT ACTUALLY EXISTS in the context (see ## WORKED EXAMPLE).
Generate {substitutes_needed} NEW replacement concepts
that are grounded in the actual scenes shown in the context below.

FACT vs INTERPRETATION REMINDER:
- FACTS = scene IDs, characters, objects, locations, actions, dialogue, events
- INTERPRETATIONS = themes, symbolism, emotional readings, causal/comparative claims
- Presenting interpretations as facts = REJECTION

GROUNDING RULES:
- Keep your CREATIVE CLAIM separate from your evidence_refs.
- The system DERIVES your evidence_refs by scanning your prose for the movie's
  actual vocabulary. So write your thesis/hook/visual_opportunity IN TERMS OF
  the catalogued objects, locations, characters, actions and scene ids in WHAT
  ACTUALLY EXISTS (copy each identifier verbatim into the prose itself).
- You may still list evidence_refs — but only exact canonical identifiers from
  the context; any invented or paraphrased value is discarded automatically.
- Every replacement concept should mention at least one scene id
  (e.g. "scene-1") and at least one real object/location/action in its prose.
- Do not repeat the rejected ideas' theses or their ungrounded refs.

Rejected concepts (do not repeat these):
{rejected_blob}

Return ONLY valid JSON (no markdown, no code fences):
{{
  "concepts": [
    {{
      "title": "...", "hook": "...", "thesis": "...", "why_interesting": "...",
      "evidence_refs": [
        {{"kind": "scene", "scene_id": "scene-1"}},
        {{"kind": "object", "value": "revolver"}}
      ],
      "visual_opportunity": "...",
      "format": "short_video_essay",
      "diversity_angle": "..."
    }}
  ]
}}

evidence_refs kinds: scene | character | object | location | action | event |
theme | mood | dialogue. Use 3-6 grounded refs per concept; at least one of them
must be a scene id. Do NOT put interpretation inside evidence_refs — only
catalogued identifiers.
"""


def build_plan_prompt(
    context: str,
    duration_sec: int = 90,
    grounding_warnings: Optional[List[str]] = None,
) -> str:
    """Prompt to build the final scene-aware director plan with STRUCTURED editorial plan.

    The concept is already decided (deterministic — the selected concept). The
    model must produce a STRUCTURED editorial plan with separate sections for
    visual, editing, and audio. Factual fields (scene IDs, timestamps, refs)
    must be grounded in the evidence scenes. Editorial fields use controlled
    vocabularies and do NOT require movie grounding.

    ``grounding_warnings`` (optional) list concrete FACTUAL terms from a previous
    plan attempt that no scene actually contains — the model must not reuse them.
    """
    # Lazy import to avoid cycles.
    from director.evidence import PLAN_EDITORIAL_TERMS
    whitelist_blob = (
        "\n## ALLOWED EDITORIAL VOCABULARY (whitelist)\n"
        "The structured fields below use controlled vocabularies. If you must "
        "write prose in any free-text field, use ONLY these terms (flat list):\n"
        + ", ".join(sorted(PLAN_EDITORIAL_TERMS))
        + "\n"
    )
    warnings_blob = ""
    if grounding_warnings:
        warnings_blob = (
            "\n## GROUNDING CORRECTION (your previous plan was audited)\n"
            "The following FACTUAL terms you used do NOT exist in any evidence "
            "scene. Remove them and re-describe using structured fields or "
            "verbatim vocabulary terms:\n"
            + "\n".join(f"- {t}" for t in grounding_warnings)
            + "\n"
        )
    return f"""
You are a director finalizing the plan (STRUCTURED plan) for the SELECTED CONCEPT
shown below, grounded ONLY in the evidence scenes also shown. The video is
{duration_sec} seconds.

MANDATORY STRUCTURE:
- The concept (title / hook / thesis) is FINAL and given to you. Copy it
  verbatim into "concept". Do not write a different concept, movie, or thesis.
- You MUST produce a STRUCTURED editorial_plan with these sections:
  * "visual":   {{scene_id, start_sec, end_sec, source_fact_refs}}
  * "editing":  {{transition, pacing, rhythm, emphasis, repetition, purpose}}
  * "audio":    {{movie_audio, narration, music}}
- Factual fields (scene_id, start_sec, end_sec, source_fact_refs) MUST be
  grounded in the evidence scenes. Use verbatim identifiers from the vocabulary.
- Editorial fields use CONTROLLED VOCABULARIES (see below) and do NOT require
  movie grounding.
- If you need free-text prose, use ONLY the ALLOWED EDITORIAL VOCABULARY.

CONTROLLED VOCABULARIES (editorial fields — no movie grounding needed):
- transition: {", ".join(sorted(PLAN_TRANSITIONS))}
- pacing: {", ".join(sorted(PLAN_PACING))}
- rhythm: {", ".join(sorted(PLAN_RHYTHM))}
- emphasis: {", ".join(sorted(PLAN_EMPHASIS))}
- repetition: {", ".join(sorted(PLAN_REPETITION))}
- purpose: {", ".join(sorted(PLAN_PURPOSE))}
- movie_audio: {", ".join(sorted(PLAN_AUDIO_MOVIE))}
- narration: {", ".join(sorted(PLAN_AUDIO_NARRATION))}
- music: {", ".join(sorted(PLAN_AUDIO_MUSIC))}

{whitelist_blob}
{warnings_blob}
Return ONLY valid JSON (no markdown, no code fences) with this structure:
{{
  "concept": {{
    "title": "copy of the selected concept title",
    "hook": "copy of the selected concept hook",
    "thesis": "copy of the selected concept thesis"
  }},
  "format": {{
    "type": "short_video_essay",
    "duration_sec": {duration_sec}
  }},
  "editorial_plan": {{
    "visual": {{
      "scene_id": "scene-1",
      "start_sec": 1.2,
      "end_sec": 3.8,
      "source_fact_refs": ["revolver", "scene-1"]
    }},
    "editing": {{
      "transition": "cut",
      "pacing": "gradual",
      "rhythm": "steady",
      "emphasis": "character",
      "repetition": "none",
      "purpose": "contrast"
    }},
    "audio": {{
      "movie_audio": "retain",
      "narration": "dominant",
      "music": "low"
    }}
  }},
  "editorial_direction": {{
    "pacing": "fallback prose if needed",
    "visual_style": "fallback prose if needed",
    "audio_style": "fallback prose if needed",
    "editing_style": "fallback prose if needed"
  }}
}}

The evidence_strategy is computed deterministically from the scenes by the
system, so you only provide concept (copied) / format / editorial_plan /
editorial_direction (prose fallback).
Base every FACTUAL claim on the evidence scenes shown. Do not invent
characters, objects, locations, or moments.

Generate now:
"""


# -- JSON extraction / validation helpers -----------------------------------

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Tolerant JSON extraction (balanced-object scan, repair of small errors)."""
    text = (text or "").strip()
    if not text:
        return None

    def _parse(s: str) -> Optional[Dict]:
        try:
            val = json.loads(s)
            return val if isinstance(val, dict) else None
        except json.JSONDecodeError:
            return None

    direct = _parse(text)
    if direct:
        return direct

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        direct = _parse(fenced.group(1).strip())
        if direct:
            return direct

    # Scan for the first balanced JSON object, string-aware.
    for idx in range(len(text)):
        if text[idx] != "{":
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(idx, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[idx : i + 1]
                    repaired = _repair(candidate)
                    parsed = _parse(repaired or candidate)
                    if parsed:
                        return parsed
                    break
    logger.warning("Failed to extract JSON from model output")
    return None


def _repair(text: str) -> Optional[str]:
    out = text
    out = re.sub(r"\bTrue\b", "true", out)
    out = re.sub(r"\bFalse\b", "false", out)
    out = re.sub(r"\bNone\b", "null", out)
    out = re.sub(r",\s*([}\]])", r"\1", out)
    try:
        json.loads(out)
        return out
    except json.JSONDecodeError:
        return None


def parse_concepts(response_text: str) -> List[Dict[str, Any]]:
    """Parse a model response into a list of concepts in the milestone schema."""
    data = extract_json(response_text)
    if not data:
        return []
    if isinstance(data, list):
        concepts = [c for c in data if isinstance(c, dict)]
    else:
        concepts = data.get("concepts", [])
        concepts = [c for c in concepts if isinstance(c, dict)] if isinstance(concepts, list) else []
    result = []
    for c in concepts:
        normalized = _normalize_concept(c)
        if normalized:
            result.append(normalized)
    return result


def _normalize_concept(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Coerce a parsed concept into the full milestone schema or None.

    ``evidence_refs`` is the authoritative grounding contract; ``required_evidence``
    is derived from it (one rendered line per ref) for downstream consumers.
    """
    if not raw:
        return None
    title = str(raw.get("title") or "").strip()
    thesis = str(raw.get("thesis") or "").strip()
    if not title or not thesis:
        return None

    refs = concept_refs(raw)
    # A concept may arrive with NO declared refs: the grounded director derives
    # deterministic evidence_refs from its prose later. Requiring refs here
    # would silently drop prose-only (and now salvageable) concepts.

    required = []
    for x in refs:
        line = render_ref(x)
        if line:
            required.append(line)

    concept = {
        "title": title,
        "hook": str(raw.get("hook") or "").strip(),
        "thesis": thesis,
        "why_interesting": str(raw.get("why_interesting") or "").strip(),
        "evidence_refs": refs,
        "required_evidence": required,
        "visual_opportunity": str(raw.get("visual_opportunity") or "").strip(),
        "format": str(raw.get("format") or "short_video_essay").strip(),
        "diversity_angle": str(raw.get("diversity_angle") or "").strip(),
    }
    return concept


def parse_plan(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse the final plan response (supports both V4 structured and legacy format)."""
    data = extract_json(response_text)
    if not isinstance(data, dict):
        return None
    concept = data.get("concept")
    # V4: structured editorial_plan (primary)
    editorial_plan = data.get("editorial_plan")
    # Legacy: free-text editorial_direction (backward compat)
    ed = data.get("editorial_direction")
    fmt = data.get("format") or {}
    if not isinstance(concept, dict):
        return None
    # At least one of editorial_plan or editorial_direction must be present
    if not isinstance(ed, dict) and not isinstance(editorial_plan, dict):
        return None
    result = {
        "concept": dict(concept),
        "format": dict(fmt) if isinstance(fmt, dict) else {},
    }
    if isinstance(editorial_plan, dict):
        result["editorial_plan"] = dict(editorial_plan)
    if isinstance(ed, dict):
        result["editorial_direction"] = dict(ed)
    return result


def is_generic_thesis(thesis: str) -> bool:
    """True if a thesis looks like a generic, non-specific AI platitude."""
    low = (thesis or "").lower().strip()
    if not low:
        return True
    return any(p in low for p in GENERIC_THESIS_PATTERNS)


def compute_diversity_metric(concepts: List[Dict[str, Any]]) -> float:
    """A crude 0..1 diversity score across thesis wording + dimension tags."""
    if len(concepts) < 2:
        return 0.0

    def wordset(c):
        return set(re.findall(r"[a-z0-9]+", (c.get("thesis", "") or "").lower()))

    theses = [wordset(c) for c in concepts]
    total_pairs = 0
    overlap_sum = 0.0
    for i in range(len(theses)):
        for j in range(i + 1, len(theses)):
            total_pairs += 1
            if not theses[i] and not theses[j]:
                continue
            inter = len(theses[i] & theses[j])
            union = len(theses[i] | theses[j]) or 1
            overlap_sum += inter / union
    token_diversity = 1.0 - (overlap_sum / max(1, total_pairs))

    angles = set()
    for c in concepts:
        a = str(c.get("diversity_angle") or "").strip().lower()
        if a:
            angles.add(a)
    angle_diversity = min(1.0, len(angles) / max(1, len(concepts)))

    return round(0.5 * token_diversity + 0.5 * angle_diversity, 3)
