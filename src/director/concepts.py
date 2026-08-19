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

MANDATORY GROUNDING (from the context above):
1. Separate your CREATIVE CLAIM from your EVIDENCE REFERENCES.
   title / hook / thesis / why_interesting are interpretation; evidence_refs
   must ONLY name identifiers that literally appear in the context.
2. Cite only SCENE ids, characters, objects, locations, actions, themes, moods,
   or dialogue that actually appear. NEVER invent anyone or anything.
3. Every evidence_ref must use exact canonical identifiers from WHAT ACTUALLY
   EXISTS and the scene cards. Prefer a scene ref ({{"kind": "scene",
   "scene_id": "scene-1"}}) whenever a specific scene carries your point.
4. The matcher is exact and token-based: "son" will NOT match just because
   "person" also appears; an object you cite must literally be listed.
5. If the movie lacks material for a thesis, do NOT force it. Pick a thesis the
   available scenes actually support.

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
      "why_interesting": "Why this angle is surprising / worth watching",
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
) -> str:
    """Prompt to replace concepts that failed evidence grounding."""
    rejected_blob = "\n".join(
        f"- [{c.get('title', '?')}] thesis={c.get('thesis', '')} "
        f"evidence_refs=[{', '.join(_refs_line(r) for r in concept_refs(c))}]"
        for c in rejected
    )
    return f"""
You are re-running a concept brainstorm. {substitutes_needed} of the previous
concepts were REJECTED because the movie's scenes did not actually contain the
evidence they claimed. Generate {substitutes_needed} NEW replacement concepts
that are grounded in the actual scenes shown in the context below.

GROUNDING RULES:
- Keep your CREATIVE CLAIM separate from your evidence_refs.
- evidence_refs must ONLY use exact canonical identifiers from the context
  (scene ids, characters, objects, locations, actions, themes, moods, dialogue).
- Every replacement concept needs at least one scene ref
  ({{"kind": "scene", "scene_id": "scene-1"}}).
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

Generate now:
"""


def build_plan_prompt(context: str, duration_sec: int = 90) -> str:
    """Prompt to build the final scene-aware director plan."""
    return f"""
You are a director finalizing the plan for the selected concept, grounded ONLY
in the evidence scenes shown below. The video is {duration_sec} seconds.

Return ONLY valid JSON (no markdown, no code fences) with this structure:
{{
  "concept": {{
    "title": "...",
    "hook": "...",
    "thesis": "..."
  }},
  "format": {{
    "type": "short_video_essay",
    "duration_sec": {duration_sec}
  }},
  "editorial_direction": {{
    "pacing": "how the pacing supports the argument",
    "visual_style": "concrete visual treatment grounded in the cited scenes",
    "audio_style": "music/sound design suggestion",
    "editing_style": "how cuts/transitions carry the thesis"
  }}
}}

The evidence_strategy is computed deterministically from the scenes by the
system, so you only provide concept / format / editorial_direction. Base every
claim in editorial_direction on the evidence scenes shown. Do not invent
characters, objects, or moments.

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
    if not refs:
        return None  # no evidence ask => can't be grounded

    required = [r for r in (render_ref(x) for x in refs) if r]

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
    """Parse the final plan response."""
    data = extract_json(response_text)
    if not isinstance(data, dict):
        return None
    concept = data.get("concept")
    ed = data.get("editorial_direction")
    fmt = data.get("format") or {}
    if not isinstance(concept, dict) or not isinstance(ed, dict):
        return None
    return {
        "concept": dict(concept),
        "format": dict(fmt) if isinstance(fmt, dict) else {},
        "editorial_direction": dict(ed),
    }


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
