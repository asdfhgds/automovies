"""Concept generation and final-plan prompting for the grounded director.

These builders talk to an ``llm`` callable of the form ``str -> str`` (the raw
model output). The director layer is responsible for wiring either the Qwen
provider (``provider.generate_text``) or a mock. Everything here is pure string
generation + tolerant JSON parsing; no model-side assumptions leak in.

The concept schema is the milestone's::

    {
      "title", "hook", "thesis", "why_interesting",
      "required_evidence": ["..."],
      "visual_opportunity", "format"
    }

plus an optional ``diversity_angle`` tag so five concepts are measured as
meaningfully different across the requested dimensions.
"""
import json
import logging
import re
from typing import Callable, Dict, Any, List, Optional

logger = logging.getLogger(__name__)

CONCEPT_REQUIRED_FIELDS = ("title", "hook", "thesis", "why_interesting",
                           "required_evidence", "visual_opportunity", "format")

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
- Cite only SCENE ids, characters, objects, locations, themes, and dialogue that
  actually appear in the context. NEVER invent anyone or anything.
- required_evidence must list specific concrete claims you would need on screen,
  each phrased as a short phrase (e.g. "revolver close-up", "man walking in
  water", "silence before dialogue"). These will be checked against the scenes.
- If the movie lacks material for a thesis, do NOT force it. Pick a thesis that
  the available scenes actually support.

DO NOT produce five versions of "the movie explores violence/problem X". Each
concept must have its own hook, thesis, and required evidence that points to
real scenes.

Return ONLY valid JSON (no markdown, no code fences) with this structure:
{{
  "concepts": [
    {{
      "title": "A SPECIFIC TITLE",
      "hook": "An engaging opening that draws the viewer in",
      "thesis": "A specific, defensible, evidence-based argument about THIS movie",
      "why_interesting": "Why this angle is surprising / worth watching",
      "required_evidence": ["short concrete claim 1", "short concrete claim 2"],
      "visual_opportunity": "Concrete visual/editing treatment you would shoot or find in scenes",
      "format": "short_video_essay",
      "diversity_angle": "the divergence dimension this concept explores"
    }}
  ]
}}
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
        f"- [{c.get('title', '?')}] {c.get('thesis', '')}"
        for c in rejected
    )
    return f"""
You are re-running a concept brainstorm. {substitutes_needed} of the previous
concepts were REJECTED because the movie's scenes did not actually contain the
evidence they claimed. Generate {substitutes_needed} NEW replacement concepts
that are grounded in the actual scenes shown in the context below.

GROUNDING RULES: Only cite SCENE ids / characters / objects / locations /
dialogue that appear in the context. required_evidence claims MUST be matched
by real scenes. Do not repeat the rejected ideas' theses.

Rejected concepts (do not repeat the same theses):
{rejected_blob}

Return ONLY valid JSON (no markdown, no code fences):
{{
  "concepts": [
    {{
      "title": "...", "hook": "...", "thesis": "...", "why_interesting": "...",
      "required_evidence": ["..."], "visual_opportunity": "...",
      "format": "short_video_essay", "diversity_angle": "..."
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
    """Coerce a parsed concept into the full milestone schema or None."""
    if not raw:
        return None
    title = str(raw.get("title") or "").strip()
    thesis = str(raw.get("thesis") or "").strip()
    if not title or not thesis:
        return None

    required = raw.get("required_evidence")
    if isinstance(required, str):
        required = [required]
    required = [str(r).strip() for r in (required or []) if str(r).strip()]
    if not required:
        return None  # no evidence ask => can't be grounded

    concept = {
        "title": title,
        "hook": str(raw.get("hook") or "").strip(),
        "thesis": thesis,
        "why_interesting": str(raw.get("why_interesting") or "").strip(),
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
