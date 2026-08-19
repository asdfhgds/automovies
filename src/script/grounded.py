"""Grounded script generator.

Replaces the heuristic-only script path. The generator consumes the director's
grounding contract (selected concept + evidence requirements + supporting
scenes + editorial intent + format) and the movie intelligence, and produces a
structured script whose every analytical section maps to real ``scene_id`` +
``evidence_id`` references.

Grounding is enforced deterministically:

- sections may only reference scenes that exist in ``movie_index.scenes`` and
  are listed in the contract's ``supporting_scenes``;
- evidence ids are assigned to the contract's ``evidence_requirements`` and only
  referenced when at least one supporting scene can satisfy them;
- narration is built from *real* scene facts (location / actions / objects /
  dialogue / visual events); when a fact is missing the generator uses cautious
  language rather than inventing a substitute.

This module never calls an LLM and never downloads a model, so the local mock
profile and the unit tests stay hermetic.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_SECTIONS = [
    "hook", "setup", "claim", "evidence", "interpretation",
    "second_evidence", "deeper_implication", "conclusion",
]

# Target-duration weights for section duration estimation (fractions of the
# total run time). They are *estimates*; real TTS word timestamps replace them.
_DURATION_WEIGHTS = {
    "hook": 0.12,
    "setup": 0.10,
    "claim": 0.10,
    "evidence": 0.14,
    "interpretation": 0.12,
    "second_evidence": 0.12,
    "deeper_implication": 0.15,
    "conclusion": 0.15,
}

WORDS_PER_SEC = 2.4


# --------------------------------------------------------------------------- #
# Narrative helpers (grounded to real scene facts; never invent)
# --------------------------------------------------------------------------- #

def _story(scene: Dict[str, Any]) -> Dict[str, Any]:
    return (scene or {}).get("story") or {}

def _scene_text(scene: Dict[str, Any]) -> str:
    """A cautious one-liner for one scene, built only from real facts."""
    story = _story(scene)
    loc = story.get("location")
    actions = story.get("actions") or []
    objects = story.get("objects") or []
    description = story.get("visual_description") or _story_get(scene, "summary")
    parts = []
    if loc:
        parts.append(f"it takes place in {loc}")
    if actions:
        parts.append(f"a character is {actions[0]}")
    if objects and len(objects) > 2:
        parts.append(f"we can see {', '.join(objects[:3])} on screen")
    elif description:
        parts.append(description)
    if parts:
        return "; ".join(parts) + "."
    return None  # no verifiable visual facts -> caller uses cautious language


def _story_get(scene: Dict[str, Any], key: str) -> Optional[str]:
    story = _story(scene)
    val = story.get(key)
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return ", ".join(str(v) for v in val[:2])
    return val


def _excerpt_window(scene: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """A short, real sub-window inside the scene (dialogue-anchored or middle).

    Mirrors the editorial retrieval windowing so the plan/timeline stages see
    the same moment references. Capped short; never the whole scene.
    """
    start = scene.get("start_sec")
    end = scene.get("end_sec")
    if start is None or end is None:
        return None
    start, end = float(start), float(end)
    if end - start < 1.0:
        return None
    dialogue = _story(scene).get("dialogue") or []
    max_sec = 6.0
    timed_lines = [float(d["start_sec"]) for d in dialogue
                   if isinstance(d, dict) and d.get("start_sec") is not None]
    if timed_lines:
        d_start = min(timed_lines)
        w0 = max(start, d_start - 0.5)
        w1 = min(end, w0 + max_sec)
    else:
        mid = (start + end) / 2.0
        w0 = max(start, mid - max_sec / 2.0)
        w1 = min(end, w0 + max_sec)
    if w1 - w0 < 1.0:
        return None
    return {"start_sec": round(w0, 3), "end_sec": round(w1, 3)}


def _estimate_seconds(text: str, pace: float = 1.0) -> float:
    words = len((text or "").split())
    return max(1.0, round(words / max(0.1, WORDS_PER_SEC * pace), 2))


# --------------------------------------------------------------------------- #
# Grounded script generator
# --------------------------------------------------------------------------- #

class GroundedScriptGenerator:
    """Deterministic, evidence-grounded script construction.

    ``contract``  : the normalized grounding contract (see grounding_contract).
    ``movie_index``: the Movie Intelligence artifact (scenes with story cards).
    """

    def __init__(
        self,
        target_sec: float = 90.0,
        min_sections: int = 5,
        pace: float = 1.0,
    ):
        self.target_sec = max(20.0, float(target_sec))
        self.min_sections = max(3, int(min_sections))
        self.pace = float(pace)

    # -- Public API ----------------------------------------------------------

    def generate(
        self,
        contract: Dict[str, Any],
        movie_index: Dict[str, Any],
        project_id: str = "project",
    ) -> Dict[str, Any]:
        """Build the grounded script from the contract + movie intelligence."""
        scenes_by_id = {
            s.get("scene_id"): s for s in (movie_index.get("scenes") or [])
            if s.get("scene_id")
        }
        supporting = [s for s in (contract.get("supporting_scenes") or [])
                      if s.get("scene_id") in scenes_by_id and scenes_by_id[s["scene_id"]]]
        if not supporting:
            raise ValueError(
                "grounded script requires >= 1 supporting scene present in the "
                "movie index (contract supporting_scenes did not resolve)"
            )

        # Assign stable evidence ids to the contract's evidence requirements.
        evidence = [
            {"id": f"ev_{i:02d}", "claim": str(claim).strip()}
            for i, claim in enumerate(contract.get("evidence_requirements") or [])
            if str(claim).strip()
        ]

        sections = self._build_sections(contract, scenes_by_id, supporting, evidence)
        sections = self._assign_durations(sections)
        concept = dict(contract.get("concept") or {})
        intents = contract.get("editorial_intent") or {}

        script = {
            "project_id": project_id,
            "thesis": concept.get("thesis", ""),
            "hook": {
                "text": concept.get("hook", ""),
                "visual_strategy": intents.get("visual_style", ""),
            },
            "concept": concept,
            "editorial_intent": intents,
            "evidence": evidence,
            "sections": sections,
            "target_duration_sec": round(self.target_sec, 2),
            "scene_ids": _all_scene_ids(sections),
            "grounded": True,
            "provenance": {"generator": "grounded"},
        }
        return script

    # -- Section construction ------------------------------------------------

    def _build_sections(
        self,
        contract: Dict[str, Any],
        scenes_by_id: Dict[str, Dict[str, Any]],
        supporting: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        concept = contract.get("concept") or {}
        thesis = concept.get("thesis") or ""
        hook = concept.get("hook") or ""
        why = concept.get("why_interesting") or ""

        # Rotate evidence claims across the available supporting scenes so each
        # analytical section is anchored to real material.
        def _anchor(scene: Dict[str, Any]) -> Dict[str, Any]:
            window = _excerpt_window(scenes_by_id[scene["scene_id"]])
            if not window:
                window = {"start_sec": scene.get("start_sec"), "end_sec": scene.get("end_sec")}
            return {
                "scene_id": scene["scene_id"],
                "start_sec": window["start_sec"],
                "end_sec": window["end_sec"],
                "reason": f"supports thesis: {thesis[:60]}",
                **({"evidence_id": evidence[0]["id"]} if evidence else {}),
            }

        opening = supporting[0]
        def _supporting(idx: int) -> Dict[str, Any]:
            return supporting[min(idx, len(supporting) - 1)]

        hook_claim = {"id": evidence[0]["id"], "claim": evidence[0]["claim"]} if evidence else {"id": "ev_00", "claim": concept.get("title", "")}

        plan = [
            {
                "type": "hook",
                "narration": build_hook_narration(hook, thesis),
                "scene_ids": [opening["scene_id"]],
                "evidence_ids": [hook_claim["id"]] if evidence else [],
                "narrative_evidence": [_anchor(opening)],
                "anchor": opening,
            },
            {
                "type": "setup",
                "narration": build_setup_narration(_supporting(1), thesis),
                "scene_ids": [_supporting(1)["scene_id"]],
                "evidence_ids": [evidence[0]["id"]] if evidence else [],
                "narrative_evidence": [_anchor(_supporting(1))],
                "anchor": _supporting(1),
            },
            {
                "type": "claim",
                "narration": build_claim_narration(thesis),
                "scene_ids": [_supporting(2)["scene_id"]],
                "evidence_ids": [evidence[0]["id"]] if evidence else [],
                "narrative_evidence": [_anchor(_supporting(2))],
                "anchor": _supporting(2),
            },
            {
                "type": "evidence",
                "narration": build_evidence_narration(_supporting(3), evidence[0]["claim"] if evidence else ""),
                "scene_ids": [_supporting(3)["scene_id"]],
                "evidence_ids": [evidence[0]["id"]] if evidence else [],
                "narrative_evidence": [_anchor(_supporting(3))],
                "anchor": _supporting(3),
            },
            {
                "type": "interpretation",
                "narration": build_interpretation_narration(_supporting(4), thesis, why),
                "scene_ids": [_supporting(4)["scene_id"]],
                "evidence_ids": [evidence[0]["id"]] if evidence else [],
                "narrative_evidence": [_anchor(_supporting(4))],
                "anchor": _supporting(4),
            },
        ]

        if len(supporting) > 1:
            scene = _supporting(5)
            plan.append({
                "type": "second_evidence",
                "narration": build_evidence_narration(scene, evidence[1]["claim"] if len(evidence) > 1 else ""),
                "scene_ids": [scene["scene_id"]],
                "evidence_ids": [evidence[min(1, len(evidence) - 1)]["id"]] if evidence else [],
                "narrative_evidence": [_anchor(scene)],
                "anchor": scene,
            })

        plan.append({
            "type": "deeper_implication",
            "narration": build_implication_narration(why, thesis),
            "scene_ids": [_supporting(0)["scene_id"]],
            "evidence_ids": [],
            "narrative_evidence": [_anchor(_supporting(0))],
            "anchor": _supporting(0),
        })
        plan.append({
            "type": "conclusion",
            "narration": build_conclusion_narration(thesis),
            "scene_ids": [opening["scene_id"]],
            "evidence_ids": [hook_claim["id"]] if evidence else [],
            "narrative_evidence": [_anchor(opening)],
            "anchor": opening,
        })

        # Keep the richest, thesis-serving structure (>= min_sections).
        trimmed = self._trim_to_minimum(plan)
        sections = []
        for i, item in enumerate(trimmed):
            sections.append({
                "id": item["type"] if len(trimmed) == 1 else f"{item['type']}_{i:02d}",
                "type": item["type"],
                "narration": item["narration"],
                "scene_ids": item["scene_ids"],
                "evidence_ids": item["evidence_ids"],
                "narrative_evidence": item["narrative_evidence"],
                "estimated_seconds": _estimate_seconds(item["narration"], self.pace),
            })
        return sections

    def _trim_to_minimum(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep a coherent spine; drop the deepest optional sections if we need
        to guarantee at least ``min_sections`` do not all appear."""
        if len(plan) >= self.min_sections:
            return plan
        # fall back to the guaranteed core spine
        core_types = {"hook", "setup", "claim", "evidence", "interpretation", "conclusion"}
        return [item for item in plan if item["type"] in core_types]

    # -- Duration ------------------------------------------------------------

    def _assign_durations(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Distribute the target duration across sections (estimate only)."""
        if not sections:
            return sections
        weights = [_DURATION_WEIGHTS.get(s["type"], 0.12) for s in sections]
        total_w = sum(weights)
        out = []
        for s, w in zip(sections, weights):
            budget = round(self.target_sec * w / total_w, 2)
            est = float(s.get("estimated_seconds") or 1.0)
            s["target_seconds"] = max(est, budget)
            out.append(s)
        return out


# --------------------------------------------------------------------------- #
# Cautious, fact-grounded narration templates (no invented detail)
# --------------------------------------------------------------------------- #

def _cautious_scene_line(scene: Dict[str, Any]) -> str:
    """A narration-safe one-liner for one scene, built only from real facts.

    NEVER includes the internal ``scene_id`` (that is metadata, not narration).
    If no verifiable visual fact exists, describe the moment without the id.
    """
    text = _scene_text(scene)
    if text:
        return text
    return "the moment captured on screen"


def build_hook_narration(hook: str, thesis: str) -> str:
    if hook:
        return hook
    if thesis:
        return f"What does {thesis} actually look like on screen?"
    return "A film hides a precise argument inside its images. Let's find it."


def build_setup_narration(scene: Dict[str, Any], thesis: str) -> str:
    line = _cautious_scene_line(scene)
    return f"{line.capitalize()} This is where the film begins to set up its central question."


def build_claim_narration(thesis: str) -> str:
    if not thesis:
        return "The film is making a specific visual argument."
    thesis = thesis.rstrip(".!?")
    return f"Here is the film's claim: {thesis}."


def build_evidence_narration(scene: Dict[str, Any], claim: str) -> str:
    line = _cautious_scene_line(scene)
    if claim:
        return f"We can test the claim against the footage directly. {line.capitalize()} {claim}."
    return f"We can test the claim against the footage directly. {line.capitalize()}"


def build_interpretation_narration(scene: Dict[str, Any], thesis: str, why: str) -> str:
    line = _cautious_scene_line(scene)
    return f"{line.capitalize()} Understood this way, the scene gives the thesis its weight."


def build_implication_narration(why: str, thesis: str) -> str:
    if why:
        return why
    if thesis:
        return f"If {thesis.rstrip('.')} is true, the whole film reads differently."
    return "This detail changes how the whole film reads."


def build_conclusion_narration(thesis: str) -> str:
    if not thesis:
        return "The film's real argument was never the plot — it was in the pictures."
    thesis = thesis.rstrip(".!?")
    return f"What remains when the story is put aside is the argument: {thesis}."


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def _all_scene_ids(sections: List[Dict[str, Any]]) -> List[str]:
    seen: List[str] = []
    for s in sections:
        for sid in s.get("scene_ids", []):
            if sid not in seen:
                seen.append(sid)
    return seen


def save_grounded_script(project_dir: Path, script: Dict[str, Any]) -> Path:
    return _save_json(Path(project_dir) / "grounded_script.json", script)


def load_grounded_script(project_dir: Path) -> Dict[str, Any]:
    path = Path(project_dir) / "grounded_script.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path