"""Director context builder: compact, fact-grounded context for the LLM.

The Movie Intelligence scene index can be thousands of lines of JSON. Dumping it
raw into the prompt drowns the model in irrelevant detail and invites
hallucination. This builder instead renders a drastically compressed, per-scene
summary plus a "only these facts exist" vocabulary, capped at a token budget.

Two kinds of output are produced:

- ``build_concept_generation_context`` — everything the director needs to
  brainstorm 5 varied concepts AND to be told what it may (and must not) invent.
- ``build_plan_context`` — a tighter slice for producing the final plan for the
  selected concept.

Both are deterministic. No LLM calls happen here.
"""
import json
from typing import Dict, Any, List, Optional, Tuple

from director.concepts import concept_refs, render_ref
from director.scene_facts import SceneFacts, _as_list


def _fmt_ts(value) -> str:
    if value is None:
        return "?–?"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def scene_summary(sf, index: int = 0) -> str:
    """One compact, labeled summary line for a single scene.

    Mirrors the milestone example::

        SCENE 18
        Time: 48.3-51.4
        Characters: Barman
        Actions: speaking, holding revolver
        Objects: revolver
        Visual: close-up, low-key lighting
        Mood: tense
        Themes: violence, confrontation
        Dialogue: ...
    """
    shown = sf.scene_id
    chars = ", ".join(sf.characters) or "unknown_character_01 (low confidence)"
    actions = ", ".join(sf.actions) or "—"
    objects = ", ".join(sf.objects) or "—"
    dialogue = sf.dialogue_text.strip()
    lines = [
        f"SCENE {shown}",
        f"Time: {_fmt_ts(sf.start_sec)}–{_fmt_ts(sf.end_sec)}",
        f"Characters: {chars}",
        f"Actions: {actions}",
        f"Objects: {objects}",
    ]
    if sf.visual_description:
        lines.append(f"Visual: {sf.visual_description}")
    if sf.cinematography:
        lines.append(f"Cinematography: {sf.cinematography}")
    if sf.mood:
        lines.append(f"Mood: {sf.mood}")
    if sf.themes:
        lines.append(f"Themes: {', '.join(sf.themes)}")
    if sf.visual_events:
        lines.append(f"Visual events: {', '.join(sf.visual_events[:4])}")
    if dialogue:
        lines.append(f"Dialogue: {dialogue[:220]}")
    return "\n".join(lines)


def scene_inventory_line(sf) -> str:
    """One-line compact inventory entry for a scene (PASS 1 — full movie)."""
    shown = sf.scene_id
    time = f"{_fmt_ts(sf.start_sec)}–{_fmt_ts(sf.end_sec)}"
    chars = ", ".join(sf.characters) if sf.characters else "—"
    loc = sf.location or "—"
    acts = ", ".join(sf.actions[:3]) if sf.actions else "—"
    objs = ", ".join(sf.objects[:4]) if sf.objects else "—"
    visual = sf.visual_description[:60] if sf.visual_description else "—"
    mood = sf.mood or "—"
    themes = ", ".join(sf.themes[:3]) if sf.themes else "—"
    return (f"{shown}: {time} | chars: {chars} | loc: {loc} | "
            f"acts: {acts} | objs: {objs} | visual: {visual} | "
            f"mood: {mood} | themes: {themes}")


class DirectorContextBuilder:
    """Builds token-limited director context from ``SceneFacts``."""

    def __init__(self, max_tokens: int = 4096, reserve_for_output: int = 2048):
        self.max_tokens = max_tokens
        self.reserve_for_output = reserve_for_output
        self.available = max_tokens - reserve_for_output

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (1 token ~ 4 chars)."""
        return len(text or "") // 4

    def _fit(self, parts: List[str], budget: int, meta: Dict[str, Any],
             meta_key: str, truncated_key: str = None) -> List[str]:
        """Append as many parts as fit within ``budget`` of remaining tokens."""
        used = 0
        included = 0
        kept = []
        for p in parts:
            t = self._estimate_tokens(p)
            if used + t > budget:
                meta[truncated_key or "truncated"] = True
                break
            kept.append(p)
            used += t
            included += 1
        meta[meta_key] = included
        return kept

    def _grounded_example(self, scene_facts: SceneFacts) -> str:
        """A worked, fully-grounded example concept built ONLY from real facts.

        Unlike a hardcoded template this example is *dynamic*: it picks the
        scene with the richest citable facts (most objects + actions + theme /
        mood), renders a complete concept whose ``evidence_refs`` are copied
        verbatim from that scene's own cards, and appends a **rejected
        contrast** — a plausible-but-absent ref the exact matcher WOULD reject —
        so the model sees the PASS/FAIL boundary. Returns an empty string when
        no scene can support an object + action + theme/mood example.
        """
        # Pick the scene with the most citable fact kinds (richness, not order).
        best = None
        best_score = -1
        for sf in scene_facts:
            objects = [o for o in sf.objects if str(o).strip()]
            actions = [a for a in sf.actions if str(a).strip()]
            themes = list(dict.fromkeys(
                t for t in sf.themes if str(t).strip()))
            if not objects or not actions:
                continue
            mood = sf.mood if str(sf.mood or "").strip() else (
                themes[0] if themes else "")
            if not mood:
                continue
            score = len(objects) + len(actions) + (1 if mood else 0) + \
                (1 if sf.location else 0)
            if score > best_score:
                best_score = score
                best = sf
        if best is None:
            return ""
        sf = best
        objects = [o for o in sf.objects if str(o).strip()]
        actions = [a for a in sf.actions if str(a).strip()]
        themes = list(dict.fromkeys(
            t for t in sf.themes if str(t).strip()))
        mood = sf.mood if str(sf.mood or "").strip() else themes[0]
        refs = [
            {"kind": "scene", "scene_id": sf.scene_id},
            {"kind": "object", "value": objects[0]},
            {"kind": "action", "value": actions[0]},
            {"kind": "theme", "value": themes[0] if themes else mood},
        ]
        if len(objects) > 1:
            refs.append({"kind": "object", "value": objects[1]})
        if len(actions) > 1:
            refs.append({"kind": "action", "value": actions[1]})
        if sf.location and str(sf.location).strip():
            refs.append({"kind": "location", "value": str(sf.location).strip()})
        example = {
            "title": f"The story of the {objects[0]}",
            "hook": f"Start with {objects[0]} already on screen.",
            "thesis": (
                f"{objects[0]} unlocks the {mood} mood in {sf.scene_id} "
                f"through {actions[0]}."
            ),
            "why_interesting": (
                "Every element here (scene, object, action, theme/mood) is "
                "copied VERBATIM from the scene cards above; the exact matcher "
                "accepts it."
            ),
            "evidence_refs": refs,
            "visual_opportunity": f"Give {objects[0]} a close-up while "
                                   f"someone does {actions[0]}.",
            "format": "short_video_essay",
            "diversity_angle": "grounding example — do not reuse verbatim",
        }

        # Deterministic REJECTED contrast: a real-but-absent phrase. We try a
        # few likely hallucination words; the first that no scene contains is
        # shown as the "would FAIL" case, so the model sees the pass boundary.
        contrast = None
        for candidate in (
            "broken clock", "kitchen table", "photograph", "apartment",
            "dinner plate", "red dress",
        ):
            if not scene_facts.is_grounded(candidate):
                contrast = candidate
                break
        example_txt = json.dumps(example, indent=2)
        if not contrast:
            return example_txt
        return (
            example_txt + "\n\n"
            "REJECTED CONTRAST (do NOT copy this pattern):\n"
            '  evidence_ref kind=object value="' + contrast + '"\n'
            '  Reason: "' + contrast + '" appears in NO scene card in this '
            "movie, so the exact matcher REJECTS it. Only cite identifiers "
            "listed verbatim in the scene cards / WHAT ACTUALLY EXISTS."
        )

    def build_movie_inventory(self, scene_facts: SceneFacts) -> str:
        """Build a compact one-line inventory of ALL scenes (PASS 1).

        Every scene gets a single line with high-value facts. This gives the
        director a complete movie overview without token-budget truncation.
        """
        lines = ["## FULL MOVIE INVENTORY (ALL SCENES)"]
        for sf in scene_facts:
            lines.append(scene_inventory_line(sf))
        return "\n".join(lines)

    def build_concept_generation_context(
        self,
        movie_metadata: Dict[str, Any],
        scene_facts: SceneFacts,
        creative_memory: str = "",
        user_topic: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Build the full context for the concept brainstorm.

        Two-level context:
        - PASS 1: Compact inventory of ALL scenes (always included)
        - PASS 2: Deep scene context for a subset (truncated if needed)
        """
        total_scenes = len(scene_facts)
        meta = {
            "total_scenes": total_scenes,
            "inventory_scenes": total_scenes,
            "detailed_scenes": 0,
            "memory_included": False,
            "example_included": False,
            "truncated": False,
        }

        parts: List[str] = []

        # 1. Movie metadata + hard grounding rules (always included, tiny).
        title = movie_metadata.get("title", "Unknown")
        duration = movie_metadata.get("duration_sec", 0)
        try:
            dur_str = f"{float(duration):.1f}s"
        except (TypeError, ValueError):
            dur_str = str(duration)
        grounding = (
            "## GROUNDING RULES (MANDATORY)\n"
            "1. Separate your CREATIVE CLAIM (title / hook / thesis /\n"
            "why_interesting) from your EVIDENCE REFERENCES (evidence_refs).\n"
            "The thesis is your interpretation; evidence_refs may ONLY name\n"
            "facts listed in this context.\n"
            "2. You may ONLY reference the characters, locations, objects,\n"
            "actions, themes, moods, and dialogue listed below. These are\n"
            "everything known to exist.\n"
            "3. If a fact is missing or uncertain, say so explicitly. NEVER\n"
            "invent a character name, line of dialogue, object, location,\n"
            "action, or scene.\n"
            "4. Refer to scenes by their SCENE id exactly as shown (e.g.\n"
            "\"scene-1\"). Prefer a scene ref whenever a specific scene carries\n"
            "your point.\n"
            "5. Every evidence_ref must be a canonical identifier from this\n"
            "context: {\"kind\": \"scene\", \"scene_id\": \"scene-1\"} or\n"
            "{\"kind\": \"object\", \"value\": \"...\"}, and so on for\n"
            "character / location / action / event / theme / mood / dialogue.\n"
            "6. The matcher is exact and token-based: \"son\" will NOT match\n"
            "just because \"person\" also appears here."
        )
        header_part = f"## MOVIE\nTitle: {title}\nDuration: {dur_str}\n\n{grounding}"
        parts.append(header_part)

        # 2. PASS 1: Full movie inventory (compact, all scenes).
        inventory = self.build_movie_inventory(scene_facts)
        parts.append(inventory)
        meta["inventory_tokens"] = self._estimate_tokens(inventory)

        # 3. Fact vocabulary (known names — hallucination guard).
        def _unique(values, limit=24):
            seen, out = set(), []
            for v in values:
                key = str(v).strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    out.append(str(v).strip())
            return out[:limit]

        actions = _unique(a for sf in scene_facts for a in sf.actions)
        themes = _unique(t for sf in scene_facts for t in sf.themes)
        moods = _unique(m for sf in scene_facts for m in _as_list(sf.mood))
        vocab_lines = [
            f"- Known characters: {', '.join(scene_facts.known_characters()) or '(none identified)'}",
            f"- Known locations: {', '.join(scene_facts.known_locations()) or '(none identified)'}",
            f"- Known objects: {', '.join(scene_facts.known_objects()) or '(none identified)'}",
            f"- Known actions: {', '.join(actions) or '(none identified)'}",
            f"- Known themes: {', '.join(themes) or '(none identified)'}",
            f"- Known moods: {', '.join(moods) or '(none identified)'}",
        ]
        vocab = "## WHAT ACTUALLY EXISTS\n" + "\n".join(vocab_lines)
        if self._estimate_tokens(vocab) <= int(self.available * 0.7):
            parts.append(vocab)

        # 4. Worked, fully-grounded example (shows the exact ref style).
        example = self._grounded_example(scene_facts)
        example_part = ""
        if example:
            example_part = (
                "## WORKED EXAMPLE (already verified against this movie)\n"
                "Treat this as a template ONLY for the evidence_refs style —\n"
                "every value below is copied verbatim from the scene cards, so\n"
                "it passes the exact matcher. Write your own thesis/hook; do\n"
                "not copy this concept.\n\n" + example
            )

        # 5. PASS 2: Deep scene context (truncated by token budget).
        reserved_now = self._estimate_tokens("\n\n".join(parts)) + (
            self._estimate_tokens(example_part) if example_part else 0
        )
        scene_budget = max(0, self.available - reserved_now)
        scene_parts = [scene_summary(sf, i) for i, sf in enumerate(scene_facts)]
        chosen = self._fit(scene_parts, scene_budget, meta, "detailed_scenes",
                           "truncated")
        if chosen:
            parts.append("## DETAILED SCENE CONTEXT\n" + "\n\n".join(chosen))
        if example_part:
            parts.append(example_part)
            meta["example_included"] = True

        # 6. Creative memory (avoid repetition) if it fits.
        if creative_memory and creative_memory.strip():
            mem_tokens = self._estimate_tokens(creative_memory)
            if self._estimate_tokens("\n\n".join(parts)) + mem_tokens <= self.available:
                parts.append(f"## PREVIOUS CONCEPTS (DO NOT REPEAT)\n{creative_memory}")
                meta["memory_included"] = True

        if user_topic:
            parts.append(f"## USER FOCUS\n{user_topic}")

        context = "\n\n".join(parts)
        meta["estimated_tokens"] = self._estimate_tokens(context)
        return context, meta

    def build_plan_context(
        self,
        concept: Dict[str, Any],
        scene_facts: SceneFacts,
        selected_scene_ids: List[str],
    ) -> str:
        """Context for producing the final plan, scoped to the chosen scenes."""
        selected = [
            sf for sf in scene_facts
            if sf.scene_id in set(selected_scene_ids)
        ]
        if not selected:
            selected = list(scene_facts.scenes)
        scene_blob = "\n\n".join(scene_summary(sf, i) for i, sf in enumerate(selected))
        ref_lines = "\n".join(
            f"- {render_ref(r)} ({r.get('kind')})"
            for r in concept_refs(concept)
        ) or "- (none)"

        # Verbatim vocabulary of the EVIDENCE scenes only — the plan model may
        # cite concrete objects/characters/locations ONLY from this list.
        def _uniq(values):
            seen, out = set(), []
            for v in values:
                key = str(v).strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    out.append(str(v).strip())
            return out

        ev_objects = _uniq(o for sf in selected for o in sf.objects)
        ev_chars = _uniq(c for sf in selected for c in sf.characters)
        ev_locs = _uniq(sf.location for sf in selected if sf.location)
        ev_actions = _uniq(a for sf in selected for a in sf.actions)
        ev_themes = _uniq(t for sf in selected for t in sf.themes)
        ev_moods = _uniq(m for sf in selected for m in _as_list(sf.mood))
        ev_vocab = (
            "## VERBATIM VOCABULARY FOR THE EVIDENCE SCENES ONLY\n"
            "You may cite concrete characters, objects, locations, actions,\n"
            "themes, or moods ONLY if they are listed here (copy verbatim, do\n"
            "not paraphrase or pluralize):\n"
            f"- Characters: {', '.join(ev_chars) or '(none identified)'}\n"
            f"- Locations: {', '.join(ev_locs) or '(none identified)'}\n"
            f"- Objects: {', '.join(ev_objects) or '(none identified)'}\n"
            f"- Actions: {', '.join(ev_actions) or '(none identified)'}\n"
            f"- Themes: {', '.join(ev_themes) or '(none identified)'}\n"
            f"- Moods: {', '.join(ev_moods) or '(none identified)'}"
        )

        # A worked, already-grounded editorial_direction example (deterministic)
        # so the model sees the only acceptable object/mood vocabulary.
        example_ref_lines = []
        for sf in selected:
            objs = [o for o in sf.objects if str(o).strip()]
            mood = sf.mood if str(sf.mood or "").strip() else (
                sf.themes[0] if sf.themes else "")
            if not objs or not mood:
                continue
            example_ref_lines.append(
                f"- Scene {sf.scene_id}: hold the \"{objs[0]}\" and the "
                f"\"{mood}\" mood."
            )
            break
        worked_example = ""
        if example_ref_lines:
            worked_example = (
                "## WORKED EDITORIAL EXAMPLE (ALREADY GROUNDED — imitate the "
                "restraint, write your own words, stay inside the whitelist)\n"
                + "\n".join(example_ref_lines)
                + "\n"
                "Pacing: slow and measured.\n"
                "Visual style: minimal, grounded in the scenes above and "
                "their stillness.\n"
                "Audio style: quiet; silence and sparse sound.\n"
                "Editing style: long takes and steady cuts, soft focus.\n"
            )

        return (
            f"## SELECTED CONCEPT\n"
            f"Title: {concept.get('title', '')}\n"
            f"Hook: {concept.get('hook', '')}\n"
            f"Thesis: {concept.get('thesis', '')}\n"
            "\n"
            f"## GROUNDED EVIDENCE REFS (already verified against the movie)\n"
            f"{ref_lines}\n"
            "\n"
            f"## EVIDENCE SCENES (the Only proven scenes)\n{scene_blob}\n"
            "\n"
            f"{ev_vocab}\n"
            "\n"
            f"{worked_example}"
        )
