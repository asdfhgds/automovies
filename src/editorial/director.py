"""Editorial director.

Turns the (already LLM-generated) thesis + hook + grounding into an *Editorial
Decision List*: a set of deliberate creative decisions about why each moment of
the edit exists, which exact movie excerpts prove it, how it is framed and cut,
and what the audio does (milestone 1: "real editorial intelligence").

Architecture (per the milestone brief, sections 11-12):

    Grounded Script / Director Plan
        ↓
    Editorial Director                (this module)
        ↓
    Editorial Decision List           (editorial/decision.py)
        ↓
    EditorialPlan                     (compiled 1:1, for the renderer)
        ↓
    Timeline Compiler → FFmpeg Renderer

The heuristic editor is *evidence-first*: it sizes the narration to the actual
reachable footage (short excerpts, never whole scenes) and only repeats a scene
with a deliberate ``scene_reuse_justification``. The Qwen editor composes the
same decision list with a real LLM (``REQUIRE_REAL_LLM=true``), parsed and
validated fail-closed.
"""
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from editorial.decision import (
    Pacing,
    EditorialDecision,
    EditorialDecisionList,
    VisualStrategy,
    AudioIntent,
    EditingIntent,
    compile_editorial_plan,
    save_decision_list,
    validate_decision_list,
)
from editorial.plan import (
    EditorialEvidence,
    EditorialPlan,
    NarrationBlock,
    NarrationDelivery,
    validate_plan,
)
from editorial.retrieval import EvidenceRetriever, blend_director_requirements
from movie_understanding import movie_memory, text_utils

PROVIDER_ENV = "EDITORIAL_PLANNER"

WORDS_PER_SEC = 2.4
MAX_EXCERPT_SEC = 6.0
MIN_EXCERPT_SEC = 1.2


def _fit_words(text: str, max_words: int) -> str:
    """Trim text to fit a narration budget without breaking mid-word.

    Prefers to end at a sentence/punctuation boundary inside the budget;
    otherwise truncates to whole words. Never returns a string that exposes
    identifiers (all inputs here are already narration-safe prose).
    """
    text = (text or "").strip()
    max_words = max(1, int(max_words))
    words = text.split()
    if len(words) <= max_words:
        return text
    cut = words[: max_words]
    # Back off to the last punctuation boundary inside the budget.
    for i in range(len(cut), 0, -1):
        token = cut[i - 1]
        if token.endswith((".", "!", "?", ";", ":")) or token.endswith(("—", "...")):
            return " ".join(cut[:i])
    return " ".join(cut).rstrip(",-;:") + "."


def _words_for(duration_sec: float, pace: float) -> int:
    """Words obtainable inside a narration window (pace-scaled)."""
    return max(4, int(duration_sec * WORDS_PER_SEC * max(0.5, min(1.5, pace))))


def _delivery_for(purpose: str) -> NarrationDelivery:
    """Performance intent per creative beat (voice direction is preserved into
    TTS via the narration extractor)."""
    table = {
        "hook": NarrationDelivery(tone="inviting", emotion="intrigued",
                                  energy=0.6, pace=0.95, dramatic_intensity=0.4),
        "establish": NarrationDelivery(tone="analytical", emotion="thoughtful",
                                       energy=0.5, pace=1.0, dramatic_intensity=0.3),
        "contrast": NarrationDelivery(tone="observational", emotion="curious",
                                      energy=0.6, pace=1.05, dramatic_intensity=0.5),
        "reaction": NarrationDelivery(tone="quiet", emotion="moved",
                                      energy=0.35, pace=0.9, dramatic_intensity=0.6),
        "detail": NarrationDelivery(tone="analytical", emotion="focused",
                                    energy=0.45, pace=0.95, dramatic_intensity=0.4),
        "escalation": NarrationDelivery(tone="analytical", emotion="intense",
                                        energy=0.7, pace=1.1, dramatic_intensity=0.6),
        "resolve": NarrationDelivery(tone="resolute", emotion="certain",
                                     energy=0.55, pace=0.95, dramatic_intensity=0.5),
        "conclusion": NarrationDelivery(tone="quiet", emotion="resolute",
                                        energy=0.4, pace=0.85, dramatic_intensity=0.7),
    }
    return table.get(purpose, NarrationDelivery())


def _purpose_visual(purpose: str) -> VisualStrategy:
    table = {
        "hook": VisualStrategy("wide", "open the film's space; let the eye find meaning"),
        "establish": VisualStrategy("wide", "establish the pattern before breaking it"),
        "contrast": VisualStrategy("cross_cut", "cut between opposing images to make the contrast visible"),
        "reaction": VisualStrategy("reaction", "hold the character's face — the inner turn has no dialogue"),
        "detail": VisualStrategy("object_detail", "show the object/motif in isolation, larger than life"),
        "escalation": VisualStrategy("movement", "let motion build energy toward the turn"),
        "resolve": VisualStrategy("match_cut", "match the motif to its consequence"),
        "conclusion": VisualStrategy("motif_return", "return to the opening image, now re-read"),
    }
    return table.get(purpose, VisualStrategy())


def _purpose_audio(purpose: str) -> AudioIntent:
    table = {
        "hook": AudioIntent(movie_audio="duck", music="low", narration="dominant",
                            silence=""),
        "establish": AudioIntent(movie_audio="duck", music="none", narration="dominant"),
        "contrast": AudioIntent(movie_audio="retain", music="low", narration="dominant"),
        "reaction": AudioIntent(movie_audio="retain", music="swell", narration="anchor",
                                silence="moment of film sound before narration"),
        "detail": AudioIntent(movie_audio="retain", music="none", narration="dominant"),
        "escalation": AudioIntent(movie_audio="retain", music="rise", narration="dominant"),
        "resolve": AudioIntent(movie_audio="retain", music="resolve", narration="dominant"),
        "conclusion": AudioIntent(movie_audio="duck", music="resolve", narration="dominant",
                                  silence="final beat held in silence"),
    }
    return table.get(purpose, AudioIntent())


def _purpose_pacing(purpose: str, duration: float) -> Pacing:
    rhythm = {"hook": "slow", "establish": "medium", "contrast": "fast",
              "reaction": "slow", "detail": "medium", "escalation": "fast",
              "resolve": "medium", "conclusion": "slow"}.get(purpose, "medium")
    return Pacing(duration_sec=duration, rhythm=rhythm)


class EditorialPlanner(ABC):
    name = "base"

    @abstractmethod
    def create_decisions(
        self,
        movie_index: dict,
        director_plan: dict,
        retriever: EvidenceRetriever,
        creative_task: str,
        target_sec: float,
        source_duration: Optional[float] = None,
    ) -> EditorialDecisionList:
        raise NotImplementedError

    def create_plan(self, movie_index: dict, director_plan: dict,
                    retriever: EvidenceRetriever,
                    creative_task: str, target_sec: float,
                    source_duration: Optional[float] = None) -> EditorialPlan:
        """Compile the decision list into the downstream EditorialPlan."""
        decisions = self.create_decisions(
            movie_index, director_plan, retriever, creative_task,
            target_sec, source_duration,
        )
        return compile_editorial_plan(decisions)


class HeuristicEditorialPlanner(EditorialPlanner):
    """Deterministic, evidence-first editorial director.

    Decisions are deliberately varied (purpose / visual strategy / audio / pacing
    differ across the arc) and sized to the *reachable* footage: total narration
    never exceeds what short excerpts can actually show, so the renderer's
    coverage gate passes without stretching or padding.
    """

    name = "heuristic"

    def create_decisions(self, movie_index: dict, director_plan: dict,
                         retriever: EvidenceRetriever,
                         creative_task: str, target_sec: float,
                         source_duration: Optional[float] = None) -> EditorialDecisionList:
        thesis = (director_plan.get("thesis") or "").strip()
        if not thesis:
            raise ValueError("director_plan.thesis is required to build editorial decisions")
        hook = (director_plan.get("hook") or "").strip() or (
            f"What if the film was hiding its real argument in plain sight? {thesis}")
        title = (director_plan.get("title") or "A Film Reveals Its Argument").strip()

        scenes = movie_index.get("scenes", [])
        if not scenes:
            raise ValueError("movie_index has no scenes to draw evidence from")

        # Reachable excerpt windows: one short clip per scene (evidence-first).
        windows = self._reachable_windows(retriever, scenes, source_duration)
        if not windows:
            raise ValueError("no reachable excerpt windows in the movie index")

        arc = self._decide_arc(len(windows), float(target_sec))
        decisions: List[EditorialDecision] = []
        used_scenes: set = set()
        total_window_sec = sum(e.end_sec - e.start_sec for e in windows)

        # Split the evidence budget across the arc (share of total narration).
        weights = self._arc_weights(arc)
        total_weight = sum(weights)
        for i, purpose in enumerate(arc):
            seg_share = weights[i] / max(1, total_weight)
            ideal_sec = max(3.0, total_window_sec * seg_share)
            evidence = self._allocate_evidence(
                windows, used_scenes, ideal_sec, purpose=arc[min(i + 1, len(arc) - 1)],
            )
            if not evidence:
                continue
            covered_sec = sum(e.end_sec - e.start_sec for e in evidence)
            # Narration fits inside what the excerpt can actually show.
            delivery = _delivery_for(purpose)
            budget_words = _words_for(covered_sec, delivery.pace)
            text = _fit_words(self._narrate(purpose, director_plan, evidence), budget_words)
            if not text:
                text = _fit_words(self._narrate("establish", director_plan, evidence),
                                  budget_words)
            decisions.append(EditorialDecision(
                segment_id=f"seg_{i:02d}",
                purpose=purpose,
                narrative_beat=self._beat(purpose, thesis, evidence),
                evidence=evidence,
                narration=NarrationBlock(text=text, delivery=delivery),
                visual_strategy=_purpose_visual(purpose),
                pacing=_purpose_pacing(purpose, covered_sec),
                audio=_purpose_audio(purpose),
                editing=self._editing_for(purpose, i, len(arc)),
            ))

        if len(decisions) < 2:
            raise ValueError("editorial director could not form a coherent arc")

        # Deliberate reuse justification for any scene appearing more than once.
        justification = self._reuse_justifications(decisions)
        dl = EditorialDecisionList(
            title=title,
            thesis=thesis,
            hook={"text": hook, "visual_strategy": "wide opening image"},
            decisions=decisions,
            audio_defaults=AudioIntent(movie_audio="retain", music="none",
                                       narration="dominant"),
            length_target_sec=float(target_sec),
            creative_task=creative_task,
            provenance={"planner": self.name,
                        "thesis_source": director_plan.get("director_provider", "unknown")},
            scene_reuse_justification=justification,
        )
        errors = validate_decision_list(dl)
        if errors:
            raise ValueError("invalid decision list: " + "; ".join(errors))
        return dl

    # ------------------------------------------------------------------

    def _reachable_windows(self, retriever: EvidenceRetriever, scenes: List[dict],
                           source_duration: Optional[float]) -> List:
        """One short, real sub-window per scene, clamped to the source."""
        query = "the film's central image and its meaning"
        hits = retriever.retrieve(query, k=len(scenes))
        out = []
        for ev in hits:
            start, end = ev.start_sec, ev.end_sec
            if source_duration is not None:
                end = min(end, float(source_duration))
                if start >= float(source_duration) - 1e-6:
                    continue
            if end - start < MIN_EXCERPT_SEC:
                continue
            out.append(type(ev)(scene_id=ev.scene_id,
                                start_sec=round(start, 3),
                                end_sec=round(end, 3),
                                reason=ev.reason))
        return out

    def _decide_arc(self, n_windows: int, target_sec: float) -> List[str]:
        """A purposeful, varied arc sized to available evidence."""
        arc = ["hook", "establish", "contrast", "detail", "escalation",
               "resolve", "conclusion"]
        if n_windows <= 4:
            return ["hook", "contrast", "detail", "conclusion"]
        if n_windows <= 7 or target_sec < 90:
            return ["hook", "establish", "contrast", "detail", "conclusion"]
        return arc

    def _arc_weights(self, arc: List[str]) -> List[float]:
        return [{"hook": 0.9, "establish": 1.0, "contrast": 1.0, "detail": 0.9,
                 "escalation": 1.0, "resolve": 0.9, "conclusion": 1.0}.get(p, 1.0)
                for p in arc]

    def _allocate_evidence(self, windows: List, used_scenes: set,
                           ideal_sec: float, purpose: str) -> List:
        """Pick distinct scenes until the segment's share is covered.

        Reuses a scene only when the pool is exhausted and the purpose wants a
        motif return / match-cut (the justification is recorded by the caller).
        """
        chosen: List = []
        covered = 0.0
        for w in windows:
            if w.scene_id in used_scenes and len(used_scenes) >= len(windows):
                pass  # pool exhausted; deliberate reuse allowed below
            elif w.scene_id in used_scenes:
                continue
            chosen.append(w)
            used_scenes.add(w.scene_id)
            covered += w.end_sec - w.start_sec
            if covered >= ideal_sec:
                break
        if not chosen:
            # Deliberate motif reuse: fall back to the strongest window.
            for w in windows:
                if w.scene_id not in used_scenes:
                    chosen.append(w)
                    used_scenes.add(w.scene_id)
                    break
        return chosen

    def _editing_for(self, purpose: str, idx: int, total: int) -> EditingIntent:
        if purpose in ("conclusion",):
            return EditingIntent(transition="fade", speed=1.0, hold=True,
                                 crop_zoom=1.0, fade_edges=True)
        if purpose in ("hook",):
            return EditingIntent(transition="cut", speed=1.0)
        if purpose == "reaction":
            return EditingIntent(transition="crossfade", hold=True, crop_zoom=1.2)
        if purpose == "contrast":
            return EditingIntent(transition="cut", speed=1.05)
        if purpose == "escalation":
            return EditingIntent(transition="cut", speed=1.1)
        return EditingIntent(transition="crossfade", speed=1.0)

    # -- narration (narration-safe prose; never scene ids / metadata) -------

    def _narrate(self, purpose: str, director_plan: dict, evidence: List) -> str:
        thesis = (director_plan.get("thesis") or "").strip().rstrip(".!?")
        hook = (director_plan.get("hook") or "").strip()
        scene_line = self._scene_line(evidence[0]) if evidence else ""
        if purpose == "hook":
            return hook or f"What if the film hides its real argument in plain sight?"
        if purpose == "establish":
            return (f"The film sets up its question in a moment most people "
                    f"skip. {scene_line} This is where the pattern begins.")
        if purpose == "contrast":
            return (f"Cut from one image to the other and the contradiction "
                    f"surfaces: {scene_line} The film is arguing, not narrating.")
        if purpose == "reaction":
            return (f"The character does not explain the turn. Their silence "
                    f"does. {scene_line}")
        if purpose == "detail":
            return f"The smallest object carries the whole idea. {scene_line}"
        if purpose == "escalation":
            return (f"The images speed up as the film closes on its answer: "
                    f"{thesis}.")
        if purpose == "resolve":
            return f"The question resolves on screen. {thesis}."
        if purpose == "conclusion":
            return (f"Strip away the plot and {thesis} is what remains — "
                    f"an argument the movie makes with pictures.")
        return f"This moment carries the film's central claim. {thesis}."

    def _beat(self, purpose: str, thesis: str, evidence: List) -> str:
        first = evidence[0].scene_id if evidence else "an opening image"
        whole = "; ".join(e.scene_id for e in evidence[:3])
        purpose_text = {
            "hook": "open the argument on the screen",
            "establish": "establish the visual pattern the thesis exposes",
            "contrast": "cut between opposing images so the contradiction is visible",
            "reaction": "show the inner turn that has no dialogue",
            "detail": "isolate the motif that proves the claim",
            "escalation": "build motion toward the turn",
            "resolve": "land the thesis with a matched image",
            "conclusion": "return to the opening image, re-read",
        }.get(purpose, "support the thesis with footage")
        return f"{purpose_text} (scenes: {whole or first})"

    def _scene_line(self, ev) -> str:
        """A narration-safe one-liner from the evidence (no scene ids)."""
        reason = (ev.reason or "").strip()
        if reason and "\n" not in reason and len(reason) < 40:
            return reason.rstrip(".")
        return "Watch how the image itself changes meaning."

    def _reuse_justifications(self, decisions: List[EditorialDecision]) -> Dict[str, str]:
        counts: Dict[str, int] = {}
        for d in decisions:
            for e in d.evidence:
                counts[e.scene_id] = counts.get(e.scene_id, 0) + 1
        return {
            sid: "deliberate motif return: the image opens and closes the argument"
            for sid, n in counts.items() if n > 1
        }


class QwenEditorialPlanner(EditorialPlanner):
    """Real-LLM editorial director.

    Composes the same Editorial Decision List with a genuine LLM
    (``provider.generate_text`` or an injected ``llm`` callable for tests).
    Gated behind ``REQUIRE_REAL_LLM=true`` so a production run can never
    silently degrade to the heuristic path: an invalid LLM response fails the
    editorial stage instead of being silently "fixed".
    """

    name = "qwen"

    def __init__(self, llm=None, strict: bool = True):
        """``llm``: callable ``prompt -> str`` (default: resolved Qwen provider).
        ``strict``: refuse heuristic fallback when the LLM output is invalid."""
        if not os.getenv("REQUIRE_REAL_LLM", "").lower() == "true":
            raise RuntimeError("QwenEditorialPlanner requires REQUIRE_REAL_LLM=true")
        self._llm = llm
        self._strict = strict

    def _resolve_llm(self):
        if self._llm is not None:
            return self._llm
        from director.provider_factory import (
            get_director_config_from_env,
            get_llm_provider_from_config,
        )
        provider = get_llm_provider_from_config(get_director_config_from_env())
        if provider is None:
            raise RuntimeError("Qwen editorial planner: no LLM provider configured")
        return provider.generate_text

    def create_decisions(self, movie_index: dict, director_plan: dict,
                         retriever: EvidenceRetriever,
                         creative_task: str, target_sec: float,
                         source_duration: Optional[float] = None) -> EditorialDecisionList:
        llm = self._resolve_llm()
        prompt = self._build_prompt(movie_index, director_plan, creative_task,
                                    target_sec)
        raw = llm(prompt)
        decisions = self._parse_decision_list(raw, movie_index, director_plan,
                                              creative_task, target_sec)
        errors = validate_decision_list(decisions)
        if errors:
            raise ValueError("Qwen editorial planner returned an invalid decision "
                             "list: " + "; ".join(errors))
        return decisions

    # ------------------------------------------------------------------

    def _build_prompt(self, movie_index: dict, director_plan: dict,
                      creative_task: str, target_sec: float) -> str:
        scenes = []
        for s in movie_index.get("scenes", [])[:40]:
            story = s.get("story") or {}
            scenes.append({
                "scene_id": s.get("scene_id"),
                "start_sec": s.get("start_sec"),
                "end_sec": s.get("end_sec"),
                "summary": (story.get("summary") or "")[:240],
                "topics": (story.get("topics") or [])[:4],
                "location": story.get("location"),
                "objects": (story.get("objects") or [])[:4],
                "dialogue": [
                    (d.get("text", "") if isinstance(d, dict) else str(d))[:120]
                    for d in (story.get("dialogue") or [])[:3]
                ],
            })
        return (
            "You are the lead editorial director of a film-essay studio. The "
            "producer gives you a thesis, a hook, and the movie's scene "
            "intelligence. You decide the EDIT, shot by shot.\n\n"
            f"Thesis: {director_plan.get('thesis', '')}\n"
            f"Hook: {director_plan.get('hook', '')}\n"
            f"Creative task: {creative_task}\n"
            f"Target duration: {target_sec:.0f}s\n\n"
            "Scenes (id, window, summary, topics, objects, dialogue):\n"
            + "\n".join(
                f"- {s['scene_id']} [{s['start_sec']}..{s['end_sec']}] "
                f"{s['location'] or ''} | {s['summary']} | "
                f"topics={s['topics']} | objects={s['objects']} | "
                f"dialogue={s['dialogue']}"
                for s in scenes[:30]
            )
            + "\n\nReturn a JSON object with schema:\n"
            "{\n"
            '  "title": str,\n'
            '  "hook": {"text": str, "visual_strategy": str},\n'
            '  "scene_reuse_justification": {"<scene_id>": "why reused"},"\n'
            '  "decisions": [\n'
            "    {\n"
            '      "segment_id": str,\n'
            '      "purpose": "hook|establish|contrast|reaction|detail|escalation|resolve|conclusion",\n'
            '      "narrative_beat": str (why this shot exists),\n'
            '      "evidence": [{"scene_id": str, "start_sec": float, "end_sec": float, "reason": str}],\n'
            '      "narration": {"text": str, "delivery": {"tone": str, "emotion": str, "energy": 0..1, "pace": 0.5..1.5}},\n'
            '      "visual_strategy": {"type": "wide|medium|close_up|reaction|object_detail|environment|cross_cut|motif_return|contrast|hold|match_cut", "description": str},\n'
            '      "pacing": {"duration_sec": float, "rhythm": "slow|medium|fast"},\n'
            '      "audio": {"movie_audio": "retain|duck|mute", "music": "none|low|rise|swell|resolve|silence", "narration": "dominant|anchor|absent", "silence": str},\n'
            '      "editing": {"transition": "cut|crossfade|fade", "speed": float, "hold": bool, "crop_zoom": float}\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Rules: every decision MUST use a real scene_id and a window that is "
            "<= 6 seconds (short excerpts, never whole scenes). Narration text "
            "must be plain spoken prose — no ids, no JSON, no metadata. Never "
            "repeat a scene without scene_reuse_justification. The whole cut must "
            "be intentionally edited: varied visual strategies, one silence/hold, "
            "an explicit audio intent per shot."
        )

    def _parse_decision_list(self, raw: str, movie_index: dict,
                             director_plan: dict, creative_task: str,
                             target_sec: float) -> EditorialDecisionList:
        from director.prompts.json_utils import extract_json

        data = extract_json(raw)
        if not data or not isinstance(data, dict):
            raise ValueError("Qwen editorial planner returned no parseable JSON")
        if not data.get("decisions"):
            raise ValueError("Qwen editorial planner returned no decisions")

        scenes = {s.get("scene_id"): s for s in movie_index.get("scenes", [])}
        decisions: List[EditorialDecision] = []
        for i, raw_d in enumerate(data["decisions"]):
            if not isinstance(raw_d, dict):
                continue
            evidence = []
            for e in raw_d.get("evidence", []):
                scene = scenes.get(str(e.get("scene_id")))
                if not scene:
                    continue
                start = float(e.get("start_sec", scene.get("start_sec", 0.0)))
                end = float(e.get("end_sec", scene.get("end_sec", 0.0)))
                start, end = sorted((start, end))
                end = min(end, start + MAX_EXCERPT_SEC)
                if end - start < MIN_EXCERPT_SEC:
                    continue
                evidence.append(EditorialEvidence(
                    scene_id=scene["scene_id"], start_sec=round(start, 3),
                    end_sec=round(end, 3),
                    reason=str(e.get("reason", "llm choice"))))
            if not evidence:
                continue  # never compile a decision with no real footage

            purpose = str(raw_d.get("purpose", "detail"))
            if purpose not in ("hook", "establish", "contrast", "reaction",
                               "detail", "escalation", "resolve", "conclusion"):
                purpose = "detail"
            nar = raw_d.get("narration") or {}
            deliv = nar.get("delivery") or {}
            try:
                energy = float(deliv.get("energy", 0.5))
                pace = float(deliv.get("pace", 0.95))
                dur = float(raw_d.get("pacing", {}).get("duration_sec", 4.0))
            except (TypeError, ValueError):
                energy, pace, dur = 0.5, 0.95, 4.0
            decisions.append(EditorialDecision(
                segment_id=str(raw_d.get("segment_id") or f"seg_{i:02d}"),
                purpose=purpose,
                narrative_beat=str(raw_d.get("narrative_beat", "")) or "support the thesis",
                evidence=evidence,
                narration=NarrationBlock(
                    text=str(nar.get("text", "")),
                    delivery=NarrationDelivery(
                        tone=str(deliv.get("tone", "analytical")),
                        emotion=str(deliv.get("emotion", "thoughtful")),
                        energy=max(0.0, min(1.0, energy)),
                        pace=max(0.5, min(1.5, pace)),
                    ),
                ),
                visual_strategy=VisualStrategy.from_dict(raw_d.get("visual_strategy")),
                pacing=Pacing.from_dict(raw_d.get("pacing")) if dur > 0 else Pacing(duration_sec=4.0),
                audio=AudioIntent.from_dict(raw_d.get("audio")),
                editing=EditingIntent.from_dict(raw_d.get("editing")),
            ))

        if len(decisions) < 2:
            raise ValueError("Qwen editorial planner: too few usable decisions")

        title = str(data.get("title") or director_plan.get("title")
                    or "A Film Reveals Its Argument")
        hook_data = data.get("hook") or {}
        hook_text = str(hook_data.get("text") or director_plan.get("hook")
                        or "Watch the images closely.")
        return EditorialDecisionList(
            title=title,
            thesis=str(director_plan.get("thesis") or data.get("thesis") or ""),
            hook={"text": hook_text,
                  "visual_strategy": str(hook_data.get("visual_strategy", ""))},
            decisions=decisions,
            audio_defaults=AudioIntent.from_dict(
                director_plan.get("editorial_intent", {}).get("audio_style")) if isinstance(
                    director_plan.get("editorial_intent", {}).get("audio_style"), dict) else AudioIntent(),
            length_target_sec=float(target_sec),
            creative_task=creative_task,
            provenance={"planner": self.name,
                        "thesis_source": director_plan.get("director_provider", "qwen")},
            scene_reuse_justification=dict(data.get("scene_reuse_justification") or {}),
        )


def editorial_planner_from_env() -> EditorialPlanner:
    """Provider factory: ``EDITORIAL_PLANNER=heuristic|qwen`` (default heuristic)."""
    provider = os.getenv(PROVIDER_ENV, "").strip().lower() or "heuristic"
    if provider == "qwen":
        return QwenEditorialPlanner()
    return HeuristicEditorialPlanner()


def create_editorial_plan(project_dir: Path, creative_task: str = "",
                          target_sec: float = 90.0,
                          planner: Optional[EditorialPlanner] = None) -> EditorialPlan:
    """Pipeline entry point: reads movie_index + director plan, writes the
    Editorial Decision List (``editorial_decisions.json``) and the compiled
    ``editorial_plan.json`` at the project root."""
    from editorial.decision import compile_editorial_plan

    project_dir = Path(project_dir)
    movie_index = movie_memory.load_movie_index(project_dir)
    director_plan = movie_memory.load_json(project_dir, "director_plan.json", {})
    retriever = EvidenceRetriever.from_project_dicts(movie_index)
    planner = planner or editorial_planner_from_env()

    source_duration = _probe_source_duration(project_dir)
    decisions = planner.create_decisions(
        movie_index, director_plan, retriever,
        creative_task or director_plan.get("creative_task", ""),
        float(target_sec),
        source_duration=source_duration,
    )
    save_decision_list(project_dir, decisions)
    plan = compile_editorial_plan(decisions)
    movie_memory.save_json(project_dir, "editorial_plan.json", plan.to_dict())
    return plan


def _probe_source_duration(project_dir: Path) -> Optional[float]:
    """Best-effort source movie duration so evidence sizing is honest about the
    footage that actually exists (never narrate past the end of the movie)."""
    meta = movie_memory.load_json(project_dir, "project_meta.json", {})
    source = meta.get("source_path")
    if not source or not Path(source).exists():
        return None
    try:
        from editor.clip_extractor import probe_duration
        return probe_duration(source) or None
    except Exception:
        return None