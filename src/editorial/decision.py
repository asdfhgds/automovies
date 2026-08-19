"""Editorial Decision List: the creative decision model produced by the
Editorial Director.

The Editorial Director is a *creative decision-making layer*, not a renderer.
For every moment of the edit it decides *why* a shot is present (narrative
beat), *which* exact movie excerpts prove it, *how* it should be framed and cut
(visual strategy, pacing, editing), and *what* the audio should do
(movie audio retain/duck/mute, music intent, silence).

The timeline compiler and FFmpeg renderer convert these decisions into a
render; they never invent editorial intent on their own (see section 11-12 of
the milestone brief).

The schema deliberately mirrors the brief's example decision record:

    segment_id, purpose, narrative_beat, evidence (scene_id + exact window),
    visual_strategy {type, description}, pacing {duration_sec, rhythm},
    audio {movie_audio, music, narration}, editing {transition, speed, hold}.
"""
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from editorial.plan import (
    EditorialEvidence,
    NarrationBlock,
    NarrationDelivery,
)

# ---------------------------------------------------------------------------
# Controlled vocabularies (keep the renderer's op set in sync)
# ---------------------------------------------------------------------------

# What the shot is meant to do (why it exists at all).
PURPOSE_TYPES = [
    "hook", "establish", "contrast", "cross_cut", "reaction", "detail",
    "motif", "escalation", "silence_hold", "resolve", "conclusion",
]

# The visual grammar the director asks for.
VISUAL_STRATEGY_TYPES = [
    "wide", "medium", "close_up", "reaction", "object_detail", "environment",
    "movement", "stillness", "cross_cut", "motif_return", "contrast",
    "hold", "match_cut",
]

AUDIO_MOVIE = ["retain", "duck", "mute"]
AUDIO_MUSIC = ["none", "low", "rise", "swell", "resolve", "silence"]
AUDIO_NARRATION = ["dominant", "anchor", "absent"]

RHYTHMS = ["slow", "medium", "fast"]
TRANSITIONS = ["cut", "crossfade", "fade"]


# ---------------------------------------------------------------------------
# Decision primitives
# ---------------------------------------------------------------------------

@dataclass
class VisualStrategy:
    """How the surviving footage should be framed/cut for this beat."""
    type: str = "wide"                       # VISUAL_STRATEGY_TYPES
    description: str = ""                     # plain-language intent

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "VisualStrategy":
        d = d or {}
        return cls(type=str(d.get("type", "wide")),
                   description=str(d.get("description", "")))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Pacing:
    """Deliberate rhythm: how long the beat runs and its tempo."""
    duration_sec: float = 4.0
    rhythm: str = "medium"                   # RHYTHMS

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Pacing":
        d = d or {}
        try:
            dur = float(d.get("duration_sec", 4.0))
        except (TypeError, ValueError):
            dur = 4.0
        return cls(duration_sec=dur, rhythm=str(d.get("rhythm", "medium")))

    def to_dict(self) -> dict:
        return {"duration_sec": round(self.duration_sec, 3), "rhythm": self.rhythm}


@dataclass
class AudioIntent:
    """Explicit audio design: never leave the film's sound to a default."""
    movie_audio: str = "retain"             # AUDIO_MOVIE
    music: str = "none"                     # AUDIO_MUSIC
    narration: str = "dominant"             # AUDIO_NARRATION
    silence: str = ""                       # e.g. "after thesis statement"

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "AudioIntent":
        d = d or {}
        return cls(
            movie_audio=str(d.get("movie_audio", "retain")),
            music=str(d.get("music", "none")),
            narration=str(d.get("narration", "dominant")),
            silence=str(d.get("silence", "")),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EditingIntent:
    """Concrete cut instructions handed to the timeline compiler."""
    transition: str = "cut"                 # TRANSITIONS
    speed: float = 1.0
    hold: bool = False
    crop_zoom: float = 1.0
    fade_edges: bool = False

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "EditingIntent":
        d = d or {}
        try:
            speed = float(d.get("speed", 1.0))
        except (TypeError, ValueError):
            speed = 1.0
        try:
            zoom = float(d.get("crop_zoom", 1.0))
        except (TypeError, ValueError):
            zoom = 1.0
        return cls(
            transition=str(d.get("transition", "cut")),
            speed=speed,
            hold=bool(d.get("hold", False)),
            crop_zoom=zoom,
            fade_edges=bool(d.get("fade_edges", False)),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EditorialDecision:
    """One deliberate editorial moment: why it exists, what it shows, how it
    is performed."""
    segment_id: str
    purpose: str = "establish"               # PURPOSE_TYPES
    narrative_beat: str = ""                 # the claim this moment proves
    evidence: List[EditorialEvidence] = field(default_factory=list)
    narration: NarrationBlock = field(default_factory=NarrationBlock)
    visual_strategy: VisualStrategy = field(default_factory=VisualStrategy)
    pacing: Pacing = field(default_factory=Pacing)
    audio: AudioIntent = field(default_factory=AudioIntent)
    editing: EditingIntent = field(default_factory=EditingIntent)

    # -- Convenience -------------------------------------------------------

    @property
    def beats(self) -> List[str]:
        return [e.scene_id for e in self.evidence]

    @classmethod
    def from_dict(cls, d: dict) -> "EditorialDecision":
        return cls(
            segment_id=str(d.get("segment_id", "")),
            purpose=str(d.get("purpose", "establish")),
            narrative_beat=str(d.get("narrative_beat", "")),
            evidence=[EditorialEvidence.from_dict(e) for e in d.get("evidence", [])],
            narration=NarrationBlock.from_dict(d.get("narration") or {}),
            visual_strategy=VisualStrategy.from_dict(d.get("visual_strategy")),
            pacing=Pacing.from_dict(d.get("pacing")),
            audio=AudioIntent.from_dict(d.get("audio")),
            editing=EditingIntent.from_dict(d.get("editing")),
        )

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "purpose": self.purpose,
            "narrative_beat": self.narrative_beat,
            "evidence": [e.to_json() for e in self.evidence],
            "narration": self.narration.to_json(),
            "visual_strategy": self.visual_strategy.to_dict(),
            "pacing": self.pacing.to_dict(),
            "audio": self.audio.to_dict(),
            "editing": self.editing.to_dict(),
        }


# ---------------------------------------------------------------------------
# Decision list
# ---------------------------------------------------------------------------

@dataclass
class EditorialDecisionList:
    """The full editorial will as decided by the real Editorial Director."""
    title: str
    thesis: str
    hook: Dict[str, str] = field(default_factory=dict)   # {"text", "visual_strategy"}
    decisions: List[EditorialDecision] = field(default_factory=list)
    audio_defaults: AudioIntent = field(default_factory=AudioIntent)
    length_target_sec: float = 90.0
    creative_task: str = ""
    provenance: Dict[str, str] = field(default_factory=dict)
    # deliberate scene reuse the director actually wants (never accidental)
    scene_reuse_justification: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "EditorialDecisionList":
        return cls(
            title=str(d.get("title", "")),
            thesis=str(d.get("thesis", "")),
            hook=dict(d.get("hook") or {}),
            decisions=[EditorialDecision.from_dict(x) for x in d.get("decisions", [])],
            audio_defaults=AudioIntent.from_dict(d.get("audio_defaults")),
            length_target_sec=float(d.get("length_target_sec", 90.0)),
            creative_task=str(d.get("creative_task", "")),
            provenance=dict(d.get("provenance") or {}),
            scene_reuse_justification=dict(d.get("scene_reuse_justification") or {}),
        )

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "thesis": self.thesis,
            "hook": self.hook,
            "decisions": [x.to_dict() for x in self.decisions],
            "audio_defaults": self.audio_defaults.to_dict(),
            "length_target_sec": round(float(self.length_target_sec), 3),
            "creative_task": self.creative_task,
            "provenance": self.provenance,
            "scene_reuse_justification": self.scene_reuse_justification,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    # -- Convenience -------------------------------------------------------

    @property
    def segment_count(self) -> int:
        return len(self.decisions)

    def all_scene_ids(self) -> List[str]:
        seen: List[str] = []
        for d in self.decisions:
            for e in d.evidence:
                if e.scene_id not in seen:
                    seen.append(e.scene_id)
        return seen


# ---------------------------------------------------------------------------
# Validation (fail closed — an invalid decision list must never reach the
# timeline compiler)
# ---------------------------------------------------------------------------

def validate_decision_list(dl: EditorialDecisionList) -> List[str]:
    """Return a list of problems (empty = valid)."""
    errors: List[str] = []
    if not dl.title.strip():
        errors.append("decision list title is empty")
    if not dl.thesis.strip():
        errors.append("decision list thesis is empty")
    if not (dl.hook.get("text") or "").strip():
        errors.append("decision list hook.text is empty")
    if not dl.decisions:
        errors.append("decision list has no decisions")
        return errors

    seen_ids = set()
    for i, d in enumerate(dl.decisions):
        tag = f"decision[{i}] ({d.segment_id or '?'})"
        if not d.segment_id:
            errors.append(f"{tag} has no segment_id")
        elif d.segment_id in seen_ids:
            errors.append(f"{tag} duplicates segment_id {d.segment_id!r}")
        seen_ids.add(d.segment_id)

        if not d.narrative_beat.strip():
            errors.append(f"{tag} has empty narrative_beat (the ''why'' is missing)")
        if d.purpose not in PURPOSE_TYPES and d.purpose:
            errors.append(f"{tag} unknown purpose {d.purpose!r}")
        if not d.evidence:
            errors.append(f"{tag} has no evidence (a shot needs a source)")
        for e in d.evidence:
            if not (e.start_sec < e.end_sec):
                errors.append(f"{tag} invalid evidence window "
                              f"{e.scene_id} [{e.start_sec},{e.end_sec}]")
            if e.end_sec - e.start_sec > 6.0 + 1e-6:
                errors.append(f"{tag} evidence window too long "
                              f"({e.end_sec - e.start_sec:.2f}s > 6s): "
                              f"use precise short excerpts")
        if not d.narration.text.strip():
            errors.append(f"{tag} has empty narration text")
        if d.visual_strategy.type not in VISUAL_STRATEGY_TYPES:
            errors.append(f"{tag} unknown visual_strategy {d.visual_strategy.type!r}")
        if d.pacing.rhythm not in RHYTHMS:
            errors.append(f"{tag} unknown rhythm {d.pacing.rhythm!r}")
        if d.pacing.duration_sec <= 0:
            errors.append(f"{tag} pacing.duration_sec must be positive")
        if d.audio.movie_audio not in AUDIO_MOVIE:
            errors.append(f"{tag} unknown audio.movie_audio {d.audio.movie_audio!r}")
        if d.audio.music not in AUDIO_MUSIC:
            errors.append(f"{tag} unknown audio.music {d.audio.music!r}")
        if d.audio.narration not in AUDIO_NARRATION:
            errors.append(f"{tag} unknown audio.narration {d.audio.narration!r}")
        if d.editing.transition not in TRANSITIONS:
            errors.append(f"{tag} unknown editing.transition {d.editing.transition!r}")

    # No accidental scene repetition: any scene used across different decisions
    # MUST be justified in scene_reuse_justification (a deliberate motif).
    reuse = {sid: dl.scene_reuse_justification.get(sid, "") for sid in dl.all_scene_ids()}
    seen_scenes: Dict[str, int] = {}
    for d in dl.decisions:
        for e in d.evidence:
            prev = seen_scenes.get(e.scene_id, 0)
            seen_scenes[e.scene_id] = prev + 1
    for sid, count in seen_scenes.items():
        if count > 1 and not (reuse.get(sid) or "").strip():
            errors.append(
                f"scene {sid!r} is used {count}x without a deliberate "
                f"scene_reuse_justification: accidental repetition is forbidden"
            )
    return errors


def save_decision_list(project_dir, decision_list: EditorialDecisionList):
    """Persist the decision list next to the editorial plan."""
    from pathlib import Path

    out = Path(project_dir) / "editorial_decisions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(decision_list.to_json(), encoding="utf-8")
    return out


def load_decision_list(project_dir) -> Optional[EditorialDecisionList]:
    from pathlib import Path

    path = Path(project_dir) / "editorial_decisions.json"
    if not path.exists():
        return None
    return EditorialDecisionList.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Compile: EditorialDecisionList -> the downstream EditorialPlan.
#
# The timeline compiler / renderer consume the (existing) EditorialPlan — the
# decision list is compiled down *without inventing anything*: every shot
# order, transition, pacing, hold, and audio decision maps 1:1 from the
# director's decision to the plan the renderer already understands.
# ---------------------------------------------------------------------------

def compile_editorial_plan(decision_list: EditorialDecisionList):
    """Map a validated decision list onto :class:`editorial.plan.EditorialPlan`.

    Raises :class:`ValueError` when the decision list is invalid (fail closed:
    a bad decision must never be silently "fixed" by the compiler).
    """
    from editorial.plan import (
        EditingDirective,
        EditorialPlan,
        EditorialSegment,
    )

    errors = validate_decision_list(decision_list)
    if errors:
        raise ValueError("cannot compile invalid decision list: " + "; ".join(errors))

    segments: List[EditorialSegment] = []
    for d in decision_list.decisions:
        vs = d.visual_strategy
        ai = d.audio
        ed = d.editing

        # Map the director's audio intent onto the renderer's supported ops.
        mute = ai.movie_audio == "mute"
        duck = 0.02 if ai.movie_audio == "duck" else 0.05

        # Visual grammar -> directive defaults (never overwrite explicit ones).
        emphasis = "detail" if vs.type in ("object_detail", "motif_return") else (
            "close_up" if vs.type in ("close_up", "reaction", "cross_cut", "match_cut")
            else "wide")
        crop_zoom = 1.15 if vs.type in ("close_up", "reaction", "object_detail",
                                        "cross_cut", "match_cut") else 1.0

        segments.append(EditorialSegment(
            id=d.segment_id,
            purpose=f"{d.purpose}: {d.narrative_beat}",
            evidence=list(d.evidence),
            narration=d.narration,
            editing=EditingDirective(
                shot_order=[e.scene_id for e in d.evidence],
                transition=ed.transition,
                speed=ed.speed,
                emphasis=emphasis,
                crop_zoom=ed.crop_zoom if ed.crop_zoom != 1.0 else crop_zoom,
                hold_sec=float(d.pacing.duration_sec) if ed.hold else 0.0,
                fade_edges=ed.fade_edges,
                mute_film_audio=mute,
            ),
            supporting_visuals=[],
        ))

    plan = EditorialPlan(
        title=decision_list.title,
        thesis=decision_list.thesis,
        hook=dict(decision_list.hook),
        segments=segments,
        length_target_sec=float(decision_list.length_target_sec),
        creative_task=decision_list.creative_task,
        provenance=dict(decision_list.provenance),
    )

    # Every segment carries a real narration block with delivery preserved.
    for seg, d in zip(segments, decision_list.decisions):
        if d.narration.delivery is None:
            seg.narration.delivery = NarrationDelivery()
    return plan