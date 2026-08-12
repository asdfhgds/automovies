"""Editorial planning models.

The EditorialPlan is the director's decision about how the argument becomes an
*edit* — each segment carries the evidence it proves, the narration that
explains it (with performance instructions), and the editing directives.
"""
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

SUPPORTED_EDITING_OPS = [
    "CUT", "CROSSFADE", "HOLD", "SPEED", "CROP", "FADE", "AUDIO_DUCK",
]
SUPPORTED_TRANSITIONS = ["cut", "crossfade", "fade"]

DEFAULT_TARGET_SEC = 90.0
MIN_EXCERPTS_PER_SEGMENT = 3
MAX_EXCERPTS_PER_SEGMENT = 6
MAX_EXCERPT_SEC = 6.0
MIN_EXCERPT_SEC = 1.2


@dataclass
class EditorialEvidence:
    """A specific movie moment selected to support one segment's argument."""
    scene_id: str
    start_sec: float
    end_sec: float
    reason: str

    @classmethod
    def from_dict(cls, d: dict) -> "EditorialEvidence":
        return cls(
            scene_id=str(d["scene_id"]),
            start_sec=float(d["start_sec"]),
            end_sec=float(d["end_sec"]),
            reason=str(d.get("reason", "")),
        )

    def to_json(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "reason": self.reason,
        }


@dataclass
class NarrationDelivery:
    """Performance instructions for one narration block.

    ``pace`` maps directly to TTS speaking_rate; energy/intensity feed
    provider-specific speed/voice adjustments. Unsupported controls are
    recorded but NOT faked; providers report ``supported`` flags.
    """
    tone: str = "analytical"
    emotion: str = "neutral"
    energy: float = 0.5
    pace: float = 1.0
    dramatic_intensity: float = 0.5
    pause_before: float = 0.0
    pause_after: float = 0.0

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "NarrationDelivery":
        d = d or {}
        return cls(
            tone=str(d.get("tone", "analytical")),
            emotion=str(d.get("emotion", "neutral")),
            energy=float(d.get("energy", 0.5)),
            pace=float(d.get("pace", 1.0)),
            dramatic_intensity=float(d.get("dramatic_intensity", 0.5)),
            pause_before=float(d.get("pause_before", 0.0)),
            pause_after=float(d.get("pause_after", 0.0)),
        )

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class EditingDirective:
    """How this segment should be cut, paced, and framed."""
    shot_order: List[str] = field(default_factory=list)   # evidence scene_ids
    transition: str = "crossfade"                          # cut | crossfade | fade
    speed: float = 1.0                                     # footage playback rate
    emphasis: str = "wide"                                 # wide | close_up | detail
    crop_zoom: float = 1.0                                 # 1.0 = full frame; >1 zooms in
    hold_sec: float = 0.0                                  # freeze on last frame
    fade_edges: bool = False                               # FADE in/out this segment
    mute_film_audio: bool = False                          # no movie dialogue here
    duck_level: float = 0.05                               # AUDIO_DUCK threshold

    @classmethod
    def from_dict(cls, d: dict) -> "EditingDirective":
        d = d or {}
        return cls(
            shot_order=list(d.get("shot_order", [])),
            transition=str(d.get("transition", "crossfade")),
            speed=float(d.get("speed", 1.0)),
            emphasis=str(d.get("emphasis", "wide")),
            crop_zoom=float(d.get("crop_zoom", 1.0)),
            hold_sec=float(d.get("hold_sec", 0.0)),
            fade_edges=bool(d.get("fade_edges", False)),
            mute_film_audio=bool(d.get("mute_film_audio", False)),
            duck_level=float(d.get("duck_level", 0.05)),
        )

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class NarrationBlock:
    text: str
    delivery: NarrationDelivery = field(default_factory=NarrationDelivery)

    @classmethod
    def from_dict(cls, d: dict) -> "NarrationBlock":
        return cls(
            text=str(d.get("text", "")),
            delivery=NarrationDelivery.from_dict(d.get("delivery")),
        )

    def to_json(self) -> dict:
        return {"text": self.text, "delivery": self.delivery.to_json()}


@dataclass
class EditorialSegment:
    id: str
    purpose: str
    evidence: List[EditorialEvidence] = field(default_factory=list)
    narration: NarrationBlock = field(default_factory=NarrationBlock)
    editing: EditingDirective = field(default_factory=EditingDirective)
    supporting_visuals: List[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "EditorialSegment":
        return cls(
            id=str(d["id"]),
            purpose=str(d.get("purpose", "")),
            evidence=[EditorialEvidence.from_dict(e) for e in d.get("evidence", [])],
            narration=NarrationBlock.from_dict(d.get("narration") or {}),
            editing=EditingDirective.from_dict(d.get("editing")),
            supporting_visuals=list(d.get("supporting_visuals", [])),
        )

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "purpose": self.purpose,
            "evidence": [e.to_json() for e in self.evidence],
            "narration": self.narration.to_json(),
            "editing": self.editing.to_json(),
            "supporting_visuals": self.supporting_visuals,
        }


@dataclass
class EditorialPlan:
    title: str
    thesis: str
    hook: dict = field(default_factory=dict)          # {text, visual_strategy}
    segments: List[EditorialSegment] = field(default_factory=list)
    length_target_sec: float = DEFAULT_TARGET_SEC
    creative_task: str = ""
    provenance: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "EditorialPlan":
        return cls(
            title=str(d.get("title", "")),
            thesis=str(d.get("thesis", "")),
            hook=dict(d.get("hook") or {}),
            segments=[EditorialSegment.from_dict(s) for s in d.get("segments", [])],
            length_target_sec=float(d.get("length_target_sec", DEFAULT_TARGET_SEC)),
            creative_task=str(d.get("creative_task", "")),
            provenance=dict(d.get("provenance") or {}),
        )

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "thesis": self.thesis,
            "hook": self.hook,
            "segments": [s.to_json() for s in self.segments],
            "length_target_sec": self.length_target_sec,
            "creative_task": self.creative_task,
            "provenance": self.provenance,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def validate_plan(plan: EditorialPlan) -> List[str]:
    """Return a list of problems (empty = valid)."""
    errors = []
    if not plan.title.strip():
        errors.append("plan.title is empty")
    if not plan.thesis.strip():
        errors.append("plan.thesis is empty")
    if not plan.hook.get("text"):
        errors.append("plan.hook.text is empty")
    if not plan.segments:
        errors.append("plan.segments is empty")
    return errors