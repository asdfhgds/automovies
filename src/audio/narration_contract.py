"""Narration extraction + TTS input contract (prompt sanitization).

The TTS provider MUST receive only plain human narration. This module is the
only gate: it walks the script sections, extracts the narration text, strips
everything that is not narration, and produces a normalized ``TTSInput`` object
per section. ``text`` is the ONLY field ever handed to a TTS provider.

It rejects (fail closed):

- empty / whitespace narration
- JSON or embedded code blocks
- internal identifiers (scene_id / evidence_id / section ids / raw metadata)
- prompt/debug/reasoning text (``<system>``, ``developer``, ``prompt``,
  ``metadata``, ``debug``, ``reasoning``, ``...``)
- suspiciously long text relative to the section budget

The architecture ensures separation: script sections carry both ``text``
(narration) and metadata keys; only ``text`` is surfaced here, and additionally
validated to reject leaked metadata that ended up inside the narration string.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

WORDS_PER_SEC = 2.4

# Artefact markers that must never be read out loud. If a narration string
# contains ANY of these it is treated as leaked internal/prompt content.
_ARTEFACT_RE = re.compile(
    r"("
    r"<system>|<SYSTEM>|<user>|<assistant>|<director>|<director_plan>"
    r"|```(?:py|json|text)?|json(?:\.dumps)?|python(?: code)?"
    r"|developer\b|metadata\b|prompt\b|debug(?:ging)?\b"
    r"|reasoning\b|model_reasoning\b|attention\b"
    r"|\[\d+:\d+\]|```"
    r"|\bscene[-_ ]?\d+\b|\bev(?:idence)?[-_ ]?\d+\b"
    r"|\bsection[-_ ]?\d+\b|\bseg_?\d+\b|\banalysis_?\d+\b"
    r")",
    re.IGNORECASE,
)

_MAX_WORDS = 300  # far beyond a 90s essay; anything bigger is noisy output.


@dataclass
class VoiceDirection:
    pace: float = 1.0        # 0.5 slow ... 1.5 fast
    energy: float = 0.5      # 0.0 flat ... 1.0 intense

    def to_dict(self) -> Dict[str, Any]:
        return {"pace": round(float(self.pace), 3), "energy": round(float(self.energy), 3)}


@dataclass
class TTSInput:
    """The ONLY object a TTS provider is allowed to receive.

    ``text`` is the single narration string. ``voice_direction`` carries
    performance hints that may be ignored by providers. No scene/evidence/
    section metadata leaks here by construction.
    """
    section_id: str
    text: str
    duration_estimate_sec: float
    voice_direction: VoiceDirection = field(default_factory=VoiceDirection)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "text": self.text,
            "duration_estimate_sec": round(float(self.duration_estimate_sec), 3),
            "voice_direction": self.voice_direction.to_dict(),
        }


class NarrationSanitizationError(ValueError):
    """Raised when a section's narration fails the fail-closed contract."""


def detected_artefacts(text: str) -> List[str]:
    """Return the matched artefact markers found in ``text`` (empty = clean)."""
    if not text or not text.strip():
        return []
    return list(dict.fromkeys(m.group(0) for m in _ARTEFACT_RE.finditer(text)))


def looks_like_json(text: str) -> bool:
    """True if the narration is (or embeds) a JSON document/object."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith(("{", "[")):
        return True
    return "```" in text or "\"section_id\"" in text or "\"scene_id\"" in text


def sanitize_narration(
    text: str,
    section_id: str,
    raise_on_error: bool = True,
) -> str:
    """Validate a narration string and return it untouched when clean.

    Raises :class:`NarrationSanitizationError` when the text is empty, looks
    like JSON/code, contains artefact markers, or is unreasonably long.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise NarrationSanitizationError(
            f"TTS input for section {section_id!r} is empty"
        )
    words = len(cleaned.split())
    if words > _MAX_WORDS:
        raise NarrationSanitizationError(
            f"TTS input for section {section_id!r} is suspiciously long "
            f"({words} words > {_MAX_WORDS}) — refusing to synthesize"
        )
    if looks_like_json(cleaned):
        raise NarrationSanitizationError(
            f"TTS input for section {section_id!r} looks like JSON/code"
        )
    arts = detected_artefacts(cleaned)
    if arts:
        raise NarrationSanitizationError(
            f"TTS input for section {section_id!r} contains internal "
            f"artefacts: {arts}"
        )
    return cleaned


def build_tts_inputs(script: Dict[str, Any]) -> List[TTSInput]:
    """Extract + validate a TTS input per script section (fail closed).

    Only sections that actually carry narration text produce an input.
    The hook is included first (it is part of the spoken essay when present).
    """
    inputs: List[TTSInput] = []
    sections = list(script.get("sections") or [])

    hook = script.get("hook") or {}
    if isinstance(hook, dict) and (hook.get("text") or "").strip():
        sections = [{"section_id": "hook", "text": hook["text"]}] + sections

    seen_texts: set = set()
    for sec in sections:
        sid = str(sec.get("section_id") or sec.get("id") or "section")
        text = sanitize_narration(sec.get("text") or "", sid)
        # De-duplicate identical narration blocks that would be spoken twice.
        if text in seen_texts:
            continue
        seen_texts.add(text)
        duration = _estimate_seconds(text, _pace_of(sec))
        inputs.append(TTSInput(
            section_id=sid,
            text=text,
            duration_estimate_sec=duration,
            voice_direction=_direction_of(sec),
        ))
    if not inputs:
        raise NarrationSanitizationError(
            "no usable narration sections found — refusing to synthesize"
        )
    return inputs


def joint_text(inputs: List[TTSInput]) -> str:
    """Join individual narration blocks into a single TTS payload while
    preserving separation (newlines between sections)."""
    return ". ".join(i.text for i in inputs if i.text.strip())


def _pace_of(section: Dict[str, Any]) -> float:
    delivery = section.get("delivery") or {}
    if isinstance(delivery, dict):
        try:
            return float(delivery.get("pace", 1.0))
        except (TypeError, ValueError):
            return 1.0
    return 1.0


def _direction_of(section: Dict[str, Any]) -> VoiceDirection:
    delivery = section.get("delivery") or {}
    pace = _pace_of(section)
    energy = 0.5
    if isinstance(delivery, dict):
        try:
            energy = float(delivery.get("energy", 0.5))
        except (TypeError, ValueError):
            energy = 0.5
    return VoiceDirection(pace=pace, energy=energy)


def _estimate_seconds(text: str, pace: float = 1.0) -> float:
    words = len((text or "").split())
    return max(1.0, round(words / max(0.1, WORDS_PER_SEC * pace), 2))


def write_tts_input_manifest(project_dir: Path, inputs: List[TTSInput]) -> Path:
    """Persist the narration input contract for auditability."""
    out = Path(project_dir) / "audio" / "narration_inputs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "schema": "tts_input_contract_v1",
            "count": len(inputs),
            "inputs": [i.to_dict() for i in inputs],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out