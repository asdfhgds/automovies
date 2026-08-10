"""Narration delivery properties for the script stage.

The creative director sets a ``tone``; we derive a full ``narration_properties``
block (tone, emotion, pace, energy, dramatic_intensity) so the TTS stage can
honor whatever the selected model supports. Environment variables
(TTS_TONE / TTS_EMOTION / TTS_PACE / TTS_ENERGY / TTS_INTENSITY) override the
derived values and are applied last.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from generation.tts_common import NarrationProperties

# tone -> (emotion, pace, energy, dramatic_intensity) defaults
_TONE_MAP = {
    "analytical": ("neutral", 1.0, 0.5, 0.4),
    "intellectual": ("thoughtful", 0.95, 0.4, 0.3),
    "humorous": ("playful", 1.15, 0.7, 0.5),
    "comedic": ("playful", 1.2, 0.7, 0.6),
    "dramatic": ("tense", 0.9, 0.6, 0.85),
    "cinematic": ("tense", 0.95, 0.6, 0.8),
    "mysterious": ("suspenseful", 0.85, 0.4, 0.7),
    "noir": ("suspenseful", 0.85, 0.4, 0.75),
    "emotional": ("warm", 0.9, 0.5, 0.6),
    "inspirational": ("uplifting", 1.05, 0.7, 0.7),
    "critical": ("sharp", 1.1, 0.7, 0.6),
    "energetic": ("energetic", 1.2, 0.85, 0.7),
    "quiet": ("calm", 0.8, 0.25, 0.3),
    "reflective": ("calm", 0.85, 0.3, 0.4),
}


def default_narration_properties(plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Derive narration_properties from a director plan's tone."""
    tone = ""
    if plan:
        tone = str(plan.get("tone") or "").lower()
    emotion, pace, energy, intensity = _TONE_MAP.get(tone, _TONE_MAP["analytical"])
    return NarrationProperties(
        tone=str(plan.get("tone") or tone or "analytical"),
        emotion=emotion,
        pace=pace,
        energy=energy,
        dramatic_intensity=intensity,
    ).to_dict()


def finalize_narration_properties(properties: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply env overrides on top of the given properties."""
    base = NarrationProperties.from_dict(properties).to_dict()
    from generation.tts_common import narration_overrides_from_env

    overrides = narration_overrides_from_env()
    merged = dict(base)
    merged.update(overrides)
    return merged


def narration_properties_from_env(plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One-call helper used by the writers and the TTS adapter."""
    env = os.getenv("TTS_NARRATION_PROPS", "false").lower() == "true"
    if env:
        return finalize_narration_properties(default_narration_properties(plan))
    return default_narration_properties(plan)
