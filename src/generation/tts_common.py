"""Shared helpers for real TTS providers.

All real providers live behind the common ``TTSProvider`` interface
(``generation.base``). This module centralizes the pieces they share:

- ``NarrationProperties``: director-controlled delivery parameters
  (tone, emotion, pace, energy, dramatic intensity).
- ``resolve_device``: pick cuda/cpu for the active runtime.
- ``probe_audio``: duration + sample-rate detection for generated files.
- ``describe``: common per-provider metadata for the benchmark manifest.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class NarrationProperties:
    """Director-controlled narration delivery parameters.

    Only the parameters a given model supports are honored; the rest are
    recorded in the manifest so the pipeline knows what was actually applied.
    """

    tone: str = "analytical"
    emotion: str = "neutral"
    pace: float = 1.0  # speaking-rate multiplier (0.5 = slow, 2.0 = fast)
    energy: float = 0.5  # 0.0..1.0
    dramatic_intensity: float = 0.5  # 0.0..1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "NarrationProperties":
        data = data or {}
        return cls(
            tone=str(data.get("tone") or "analytical"),
            emotion=str(data.get("emotion") or "neutral"),
            pace=_to_float(data.get("pace"), 1.0, lo=0.3, hi=2.5),
            energy=_to_float(data.get("energy"), 0.5, lo=0.0, hi=1.0),
            dramatic_intensity=_to_float(data.get("dramatic_intensity"), 0.5, lo=0.0, hi=1.0),
        )

    @classmethod
    def from_env(cls) -> "NarrationProperties":
        """Apply TTS_TONE / TTS_EMOTION / TTS_PACE / TTS_ENERGY / TTS_INTENSITY overrides.

        Only variables that are actually set in the environment are applied;
        unset variables leave the base properties untouched.
        """
        return cls.from_dict(narration_overrides_from_env())


def narration_overrides_from_env() -> Dict[str, Any]:
    """Return a dict of narration overrides from set TTS_* environment variables.

    Unset variables are omitted so callers can merge them over a base set of
    narration properties without clobbering values with defaults.
    """
    overrides: Dict[str, Any] = {}
    for env, key in (
        ("TTS_TONE", "tone"),
        ("TTS_EMOTION", "emotion"),
        ("TTS_PACE", "pace"),
        ("TTS_ENERGY", "energy"),
        ("TTS_INTENSITY", "dramatic_intensity"),
    ):
        value = os.getenv(env)
        if value is not None and value.strip() != "":
            overrides[key] = value
    return overrides


def _to_float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, f))


def resolve_device(requested: Optional[str] = None) -> str:
    """Resolve ``auto``/``cpu``/``cuda`` to an actual device string."""
    device = (requested or os.getenv("TTS_DEVICE", "auto")).lower()
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def probe_audio(path: Path) -> Dict[str, float]:
    """Return ``{duration_sec, sample_rate}`` for an audio file via ffprobe."""
    import json
    import subprocess

    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=duration,sample_rate",
            "-show_entries", "format=duration",
            "-of", "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams") or [{}]
    stream = streams[0]
    duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
    sample_rate = float(stream.get("sample_rate") or 0)
    return {"duration_sec": max(0.0, duration), "sample_rate": sample_rate}


def write_audio(samples, sample_rate: int, output_path: Path) -> Dict[str, Any]:
    """Persist a torch tensor / numpy array as 24-bit-safe WAV via soundfile."""
    import numpy as np

    if hasattr(samples, "detach"):  # torch tensor
        samples = samples.detach().cpu().numpy()
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import soundfile as sf

    sf.write(output_path, samples, sample_rate, subtype="PCM_16")
    return {"duration_sec": float(len(samples) / sample_rate), "sample_rate": sample_rate}


def supported_voices(provider: str) -> List[str]:
    """Return the known built-in voices for a provider (empty if dynamic)."""
    if provider == "kokoro":
        return [
            "af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore", "af_sarah",
            "af_sky", "am_adam", "am_michael", "am_fenrir", "am_puck",
            "bf_emma", "bm_george", "bm_lewis",
        ]
    if provider == "qwen3_tts":
        return ["Chelsie", "George", "Koren"]
    return []


def load_voice_bytes(voice: str, language: str, provider: str) -> Optional[bytes]:
    """Load a voice's weights for providers that bundle per-voice .pt files.

    Kokoro downloads voice files from the ``hexgrad/Kokoro-82M`` repo when the
    requested voice is not already cached. Returns the raw bytes so the provider
    can save them into a temp file for the pipeline.
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None
    try:
        path = hf_hub_download(
            repo_id="hexgrad/Kokoro-82M",
            filename=f"voices/{voice}.pt",
        )
        return Path(path).read_bytes()
    except Exception:
        return None


def emotion_to_kokoro_voice(emotion: str, tone: str, default: str = "am_adam") -> str:
    """Map a requested emotion/tone to a Kokoro voice with a suitable texture.

    Kokoro voices have fixed personalities; this is a documented approximation,
    not a true emotion synthesizer. Neutral analytical defaults to ``am_adam``.
    """
    key = f"{tone} {emotion}".lower()
    if any(w in key for w in ("dark", "intense", "dramatic", "angry", "menacing", "thriller")):
        return "am_fenrir"
    if any(w in key for w in ("calm", "soft", "gentle", "serene", "sad", "melancholy")):
        return "am_michael"
    if any(w in key for w in ("energetic", "exciting", "upbeat", "enthusiastic", "comedy", "funny")):
        return "af_bella"
    if any(w in key for w in ("mysterious", "suspense", "noir", "cryptic")):
        return "af_kore"
    return default


def pace_to_kokoro_speed(pace: float, energy: float, dramatic_intensity: float) -> float:
    """Combine narration pace/energy/intensity into a Kokoro ``speed`` in [0.5, 2.0]."""
    speed = pace
    speed += (energy - 0.5) * 0.3
    speed += (dramatic_intensity - 0.5) * 0.2
    return max(0.5, min(2.0, round(speed, 3)))
