"""Qwen3-TTS provider (QwenLM/qwen3-tts, open-source, on-device).

Wraps the ``qwen3_tts`` package (Qwen3-TTS, ~2B) behind the common
``TTSProvider`` interface. Qwen3-TTS supports a small set of built-in voices
(``Chelsie``, ``George``, ``Koren``), multiple languages, and 24 kHz output.

It does not expose per-call emotion/pace control in the released package; those
narration properties are recorded as unsupported while ``voice`` and ``language``
are honored.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from .base import TTSProvider
from .tts_common import NarrationProperties, probe_audio, resolve_device

QWEN3_SAMPLE_RATE = 24000
QWEN3_DEFAULT_MODEL = "Qwen/Qwen3-TTS"


class Qwen3TTSProvider(TTSProvider):
    """Real Qwen3-TTS text-to-speech provider."""

    name = "qwen3_tts"

    _model = None  # class-level cache
    _model_name: Optional[str] = None

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        config = config or {}
        self.model_name = config.get("model", QWEN3_DEFAULT_MODEL)
        self.device = resolve_device(device or config.get("device"))
        self.dtype = config.get("dtype", "fp16" if self.device == "cuda" else "fp32")
        self.default_voice = config.get("voice", "Chelsie")

    @classmethod
    def is_available(cls) -> bool:
        import importlib.util

        return importlib.util.find_spec("qwen3_tts") is not None

    @classmethod
    def release_model(cls):
        cls._model = None
        cls._model_name = None

    def _load(self):
        if Qwen3TTSProvider._model is not None:
            return Qwen3TTSProvider._model
        try:
            from qwen3_tts import Qwen3TTS
        except Exception as e:  # pragma: no cover - depends on install
            raise RuntimeError(
                "Qwen3-TTS requires the 'qwen3_tts' package. Install it with "
                "`pip install qwen3_tts`."
            ) from e
        model = Qwen3TTS(device=self.device, dtype=self.dtype)
        Qwen3TTSProvider._model = model
        Qwen3TTSProvider._model_name = self.model_name
        return model

    def synthesize(
        self,
        text: str,
        voice: str = "default",
        language: str = "en",
        emotion: str = "neutral",
        speaking_rate: float = 1.0,
        pitch: float = 1.0,
        output_path: Optional[Path] = None,
        narration: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        NarrationProperties.from_dict(narration)  # validated; mostly unsupported
        if voice in ("default", "", None):
            voice = self.default_voice

        if output_path is None:
            output_path = Path("qwen3_voice.wav")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = self._load()
        t0 = time.monotonic()
        try:
            model.synthesize(text, voice=voice, language=language, output_path=str(output_path))
        except Exception as e:
            raise RuntimeError(f"Qwen3-TTS synthesis failed: {e}") from e
        generation_time = time.monotonic() - t0

        probed = probe_audio(output_path)
        return {
            "audio_path": output_path,
            "duration_sec": probed.get("duration_sec", 0.0),
            "sample_rate": int(probed.get("sample_rate") or QWEN3_SAMPLE_RATE),
            "voice": voice,
            "language": language,
            "provider": self.name,
            "model": self.model_name,
            "device": self.device,
            "generation_time_sec": round(generation_time, 3),
            "model_load_time_sec": None,
            "supported": {
                "emotion": False,
                "pace": False,
                "pitch": False,
                "energy": False,
                "dramatic_intensity": False,
                "voice": True,
                "language": True,
            },
            "mock": False,
        }
