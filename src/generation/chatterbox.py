"""Chatterbox TTS provider (resemble-ai/chatterbox, open-source, on-device).

Wraps the ``chatterbox`` package (a ~1B zero-shot voice-cloning model) behind the
common ``TTSProvider`` interface. Chatterbox clones the speaker from a reference
audio clip, so ``voice`` is interpreted as a path to a reference ``.wav``/``.mp3``
(use the ``TTS_VOICE_PATH`` env var or pass a path as ``voice``).

Sample rate: 16 kHz. Chatterbox does not expose per-call emotion/pace control in
the base package; the provider records those as unsupported while still honoring
``voice`` cloning and language selection where the model allows.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .base import TTSProvider
from .tts_common import NarrationProperties, probe_audio, resolve_device, write_audio

CHATTERBOX_SAMPLE_RATE = 16000
CHATTERBOX_DEFAULT_MODEL = "resembleai/chatterbox"


class ChatterboxTTSProvider(TTSProvider):
    """Real Chatterbox (voice-cloning) text-to-speech provider."""

    name = "chatterbox"

    _model = None  # class-level cache
    _model_name: Optional[str] = None

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        config = config or {}
        self.model_name = config.get("model", CHATTERBOX_DEFAULT_MODEL)
        self.device = resolve_device(device or config.get("device"))
        self.dtype = config.get("dtype", "fp16" if self.device == "cuda" else "fp32")
        self.default_voice = os.getenv("TTS_VOICE_PATH") or config.get("voice_path")

    @classmethod
    def is_available(cls) -> bool:
        import importlib.util

        return importlib.util.find_spec("chatterbox") is not None

    @classmethod
    def release_model(cls):
        cls._model = None
        cls._model_name = None

    def _load(self):
        if ChatterboxTTSProvider._model is not None:
            return ChatterboxTTSProvider._model
        try:
            from chatterbox import Chatterbox
        except Exception as e:  # pragma: no cover - depends on install
            raise RuntimeError(
                "Chatterbox TTS requires the 'chatterbox' package. Install it "
                "with `pip install chatterbox-tts`."
            ) from e
        model = Chatterbox.from_pretrained(
            self.model_name, device=self.device, dtype=self.dtype
        )
        ChatterboxTTSProvider._model = model
        ChatterboxTTSProvider._model_name = self.model_name
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
        NarrationProperties.from_dict(narration)  # validated; most are unsupported
        voice_path = voice
        if voice in ("default", "", None):
            voice_path = self.default_voice
        if not voice_path:
            raise RuntimeError(
                "Chatterbox clones the speaker from a reference clip. Provide a "
                "voice reference via TTS_VOICE_PATH or pass `voice=<path>`."
            )
        voice_path = Path(voice_path)
        if not voice_path.exists():
            raise FileNotFoundError(f"Chatterbox voice reference not found: {voice_path}")

        if output_path is None:
            output_path = Path("chatterbox_voice.wav")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = self._load()
        t0 = time.monotonic()
        try:
            audio = model.infer(text, voice_path=voice_path, output_path=str(output_path))
        except TypeError:
            # Some builds return audio without an output_path kwarg.
            audio = model.infer(text, voice_path=voice_path)
            write_audio(audio, CHATTERBOX_SAMPLE_RATE, output_path)
        except Exception as e:
            raise RuntimeError(f"Chatterbox synthesis failed: {e}") from e
        generation_time = time.monotonic() - t0

        if not output_path.exists():
            write_audio(audio, CHATTERBOX_SAMPLE_RATE, output_path)
        probed = probe_audio(output_path)

        return {
            "audio_path": output_path,
            "duration_sec": probed.get("duration_sec", 0.0),
            "sample_rate": int(probed.get("sample_rate") or CHATTERBOX_SAMPLE_RATE),
            "voice": str(voice_path),
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
                "voice_clone": True,
            },
            "mock": False,
        }
