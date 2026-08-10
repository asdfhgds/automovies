"""Kokoro TTS provider (open-source, on-device).

Wraps the ``kokoro`` package (hexgrad/Kokoro-82M, ~82M params) behind the common
``TTSProvider`` interface. Runs on CPU or CUDA, writes a 24 kHz WAV, and records
the subset of narration properties Kokoro can honor:

- ``pace`` / ``energy`` / ``dramatic_intensity`` -> ``speed`` (0.5..2.0)
- ``tone`` / ``emotion`` -> a fixed-personality voice approximation
  (Kokoro voices are pre-trained; the emotion map is documented as an
  approximation, not a real emotion synthesizer)

Lazy-loads the model once and caches it at class level so the benchmark and the
pipeline run share a single model in memory.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from .base import TTSProvider
from .tts_common import (
    NarrationProperties,
    emotion_to_kokoro_voice,
    pace_to_kokoro_speed,
    probe_audio,
    resolve_device,
    write_audio,
)

KOKORO_SAMPLE_RATE = 24000
KOKORO_REPO = "hexgrad/Kokoro-82M"


class KokoroTTSProvider(TTSProvider):
    """Real Kokoro-82M text-to-speech provider."""

    name = "kokoro"

    _pipeline = None  # class-level cache: one loaded model per process
    _device: Optional[str] = None

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        repo_id: str = KOKORO_REPO,
    ):
        config = config or {}
        self.repo_id = config.get("repo_id", repo_id)
        self.lang_code = config.get("lang_code", "a")  # 'a' = American English
        self.default_voice = config.get("voice", "am_adam")
        self.model = config.get("model", KOKORO_REPO)
        self.device = resolve_device(device or config.get("device"))

    @classmethod
    def is_available(cls) -> bool:
        import importlib.util

        return importlib.util.find_spec("kokoro") is not None

    @classmethod
    def release_model(cls):
        cls._pipeline = None
        cls._device = None

    def _load(self):
        if KokoroTTSProvider._pipeline is not None:
            self.device = KokoroTTSProvider._device or self.device
            return KokoroTTSProvider._pipeline
        try:
            from kokoro import KPipeline
        except Exception as e:  # pragma: no cover - depends on install
            raise RuntimeError(
                "Kokoro TTS requires the 'kokoro' package. Install it with "
                "`pip install kokoro` (plus espeak-ng phonemizer)."
            ) from e
        pipeline = KPipeline(
            lang_code=self.lang_code,
            repo_id=self.repo_id,
            device=self.device,
        )
        KokoroTTSProvider._pipeline = pipeline
        KokoroTTSProvider._device = self.device
        return pipeline

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
        props = NarrationProperties.from_dict(narration)
        if voice in ("default", "", None):
            voice = self.default_voice
        voice = emotion_to_kokoro_voice(props.emotion, props.tone, default=voice)
        speed = pace_to_kokoro_speed(
            pace=speaking_rate * props.pace,
            energy=props.energy,
            dramatic_intensity=props.dramatic_intensity,
        )

        if output_path is None:
            output_path = Path("kokoro_voice.wav")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pipeline = self._load()
        load_t0 = time.monotonic()
        segments = []
        t0 = time.monotonic()
        try:
            for result in pipeline(text=text, voice=voice, speed=speed):
                segments.append(result.audio)
        except Exception as e:
            raise RuntimeError(f"Kokoro synthesis failed: {e}") from e
        generation_time = time.monotonic() - t0

        import torch

        if segments:
            audio = torch.cat(segments)
        else:
            audio = torch.zeros(int(KOKORO_SAMPLE_RATE * 0.5))

        written = write_audio(audio, KOKORO_SAMPLE_RATE, output_path)
        probed = probe_audio(output_path)
        duration = probed.get("duration_sec", written.get("duration_sec", 0.0))
        sample_rate = int(probed.get("sample_rate") or written.get("sample_rate") or KOKORO_SAMPLE_RATE)

        return {
            "audio_path": output_path,
            "duration_sec": duration,
            "sample_rate": sample_rate,
            "voice": voice,
            "language": language,
            "provider": self.name,
            "model": self.model,
            "device": self.device,
            "generation_time_sec": round(generation_time, 3),
            "model_load_time_sec": None,
            "supported": {
                "emotion": True,  # approximated through voice mapping
                "pace": True,     # speed
                "pitch": False,
                "energy": True,   # folded into speed
                "dramatic_intensity": True,  # folded into speed
            },
            "mock": False,
        }
