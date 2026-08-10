"""Unit tests for real TTS providers (no model loading, no GPU)."""
import importlib.util
from pathlib import Path

import pytest

from generation.chatterbox import ChatterboxTTSProvider
from generation.kokoro import KokoroTTSProvider
from generation.provider_factory import available_tts_providers
from generation.qwen_tts import Qwen3TTSProvider
from generation.tts_common import (
    NarrationProperties,
    emotion_to_kokoro_voice,
    pace_to_kokoro_speed,
    resolve_device,
    supported_voices,
)


def test_narration_properties_defaults():
    props = NarrationProperties()
    assert props.tone == "analytical"
    assert props.emotion == "neutral"
    assert props.pace == 1.0
    assert props.energy == 0.5
    assert props.dramatic_intensity == 0.5
    d = props.to_dict()
    assert d["pace"] == 1.0


def test_narration_properties_from_dict_clamps():
    props = NarrationProperties.from_dict(
        {"tone": "dramatic", "emotion": "tense", "pace": "9", "energy": "2", "dramatic_intensity": "-1"}
    )
    assert props.pace == 2.5
    assert props.energy == 1.0
    assert props.dramatic_intensity == 0.0


def test_emotion_to_kokoro_voice_maps_tones():
    assert emotion_to_kokoro_voice("tense", "dramatic") == "am_fenrir"
    assert emotion_to_kokoro_voice("calm", "quiet") == "am_michael"
    assert emotion_to_kokoro_voice("neutral", "analytical", default="am_adam") == "am_adam"


def test_pace_to_kokoro_speed_is_clamped():
    assert pace_to_kokoro_speed(1.0, 0.5, 0.5) == 1.0
    fast = pace_to_kokoro_speed(2.0, 1.0, 1.0)
    assert fast <= 2.0
    slow = pace_to_kokoro_speed(0.1, 0.0, 0.0)
    assert slow >= 0.5


def test_resolve_device_returns_known_values():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_supported_voices_lists():
    assert "am_adam" in supported_voices("kokoro")
    assert "Chelsie" in supported_voices("qwen3_tts")


def test_provider_classes_define_name_and_availability():
    for cls in (KokoroTTSProvider, ChatterboxTTSProvider, Qwen3TTSProvider):
        assert isinstance(cls.name, str)
        assert callable(cls.is_available)


def test_availability_does_not_hang_or_load_models():
    # find_spec based: must not import torch/kokoro (which keeps the process alive)
    info = available_tts_providers()
    assert "kokoro" in info
    assert "chatterbox" in info
    assert "qwen3_tts" in info


def test_unavailable_provider_raises_clear_error_on_synthesize(tmp_path):
    if importlib.util.find_spec("chatterbox") is not None:
        pytest.skip("chatterbox installed; skipping not-installed error test")
    provider = ChatterboxTTSProvider({"device": "cpu"})
    with pytest.raises(RuntimeError):
        provider._load()


def test_chatterbox_requires_voice_reference(tmp_path):
    provider = ChatterboxTTSProvider({"device": "cpu", "voice_path": None})
    with pytest.raises(RuntimeError):
        provider.synthesize("hello", output_path=tmp_path / "out.wav")


def test_kokoro_constructor_resolves_config():
    provider = KokoroTTSProvider({"voice": "am_michael", "device": "cpu"})
    assert provider.default_voice == "am_michael"
    assert provider.device == "cpu"
    assert provider.model == "hexgrad/Kokoro-82M"
