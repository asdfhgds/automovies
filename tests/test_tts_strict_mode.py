"""REQUIRE_REAL_TTS strict-mode unit tests."""
import os

import pytest

from generation.mock import MockTTSProvider
from utils.strict import require_real_tts, tts_strict_mode_enabled


@pytest.fixture
def strict_tts(monkeypatch):
    monkeypatch.setenv("REQUIRE_REAL_TTS", "true")
    return monkeypatch


def test_tts_strict_mode_enabled(monkeypatch):
    monkeypatch.setenv("REQUIRE_REAL_TTS", "true")
    assert tts_strict_mode_enabled()
    monkeypatch.delenv("REQUIRE_REAL_TTS")
    assert not tts_strict_mode_enabled()


def test_require_real_tts_rejects_none(strict_tts):
    with pytest.raises(RuntimeError, match="REQUIRE_REAL_TTS"):
        require_real_tts(None, "TTS")


def test_require_real_tts_rejects_mock(strict_tts):
    mock = MockTTSProvider()
    with pytest.raises(RuntimeError, match="mock audio"):
        require_real_tts(mock, "TTS")


def test_require_real_tts_passes_real_provider(strict_tts):
    class FakeReal:
        name = "kokoro"

    assert require_real_tts(FakeReal(), "TTS").name == "kokoro"


def test_require_real_tts_is_noop_when_not_strict():
    os.environ.pop("REQUIRE_REAL_TTS", None)
    mock = MockTTSProvider()
    assert require_real_tts(mock, "TTS") is mock
