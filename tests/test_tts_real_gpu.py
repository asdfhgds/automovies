"""Real-model TTS tests (GPU, explicitly gated).

These tests run a real open-source TTS model (Kokoro by default) on CUDA.
They are skipped unless ALL of the following hold:

- STUDIO_RUN_REAL_TESTS=1 (explicit opt-in; see PROJECT_STATUS.md)
- a CUDA GPU is available
- the TTS package is installed

The default `pytest` run keeps them deselected/skipped.
"""
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

kokoro_installed = importlib.util.find_spec("kokoro") is not None
chatterbox_installed = importlib.util.find_spec("chatterbox") is not None
qwen3_installed = importlib.util.find_spec("qwen3_tts") is not None


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _run_real_tests() -> bool:
    return os.getenv("STUDIO_RUN_REAL_TESTS", "0") == "1"


REAL_TTS_REQUIRED = not (_run_real_tests() and _cuda_available())

pytestmark = pytest.mark.skipif(
    REAL_TTS_REQUIRED,
    reason="requires STUDIO_RUN_REAL_TESTS=1 + CUDA GPU (real TTS, explicitly gated)",
)


@pytest.mark.llm_integration
def test_kokoro_synthesizes_real_audio_on_gpu(tmp_path):
    from generation.kokoro import KokoroTTSProvider

    provider = KokoroTTSProvider({"device": "cuda"})
    out = tmp_path / "voice.wav"
    result = provider.synthesize(
        "Welcome to this deep dive into the movie.",
        voice="am_adam",
        narration={"tone": "dramatic", "emotion": "tense", "pace": 0.95},
        output_path=out,
    )
    assert out.exists() and out.stat().st_size > 0
    assert result["provider"] == "kokoro"
    assert result["device"] == "cuda"
    assert result["duration_sec"] > 0
    assert result["sample_rate"] == 24000
    assert result["mock"] is False
    assert result["supported"]["pace"] is True


@pytest.mark.llm_integration
def test_benchmark_kokoro_records_manifest_fields(tmp_path):
    from generation.tts_benchmark import benchmark_tts

    report = benchmark_tts(
        text="Benchmarking the narrator on the GPU.",
        providers=["kokoro"],
        output_dir=tmp_path,
        include_mock=False,
    )
    entry = report["results"][0]
    assert entry["status"] == "ok"
    assert entry["provider"] == "kokoro"
    assert entry["device"] == "cuda"
    assert entry["model"] == "hexgrad/Kokoro-82M"
    assert entry["generation_time_sec"] is not None
    assert entry["duration_sec"] > 0
    assert entry["sample_rate"] == 24000


@pytest.mark.llm_integration
def test_require_real_tts_passes_kokoro(tmp_path):
    from generation.kokoro import KokoroTTSProvider
    from utils.strict import require_real_tts

    provider = KokoroTTSProvider({"device": "cuda"})
    assert require_real_tts(provider, "TTS") is provider
