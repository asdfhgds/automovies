"""TTS benchmark tests (mock-only, fast)."""
import json

import pytest

from generation.tts_benchmark import benchmark_tts, load_benchmark_report


def test_benchmark_runs_mock_and_records_schema(tmp_path):
    report = benchmark_tts(
        text="Benchmark me please.",
        providers=["mock"],
        output_dir=tmp_path,
        include_mock=True,
        narration={"tone": "analytical", "emotion": "neutral", "pace": 1.0},
    )
    assert report["results"]
    entry = report["results"][0]
    assert entry["provider"] == "mock"
    assert entry["status"] == "ok"
    assert entry["device"] == "cpu"
    assert entry["generation_time_sec"] is not None
    assert entry["duration_sec"] > 0
    assert entry["sample_rate"] == 44100
    assert entry["mock"] is True
    assert (tmp_path / "benchmark_mock.wav").exists()

    loaded = load_benchmark_report(tmp_path)
    assert loaded is not None
    assert loaded["results"][0]["provider"] == "mock"


def test_benchmark_reports_unavailable_providers(tmp_path):
    from generation.provider_factory import available_tts_providers

    availability = available_tts_providers()
    # only benchmark providers that are NOT installed here so the fast suite
    # never loads a real model on CPU
    candidates = [name for name, info in availability.items() if not info.get("available")]
    if not candidates:
        pytest.skip("all real TTS providers installed; CPU benchmark not allowed")
    report = benchmark_tts(
        providers=candidates,
        output_dir=tmp_path,
        include_mock=False,
    )
    names = {r["provider"]: r["status"] for r in report["results"]}
    for name in candidates:
        assert names.get(name) == "unavailable"
        assert any(r["provider"] == name and r["error"] for r in report["results"])
    # report must be persisted
    assert (tmp_path / "tts_benchmark.json").exists()
