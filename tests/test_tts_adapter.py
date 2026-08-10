"""TTS adapter tests: meta writing, narration props, strict rejection."""
import json
import os

import pytest

from audio.tts_adapter import synthesize_voice


def _write_project(tmp_path, text="Narration text for testing."):
    project = tmp_path / "project"
    (project / "audio").mkdir(parents=True)
    script = {
        "voiceover_text": text,
        "sections": [
            {"section_id": "intro", "text": text, "estimated_seconds": 3, "scene_ids": []}
        ],
        "narration_properties": {
            "tone": "dramatic",
            "emotion": "tense",
            "pace": 0.9,
            "energy": 0.6,
            "dramatic_intensity": 0.8,
        },
    }
    (project / "script.json").write_text(json.dumps(script), encoding="utf-8")
    return project


def test_synthesize_voice_writes_meta_with_mock(tmp_path, monkeypatch):
    monkeypatch.setenv("TTS_PROVIDER", "mock")
    project = _write_project(tmp_path)
    voice = synthesize_voice(project)
    assert voice.exists()
    meta = json.loads((project / "audio" / "tts_meta.json").read_text(encoding="utf-8"))
    assert meta["voice_provider"] == "mock"
    assert meta["mock"] is True
    assert meta["duration_sec"] > 0
    assert meta["sample_rate"] == 44100
    assert meta["narration_properties"]["tone"] == "dramatic"
    assert meta["narration_properties"]["dramatic_intensity"] == 0.8


def test_synthesize_voice_strict_rejects_mock(tmp_path, monkeypatch):
    monkeypatch.setenv("TTS_PROVIDER", "mock")
    monkeypatch.setenv("REQUIRE_REAL_TTS", "true")
    project = _write_project(tmp_path)
    with pytest.raises(RuntimeError, match="REQUIRE_REAL_TTS"):
        synthesize_voice(project)
    assert not (project / "audio" / "tts_meta.json").exists()


def test_synthesize_voice_raises_without_script(tmp_path, monkeypatch):
    monkeypatch.setenv("TTS_PROVIDER", "mock")
    project = tmp_path / "empty"
    project.mkdir()
    with pytest.raises(FileNotFoundError):
        synthesize_voice(project)
