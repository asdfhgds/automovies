"""Unit tests for strict GPU mode and the Qwen script writer's pure logic.

No model loading, no network: these tests exercise the strict-mode guards and
the JSON/normalization helpers that a real Qwen run relies on.
"""
import json
import os
from pathlib import Path

import pytest

from utils import strict as strict_mod
from script import qwen_writer


# ---------------------------------------------------------------------------
# QwenProvider model cache (OOM prevention: director + script share one load)
# ---------------------------------------------------------------------------

def test_qwen_reuses_cached_model(monkeypatch):
    """A second QwenProvider for the same model must NOT load the model again.

    The class-level cache is what keeps the model weights resident only once on a
    16GB T4 (director + script stages). We prove the short-circuit by failing the
    test if `transformers` is ever imported during init.
    """
    from director.providers import qwen as qwen_module

    fake_model, fake_tok = object(), object()
    key = ("Qwen/Qwen3-4B-Instruct-2507", "cpu", "torch.float32", False)
    qwen_module._MODEL_CACHE[key] = (fake_model, fake_tok, "cpu")

    real_import = __import__

    def guard(name, *a, **k):
        if name == "transformers" or name.startswith("transformers."):
            raise RuntimeError("transformers imported on a cache hit (model loaded twice!)")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", guard)

    provider = qwen_module.QwenProvider(model="Qwen/Qwen3-4B-Instruct-2507", device="cpu", dtype="float32")
    provider._initialize()

    assert provider.model is fake_model
    assert provider.tokenizer is fake_tok
    assert provider._initialized is True
    assert provider.device_resolved == "cpu"
    assert provider.model_load_time_sec == 0.0


def test_qwen_release_model_clears_cache():
    from director.providers import qwen as qwen_module

    qwen_module._MODEL_CACHE["something"] = ("a", "b", "c")
    qwen_module.QwenProvider.release_model()
    assert qwen_module._MODEL_CACHE == {}


def test_qwen_release_model_tolerates_no_torch(monkeypatch):
    """release_model must not crash on boxes without CUDA/torch usable."""
    from director.providers import qwen as qwen_module

    qwen_module._MODEL_CACHE["x"] = 1

    def boom(*a, **k):
        raise ImportError("torch intentionally unavailable")

    monkeypatch.setattr("builtins.__import__", boom)
    qwen_module.QwenProvider.release_model()  # should not raise
    assert qwen_module._MODEL_CACHE == {}


# ---------------------------------------------------------------------------
# Strict mode helpers
# ---------------------------------------------------------------------------

def _set_env(monkeypatch, **kw):
    for k, v in kw.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)


def test_strict_mode_enabled_toggle(monkeypatch):
    _set_env(monkeypatch, REQUIRE_REAL_LLM=None)
    assert strict_mod.strict_mode_enabled() is False
    _set_env(monkeypatch, REQUIRE_REAL_LLM="true")
    assert strict_mod.strict_mode_enabled() is True
    _set_env(monkeypatch, REQUIRE_REAL_LLM="TRUE")
    assert strict_mod.strict_mode_enabled() is True
    _set_env(monkeypatch, REQUIRE_REAL_LLM="false")
    assert strict_mod.strict_mode_enabled() is False


def test_require_real_provider_rejects_none():
    with pytest.raises(RuntimeError, match="REQUIRE_REAL_LLM"):
        strict_mod.require_real_provider(None, "Director")


def test_require_real_provider_rejects_mock(monkeypatch):
    _set_env(monkeypatch, REQUIRE_REAL_LLM="true")
    from director.providers.mock_llm import MockLLMProvider
    with pytest.raises(RuntimeError, match="REQUIRE_REAL_LLM=true"):
        strict_mod.require_real_provider(MockLLMProvider(), "Director")


def test_require_real_provider_accepts_real():
    class _Real:
        pass

    assert strict_mod.require_real_provider(_Real(), "Director") is not None


def test_require_cuda_raises_without_gpu():
    # CI/local has no CUDA; assert we refuse instead of silently running on CPU.
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        strict_mod.require_cuda()


def test_llm_provider_factory_strict_forbids_mock(monkeypatch):
    _set_env(monkeypatch, REQUIRE_REAL_LLM="true")
    from director.provider_factory import get_llm_provider_from_config
    with pytest.raises(RuntimeError, match="DIRECTOR_PROVIDER=qwen"):
        get_llm_provider_from_config({"provider": "mock"})


def test_llm_provider_factory_non_strict_mock(monkeypatch):
    _set_env(monkeypatch, REQUIRE_REAL_LLM=None)
    from director.provider_factory import get_llm_provider_from_config
    provider = get_llm_provider_from_config({"provider": "mock"})
    assert provider is not None
    from director.providers.mock_llm import MockLLMProvider
    assert isinstance(provider, MockLLMProvider)


def test_llm_provider_factory_strict_qwen_needs_cuda(monkeypatch):
    # Local box has no CUDA: strict qwen config must raise even before inference.
    _set_env(monkeypatch, REQUIRE_REAL_LLM="true")
    from director.provider_factory import get_llm_provider_from_config
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        get_llm_provider_from_config({"provider": "qwen", "device": "auto"})


# ---------------------------------------------------------------------------
# Qwen script writer helpers (pure, no model)
# ---------------------------------------------------------------------------

def test_extract_json_direct():
    assert qwen_writer._extract_json('{"sections": []}') == {"sections": []}


def test_extract_json_fenced():
    raw = "```json\n{\"sections\": [{\"a\": 1}]}\n```"
    assert qwen_writer._extract_json(raw) == {"sections": [{"a": 1}]}


def test_extract_json_within_text():
    raw = "Sure! Here it is: {\"sections\": []} -- hope that helps"
    assert qwen_writer._extract_json(raw) == {"sections": []}


def test_extract_json_invalid_returns_none():
    assert qwen_writer._extract_json("no json here") is None


def _structure():
    return [
        {"id": "intro", "goal": "Hook", "target_seconds": 15},
        {"id": "analysis", "goal": "Analyze", "target_seconds": 30},
        {"id": "closing", "goal": "Close", "target_seconds": 10},
    ]


def _selected():
    return [
        {"scene_id": "s1", "start_sec": 0.0, "end_sec": 5.0},
        {"scene_id": "s2", "start_sec": 5.0, "end_sec": 10.0},
    ]


def test_normalize_sections_basic():
    raw = [
        {"section_id": "intro", "text": "Open with thesis", "estimated_seconds": 15, "scene_ids": ["s1"]},
        {"section_id": "analysis", "text": "Evidence here", "estimated_seconds": 30, "scene_ids": ["s1", "s2"]},
    ]
    sections = qwen_writer._normalize_sections(raw, _structure(), _selected())
    assert len(sections) == 2
    assert sections[0]["estimated_seconds"] == 15
    assert sections[1]["scene_ids"] == ["s1", "s2"]


def test_normalize_sections_filters_invalid():
    raw = [
        {"section_id": "", "text": "  ", "estimated_seconds": "bad", "scene_ids": []},
        {"section_id": None, "text": "Good text", "estimated_seconds": 5, "scene_ids": ["nonexistent"]},
    ]
    sections = qwen_writer._normalize_sections(raw, _structure(), _selected())
    assert len(sections) == 1
    assert sections[0]["text"] == "Good text"
    assert sections[0]["estimated_seconds"] == 5
    assert sections[0]["scene_ids"] == ["s1"]  # unknown reduced to known first
    assert sections[0]["section_id"]  # defaulted


def test_normalize_sections_clamps_seconds():
    raw = [{"section_id": "a", "text": "x", "estimated_seconds": 99999, "scene_ids": ["s1"]}]
    sections = qwen_writer._normalize_sections(raw, _structure(), _selected())
    assert sections[0]["estimated_seconds"] <= 300


def test_normalize_sections_empty_raises():
    with pytest.raises(ValueError):
        qwen_writer._normalize_sections([], _structure(), _selected())


def test_build_prompt_contains_thesis_and_scenes():
    prompt = qwen_writer._build_prompt(
        "Contrast drives meaning", "analytical", _structure(), _selected(), {"s1": {"transcript": "hello"}}
    )
    assert "Contrast drives meaning" in prompt
    assert "s1" in prompt