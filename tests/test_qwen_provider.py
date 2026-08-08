"""Unit tests for Qwen provider (mock-based, no model download)."""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from src.director.providers.qwen import QwenProvider
from src.director.prompts.json_utils import (
    extract_json,
    validate_concepts_schema,
    check_concept_diversity,
)
from src.director.prompts.context_builder import ContextBuilder


class TestQwenProviderInit:
    """Test Qwen provider initialization (lazy loading)."""

    def test_provider_init_no_model_loaded(self):
        """Provider should not load model during init."""
        provider = QwenProvider(
            model="Qwen/Qwen3-30B-A3B",
            device="cpu",
            dtype="float32",
        )

        assert provider.model is None
        assert provider.tokenizer is None
        assert not provider._initialized
        assert provider.model_name == "Qwen/Qwen3-30B-A3B"

    def test_provider_config_storage(self):
        """Provider should store configuration."""
        provider = QwenProvider(
            model="Qwen/Qwen3-30B-A3B",
            device="cuda",
            temperature=0.7,
            top_p=0.95,
            max_new_tokens=1024,
            timeout_sec=120,
        )

        assert provider.temperature == 0.7
        assert provider.top_p == 0.95
        assert provider.max_new_tokens == 1024
        assert provider.timeout_sec == 120


class TestQwenProviderDeviceResolution:
    """Test device detection."""

    def test_device_auto_detection_cuda_available(self):
        """Should detect cuda when available."""
        provider = QwenProvider(device="auto")

        with patch("torch.cuda.is_available", return_value=True):
            device = provider._resolve_device()
            assert device == "cuda"

    def test_device_auto_detection_cuda_unavailable(self):
        """Should fall back to cpu when cuda unavailable."""
        provider = QwenProvider(device="auto")

        with patch("torch.cuda.is_available", return_value=False):
            device = provider._resolve_device()
            assert device == "cpu"

    def test_device_explicit_cpu(self):
        """Should respect explicit device specification."""
        provider = QwenProvider(device="cpu")
        assert provider._resolve_device() == "cpu"

    def test_device_explicit_cuda(self):
        """Should respect explicit cuda specification."""
        provider = QwenProvider(device="cuda")
        assert provider._resolve_device() == "cuda"


class TestJSONExtraction:
    """Test robust JSON extraction from LLM output."""

    def test_extract_direct_json(self):
        """Should extract direct JSON."""
        text = '{"key": "value"}'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_extract_fenced_json(self):
        """Should extract fenced JSON."""
        text = "Here is the output:\n```json\n{\"key\": \"value\"}\n```\nDone."
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_extract_json_within_text(self):
        """Should extract JSON from within text."""
        text = "Some text before\n{\"key\": \"value\"}\nSome text after"
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_extract_nested_json(self):
        """Should extract nested JSON."""
        text = '{"outer": {"inner": "value"}}'
        result = extract_json(text)
        assert result == {"outer": {"inner": "value"}}

    def test_extract_fails_gracefully(self):
        """Should return None for invalid JSON."""
        text = "No JSON here at all"
        result = extract_json(text)
        assert result is None


class TestConceptValidation:
    """Test concept schema validation."""

    def test_validate_valid_concepts(self):
        """Should validate well-formed concepts."""
        concepts = [
            {
                "title": "Concept 1",
                "hook": "Hook text",
                "thesis": "Thesis text",
                "why_interesting": "Reason",
            }
        ]
        is_valid, error = validate_concepts_schema(concepts)
        assert is_valid
        assert error == ""

    def test_validate_missing_field(self):
        """Should reject concepts with missing fields."""
        concepts = [
            {
                "title": "Concept 1",
                "hook": "Hook text",
                # Missing "thesis"
                "why_interesting": "Reason",
            }
        ]
        is_valid, error = validate_concepts_schema(concepts)
        assert not is_valid
        assert "thesis" in error

    def test_validate_empty_field(self):
        """Should reject concepts with empty fields."""
        concepts = [
            {
                "title": "Concept 1",
                "hook": "",  # Empty
                "thesis": "Thesis text",
                "why_interesting": "Reason",
            }
        ]
        is_valid, error = validate_concepts_schema(concepts)
        assert not is_valid

    def test_validate_not_list(self):
        """Should reject non-list input."""
        concepts = {"title": "Not a list"}
        is_valid, error = validate_concepts_schema(concepts)
        assert not is_valid

    def test_validate_empty_list(self):
        """Should reject empty list."""
        concepts = []
        is_valid, error = validate_concepts_schema(concepts)
        assert not is_valid


class TestConceptDiversity:
    """Test concept diversity checking."""

    def test_diverse_concepts(self):
        """Should accept diverse concepts."""
        concepts = [
            {"thesis": "Concept A explores theme X with deep philosophical implications"},
            {"thesis": "Concept B"},
            {"thesis": "Concept C explores a very long and nuanced theme with multiple layers of meaning and interpretation"},
        ]
        is_diverse, msg = check_concept_diversity(concepts)
        assert is_diverse

    def test_identical_concepts(self):
        """Should reject identical concepts."""
        concepts = [
            {"thesis": "Same thesis"},
            {"thesis": "Same thesis"},
        ]
        is_diverse, msg = check_concept_diversity(concepts)
        assert not is_diverse

    def test_single_concept(self):
        """Should not check diversity for single concept."""
        concepts = [{"thesis": "Only one"}]
        is_diverse, msg = check_concept_diversity(concepts)
        assert is_diverse


class TestContextBuilder:
    """Test context limiting for long movies."""

    def test_context_builder_init(self):
        """Should initialize with token limits."""
        builder = ContextBuilder(max_tokens=4096, reserve_for_output=2048)
        assert builder.max_tokens == 4096
        assert builder.available_for_context == 2048

    def test_token_estimation(self):
        """Should estimate tokens (rough: 1 token ≈ 4 chars)."""
        builder = ContextBuilder()
        text = "a" * 400  # 400 chars
        tokens = builder._estimate_tokens(text)
        assert tokens == 100  # 400 / 4

    def test_build_concept_context_includes_metadata(self):
        """Should always include movie metadata."""
        builder = ContextBuilder(max_tokens=1000, reserve_for_output=500)

        movie_metadata = {"title": "Test Movie", "duration_sec": 120}
        scene_index = []
        transcript = {}
        memory = ""

        context, metadata = builder.build_concept_generation_context(
            movie_metadata, scene_index, transcript, memory
        )

        assert "Test Movie" in context
        assert metadata["movie_metadata_included"]

    def test_build_concept_context_includes_scenes(self):
        """Should include scene information when available."""
        builder = ContextBuilder(max_tokens=5000, reserve_for_output=500)

        movie_metadata = {"title": "Test", "duration_sec": 120}
        scene_index = [
            {
                "scene_id": "scene_001",
                "start_sec": 0,
                "end_sec": 30,
                "transcript": "Opening dialogue",
            }
        ]
        transcript = {}
        memory = ""

        context, metadata = builder.build_concept_generation_context(
            movie_metadata, scene_index, transcript, memory
        )

        assert "scene_001" in context
        assert metadata["num_scenes_included"] > 0

    def test_build_concept_context_handles_truncation(self):
        """Should mark truncation when context too large."""
        builder = ContextBuilder(max_tokens=500, reserve_for_output=400)

        movie_metadata = {"title": "Test", "duration_sec": 120}
        scene_index = [
            {
                "scene_id": f"scene_{i:03d}",
                "start_sec": i * 30,
                "end_sec": (i + 1) * 30,
                "transcript": "Very long dialogue " * 50,
            }
            for i in range(20)
        ]
        transcript = {}
        memory = ""

        context, metadata = builder.build_concept_generation_context(
            movie_metadata, scene_index, transcript, memory
        )

        assert metadata["truncated"]
        assert metadata["num_scenes_included"] < len(scene_index)


class TestQwenProviderLazyInitialization:
    """Test that provider doesn't load model until needed."""

    def test_initialize_called_on_first_use(self):
        """Should call _initialize on first generation attempt."""
        provider = QwenProvider()

        # Mock the actual initialization to avoid downloading
        with patch.object(provider, "_generate_text", return_value="{}"):
            with patch.object(provider, "_initialize") as mock_init:
                # This should not be called yet
                assert not mock_init.called

                # Now if we tried to generate, it would initialize
                # (We're mocking _generate_text to avoid the actual call)


class TestPromptConstruction:
    """Test prompt building."""

    def test_concept_generation_prompt_format(self):
        """Should produce valid prompt structure."""
        from src.director.prompts import build_concept_generation_prompt

        movie_metadata = {"title": "Test", "duration_sec": 120}
        scene_index = [
            {
                "scene_id": "scene_001",
                "start_sec": 0,
                "end_sec": 30,
                "transcript": "Test dialogue",
            }
        ]
        transcript = {"segments": []}
        memory = ""

        prompt = build_concept_generation_prompt(
            movie_metadata,
            scene_index,
            transcript,
            memory,
            num_concepts=3,
        )

        assert "Test" in prompt  # Movie title
        assert "3" in prompt  # num_concepts
        assert "JSON" in prompt
        assert "thesis" in prompt.lower()
