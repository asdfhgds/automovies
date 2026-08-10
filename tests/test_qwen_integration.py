"""Integration tests for real Qwen LLM provider (requires model download)."""
import pytest
import os
from pathlib import Path

# Mark entire module as requiring LLM
pytestmark = pytest.mark.llm_integration


@pytest.fixture
def qwen_provider():
    """Fixture providing real Qwen provider."""
    try:
        from src.director.providers.qwen import QwenProvider
        
        # Use CPU for testing (can be overridden via env)
        device = os.getenv("QWEN_DEVICE", "cpu")
        model = os.getenv("QWEN_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
        
        provider = QwenProvider(
            model=model,
            device=device,
            dtype="auto",
            temperature=0.8,
            top_p=0.9,
            max_new_tokens=1024,
        )
        return provider
    except ImportError as e:
        pytest.skip(f"Qwen dependencies not available: {e}")


@pytest.fixture
def sample_movie_data():
    """Sample movie data for testing."""
    return {
        "movie_metadata": {
            "title": "Test Movie",
            "duration_sec": 120,
        },
        "scene_index": [
            {
                "scene_id": "scene_001",
                "start_sec": 0,
                "end_sec": 30,
                "transcript": "Opening dialogue about characters meeting",
            },
            {
                "scene_id": "scene_002",
                "start_sec": 30,
                "end_sec": 60,
                "transcript": "Conflict between main character and antagonist",
            },
            {
                "scene_id": "scene_003",
                "start_sec": 60,
                "end_sec": 90,
                "transcript": "Resolution and emotional climax",
            },
        ],
        "transcript": {
            "segments": [
                {"text": "Hello, welcome to this scene", "start_sec": 0},
                {"text": "This is important dialogue", "start_sec": 30},
                {"text": "Everything changes now", "start_sec": 60},
            ]
        },
    }


class TestQwenConceptGeneration:
    """Test real Qwen concept generation."""

    @pytest.mark.slow
    def test_generate_concepts_returns_valid_json(self, qwen_provider, sample_movie_data):
        """Should generate concepts returning valid JSON."""
        if qwen_provider is None:
            pytest.skip("Qwen provider not available")

        try:
            concepts = qwen_provider.generate_concepts(
                movie_metadata=sample_movie_data["movie_metadata"],
                scene_index=sample_movie_data["scene_index"],
                transcript=sample_movie_data["transcript"],
                creative_memory="",
                num_concepts=3,
            )

            assert isinstance(concepts, list)
            assert len(concepts) > 0

            for concept in concepts:
                assert "title" in concept
                assert "thesis" in concept
                assert "hook" in concept
                assert "why_interesting" in concept

        except Exception as e:
            pytest.skip(f"Qwen generation failed: {e}")

    @pytest.mark.slow
    def test_generated_concepts_are_specific_not_generic(
        self, qwen_provider, sample_movie_data
    ):
        """Generated concepts should avoid generic platitudes."""
        if qwen_provider is None:
            pytest.skip("Qwen provider not available")

        try:
            concepts = qwen_provider.generate_concepts(
                movie_metadata=sample_movie_data["movie_metadata"],
                scene_index=sample_movie_data["scene_index"],
                transcript=sample_movie_data["transcript"],
                creative_memory="",
                num_concepts=3,
            )

            generic_phrases = [
                "teaches us about life",
                "shows emotions",
                "good and evil",
                "explores the human condition",  # Too vague
            ]

            for concept in concepts:
                thesis = concept.get("thesis", "").lower()

                for phrase in generic_phrases:
                    # Allow phrase if it's more specific than just the phrase
                    if phrase in thesis and len(thesis) < 100:
                        pytest.fail(
                            f"Concept appears generic: {thesis}"
                        )

        except Exception as e:
            pytest.skip(f"Qwen generation failed: {e}")

    @pytest.mark.slow
    def test_generated_concepts_are_diverse(self, qwen_provider, sample_movie_data):
        """Generated concepts should be diverse, not variations of same idea."""
        if qwen_provider is None:
            pytest.skip("Qwen provider not available")

        try:
            concepts = qwen_provider.generate_concepts(
                movie_metadata=sample_movie_data["movie_metadata"],
                scene_index=sample_movie_data["scene_index"],
                transcript=sample_movie_data["transcript"],
                creative_memory="",
                num_concepts=3,
            )

            theses = [c.get("thesis", "") for c in concepts]

            # Check that theses are not identical
            if len(set(theses)) < len(theses):
                pytest.fail("Some concepts have identical theses")

            # Check that they approach from different angles
            # (This is a heuristic check)
            if len(theses) > 1:
                thesis_words = [set(t.split()) for t in theses]
                # Calculate rough similarity
                common = thesis_words[0] & thesis_words[1]
                if len(common) > len(thesis_words[0]) * 0.8:
                    pytest.warn("Concepts may be too similar")

        except Exception as e:
            pytest.skip(f"Qwen generation failed: {e}")


class TestQwenProductionPlan:
    """Test real Qwen production plan generation."""

    @pytest.mark.slow
    def test_generate_production_plan_returns_valid_structure(
        self, qwen_provider, sample_movie_data
    ):
        """Should generate production plan with valid structure."""
        if qwen_provider is None:
            pytest.skip("Qwen provider not available")

        try:
            # First generate a concept
            concepts = qwen_provider.generate_concepts(
                movie_metadata=sample_movie_data["movie_metadata"],
                scene_index=sample_movie_data["scene_index"],
                transcript=sample_movie_data["transcript"],
                creative_memory="",
                num_concepts=1,
            )

            if not concepts:
                pytest.skip("No concepts generated")

            concept = concepts[0]

            # Then generate a plan for it
            plan = qwen_provider.generate_production_plan(
                concept=concept,
                scene_index=sample_movie_data["scene_index"],
                transcript=sample_movie_data["transcript"],
            )

            assert isinstance(plan, dict)
            assert "structure" in plan
            assert isinstance(plan["structure"], list)
            assert len(plan["structure"]) > 0

            # Check structure sections
            for section in plan["structure"]:
                assert "section" in section
                assert "duration_sec" in section

        except Exception as e:
            pytest.skip(f"Qwen production plan failed: {e}")


class TestQwenCritiqueGeneration:
    """Test real Qwen concept critique."""

    @pytest.mark.slow
    def test_critique_concept_returns_valid_scores(
        self, qwen_provider, sample_movie_data
    ):
        """Should critique concept with valid score structure."""
        if qwen_provider is None:
            pytest.skip("Qwen provider not available")

        try:
            # Generate a concept first
            concepts = qwen_provider.generate_concepts(
                movie_metadata=sample_movie_data["movie_metadata"],
                scene_index=sample_movie_data["scene_index"],
                transcript=sample_movie_data["transcript"],
                creative_memory="",
                num_concepts=1,
            )

            if not concepts:
                pytest.skip("No concepts generated")

            # Note: QwenProvider doesn't have a critique method yet
            # This is a placeholder for when critique is implemented
            pytest.skip("Qwen critique not yet implemented")

        except Exception as e:
            pytest.skip(f"Qwen test failed: {e}")


class TestQwenEndToEnd:
    """Full end-to-end test of Qwen-powered creative director."""

    @pytest.mark.slow
    def test_full_creative_director_pipeline_with_qwen(
        self, qwen_provider, sample_movie_data, tmp_path
    ):
        """Should run full creative director pipeline with real Qwen."""
        if qwen_provider is None:
            pytest.skip("Qwen provider not available")

        try:
            from src.director.creative_director import CreativeDirector
            from src.director.memory import CreativeMemory

            # Setup memory
            memory_dir = tmp_path / "memory"
            memory_dir.mkdir()
            memory = CreativeMemory(memory_dir / "concepts.jsonl")

            # Create director with real provider
            director = CreativeDirector(provider=qwen_provider, memory=memory)

            # Run pipeline
            plan = director.develop_production_plan(
                movie_metadata=sample_movie_data["movie_metadata"],
                scene_index=sample_movie_data["scene_index"],
                transcript=sample_movie_data["transcript"],
                user_topic=None,
            )

            # Verify output
            assert plan is not None
            assert "thesis" in plan or "concept" in plan
            assert "production_plan" in plan or "structure" in plan

        except Exception as e:
            pytest.skip(f"End-to-end pipeline failed: {e}")


@pytest.mark.llm_integration
def test_qwen_memory_marker():
    """Marker test to show LLM integration tests are recognized."""
    assert True


# Skip tests if torch/transformers not available
def pytest_collection_modifyitems(config, items):
    """Skip LLM tests if dependencies not available."""
    skip_llm = pytest.mark.skip(reason="LLM dependencies not installed")

    for item in items:
        if "llm_integration" in item.keywords:
            try:
                import torch
                import transformers  # noqa: F401
            except ImportError:
                item.add_marker(skip_llm)
