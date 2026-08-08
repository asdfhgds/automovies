"""Tests for creative director components."""
import json
import pytest
from pathlib import Path
from director.memory import CreativeMemory
from director.critic import ConceptCritic
from director.creative_director import CreativeDirector
from director.providers.mock_llm import MockLLMProvider


@pytest.fixture
def temp_memory_dir(tmp_path):
    """Temporary memory directory."""
    return tmp_path / "memory"


class TestCreativeMemory:
    """Test creative memory."""

    def test_add_concept(self, temp_memory_dir):
        """Test adding a concept to memory."""
        memory = CreativeMemory(temp_memory_dir)
        memory.add_concept(
            title="Test Concept",
            thesis="Test thesis",
            tone="analytical",
            structure=[],
            visual_strategy="test",
            duration_sec=60,
            movie_title="Test Movie",
            themes=["theme1", "theme2"],
        )
        concepts = memory.get_all_concepts()
        assert len(concepts) == 1
        assert concepts[0]["title"] == "Test Concept"

    def test_get_concepts_summary(self, temp_memory_dir):
        """Test getting concepts summary."""
        memory = CreativeMemory(temp_memory_dir)
        memory.add_concept(
            title="Concept 1",
            thesis="Thesis 1",
            tone="analytical",
            structure=[],
            visual_strategy="test",
            duration_sec=60,
            movie_title="Movie 1",
            themes=["theme1"],
        )
        summary = memory.get_concepts_summary()
        assert "Concept 1" in summary
        assert "Thesis 1" in summary

    def test_clear_memory(self, temp_memory_dir):
        """Test clearing memory."""
        memory = CreativeMemory(temp_memory_dir)
        memory.add_concept(
            title="Concept",
            thesis="Thesis",
            tone="analytical",
            structure=[],
            visual_strategy="test",
            duration_sec=60,
            movie_title="Movie",
            themes=["theme"],
        )
        memory.clear_memory()
        concepts = memory.get_all_concepts()
        assert len(concepts) == 0


class TestConceptCritic:
    """Test concept critic."""

    def test_critique_strong_concept(self):
        """Test critiquing a strong concept."""
        concept = {
            "thesis": "This film explores how moral ambiguity shapes character decisions through a series of impossible choices.",
            "why_interesting": "Examines the psychological cost of compromise",
            "visual_strategy": "Use contrasting shot compositions to show internal conflict.",
            "supporting_scene_types": ["decision", "consequence"],
        }
        critique = ConceptCritic.critique(concept)
        assert critique["overall"] > 0.5
        assert "critique" in critique

    def test_critique_weak_concept(self):
        """Test critiquing a weak concept."""
        concept = {
            "thesis": "Focused analysis.",
            "why_interesting": "",
            "visual_strategy": "",
            "supporting_scene_types": [],
        }
        critique = ConceptCritic.critique(concept)
        assert critique["overall"] < 0.5
        assert "Weak" in critique["critique"]

    def test_score_originality(self):
        """Test originality scoring."""
        generic = {"thesis": "focused analysis of key moment"}
        specific = {"thesis": "explores how justice becomes compromise through incremental moral erosion"}
        
        generic_score = ConceptCritic._score_originality(generic)
        specific_score = ConceptCritic._score_originality(specific)
        
        assert specific_score > generic_score


class TestMockLLMProvider:
    """Test mock LLM provider."""

    def test_generate_concepts(self):
        """Test concept generation."""
        provider = MockLLMProvider()
        concepts = provider.generate_concepts(
            movie_metadata={"title": "Test Movie"},
            scene_index=[{"scene_id": "scene-1", "transcript": "test dialogue"}],
            transcript={"segments": []},
            creative_memory="",
            num_concepts=2,
        )
        assert len(concepts) == 2
        assert all("title" in c for c in concepts)
        assert all("thesis" in c for c in concepts)

    def test_generate_production_plan(self):
        """Test production plan generation."""
        provider = MockLLMProvider()
        concept = {
            "title": "Test Concept",
            "thesis": "Test thesis",
            "tone": "analytical",
            "visual_strategy": "Test strategy",
            "estimated_duration_sec": 90,
        }
        plan = provider.generate_production_plan(
            concept=concept,
            scene_index=[],
            transcript={},
        )
        assert "structure" in plan
        assert "visual_strategy" in plan
        assert plan["format"]["duration_sec"] == 90


class TestCreativeDirector:
    """Test creative director orchestration."""

    def test_develop_production_plan(self, temp_memory_dir):
        """Test full production plan development."""
        director = CreativeDirector(memory_dir=temp_memory_dir)
        
        result = director.develop_production_plan(
            movie_metadata={"title": "Test Movie"},
            scene_index=[{"scene_id": "scene-1", "transcript": "Test dialogue"}],
            transcript={"segments": [{"text": "Test"}]},
            num_concepts=2,
        )
        
        assert "generated_concepts" in result
        assert "critiques" in result
        assert "selected_concept" in result
        assert "production_plan" in result
        assert len(result["generated_concepts"]) == 2

    def test_integration_with_deterministic_director(self):
        """Test that creative director produces specific ideas."""
        director = CreativeDirector()
        
        result = director.develop_production_plan(
            movie_metadata={"title": "Test Movie"},
            scene_index=[
                {"scene_id": "scene-1", "transcript": "Character discusses moral choice"},
            ],
            transcript={"segments": [{"text": "Dialogue"}]},
        )
        
        selected = result["selected_concept"]
        # Should be a specific idea, not generic
        assert len(selected["thesis"].split()) > 5
        assert "focused analysis" not in selected["thesis"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
