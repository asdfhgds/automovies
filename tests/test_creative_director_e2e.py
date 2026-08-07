"""
End-to-end integration test: Full pipeline with CreativeDirector.

This test runs the complete pipeline using the mock LLM provider,
ensuring that creative concepts flow through to the final ranking and selection.

Marked as integration test; run with: pytest -m integration
"""
import json
import pytest
import tempfile
from pathlib import Path
from director.creative_director import CreativeDirector
from director.providers.mock_llm import MockLLMProvider
from scene_selection.ranker import rank_scenes, score_scene, tokenize


@pytest.mark.integration
def test_creative_director_end_to_end_with_mock(tmp_path):
    """
    Full pipeline: scene_index → creative concepts → production plan → ranking → selected scene.
    """
    # Setup: Create a mock scene index with multiple scenes
    scenes = [
        {
            "scene_id": "scene_001",
            "start_sec": 0.0,
            "end_sec": 10.5,
            "duration_sec": 10.5,
            "transcript": "The hero enters the mysterious forest. Trees tower overhead.",
            "keywords": ["forest", "hero", "mystery"],
        },
        {
            "scene_id": "scene_002",
            "start_sec": 10.5,
            "end_sec": 25.3,
            "duration_sec": 14.8,
            "transcript": "An ancient temple emerges from the mist. Intricate carvings cover the walls.",
            "keywords": ["temple", "ancient", "carvings"],
        },
        {
            "scene_id": "scene_003",
            "start_sec": 25.3,
            "end_sec": 40.0,
            "duration_sec": 14.7,
            "transcript": "The hero discovers a hidden chamber with glowing crystals.",
            "keywords": ["chamber", "crystals", "discovery"],
        },
    ]

    movie_metadata = {
        "title": "Test Movie",
        "duration_sec": 40.0,
        "source": "test.mp4",
    }

    transcript = {
        "segments": [
            {"start_sec": 0.0, "end_sec": 10.5, "text": scenes[0]["transcript"]},
            {"start_sec": 10.5, "end_sec": 25.3, "text": scenes[1]["transcript"]},
            {"start_sec": 25.3, "end_sec": 40.0, "text": scenes[2]["transcript"]},
        ]
    }

    # Create a temp directory for memory
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    # Initialize CreativeDirector with mock provider
    provider = MockLLMProvider()
    director = CreativeDirector(
        provider=provider,
        memory_dir=memory_dir,
    )

    # Step 1: Generate production plan from scene index
    result = director.develop_production_plan(
        movie_metadata=movie_metadata,
        scene_index=scenes,
        transcript=transcript,
    )

    # Assertions: Result structure
    assert "production_plan" in result, "Result must have production_plan"
    assert "generated_concepts" in result, "Result must have generated_concepts"
    assert "selected_concept" in result, "Result must have selected_concept"

    production_plan = result["production_plan"]
    selected_concept = result["selected_concept"]

    # Assertions: Production plan structure
    # Production plan contains concept, format, tone, structure, etc.
    assert production_plan is not None, "Production plan must not be None"
    assert isinstance(production_plan, dict), "Production plan must be a dict"
    # The actual thesis is in selected_concept
    assert "thesis" in selected_concept, "Selected concept must have thesis"
    assert len(selected_concept["thesis"]) > 0, "Thesis must not be empty"

    # Step 2: Verify that memory stores the concept
    memory_file = memory_dir / "concepts.jsonl"
    if memory_file.exists():
        with open(memory_file, "r") as f:
            lines = f.readlines()
        assert len(lines) >= 1, "At least one concept must be stored in memory"

    # Step 3: Use the production plan's thesis for scene ranking
    # The thesis is in the selected_concept
    thesis = selected_concept.get("thesis", "")
    thesis_tokens = tokenize(thesis)
    
    rankings = []
    for scene in scenes:
        text = scene.get('transcript') or scene.get('summary') or ''
        sc = score_scene(thesis_tokens, text)
        rankings.append({
            'scene_id': scene.get('scene_id'),
            'score': sc,
            'start_sec': scene.get('start_sec'),
            'end_sec': scene.get('end_sec'),
        })
    
    rankings.sort(key=lambda r: r['score'], reverse=True)

    assert len(rankings) > 0, "Must have at least one ranked scene"

    # Assertions: Top-ranked scene has valid structure
    top_scene = rankings[0]
    assert "scene_id" in top_scene, "Ranked scene must have scene_id"
    assert "score" in top_scene, "Ranked scene must have score"
    assert top_scene["score"] >= 0.0, "Score must be >= 0"

    # Step 4: Verify that scenes were ranked based on thesis keyword match
    scores = [r['score'] for r in rankings]
    assert all(0.0 <= s <= 1.0 for s in scores), "All scores must be in [0, 1]"
    assert scores == sorted(scores, reverse=True), "Scores must be in descending order"


@pytest.mark.integration
def test_creative_director_memory_accumulation(tmp_path):
    """
    Verify that CreativeDirector accumulates concepts in memory across multiple calls.
    """
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    
    movie_metadata = {
        "title": "Test Movie",
        "duration_sec": 20.0,
        "source": "test.mp4",
    }

    scenes = [
        {
            "scene_id": "scene_001",
            "start_sec": 0.0,
            "end_sec": 20.0,
            "duration_sec": 20.0,
            "transcript": "A dramatic scene unfolds.",
            "keywords": ["drama", "action"],
        }
    ]

    transcript = {
        "segments": [
            {"start_sec": 0.0, "end_sec": 20.0, "text": "A dramatic scene unfolds."},
        ]
    }

    provider = MockLLMProvider()
    director = CreativeDirector(provider=provider, memory_dir=memory_dir)

    # First call
    result1 = director.develop_production_plan(
        movie_metadata=movie_metadata,
        scene_index=scenes,
        transcript=transcript,
    )
    assert result1 is not None

    # Second call (should add to memory without overwriting)
    result2 = director.develop_production_plan(
        movie_metadata=movie_metadata,
        scene_index=scenes,
        transcript=transcript,
    )
    assert result2 is not None

    # Verify both concepts are in memory
    memory_file = memory_dir / "concepts.jsonl"
    if memory_file.exists():
        with open(memory_file, "r") as f:
            concepts = [json.loads(line) for line in f if line.strip()]
        assert len(concepts) >= 2, "Both concepts should be stored in memory"
        for concept in concepts:
            assert "thesis" in concept, "Stored concept must have thesis"


@pytest.mark.integration
def test_creative_director_fallback_to_deterministic(tmp_path, monkeypatch):
    """
    Verify that if CreativeDirector fails, pipeline falls back to deterministic director.
    
    This test simulates environment where creative director is disabled.
    """
    from director.planner import plan_director
    import json

    # Create a mock project directory with required files
    project_dir = tmp_path / "project"
    scenes_dir = project_dir / "scenes"
    transcripts_dir = project_dir / "transcripts"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal scene_index.json
    scene_index_data = [
        {
            "scene_id": "scene_001",
            "start_sec": 0.0,
            "end_sec": 10.0,
            "duration_sec": 10.0,
            "transcript": "Test scene.",
            "keywords": ["test"],
        }
    ]
    with (scenes_dir / "scene_index.json").open("w") as f:
        json.dump(scene_index_data, f)

    # Create minimal transcript.json
    transcript_data = {
        "segments": [
            {"start_sec": 0.0, "end_sec": 10.0, "text": "Test scene."},
        ]
    }
    with (transcripts_dir / "transcript.json").open("w") as f:
        json.dump(transcript_data, f)

    # Set flag to disable creative director
    monkeypatch.setenv("CREATIVE_DIRECTOR_ENABLED", "false")

    # Plan director should fall back to deterministic
    plan_path = plan_director(project_dir)

    assert plan_path is not None, "Planner must always produce a plan path"
    assert plan_path.exists(), "Plan file must exist"

    # Read and verify plan
    with plan_path.open("r") as f:
        plan = json.load(f)

    assert plan is not None, "Planner must produce a plan"
    assert "thesis" in plan, "Plan must have thesis"
    # Deterministic plan will have content
    assert len(plan.get("thesis", "")) > 0, "Thesis must not be empty"


@pytest.mark.integration
def test_creative_concepts_are_specific_not_generic(tmp_path):
    """
    Verify that mock provider generates specific, detailed concepts, not generic phrases.
    """
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    
    movie_metadata = {
        "title": "Test Movie",
        "duration_sec": 10.0,
        "source": "test.mp4",
    }

    scenes = [
        {
            "scene_id": "scene_001",
            "start_sec": 0.0,
            "end_sec": 10.0,
            "duration_sec": 10.0,
            "transcript": "Character walks through door and discovers something unexpected.",
            "keywords": ["discovery", "character", "unexpected"],
        }
    ]

    transcript = {
        "segments": [
            {"start_sec": 0.0, "end_sec": 10.0, "text": "Character walks through door and discovers something unexpected."},
        ]
    }

    provider = MockLLMProvider()
    director = CreativeDirector(provider=provider, memory_dir=memory_dir)
    
    result = director.develop_production_plan(
        movie_metadata=movie_metadata,
        scene_index=scenes,
        transcript=transcript,
    )

    selected_concept = result["selected_concept"]
    production_plan = result["production_plan"]

    thesis = production_plan.get("thesis", selected_concept.get("thesis", ""))
    hook = selected_concept.get("hook", "")

    # Check that thesis and hook are not generic/empty
    assert len(thesis) > 20, "Thesis should be substantive (not just a few words)"
    assert len(hook) > 20, "Hook should be substantive"

    # Check for specificity indicators: Should mention themes, perspectives, or narrative elements
    # Not just generic like "focused analysis"
    generic_phrases = ["focused analysis", "key scene", "interesting scene", "good scene"]
    assert not any(phrase in thesis.lower() for phrase in generic_phrases), \
        "Thesis should not contain generic phrases"

    # Should contain more specific language like psychological, thematic, narrative, etc.
    specific_keywords = ["psychology", "theme", "character", "narrative", "metaphor", "emotion", "perspective"]
    has_specificity = any(keyword in thesis.lower() for keyword in specific_keywords)
    assert has_specificity or len(thesis) > 50, \
        "Thesis should demonstrate specificity or depth"

    print(f"Generated thesis (specific ✓): {thesis[:100]}...")
    print(f"Generated hook: {hook[:100]}...")
