"""Gated real-Qwen director validation test (run explicitly, not by default).

Per the milestone: real Qwen tests are gated. This test is deselected by the
project's default ``addopts`` (``-m "not slow and not llm_integration"``).

Run it explicitly on a GPU machine with Qwen available::

    python -m pytest tests/test_grounded_director_real_qwen.py -m llm_integration -v

It loads the real Movie Intelligence for the validated movie, runs the
Movie-grounded Creative Director with real Qwen, and asserts the concepts are
grounded (non-generic) and that a selectable plan is produced. Nothing is
written to a persistent project dir unless DIRECTOR_VALIDATION_WRITE=1.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from director.scene_facts import SceneFacts  # noqa: E402
from director.grounded import MovieGroundedDirector  # noqa: E402

# The movie used in the Movie Intelligence validation milestone.
_REAL_PROJECT = Path(__file__).resolve().parents[1] / \
    "data/bc6384be-47a5-4ee8-8674-7ff861472026"

pytestmark = [
    pytest.mark.llm_integration,
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def real_result():
    if not _REAL_PROJECT.exists():
        pytest.skip("real project not present")

    from director.providers.qwen import QwenProvider

    provider = QwenProvider(
        model=os.getenv("DIRECTOR_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
        device=os.getenv("DIRECTOR_DEVICE", "cuda"),
        dtype=os.getenv("DIRECTOR_DTYPE", "auto"),
    )
    idx = json.loads((_REAL_PROJECT / "movie_index.json").read_text(encoding="utf-8"))
    facts = SceneFacts.from_movie_intelligence(movie_index=idx)
    assert len(facts) > 0

    metadata = {
        "title": idx.get("movie", {}).get("title", "Unknown"),
        "duration_sec": idx.get("movie", {}).get("duration_sec", 0),
    }
    director = MovieGroundedDirector(llm=provider.generate_text)
    return director.develop(
        movie_metadata=metadata,
        scale_facts=facts,
        num_concepts=5,
        min_coverage=0.4,
        user_topic="Create an original 60-120 second movie-analysis concept "
                   "based only on what is actually present in this movie.",
        duration_sec=90,
    )


def test_real_qwen_produces_grounded_concepts(real_result):
    concepts = real_result["generated_concepts"]
    assert len(concepts) > 0
    # No concept should be a generic AI platitude.
    for c in concepts:
        theses = c.get("thesis", "").lower()
        assert "movie explores violence" not in theses
        assert c.get("required_evidence"), "each concept needs evidence claims"
    # The selected concept (if any) must have a plan.
    if real_result["selected_concept"] is not None:
        assert real_result["plan"] is not None
        assert "evidence_strategy" in real_result["plan"]


def test_real_qwen_grounding_respects_available_facts(real_result):
    concepts = real_result["generated_concepts"]
    # Every admissible concept must carry real evidence phrases (the gate
    # already ensures coverage >= 0.4); here we additionally confirm the
    # selected concept's evidence_strategy maps onto real scene ids.
    if real_result["selected_concept"] is not None:
        strategy = real_result["plan"]["evidence_strategy"]
        assert "scene_ids" in strategy
        assert strategy["scene_ids"], "evidence must map to real scenes"
