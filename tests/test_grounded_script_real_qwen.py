"""Gated real-Qwen validation of the grounded script + editorial plan chain.

Companion to ``tests/test_grounded_director_real_qwen.py``. Builds on real
Movie Intelligence + real Qwen Grounded Director -> Grounding Contract ->
Grounded Script -> Grounded Editorial Plan, and asserts the script's analytical
sections always cite real scenes/evidence windows.

Run explicitly on a GPU machine with Qwen available::

    python -m pytest tests/test_grounded_script_real_qwen.py -m llm_integration -v
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from director.scene_facts import SceneFacts  # noqa: E402
from director.grounded import MovieGroundedDirector  # noqa: E402
from director.grounding_contract import build_grounding_contract, contract_is_valid  # noqa: E402
from script.grounded import GroundedScriptGenerator  # noqa: E402
from editorial.grounded import GroundedEditorialPlanner  # noqa: E402
from editorial.plan import validate_plan  # noqa: E402

# The movie used in the Movie Intelligence validation milestone.
_REAL_PROJECT = Path(__file__).resolve().parents[1] / \
    "data/bc6384be-47a5-4ee8-8674-7ff861472026"

pytestmark = [
    pytest.mark.llm_integration,
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def real_chain():
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
    result = director.develop(
        movie_metadata=metadata,
        scale_facts=facts,
        num_concepts=5,
        min_coverage=0.4,
        user_topic="Create an original 60-120 second movie-analysis concept "
                   "based only on what is actually present in this movie.",
        duration_sec=90,
    )
    assert result["selected_concept"] is not None, "selectable concept required"
    if result["plan"] is None:
        # Strict plan gate: no grounded plan is emitted, so the script chain
        # must not be forced to proceed from invented prose.
        pytest.skip("grounded plan was honestly rejected by the strict gate")
    assert result["plan"]["evidence_strategy"]["scene_ids"], \
        "grounded plan must map to real scenes"

    contract = build_grounding_contract(result, facts, movie_index=idx)
    assert contract_is_valid(contract) == []

    generator = GroundedScriptGenerator(target_sec=90)
    script = generator.generate(contract, movie_index=idx, project_id="real")

    return {
        "movie_index": idx,
        "contract": contract,
        "script": script,
    }


def test_real_script_sections_cite_real_scenes(real_chain):
    script = real_chain["script"]
    scene_ids = {s["scene_id"] for s in real_chain["movie_index"]["scenes"]}
    assert script["grounded"] is True
    assert len(script["sections"]) >= 5
    for section in script["sections"]:
        assert set(section["scene_ids"]).issubset(scene_ids)
        for ev in section["narrative_evidence"]:
            assert ev["scene_id"] in scene_ids
            assert 0 < ev["end_sec"] - ev["start_sec"] <= 6.0 + 1e-6
        assert section["narration"], "every section has narration"


def test_real_contract_references_only_known_scenes(real_chain):
    scene_ids = {s["scene_id"] for s in real_chain["movie_index"]["scenes"]}
    for ref in real_chain["contract"]["supporting_scenes"]:
        assert ref["scene_id"] in scene_ids


def test_real_editorial_plan_from_grounded_script(real_chain):
    script = real_chain["script"]
    idx = real_chain["movie_index"]
    scorer = SceneFacts.from_movie_intelligence(movie_index=idx)
    planner = GroundedEditorialPlanner(script=script)
    director_plan = {
        "thesis": real_chain["contract"]["concept"]["thesis"],
        "title": real_chain["contract"]["concept"]["title"],
        "director_provider": "grounded",
    }
    plan = planner.create_plan(idx, director_plan, target_sec=90)
    assert validate_plan(plan) == []
    assert len(plan.segments) == len(script["sections"])
    for seg in plan.segments:
        assert seg.narration.text
        for ev in seg.evidence:
            assert ev.end_sec - ev.start_sec <= 6.0 + 1e-6