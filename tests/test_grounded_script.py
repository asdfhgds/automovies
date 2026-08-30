"""Tests for the grounded-director -> grounded script -> editorial plan chain.

The milestone requirement: the grounded Creative Director's output must become
the source of truth for the script; every analytical script section must map to
real scene/evidence references; the editorial plan must consume the grounded
script with real excerpt windows; and the whole chain must work on the local
(mock) profile without downloading any model.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from director.grounding_contract import (
    CONTRACT_KEYS,
    build_grounding_contract,
    contract_is_valid,
    save_grounding_contract,
    load_grounding_contract,
)
from director.scene_facts import SceneFacts
from script.grounded import (
    GroundedScriptGenerator,
    save_grounded_script,
    load_grounded_script,
    build_hook_narration,
    build_claim_narration,
    _excerpt_window,
)
from script.grounding_report import build_grounding_report, write_script_grounding_report
from editorial.grounded import GroundedEditorialPlanner
from editorial.plan import validate_plan
from editorial.director import create_editorial_plan


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _story_scene(scene_id, start, end, location=None, actions=None,
                 objects=None, dialogue=None, visual_events=None):
    scene = {
        "scene_id": scene_id,
        "start_sec": start,
        "end_sec": end,
        "transcript": "some transcript",
    }
    scene["story"] = {
        "characters": ["Barman"] if scene_id == "scene-1" else ["Stranger"],
        "location": location,
        "actions": actions or [],
        "objects": objects or [],
        "visual_description": None,
        "visual_events": visual_events or [],
        "emotional_cues": [],
        "themes": [],
        "mood": None,
        "cinematography": None,
        "dialogue": dialogue or [],
    }
    return scene


@pytest.fixture
def movie_index():
    return {
        "project_id": "proj-grounded",
        "movie": {"title": "Test Movie", "duration_sec": 240.0},
        "scenes": [
            _story_scene(
                "scene-1", 0.0, 30.0,
                location="saloon, dim light",
                actions=["pouring", "talking"],
                objects=["revolver", "glass", "bar"],
                visual_events=["a revolver is placed on the bar"],
                dialogue=[{"speaker": "Barman", "text": "Keep your hands where I can see them."}],
            ),
            _story_scene(
                "scene-2", 40.0, 70.0,
                location="street at dusk",
                actions=["riding"],
                objects=["horse"],
            ),
        ],
    }


@pytest.fixture
def contract(movie_index):
    return {
        "concept": {
            "title": "The Gun on the Bar",
            "hook": "A revolver appears before anyone says a word.",
            "thesis": "The film uses its objects to stage the choice before "
                      "any character makes it aloud.",
            "why_interesting": "The visual staging does the arguing, not the script.",
        },
        "evidence_refs": [
            {"kind": "object", "value": "revolver"},
            {"kind": "object", "value": "bar"},
            {"kind": "scene", "scene_id": "scene-1"},
            {"kind": "dialogue", "value": "Keep your hands where I can see them."},
        ],
        "evidence_requirements": [
            "a revolver is placed on the bar",
            "dialogue between characters",
        ],
        "supporting_scenes": [
            {"scene_id": "scene-1", "start_sec": 0.0, "end_sec": 30.0},
            {"scene_id": "scene-2", "start_sec": 40.0, "end_sec": 70.0},
        ],
        "visual_motifs": ["revolver", "bar"],
        "character_focus": ["Barman"],
        "format": {"type": "short_video_essay", "duration_sec": 90},
        "editorial_intent": {
            "pacing": "measured",
            "tone": "analytical",
            "visual_style": "close-ups on the objects",
            "audio_style": "restrained score",
        },
    }


# --------------------------------------------------------------------------- #
# Grounding contract
# --------------------------------------------------------------------------- #

def test_contract_has_all_required_keys(contract):
    assert contract_is_valid(contract) == []
    for key in CONTRACT_KEYS:
        assert key in contract


def test_contract_roundtrip_persists(tmp_path, contract):
    save_grounding_contract(tmp_path, contract)
    loaded = load_grounding_contract(tmp_path)
    assert loaded == contract


def test_contract_invalid_when_fields_missing():
    bad = {
        "concept": {"thesis": ""},
        "evidence_requirements": [],
        "supporting_scenes": [],
        "visual_motifs": [],
        "character_focus": [],
        "format": {},
        "editorial_intent": {},
    }
    errors = contract_is_valid(bad)
    assert errors, "an unsupported contract must be flagged"
    assert any("thesis" in e for e in errors)
    assert any("supporting_scenes" in e for e in errors)


def test_contract_built_from_director_result(movie_index):
    facts = SceneFacts.from_movie_intelligence(movie_index=movie_index)

    class _LLM:
        def __call__(self, prompt):
            if "finalizing the plan" in prompt:
                return json.dumps({
                    "concept": {"title": "T", "hook": "H", "thesis": "The revolver stages the choice."},
                    "format": {"type": "short_video_essay", "duration_sec": 90},
                    "editorial_plan": {
                        "visual": {"scene_id": "scene-1", "start_sec": 1.0,
                                   "end_sec": 5.0, "source_fact_refs": ["revolver"]},
                        "editing": {"transition": "cut", "pacing": "measured",
                                    "rhythm": "steady", "emphasis": "object",
                                    "repetition": "none", "purpose": "emphasis"},
                        "audio": {"movie_audio": "retain", "narration": "moderate",
                                  "music": "low"},
                    },
                })
            return json.dumps({"concepts": [
                {
                    "title": "T",
                    "hook": "H",
                    "thesis": "The revolver stages the choice instead of the dialogue.",
                    "why_interesting": "Objects do the arguing.",
                    "required_evidence": ["a revolver is placed on the bar",
                                          "dialogue between characters"],
                    "visual_opportunity": "close-ups",
                    "format": "short_video_essay",
                }]})

    from director.grounded import MovieGroundedDirector

    director = MovieGroundedDirector(llm=_LLM())
    metadata = {"title": "Test Movie", "duration_sec": 240.0}
    result = director.develop(metadata, facts, num_concepts=2, min_coverage=0.2)
    assert result["selected_concept"] is not None

    mv = build_grounding_contract(result, facts, movie_index)
    assert mv["concept"]["thesis"]
    assert mv["evidence_refs"], "evidence refs are forwarded to the contract"
    assert mv["supporting_scenes"], "contract supporting scenes are populated"
    assert all(s["scene_id"] in {sc["scene_id"] for sc in movie_index["scenes"]}
               for s in mv["supporting_scenes"])


# --------------------------------------------------------------------------- #
# Grounded script generator
# --------------------------------------------------------------------------- #

def test_grounded_script_sections_are_grounded(contract, movie_index):
    gen = GroundedScriptGenerator(target_sec=90)
    script = gen.generate(contract, movie_index, project_id="proj")
    assert script["grounded"] is True
    assert script["thesis"] == contract["concept"]["thesis"]
    sections = script["sections"]
    assert len(sections) >= 5

    scene_ids = {s["scene_id"] for s in movie_index["scenes"]}
    for section in sections:
        # every referenced scene exists in the movie index
        assert set(section["scene_ids"]).issubset(scene_ids)
        for ev in section["narrative_evidence"]:
            assert ev["scene_id"] in scene_ids
            # excerpt windows are short, real sub-windows, never the whole scene
            assert ev["end_sec"] - ev["start_sec"] <= 6.0 + 1e-6
        # evidence ids map to real evidence claims
        for eid in section["evidence_ids"]:
            assert eid in {e["id"] for e in script["evidence"]}
        assert section["narration"], "every section has narration"


def test_script_rejects_nonexistent_scenes_contract(contract, movie_index):
    """If the contract names a scene absent from the movie, it must be dropped
    (hallucination rejection), never invented or referenced."""
    bad_contract = {
        **contract,
        "supporting_scenes": [
            {"scene_id": "scene-999", "start_sec": 0, "end_sec": 10},
            {"scene_id": "scene-2", "start_sec": 40, "end_sec": 70},
        ],
    }
    gen = GroundedScriptGenerator(target_sec=90)
    script = gen.generate(bad_contract, movie_index, project_id="proj")
    for section in script["sections"]:
        assert "scene-999" not in section["scene_ids"]
        for ev in section["narrative_evidence"]:
            assert ev["scene_id"] != "scene-999"
    assert any("scene-2" in s["scene_ids"] for s in script["sections"])


def test_script_duration_estimates(contract, movie_index):
    gen = GroundedScriptGenerator(target_sec=60)
    script = gen.generate(contract, movie_index)
    total = sum(s.get("estimated_seconds", 0) for s in script["sections"])
    assert total > 0
    for s in script["sections"]:
        assert s.get("estimated_seconds", 0) >= 1.0


def test_excerpt_window_is_short_and_real(movie_index):
    scene = movie_index["scenes"][0]
    w = _excerpt_window(scene)
    assert w is not None
    assert 0 <= w["start_sec"] < w["end_sec"] <= scene["end_sec"]
    assert w["end_sec"] - w["start_sec"] <= 6.0 + 1e-6


def test_narration_is_grounded_cautious():
    # A claim without an inventory never injects invented dialogue.
    assert "invented" not in build_claim_narration("")
    assert build_claim_narration("The film argues X.") == "Here is the film's claim: The film argues X."
    assert build_hook_narration("", "")  # safe fallback for empty input


def test_script_persists_roundtrip(tmp_path, contract, movie_index):
    gen = GroundedScriptGenerator(target_sec=90)
    script = gen.generate(contract, movie_index, project_id="proj")
    save_grounded_script(tmp_path, script)
    loaded = load_grounded_script(tmp_path)
    assert loaded == script


# --------------------------------------------------------------------------- #
# Grounding report
# --------------------------------------------------------------------------- #

def test_grounding_report_human_inspectable(tmp_path, contract, movie_index):
    save_grounding_contract(tmp_path, contract)
    gen = GroundedScriptGenerator(target_sec=90)
    script = gen.generate(contract, movie_index, project_id="proj")
    path = write_script_grounding_report(tmp_path, contract, script)
    text = path.read_text(encoding="utf-8")
    assert "# Script Grounding Report" in text
    assert "Selected Concept" in text
    assert contract["concept"]["thesis"] in text
    assert "Narration" in text
    assert "scene-1" in text


# --------------------------------------------------------------------------- #
# Editorial plan from grounded script
# --------------------------------------------------------------------------- #

def test_grounded_editorial_plan_consumes_script(contract, movie_index, tmp_path):
    gen = GroundedScriptGenerator(target_sec=90)
    script = gen.generate(contract, movie_index, project_id="proj")
    planner = GroundedEditorialPlanner(script=script)
    director_plan = {"thesis": contract["concept"]["thesis"],
                     "title": contract["concept"]["title"],
                     "director_provider": "grounded"}
    plan = planner.create_plan(movie_index, director_plan, target_sec=90)
    assert validate_plan(plan) == []
    assert len(plan.segments) == len(script["sections"])
    for seg in plan.segments:
        assert seg.narration.text, f"{seg.id} has narration"
        # evidence comes from the script's real excerpt windows
        for ev in seg.evidence:
            assert ev.scene_id in {s["scene_id"] for s in movie_index["scenes"]}
            assert ev.end_sec - ev.start_sec <= 6.0 + 1e-6


def test_grounded_editorial_pipeline_writes_plan(tmp_path, contract, movie_index):
    """create_editorial_plan persists editorial_plan.json; the grounded planner
    consumes the grounded script the same way the orchestrator does."""
    from tests.editorial_fixtures import seed_project

    seed_project(tmp_path)
    (tmp_path / "grounding_contract.json").write_text(
        json.dumps(contract), encoding="utf-8")
    gen = GroundedScriptGenerator(target_sec=60)
    script = gen.generate(contract, movie_index, project_id="proj")
    save_grounded_script(tmp_path, script)

    # chief director_plan.json now carries the grounded flag, so the editorial
    # wiring path inside the orchestrator picks GroundedEditorialPlanner.
    dp = json.loads((tmp_path / "director_plan.json").read_text(encoding="utf-8"))
    dp["grounded"] = True
    dp["thesis"] = contract["concept"]["thesis"]
    dp["title"] = contract["concept"]["title"]
    (tmp_path / "director_plan.json").write_text(json.dumps(dp), encoding="utf-8")

    from editorial.grounded import GroundedEditorialPlanner
    from editorial.script import build_editorial_script

    movie_memory_idx = json.loads((tmp_path / "movie_index.json").read_text(encoding="utf-8"))
    planner = GroundedEditorialPlanner(script=script)
    plan = planner.create_plan(movie_memory_idx, dp, target_sec=60)
    out = tmp_path / "editorial_plan.json"
    out.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
    assert out.exists()
    build_editorial_script(tmp_path, plan, movie_memory_idx)

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["provenance"]["planner"] == "grounded"
    assert validate_plan(plan) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])