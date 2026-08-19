"""Tests for the structured evidence contract (``evidence_refs``).

Covers the grounded-director milestone fixes:

- exact-scene-id evidence refs (with tolerant id normalization),
- canonical object/location/character identifiers + conservative aliases,
- exact token matching — NEVER arbitrary substring (``son`` != ``person``),
- the extended ``concept_evidence`` result (requested / matched / missing refs),
- coverage labeling (HIGH >= 0.7 / MED >= 0.4 / LOW),
- bounded regeneration: one initial batch, at most one retry; otherwise FAIL
  (no selected concept, no plan).
"""
import json

import pytest

from director.scene_facts import SceneFacts
from director.evidence import EvidenceAnalyzer
from director.grounded import MovieGroundedDirector
from director.concepts import parse_concepts, concept_refs, render_ref


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _scene(scene_id, start=0.0, end=10.0, **facts):
    scene = {
        "scene_id": scene_id,
        "start_sec": start,
        "end_sec": end,
        "transcript": facts.pop("transcript", ""),
    }
    scene["story"] = {
        "characters": facts.pop("characters", []),
        "location": facts.pop("location", None),
        "actions": facts.pop("actions", []),
        "objects": facts.pop("objects", []),
        "visual_description": facts.pop("visual_description", None),
        "visual_events": facts.pop("visual_events", []),
        "emotional_cues": facts.pop("emotional_cues", []),
        "themes": facts.pop("themes", []),
        "mood": facts.pop("mood", None),
        "cinematography": facts.pop("cinematography", None),
        "dialogue": facts.pop("dialogue", []),
    }
    return scene


@pytest.fixture
def movie_index():
    return {
        "movie": {"title": "Test Western", "duration_sec": 180.0},
        "scenes": [
            _scene(
                "scene-1", 0.0, 30.0,
                characters=["Barman"],
                location="saloon, dim light",
                actions=["pouring", "talking"],
                objects=["revolver", "glass"],
                visual_description="close-up of the barman's hands",
                visual_events=["a revolver is placed on the bar"],
                themes=["tension", "confrontation"],
                mood="tense",
                dialogue=[{"speaker": "Barman",
                           "text": "Keep your hands where I can see them."}],
            ),
            _scene(
                "scene-2", 30.0, 60.0,
                characters=["Stranger"],
                location="outdoor, street at dusk",
                actions=["riding"],
                objects=["horse", "dust"],
                visual_description="wide shot of a lone rider",
                themes=["solitude"],
                mood="somber",
            ),
            _scene(
                "scene-3", 60.0, 90.0,
                location="riverbank",
                actions=["walking"],
                objects=["water", "counter with various items"],
                visual_description="a person walks through shallow water",
                themes=["nature"],
                mood="serene",
            ),
        ],
    }


@pytest.fixture
def facts(movie_index):
    return SceneFacts.from_movie_intelligence(movie_index=movie_index)


@pytest.fixture
def analyzer(facts):
    return EvidenceAnalyzer(facts)


# --------------------------------------------------------------------------- #
# 1. Exact scene-id grounding (exact-ID first)
# --------------------------------------------------------------------------- #

class TestSceneRefs:
    def test_scene_ref_matches_exact_scene_id(self, analyzer):
        concept = {"thesis": "t", "evidence_refs": [
            {"kind": "scene", "scene_id": "scene-2"}]}
        ev = analyzer.concept_evidence(concept)
        assert ev["matched_scenes"] == ["scene-2"]
        assert ev["missing_refs"] == []

    def test_scene_ref_tolerates_id_separator_variants(self, analyzer):
        for variant in ("scene2", "scene 2", "SCENE-2"):
            ev = analyzer.concept_evidence(
                {"thesis": "t", "evidence_refs": [
                    {"kind": "scene", "scene_id": variant}]})
            assert ev["matched_scenes"] == ["scene-2"], variant

    def test_scene_ref_rejects_nonexistent_scene(self, analyzer):
        concept = {"thesis": "t", "evidence_refs": [
            {"kind": "scene", "scene_id": "scene-99"}]}
        ev = analyzer.concept_evidence(concept)
        assert ev["matched_scenes"] == []
        assert len(ev["missing_refs"]) == 1
        assert analyzer.is_sufficient(concept, min_coverage=0.4) is False


# --------------------------------------------------------------------------- #
# 2. Canonical object / location / character identifiers (+ conservative aliases)
# --------------------------------------------------------------------------- #

class TestCanonicalVocabulary:
    def test_object_ref_matches_with_significant_token_alias(self, analyzer):
        # "counter" is a content token of the canonical "counter with various items".
        concept = {"thesis": "t", "evidence_refs": [
            {"kind": "object", "value": "counter"}]}
        ev = analyzer.concept_evidence(concept)
        assert ev["matched_scenes"] == ["scene-3"]
        assert ev["missing_refs"] == []

    def test_object_ref_exact_canonical(self, analyzer):
        concept = {"thesis": "t", "evidence_refs": [
            {"kind": "object", "value": "revolver"}]}
        ev = analyzer.concept_evidence(concept)
        assert ev["matched_scenes"] == ["scene-1"]

    def test_object_ref_rejects_absent_object(self, analyzer):
        concept = {"thesis": "t", "evidence_refs": [
            {"kind": "object", "value": "telepathy beam"}]}
        ev = analyzer.concept_evidence(concept)
        assert ev["missing_refs"] and not ev["matched_scenes"]
        assert analyzer.is_sufficient(concept, min_coverage=0.4) is False

    def test_character_ref_matches_and_rejects(self, analyzer):
        ok = analyzer.concept_evidence({"thesis": "t", "evidence_refs": [
            {"kind": "character", "value": "Barman"}]})
        assert ok["matched_scenes"] == ["scene-1"]
        bad = analyzer.concept_evidence({"thesis": "t", "evidence_refs": [
            {"kind": "character", "value": "Kay Corleone"}]})
        assert bad["missing_refs"] and not bad["matched_scenes"]

    def test_location_ref_matches_via_alias_token(self, analyzer):
        # "saloon" is a content token of location "saloon, dim light".
        concept = {"thesis": "t", "evidence_refs": [
            {"kind": "location", "value": "saloon"}]}
        ev = analyzer.concept_evidence(concept)
        assert ev["matched_scenes"] == ["scene-1"]


# --------------------------------------------------------------------------- #
# 3. No arbitrary substring grounding
# --------------------------------------------------------------------------- #

class TestNoSubstringMatching:
    def test_son_does_not_match_person(self, facts, analyzer):
        # scene-3 facts contain the literal word "person".
        assert facts.is_grounded("person") is True
        assert facts.is_grounded("son") is False

        concept = {"thesis": "t", "evidence_refs": [
            {"kind": "object", "value": "son"}]}
        ev = analyzer.concept_evidence(concept)
        assert ev["missing_refs"] and not ev["matched_scenes"]

    def test_known_entity_guards_are_token_based(self, facts):
        assert facts.is_known_object("glass") is True
        assert facts.is_known_object("glassrooster") is False
        assert facts.is_known_character("Barman") is True
        assert facts.is_known_character("Vito") is False


# --------------------------------------------------------------------------- #
# 4. Rejection gate / contract completeness
# --------------------------------------------------------------------------- #

class TestEvidenceContract:
    def test_concept_without_evidence_refs_fails_gate(self, analyzer):
        concept = {"thesis": "a specific claim", "visual_opportunity": "x"}
        assert analyzer.concept_evidence(concept)["requested_refs"] == []
        assert analyzer.is_sufficient(concept, min_coverage=0.4) is False

    def test_evidence_refs_derived_from_legacy_and_vice_versa(self):
        legacy = parse_concepts(json.dumps({"concepts": [{
            "title": "T", "hook": "H", "thesis": "a specific thesis",
            "why_interesting": "W", "required_evidence": ["revolver"],
            "visual_opportunity": "V", "format": "f",
        }]}))
        c = legacy[0]
        assert c["evidence_refs"] == [{"kind": "text", "value": "revolver"}]
        assert c["required_evidence"] == ["revolver"]

        structured = parse_concepts(json.dumps({"concepts": [{
            "title": "T", "hook": "H", "thesis": "a specific thesis",
            "why_interesting": "W",
            "evidence_refs": [
                {"kind": "scene", "scene_id": "scene-1"},
                {"kind": "object", "value": "revolver"},
            ],
            "visual_opportunity": "V", "format": "f",
        }]}))
        c2 = structured[0]
        assert c2["evidence_refs"][0] == {"kind": "scene", "scene_id": "scene-1"}
        assert c2["required_evidence"] == ["scene-1", "revolver"]
        # render round-trips the refs.
        assert [render_ref(r) for r in concept_refs(c2)] == c2["required_evidence"]

    def test_concept_evidence_structured_fields(self, analyzer):
        refs = [
            {"kind": "scene", "scene_id": "scene-1"},
            {"kind": "object", "value": "revolver"},
            {"kind": "object", "value": "telepathy beam"},
        ]
        ev = analyzer.concept_evidence({"thesis": "t", "evidence_refs": refs})
        assert ev["requested_refs"] == refs
        assert len(ev["matched_refs"]) == 2
        assert len(ev["missing_refs"]) == 1
        assert ev["matched_scenes"] == ["scene-1"]
        assert ev["supporting_scene_ids"] == ev["matched_scenes"]
        for m in ev["matched_refs"]:
            assert m["matched_scenes"]

    def test_coverage_labeling(self, analyzer):
        high = analyzer.concept_evidence({"thesis": "t", "evidence_refs": [
            {"kind": "object", "value": "revolver"},
            {"kind": "theme", "value": "confrontation"}]})
        assert high["coverage"] == "HIGH"
        mid = analyzer.concept_evidence({"thesis": "t", "evidence_refs": [
            {"kind": "object", "value": "revolver"},
            {"kind": "object", "value": "telepathy beam"}]})
        assert mid["coverage"] == "MED"
        low = analyzer.concept_evidence({"thesis": "t", "evidence_refs": [
            {"kind": "object", "value": "flying saucer"},
            {"kind": "object", "value": "telepathy beam"}]})
        assert low["coverage"] == "LOW"


# --------------------------------------------------------------------------- #
# 5. Bounded regeneration: initial batch + single retry, then FAIL
# --------------------------------------------------------------------------- #

class _StubbornLLM:
    """Always returns ungrounded concepts (initial and retry)."""

    def __init__(self, concepts):
        self._concepts = concepts

    def __call__(self, prompt):
        return json.dumps({"concepts": self._concepts})


class TestBoundedRegeneration:
    def test_all_ungrounded_after_one_retry_fails_safely(self, facts):
        bad = {
            "title": "Bad", "hook": "h",
            "thesis": "a specific claim about aliens",
            "why_interesting": "w", "required_evidence": ["flying saucer"],
            "visual_opportunity": "x", "format": "f",
        }
        director = MovieGroundedDirector(_StubbornLLM([bad]))
        res = director.develop({"title": "T", "duration_sec": 90}, facts,
                               num_concepts=1, min_coverage=0.4)
        assert res["selected_concept"] is None
        assert res["plan"] is None
        stats = res["llm_stats"]
        assert stats["regeneration_rounds"] == 1
        assert stats["substitutes_generated"] == 1
        assert stats["llm_calls"] == 2  # brainstorm + single retry (no plan)

    def test_evidence_strategy_uses_only_matched_scenes(self, facts):
        concept = {"thesis": "t", "evidence_refs": [
            {"kind": "scene", "scene_id": "scene-1"},
            {"kind": "scene", "scene_id": "scene-3"},
            {"kind": "object", "value": "telepathy beam"},
        ]}
        strat = EvidenceAnalyzer(facts).build_evidence_strategy(concept)
        assert set(strat["scene_ids"]) == {"scene-1", "scene-3"}
        assert "scene-2" not in strat["scene_ids"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))