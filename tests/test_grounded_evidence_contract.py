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


@pytest.fixture
def run_movie_index():
    """Faithful miniature of the T4 failing run's movie: real dialogue with
    "time", real objects a clock-thesis might borrow (mirror, face), a hedged
    vehicle location, and — for the Run-2 thesis-noun leak regression — the
    live "looking around" / "another person partially visible" identifiers a
    clock thesis could borrow via the shared words "around" / "visible"."""
    return {
        "movie": {"title": "Real Movie", "duration_sec": 180.0},
        "scenes": [
            _scene(
                "scene-1", 0.0, 30.0,
                location="indoor, convenience store",
                objects=["man in plaid shirt", "woman in denim jacket"],
                dialogue=[{"speaker": "A",
                           "text": "What time do you go to bed?"}],
            ),
            _scene(
                "scene-2", 30.0, 60.0,
                location="indoor, inside a vehicle (likely a bus or train)",
                objects=["window showing a snowy landscape",
                         "bus interior with hanging items",
                         "another person partially visible",
                         "looking around"],
            ),
            _scene(
                "scene-3", 60.0, 90.0,
                location="indoor, bathroom, personal space",
                objects=["mirror", "woman's face"],
            ),
        ],
    }


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

    def test_scene_refs_alone_do_not_pass_gate(self, analyzer):
        # A concept citing only real scene ids carries no verifiable claims.
        concept = {"thesis": "t", "evidence_refs": [
            {"kind": "scene", "scene_id": "scene-1"},
            {"kind": "scene", "scene_id": "scene-2"}]}
        ev = analyzer.concept_evidence(concept)
        assert ev["matched_scenes"] == ["scene-1", "scene-2"]
        assert ev["claim_refs"] == []
        assert analyzer.is_sufficient(concept, min_coverage=0.4) is False

    def test_claim_gate_rejects_invented_claims_even_with_matched_scene(
            self, analyzer):
        # scene-1 is real, but "son" and "flying saucer" do not exist in this
        # movie — this is exactly the FAIL-run pattern (grounded only by scene
        # ids while every claim is invented).
        concept = {"thesis": "t", "evidence_refs": [
            {"kind": "scene", "scene_id": "scene-1"},
            {"kind": "character", "value": "son"},
            {"kind": "object", "value": "flying saucer"}]}
        ev = analyzer.concept_evidence(concept)
        assert ev["matched_scenes"] == ["scene-1"]
        assert ev["claim_matched"] == 0
        assert ev["claim_ratio"] == 0.0
        assert analyzer.is_sufficient(concept, min_coverage=0.4) is False

    def test_claim_gate_admits_grounded_claims(self, analyzer):
        concept = {"thesis": "t", "evidence_refs": [
            {"kind": "scene", "scene_id": "scene-1"},
            {"kind": "object", "value": "revolver"},
            {"kind": "theme", "value": "confrontation"}]}
        ev = analyzer.concept_evidence(concept)
        assert ev["claim_matched"] == 2
        assert ev["claim_coverage"] == "HIGH"
        assert analyzer.is_sufficient(concept, min_coverage=0.4) is True

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
# 4b. Deterministic ref derivation from prose (anti-hallucination core)
# --------------------------------------------------------------------------- #

class TestDeriveRefsFromProse:
    """The model's declared refs are advisory; evidence_refs are synthesized by
    scanning the concept's prose for the movie's ACTUAL known vocabulary. This
    is what makes the generation pipeline impossible to fool with invented
    nouns (kitchen, notebook, father...) or compound invented actions."""

    def test_derives_grounded_object_and_scene_from_prose(self, analyzer):
        concept = {
            "title": "The Revolver",
            "hook": "Why does the revolver never get fired?",
            "thesis": "a claim about the revolver sitting on the bar",
            "why_interesting": "w",
            "visual_opportunity": "close-up on the glass",
            # declared refs are IGNORED — they are inventions
            "required_evidence": ["flying saucer", "telepathy beam"],
        }
        refs = analyzer.derive_refs(concept)
        kinds = {r["kind"] for r in refs}
        values = {r.get("value", "") for r in refs}
        assert "object" in kinds
        assert "revolver" in values
        assert "glass" in values
        assert "flying saucer" not in values
        assert "telepathy beam" not in values
        assert any(r["kind"] == "scene" for r in refs)

    def test_invented_family_drama_derives_no_refs(self, analyzer):
        """The exact real-run failure: prose about a fictional family drama
        (kitchen, notebook, watch, father, oven) must derive NOTHING."""
        concept = {
            "title": "The Kitchen as Site of Emotional Containment",
            "hook": "h",
            "thesis": "father adjusts the oven temperature while a child "
                      "writes in a notebook by the door",
            "why_interesting": "w",
            "visual_opportunity": "close-up of the oven dial",
            "required_evidence": ["oven", "notebook"],
        }
        assert analyzer.derive_refs(concept) == []

    def test_compound_action_does_not_masquerade_as_verbatim(self, analyzer):
        """A narrative sentence like 'mother looks at father without speaking'
        is NOT a vocabulary identifier and must not produce an action ref."""
        concept = {
            "title": "T", "hook": "h",
            "thesis": "the mother looks at the father without speaking",
            "why_interesting": "w", "visual_opportunity": "x",
        }
        refs = analyzer.derive_refs(concept)
        assert not any(r["kind"] == "action" for r in refs)

    def test_derived_refs_all_match(self, analyzer):
        """Every derived ref must be groundable — this is the invariant that
        makes the strict gate pass on REAL (not hallucinated) claims."""
        concept = {
            "title": "Saloon", "hook": "h",
            "thesis": "the saloon barman pours a drink while a rider is "
                      "waiting outside",
            "why_interesting": "w",
            "visual_opportunity": "saloon and the horse in the street",
        }
        refs = analyzer.derive_refs(concept)
        assert refs, "expected at least location/object/action refs"
        for r in refs:
            assert analyzer._match_ref(r), f"derived ref {r} is not grounded"
        # abstract-only concepts fail the concrete gate on derived refs
        ev = analyzer.concept_evidence({
            "thesis": concept["thesis"], "evidence_refs": refs})
        assert analyzer.is_sufficient_refs(ev, min_coverage=0.4) is True

    def test_scene_ref_synthesized_when_absent(self, analyzer):
        concept = {
            "title": "Revolver", "hook": "h",
            "thesis": "a claim centred on the revolver", "why_interesting": "w",
            "visual_opportunity": "x",
        }
        refs = analyzer.derive_refs(concept)
        scenes = [r for r in refs if r["kind"] == "scene"]
        assert len(scenes) == 1
        assert scenes[0]["scene_id"] == "scene-1"


class TestDeriveRefsResistsStopwordLeak:
    """Regression: the real-Qwen T4 run produced IDENTICAL generic refs for
    every concept because ``derive_refs`` matched on raw tokens — stopwords
    like "in"/"with" inside a vocabulary phrase ("man IN plaid shirt", "counter
    WITH various items") fired against ordinary prose containing the same
    function word. Six hallucinated theses (about a clock / revolver / drawing
    that exist in NO scene) all reported HIGH/HIGH coverage.

    Derivation must match only SIGNIFICANT (content) tokens: a concept whose
    prose mentions none of a phrase's real content words derives nothing from
    it, even when the prose contains "in" / "with" / "a"."""

    def _concept(self, thesis, visual):
        return {
            "title": "T", "hook": "the camera hovers over a quiet moment",
            "thesis": thesis, "why_interesting": "w",
            "visual_opportunity": visual,
        }

    def test_no_derived_ref_via_shared_function_word(self, analyzer):
        # The exact T4 pattern: a thesis about a "cracked clock" — an object
        # that exists in NO scene — must NOT harvest "man in plaid shirt" or
        # "counter with various items" just because the prose contains "in".
        concept = self._concept(
            thesis="the cracked clock in the shop reflects a failing sense of "
                   "time, as shown by repeated failed attempts to reset it",
            visual="slow zoom on the cracked face as the hands move forward",
        )
        refs = analyzer.derive_refs(concept)
        values = {r.get("value", "") for r in refs}
        # The fixture vocabulary has no clock; "man in plaid shirt" is NOT in
        # this fixture, so this is the generic-leak guard: shared function
        # words must not fabricate object/location refs for absent objects.
        assert not any("clock" in v for v in values)

    def test_stopwords_in_vocab_do_not_match_generic_prose(self, analyzer):
        # "counter with various items" exists in the fixture (scene-3). A
        # concept about a clock must not match it via the shared word "with"
        # or "items". It should only be derived when the prose names a content
        # token of the phrase itself.
        concept = self._concept(
            thesis="the cracked clock betrays a decaying sense of time",
            visual="extreme close-up of the hands turning",
        )
        refs = analyzer.derive_refs(concept)
        values = {r.get("value", "") for r in refs}
        assert "counter with various items" not in values
        # ...and the derived refs must still be honest/grounded.
        for r in refs:
            assert analyzer._match_ref(r), f"derived ref {r} is not grounded"

    def test_content_token_still_derives_empty_room_concept_rejected(self, analyzer):
        # An "empty room" thesis whose central object is ABSENT (no "empty"
        # anywhere) must not be admitted merely because prose contains common
        # words. There is no empty-room object in the fixture.
        concept = self._concept(
            thesis="a room labelled empty while a shadow remains visible",
            visual="framing of the room with no figures",
        )
        refs = analyzer.derive_refs(concept)
        assert not any("room" in str(r.get("value", "")) for r in refs)
        ev = analyzer.concept_evidence({"thesis": concept["thesis"],
                                        "evidence_refs": refs})
        assert analyzer.is_sufficient_refs(ev, min_coverage=0.4) is False


class TestClaimMustGround:
    """Regression: the real-Qwen T4 run produced theses about objects the movie
    does NOT contain (a clock, a train platform, a drawing), each grounding
    only via an incidental shared word. Two independent leaks were found and
    fixed:

    1. stopwords — fixed by significant-token matching (previous class);
    2. a single shared CONTENT token — a "sense of time" thesis matched the
       dialogue line "What time do you go to bed?"; a "train platform" thesis
       matched the hedged location label "indoor, inside a vehicle (likely a
       bus or train)".

    The gate must require the THESIS (title + thesis, the claim substance) to
    ground on its own — decorative hook/visual prose cannot rescue it.
    """

    @pytest.fixture
    def run_analyzer(self, run_movie_index):
        return EvidenceAnalyzer(SceneFacts.from_movie_intelligence(
            movie_index=run_movie_index))

    def test_time_thesis_does_not_ground_on_dialogue(self, run_analyzer):
        # The T4 "Broken Clock" thesis: no clock anywhere, but "time" appears
        # in a real dialogue line. A single shared content word must not bridge
        # thesis -> dialogue. Dialogue needs a STRONG overlap (>= half the
        # line, min 2 tokens), so "time" alone derives nothing.
        concept = {
            "title": "The Broken Clock",
            "hook": "why does time break things?",
            "thesis": "the cracked clock betrays a failing sense of time",
            "why_interesting": "w",
            "visual_opportunity": "extreme close-up of the hands turning",
        }
        refs = run_analyzer.derive_refs(concept)
        values = {r.get("value", "") for r in refs}
        assert not any("time" in str(v) for v in values)
        # The thesis (claim substance) alone cannot ground -> reject.
        assert run_analyzer.is_claim_sufficient(concept, min_coverage=0.4) is False

    def test_visual_prose_cannot_rescue_unclaimed_object(self, run_analyzer):
        # Even if the visual_opportunity mentions the mirror (a REAL object),
        # the THESIS is about a clock that does not exist. The claim-grounded
        # gate must reject it: decorative fields are not the claim.
        concept = {
            "title": "The Broken Clock",
            "hook": "h",
            "thesis": "the cracked clock betrays a failing sense of time",
            "why_interesting": "w",
            "visual_opportunity": "the mirror catches the woman's face",
        }
        full_refs = run_analyzer.derive_refs(concept)
        full_values = {r.get("value", "") for r in full_refs}
        # The full-prose derivation legitimately finds the real mirror/face...
        assert "mirror" in full_values
        # ...but the milestone gate requires the CLAIM to ground, and it does not.
        assert run_analyzer.is_claim_sufficient(concept, min_coverage=0.4) is False

    def test_train_platform_thesis_ignores_hedged_location(self, run_analyzer):
        # "indoor, inside a vehicle (likely a bus or train)" contains "train",
        # but it is a hedged GUESS, not confirmed content. A thesis about a
        # train platform must not ground on it.
        concept = {
            "title": "The Train Platform",
            "hook": "escape",
            "thesis": "the train platform becomes a symbol of escape",
            "why_interesting": "w",
            "visual_opportunity": "wide shot of the platform",
        }
        refs = run_analyzer.derive_refs(concept)
        values = {r.get("value", "") for r in refs}
        assert not any("train" in str(v) for v in values)
        assert run_analyzer.is_claim_sufficient(concept, min_coverage=0.4) is False

    def test_hedged_or_alternative_location_never_grounds_claim(
            self, run_analyzer):
        # scene-2's location "indoor, inside a vehicle (likely a bus or train)"
        # is a HEDGED label (uncertain guess + "or" alternative). A thesis
        # naming ONLY a hedged location's word must derive no LOCATION ref from
        # it, and — when the word is absent from all real content (no "train" /
        # "trolley" object exists) — the claim cannot be grounded at all.
        for word in ("train", "trolley"):
            concept = {
                "title": "Vehicle",
                "hook": "h",
                "thesis": f"the {word} becomes the film's image of movement",
                "why_interesting": "w",
                "visual_opportunity": f"wide shot of the {word}",
            }
            refs = run_analyzer.derive_refs(concept, fields=("title", "thesis"))
            locs = [r for r in refs if r["kind"] == "location"]
            assert locs == [], f"{word} must not ground on a hedged location"
            assert run_analyzer.is_claim_sufficient(
                concept, min_coverage=0.4) is False

    def test_confident_location_still_grounds_claim(self, run_analyzer):
        # The tightening must not reject fully confirmed locations: scene-1's
        # plain "indoor, convenience store" is a confident label.
        concept = {
            "title": "The Convenience Store",
            "hook": "h",
            "thesis": "the convenience store becomes the film's image of commerce",
            "why_interesting": "w",
            "visual_opportunity": "the store shelves with various items",
        }
        refs = run_analyzer.derive_refs(concept, fields=("title", "thesis"))
        locs = [r for r in refs if r["kind"] == "location"]
        assert any("convenience store" in str(r.get("value", ""))
                   for r in locs)
        assert run_analyzer.is_claim_sufficient(
            concept, min_coverage=0.4) is True

    def test_grounded_thesis_still_passes_claim_gate(self, run_analyzer):
        # The tightening must not reject genuinely grounded concepts: a thesis
        # whose claim floats directly on real on-screen objects.
        concept = {
            "title": "The Mirror",
            "hook": "h",
            "thesis": "the mirror and the woman's face frame a private ritual",
            "why_interesting": "w",
            "visual_opportunity": "the woman's face in the mirror",
        }
        refs = run_analyzer.derive_refs(concept)
        values = {r.get("value", "") for r in refs}
        assert "mirror" in values
        assert "woman's face" in values
        assert run_analyzer.is_claim_sufficient(concept, min_coverage=0.4) is True


class TestThesisNounLeakClosed:
    """Regression for the real-Qwen T4 Run 2 leak: the selected "Clock That
    Never Ticks" thesis passed the claim gate because ``derive_refs`` matched
    multi-token vocab items on ANY single shared token. A thesis whose prose
    said "clock face ... is visible ... constructed around" harvested LIVE
    refs ("woman's face", "another person partially visible", "looking
    around") purely via the shared generic words "face" / "visible" / "around".

    Multi-token vocabulary items now require at least TWO of their own content
    tokens to appear in the prose, so a real peripheral noun can no longer be
    borrowed to admit a thesis whose star object (a clock) exists nowhere.
    """

    @pytest.fixture
    def leak_analyzer(self, run_movie_index):
        return EvidenceAnalyzer(SceneFacts.from_movie_intelligence(
            movie_index=run_movie_index))

    def test_clock_prose_does_not_borrow_live_refs(self, leak_analyzer):
        # The exact Run-2 shape: a thesis about an absent clock whose prose
        # happens to contain "around" (claim-level bridge) plus "face" and
        # "visible" (full-prose bridges).
        concept = {
            "title": "The Clock That Never Ticks",
            "hook": "h",
            "thesis": (
                "the film's narrative is constructed around a single "
                "unchanging time frame, showing a failing sense of time in "
                "scene-1 and scene-3"
            ),
            "why_interesting": "w",
            "visual_opportunity": (
                "the clock face is visible at the fixed 12:00 mark, synced "
                "across cuts"
            ),
        }
        refs = leak_analyzer.derive_refs(concept)
        values = {r.get("value", "") for r in refs}
        # These live refs used to be harvested from the four shared words.
        assert "woman's face" not in values
        assert "another person partially visible" not in values
        assert "looking around" not in values
        # The claim substance (title + thesis) itself grounds nothing here.
        assert leak_analyzer.is_claim_sufficient(
            concept, min_coverage=0.4) is False

    def test_multitoken_vocab_still_derives_on_two_own_tokens(
            self, leak_analyzer):
        # The tightening must not over-reject: a concept that REALLY names two
        # of a vocab item's own content tokens still derives it, and a claim
        # built on that real content is admissible.
        concept = {
            "title": "Visible Person", "hook": "h", "why_interesting": "w",
            "thesis": "a person barely visible in the frame",
            "visual_opportunity": "x",
        }
        refs = leak_analyzer.derive_refs(concept)
        values = {r.get("value", "") for r in refs}
        assert "another person partially visible" in values
        assert leak_analyzer.is_claim_sufficient(
            concept, min_coverage=0.4) is True


class TestHedgedLocationStrip:
    def test_strip_hedged_location_clause(self):
        assert EvidenceAnalyzer._strip_hedged_location_clause(
            "indoor, inside a vehicle (likely a bus or train)") == (
            "indoor, inside a vehicle")
        assert EvidenceAnalyzer._strip_hedged_location_clause(
            "indoor, small room, possibly a diner or a bar") == ("indoor, small room")
        assert EvidenceAnalyzer._strip_hedged_location_clause(
            "outdoor, riverbank, natural setting") == (
            "outdoor, riverbank, natural setting")

    def test_is_location_confident(self):
        # Soft hedge words disqualify the label from claim grounding.
        assert EvidenceAnalyzer._is_location_confident(
            "outdoor, riverbank, natural setting") is True
        assert EvidenceAnalyzer._is_location_confident(
            "indoor, convenience store") is True
        assert EvidenceAnalyzer._is_location_confident(
            "indoor, inside a vehicle (likely a bus or train)") is False
        assert EvidenceAnalyzer._is_location_confident(
            "indoor, small room, possibly a diner or a bar") is False
        # Hard alternatives disqualify too, even without a soft hedge word.
        assert EvidenceAnalyzer._is_location_confident(
            "indoor, small shop or garage, setting appears to be a workshop "
            "or storage area") is False


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

    def test_plan_concept_is_deterministic_from_selected(self, facts):
        """The plan's concept block is the selected concept — the plan LLM must
        never re-imagine a different movie (the FAIL run invented a hospital
        film for a cowboy-movie concept)."""
        class _LLM:
            def __init__(self):
                self.calls = []

            def __call__(self, prompt):
                self.calls.append(prompt)
                if "finalizing the plan" in prompt:
                    return json.dumps({
                        "concept": {
                            "title": "A Different Movie",
                            "hook": "invented",
                            "thesis": "an invented thesis about a hospital clock",
                        },
                        "format": {"type": "short_video_essay",
                                   "duration_sec": 90},
                        "editorial_direction": {
                            "pacing": "slow",
                            "visual_style": "close-up on the revolver while "
                                            "the barman talks",
                            "audio_style": "minimal",
                            "editing_style": "quiet cuts",
                        },
                    })
                return json.dumps({"concepts": [{
                    "title": "Real One", "hook": "h",
                    "thesis": "a specific grounded claim about the saloon",
                    "why_interesting": "w",
                    "evidence_refs": [
                        {"kind": "scene", "scene_id": "scene-1"},
                        {"kind": "object", "value": "revolver"},
                    ],
                    "visual_opportunity": "close-up", "format": "f",
                }]})

        llm = _LLM()
        director = MovieGroundedDirector(llm)
        res = director.develop({"title": "T", "duration_sec": 90}, facts,
                               num_concepts=1, min_coverage=0.4)
        assert res["selected_concept"] is not None
        plan = res["plan"]
        # Even though the mock plan LLM returned a different movie, the plan's
        # concept must be the selected concept, verbatim.
        assert plan["concept"]["title"] == "Real One"
        assert plan["concept"]["thesis"] == "a specific grounded claim about the saloon"
        # The plan prompt must forbid re-imagining.
        assert "copy of the selected concept" in llm.calls[-1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))