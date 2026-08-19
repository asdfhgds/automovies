"""Tests for the Movie-grounded Creative Director and its evidence pipeline.

The LLM is mocked in every test (per the milestone: mock the LLM, gate real
Qwen). Uses small, synthetic SceneFacts so grounding is deterministic.
"""
import json
from pathlib import Path

import pytest

from director.scene_facts import SceneFacts, SceneFact
from director.context_builder import DirectorContextBuilder, scene_summary
from director.evidence import EvidenceAnalyzer
from director.grounded import MovieGroundedDirector
from director.concepts import (
    build_generation_prompt,
    build_plan_prompt,
    build_rejection_prompt,
    parse_concepts,
    parse_plan,
    compute_diversity_metric,
    is_generic_thesis,
)
from director.report import build_report


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _scene(scene_id, start, end, **facts):
    """Helper to construct a movie_index-style scene dict."""
    scene = {
        "scene_id": scene_id,
        "start_sec": start,
        "end_sec": end,
        "transcript": facts.pop("transcript", ""),
    }
    story = {
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
    scene["story"] = story
    return scene


@pytest.fixture
def movie_index():
    return {
        "project_id": "proj-test",
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
                dialogue=[{"speaker": "Barman", "text": "Keep your hands where I can see them."}],
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
                objects=["water"],
                visual_description="a man walks through shallow water",
                themes=["nature"],
                mood="serene",
            ),
        ],
    }


@pytest.fixture
def facts(movie_index):
    return SceneFacts.from_movie_intelligence(movie_index=movie_index)


@pytest.fixture
def metadata(movie_index):
    return dict(movie_index["movie"])


class MockLLM:
    """Deterministic mock LLM for grounded-director tests.

    Expressive enough to (a) return N concepts on the brainstorm call,
    (b) return a plan when asked for one, and (c) optionally reject concepts
    on regeneration so the rejection path can be exercised.
    """

    def __init__(self, concepts=None, plan=None, regenerate_to=None):
        self._concepts = concepts or []
        self._plan = plan or {}
        self._regenerate_to = regenerate_to
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        if "re-running" in prompt:
            return json.dumps({"concepts": self._regenerate_to or []})
        if "finalizing the plan" in prompt:
            return json.dumps(self._plan)
        return json.dumps({"concepts": self._concepts})


# --------------------------------------------------------------------------- #
# Context builder
# --------------------------------------------------------------------------- #

class TestContextBuilder:
    def test_builds_compact_summary(self, facts):
        text = scene_summary(facts.scenes[0])
        assert "scene-1" in text
        assert "Barman" in text
        assert "revolver" in text
        assert "Keep your hands" in text  # dialogue included

    def test_context_grounding_rules_present(self, facts, metadata):
        cb = DirectorContextBuilder()
        context, meta = cb.build_concept_generation_context(metadata, facts)
        assert "ACTUAL SCENES" in context
        assert "scene-1" in context
        assert "GROUNDING RULES" in context
        assert meta["total_scenes"] == 3
        assert meta["scenes_included"] == 3

    def test_context_truncation_respects_token_budget(self, facts, metadata):
        many = SceneFacts(
            [facts.scenes[i % len(facts.scenes)] for i in range(60)]
        )
        cb = DirectorContextBuilder(max_tokens=600, reserve_for_output=400)
        context, meta = cb.build_concept_generation_context(metadata, many)
        # Budget = 200 tokens => must truncate, not include all 60 scenes.
        assert meta["truncated"] is True
        assert meta["scenes_included"] < 60
        assert meta["total_scenes"] == 60

    def test_unknown_character_flagged_not_invented(self, facts):
        empty = SceneFacts.from_movie_intelligence(
            scenes=[_scene("scene-1", 0, 5, location="empty")])
        text = scene_summary(empty.scenes[0])
        assert "unknown_character_01" in text

    def test_grounded_example_present_and_passable(self, facts, metadata):
        cb = DirectorContextBuilder()
        context, meta = cb.build_concept_generation_context(metadata, facts)
        assert meta.get("example_included") is True
        assert "## WORKED EXAMPLE" in context
        # Extract the embedded example JSON and verify it sorts HIGH by the
        # deterministic matcher (it must be a demonstrably-correct template).
        seg = context[context.find("## WORKED EXAMPLE"):]
        obj = seg[seg.find("{"):]
        obj = obj[:obj.rfind("}") + 1]
        example = json.loads(obj)
        ev = EvidenceAnalyzer(facts).concept_evidence(example)
        assert ev["coverage"] == "HIGH"
        assert ev["claim_coverage"] == "HIGH"

    def test_grounded_example_skipped_when_no_citable_facts(self):
        empty = SceneFacts.from_movie_intelligence(
            scenes=[_scene("scene-1", 0, 5, location="empty")])
        cb = DirectorContextBuilder()
        context, meta = cb.build_concept_generation_context(
            {"title": "Empty"}, empty)
        assert meta.get("example_included", False) is False
        assert "## WORKED EXAMPLE" not in context

    def test_plan_context_has_verbatim_vocab_and_grounded_editorial(self,
                                                                    facts, metadata):
        from director.concepts import concept_refs
        concept = {
            "title": "C1", "hook": "h", "thesis": "a grounded claim",
            "why_interesting": "w",
            "evidence_refs": [
                {"kind": "scene", "scene_id": "scene-1"},
                {"kind": "object", "value": "revolver"},
            ],
            "required_evidence": ["revolver", "scene-1"],
            "visual_opportunity": "close-up", "format": "short_video_essay",
        }
        cb = DirectorContextBuilder()
        ctx = cb.build_plan_context(concept, facts, ["scene-1"])
        assert "## VERBATIM VOCABULARY FOR THE EVIDENCE SCENES ONLY" in ctx
        assert "## WORKED EDITORIAL EXAMPLE" in ctx
        assert "revolver" in ctx
        assert "mirror" not in ctx  # scene-2 facts must NOT leak into scene-1
        # The worked editorial example must not cite out-of-scope props.
        audit = EvidenceAnalyzer(facts).plan_grounding(
            {"visual_style": ctx}, ["scene-1"])
        assert "mirror" not in audit["invented_terms"] + audit["elsewhere_terms"]


# --------------------------------------------------------------------------- #
# Scene facts / hallucination prevention
# --------------------------------------------------------------------------- #

class TestSceneFacts:
    def test_known_vocabulary(self, facts):
        assert "Barman" in facts.known_characters()
        assert "Stranger" in facts.known_characters()
        assert any("saloon" in l for l in facts.known_locations())
        assert "revolver" in facts.known_objects()

    def test_hallucination_guard_rejects_invented_facts(self, facts):
        assert facts.is_grounded("revolver") is True
        assert facts.is_grounded("flying saucer") is False
        assert facts.is_known_character("Barman") is True
        assert facts.is_known_character("Sherlock Holmes") is False

    def test_loads_from_bare_scene_list(self):
        sf = SceneFacts.from_movie_intelligence(
            scenes=[_scene("s1", 0, 5, objects=["rock"])]
        )
        assert sf.known_objects() == ["rock"]


# --------------------------------------------------------------------------- #
# Evidence availability + rejection
# --------------------------------------------------------------------------- #

class TestEvidence:
    def test_concept_with_real_evidence_high(self, facts):
        analyzer = EvidenceAnalyzer(facts)
        concept = {
            "thesis": "a specific grounded thesis about tension",
            "required_evidence": ["revolver", "confrontation"],
            "visual_opportunity": "close-up of the revolver on the bar",
        }
        ev = analyzer.concept_evidence(concept)
        assert ev["coverage"] == "HIGH"
        assert ev["matched_claims"] >= len(ev["unmatched_claims"])

    def test_concept_with_invented_evidence_rejected(self, facts):
        analyzer = EvidenceAnalyzer(facts)
        concept = {
            "thesis": "a specific claim about aliens",
            "required_evidence": ["flying saucer", "mind control ray"],
            "visual_opportunity": "none",
        }
        assert analyzer.is_sufficient(concept, min_coverage=0.4) is False

    def test_evidence_strategy_maps_to_real_scenes(self, facts):
        analyzer = EvidenceAnalyzer(facts)
        concept = {
            "thesis": "tension via the revolver confrontation",
            "required_evidence": ["revolver"],
            "visual_opportunity": "close-up",
        }
        strat = analyzer.build_evidence_strategy(concept)
        assert "scene-1" in strat["scene_ids"]
        assert "revolver" in " ".join(strat["visual_motifs"])


class TestPlanGroundingAudit:
    """Deterministic audit of plan editorial_direction prose."""

    def test_grounded_plan_is_sufficient(self, facts):
        analyzer = EvidenceAnalyzer(facts)
        ed = {
            "pacing": "slow and measured",
            "visual_style": "close-up on the revolver and the glass while the "
                            "barman is talking",
            "audio_style": "minimal",
            "editing_style": "long takes and quiet cuts",
        }
        audit = analyzer.plan_grounding(ed, ["scene-1"])
        assert audit["sufficient"] is True
        assert audit["coverage"] >= audit["min_coverage"]
        assert "revolver" in " ".join(audit["grounded_terms"])

    def test_invented_plan_terms_flagged(self, facts):
        analyzer = EvidenceAnalyzer(facts)
        ed = {
            "visual_style": "empty chairs, an open window and silhouettes in "
                            "the saloon",
            "editing_style": "quiet cuts",
        }
        audit = analyzer.plan_grounding(ed, ["scene-1"])
        assert audit["sufficient"] is False
        assert "chairs" in audit["invented_terms"]
        assert "window" in audit["invented_terms"]
        assert "silhouettes" in audit["invented_terms"]

    def test_out_of_scope_terms_flagged_separately(self, facts):
        analyzer = EvidenceAnalyzer(facts)
        # "horse" exists in the movie but NOT in evidence scene-1.
        ed = {"visual_style": "a lone horse in the saloon light"}
        audit = analyzer.plan_grounding(ed, ["scene-1"])
        assert "horse" in audit["elsewhere_terms"]
        assert "horse" not in audit["invented_terms"]
        assert "saloon" in audit["grounded_terms"]

    def test_editorial_craft_terms_not_flagged(self, facts):
        analyzer = EvidenceAnalyzer(facts)
        ed = {"visual_style": "slow zooms, crossfades and minimal ambient sound"}
        audit = analyzer.plan_grounding(ed, ["scene-1"])
        assert audit["invented_terms"] == []


# --------------------------------------------------------------------------- #
# Concept generation / parsing / diversity
# --------------------------------------------------------------------------- #

class TestConcepts:
    def test_generation_prompt_requests_diversity(self):
        prompt = build_generation_prompt("CTX", num_concepts=5)
        assert "philosophy" in prompt
        assert "narrative_structure" in prompt
        assert "evidence_refs" in prompt
        assert "structured" in prompt or "kind" in prompt
        assert "5" in prompt

    def test_parse_concepts(self):
        raw = json.dumps({"concepts": [{
            "title": "T", "hook": "H", "thesis": "a specific thesis",
            "why_interesting": "W", "required_evidence": ["revolver"],
            "visual_opportunity": "V", "format": "short_video_essay",
            "diversity_angle": "irony",
        }]})
        concepts = parse_concepts(raw)
        assert len(concepts) == 1
        assert concepts[0]["required_evidence"] == ["revolver"]

    def test_parse_plan(self):
        raw = json.dumps({
            "concept": {"title": "T", "hook": "H", "thesis": "s"},
            "format": {"type": "short_video_essay", "duration_sec": 90},
            "editorial_direction": {
                "pacing": "slow", "visual_style": "wide",
                "audio_style": "minimal", "editing_style": "cuts",
            },
        })
        plan = parse_plan(raw)
        assert plan["editorial_direction"]["pacing"] == "slow"

    def test_diversity_metric_separates_batches(self):
        a = [{"thesis": "quiet violence as character", "diversity_angle": "irony"},
             {"thesis": "the river as landscape mirror", "diversity_angle": "symbolism"},
             {"thesis": "framing solitude in wide shots", "diversity_angle": "cinematography"}]
        same = [{"thesis": "x" * 5, "diversity_angle": "irony"} for _ in range(3)]
        assert compute_diversity_metric(a) > compute_diversity_metric(same)

    def test_generic_thesis_detected(self):
        assert is_generic_thesis("This movie explores violence.") is True
        assert is_generic_thesis("specific evidence-based claim") is False

    def test_rejection_prompt_shows_failed_refs(self):
        rejected = [{
            "title": "Bad", "thesis": "claim about a flying saucer",
            "evidence_refs": [{"kind": "object", "value": "flying saucer"}],
        }]
        prompt = build_rejection_prompt(
            "CTX", rejected, substitutes_needed=1,
            ref_failures=[["flying saucer"]],
        )
        assert "flying saucer" in prompt
        assert "NOT FOUND in the movie" in prompt

    def test_rejection_prompt_without_failures_still_lists_concepts(self):
        rejected = [{"title": "Bad", "thesis": "s",
                     "evidence_refs": [{"kind": "object", "value": "x"}]}]
        prompt = build_rejection_prompt("CTX", rejected, substitutes_needed=1)
        assert "re-running" in prompt
        assert "Bad" in prompt

    def test_plan_prompt_warns_against_invented_terms(self):
        prompt = build_plan_prompt("CTX", grounding_warnings=["chairs", "city"])
        assert "GROUNDING CORRECTION" in prompt
        assert "chairs" in prompt
        assert "city" in prompt
        prompt2 = build_plan_prompt("CTX")
        assert "GROUNDING CORRECTION" not in prompt2


# --------------------------------------------------------------------------- #
# Creative memory
# --------------------------------------------------------------------------- #

class TestMemoryIntegration:
    def test_selected_concept_stored(self, facts, metadata, tmp_path):
        mock = MockLLM(concepts=[{
            "title": "Grounded Idea",
            "hook": "hook", "thesis": "a specific grounded claim",
            "why_interesting": "w", "required_evidence": ["revolver"],
            "visual_opportunity": "close-up", "format": "short_video_essay",
        }])
        director = MovieGroundedDirector(mock, memory_dir=tmp_path / "mem")
        res = director.develop(metadata, facts, num_concepts=1, min_coverage=0.4)
        concepts = director.memory.get_all_concepts()
        assert len(concepts) == 1
        assert concepts[0]["title"] == "Grounded Idea"


# --------------------------------------------------------------------------- #
# Director-plan schema
# --------------------------------------------------------------------------- #

class TestPlanSchema:
    def _run(self, facts, metadata, mock_llm, tmp_path):
        director = MovieGroundedDirector(mock_llm, memory_dir=tmp_path / "mem")
        return director.develop(metadata, facts, num_concepts=2, min_coverage=0.4)

    def test_plan_has_required_keys(self, facts, metadata, tmp_path):
        mock = MockLLM(
            concepts=[{
                "title": "C1", "hook": "h", "thesis": "a specific grounded claim",
                "why_interesting": "w", "required_evidence": ["revolver"],
                "visual_opportunity": "close-up", "format": "short_video_essay",
            }],
            plan={
                "concept": {"title": "C1", "hook": "h", "thesis": "s"},
                "format": {"type": "short_video_essay", "duration_sec": 90},
                "editorial_direction": {
                    "pacing": "slow", "visual_style": "wide",
                    "audio_style": "minimal", "editing_style": "cut",
                },
            },
        )
        res = self._run(facts, metadata, mock, tmp_path)
        plan = res["plan"]
        assert "concept" in plan
        assert "evidence_strategy" in plan
        assert "format" in plan
        assert "editorial_direction" in plan
        assert plan["format"]["duration_sec"] == 90
        # Evidence strategy is deterministic and grounded.
        assert "scene-1" in plan["evidence_strategy"]["scene_ids"]


# --------------------------------------------------------------------------- #
# Full director flow incl. rejection
# --------------------------------------------------------------------------- #

class TestMovieGroundedDirector:
    def test_full_flow_selects_grounded_concept(self, facts, metadata, tmp_path):
        mock = MockLLM(
            concepts=[
                # admissible: cites real evidence
                {"title": "Real One", "hook": "h", "thesis": "a specific "
                 "grounded claim about the saloon",
                 "why_interesting": "w", "required_evidence": ["revolver"],
                 "visual_opportunity": "close-up", "format": "f"},
                # generic: must be rejected
                {"title": "Generic", "hook": "g", "thesis": "This movie "
                 "explores violence.", "why_interesting": "w",
                 "required_evidence": ["nonsense term"],
                 "visual_opportunity": "x", "format": "f"},
            ],
            plan={
                "concept": {"title": "Real One", "hook": "h", "thesis": "s"},
                "format": {"type": "short_video_essay", "duration_sec": 90},
                "editorial_direction": {"pacing": "p", "visual_style": "v",
                                        "audio_style": "a", "editing_style": "e"},
            },
        )
        director = MovieGroundedDirector(mock, memory_dir=tmp_path / "mem")
        res = director.develop(metadata, facts, num_concepts=2, min_coverage=0.4)
        titles = [c["title"] for c in res["generated_concepts"]]
        assert "Real One" in titles
        assert res["selected_concept"]["title"] == "Real One"
        assert res["selected_concept_index"] == 0

    def test_concept_without_evidence_rejected_and_replaced(self, facts, metadata, tmp_path):
        mock = MockLLM(
            concepts=[{
                "title": "Bad", "hook": "h", "thesis": "a specific claim about "
                "alien mind control", "why_interesting": "w",
                "required_evidence": ["flying saucer", "telepathy beam"],
                "visual_opportunity": "x", "format": "f",
            }],
            regenerate_to=[{
                "title": "Good Replacement", "hook": "h",
                "thesis": "a specific claim grounded in the saloon scene",
                "why_interesting": "w", "required_evidence": ["revolver"],
                "visual_opportunity": "close-up", "format": "f",
            }],
        )
        director = MovieGroundedDirector(mock, memory_dir=tmp_path / "mem")
        res = director.develop(metadata, facts, num_concepts=1, min_coverage=0.5)
        assert any(c["title"] == "Good Replacement"
                   for c in res["generated_concepts"])
        # The rejected concept is surfaced in the report data.
        assert any(c["title"] == "Bad" for c in res["rejected_concepts"])

    def test_report_renders_candidates_and_rejected(self, facts, metadata, tmp_path):
        mock = MockLLM(
            concepts=[
                {
                    "title": "Real", "hook": "h", "thesis": "a specific grounded "
                    "claim about the saloon", "why_interesting": "w",
                    "required_evidence": ["revolver"],
                    "visual_opportunity": "closeup", "format": "f",
                },
                {
                    "title": "Bogus", "hook": "g", "thesis": "a specific claim "
                    "about a flying saucer", "why_interesting": "gg",
                    "required_evidence": ["flying saucer"],
                    "visual_opportunity": "x", "format": "f",
                },
            ],
        )
        director = MovieGroundedDirector(mock, memory_dir=tmp_path / "mem")
        res = director.develop(metadata, facts, num_concepts=2, min_coverage=0.5)
        md = build_report(
            movie_title="T",
            concepts=res["generated_concepts"],
            rejected=res["rejected_concepts"],
            selected=res["selected_concept"],
            selected_index=res["selected_concept_index"],
            analyzer=EvidenceAnalyzer(facts),
            plan=res["plan"],
            diversity_metric=res["diversity_metric"],
        )
        assert "Candidate A" in md
        assert "SELECTED CONCEPT" in md
        assert "Rejected Concepts" in md

    def test_no_concepts_leads_to_empty_safe_result(self, facts, metadata, tmp_path):
        mock = MockLLM(concepts=[], plan={})
        director = MovieGroundedDirector(mock, memory_dir=tmp_path / "mem")
        res = director.develop(metadata, facts, num_concepts=3, min_coverage=0.4)
        assert res["selected_concept"] is None
        assert res["plan"] is None
        assert res["rejected_concepts"] == []

    def test_plan_with_invented_editorial_is_regenerated_once(self, facts,
                                                              metadata,
                                                              tmp_path):
        """A plan whose editorial_direction names props not in the evidence
        scenes triggers ONE corrective regeneration, then the audit is recorded
        (never forced through silently)."""

        class _SeqLLM:
            def __init__(self):
                self.calls = []
                self._plan_calls = 0

            def __call__(self, prompt):
                self.calls.append(prompt)
                if "finalizing the plan" in prompt:
                    self._plan_calls += 1
                    if self._plan_calls == 1:
                        ed = {
                            "pacing": "slow",
                            "visual_style": "empty chairs and a flying saucer "
                                            "in the saloon",
                            "audio_style": "minimal",
                            "editing_style": "quiet cuts",
                        }
                    else:
                        ed = {
                            "pacing": "slow",
                            "visual_style": "close-up on the revolver while "
                                            "the barman talks",
                            "audio_style": "minimal",
                            "editing_style": "quiet cuts",
                        }
                    return json.dumps({
                        "concept": {"title": "C", "hook": "h", "thesis": "s"},
                        "format": {"type": "short_video_essay",
                                   "duration_sec": 90},
                        "editorial_direction": ed,
                    })
                return json.dumps({"concepts": [{
                    "title": "C", "hook": "h",
                    "thesis": "a specific grounded claim about the saloon",
                    "why_interesting": "w",
                    "required_evidence": ["revolver"],
                    "visual_opportunity": "close-up",
                    "format": "short_video_essay",
                }]})

        mock = _SeqLLM()
        director = MovieGroundedDirector(mock, memory_dir=tmp_path / "mem")
        res = director.develop(metadata, facts, num_concepts=1, min_coverage=0.4)
        plan = res["plan"]
        # 3 calls: brainstorm + first plan + corrective plan retry.
        assert res["llm_stats"]["llm_calls"] == 3
        assert plan is not None
        audit = plan["grounding_audit"]
        assert audit["sufficient"] is True
        assert "chairs" not in audit["invented_terms"]
        assert "revolver" in " ".join(audit["grounded_terms"])
        # The corrective prompt carried the exact offending terms.
        assert "flying" in mock.calls[-1] or "saucer" in mock.calls[-1]

    def test_plan_regeneration_bounded_at_one(self, facts, metadata, tmp_path):
        """If the model keeps hallucinating, we record ONE retry and record the
        audit honestly rather than looping or silently forcing content."""

        class _StubbornLLM:
            def __init__(self):
                self.calls = []

            def __call__(self, prompt):
                self.calls.append(prompt)
                if "finalizing the plan" in prompt:
                    return json.dumps({
                        "concept": {"title": "C", "hook": "h", "thesis": "s"},
                        "format": {"type": "short_video_essay",
                                   "duration_sec": 90},
                        "editorial_direction": {
                            "pacing": "slow",
                            "visual_style": "a flying saucer over the saloon",
                            "audio_style": "minimal",
                            "editing_style": "quiet cuts",
                        },
                    })
                return json.dumps({"concepts": [{
                    "title": "C", "hook": "h",
                    "thesis": "a specific grounded claim about the saloon",
                    "why_interesting": "w",
                    "required_evidence": ["revolver"],
                    "visual_opportunity": "close-up",
                    "format": "short_video_essay",
                }]})

        director = MovieGroundedDirector(_StubbornLLM(),
                                         memory_dir=tmp_path / "mem")
        res = director.develop(metadata, facts, num_concepts=1, min_coverage=0.4)
        assert res["llm_stats"]["llm_calls"] == 3  # bounded: 1 corrective retry
        audit = res["plan"]["grounding_audit"]
        assert audit["sufficient"] is False  # honestly recorded, not faked
        assert "saucer" in audit["invented_terms"]

    def test_write_report_writes_director_reasoning_md(self, facts, metadata, tmp_path):
        """The public write_report alias produces reports/director_reasoning.md."""
        mock = MockLLM(
            concepts=[{
                "title": "Real One", "hook": "h", "thesis": "a specific grounded "
                "claim about the saloon", "why_interesting": "w",
                "required_evidence": ["revolver"],
                "visual_opportunity": "close-up", "format": "f",
            }],
            plan={
                "concept": {"title": "Real One", "hook": "h", "thesis": "s"},
                "format": {"type": "short_video_essay", "duration_sec": 90},
                "editorial_direction": {"pacing": "p", "visual_style": "v",
                                        "audio_style": "a", "editing_style": "e"},
            },
        )
        director = MovieGroundedDirector(mock, memory_dir=tmp_path / "mem")
        res = director.develop(metadata, facts, num_concepts=1, min_coverage=0.4)
        path = director.write_report(tmp_path, res)
        assert path.name == "director_reasoning.md"
        text = path.read_text(encoding="utf-8")
        assert "Director Reasoning Report" in text
        assert "Real One" in text


# --------------------------------------------------------------------------- #
# Run bookkeeping (validation-report counters)
# --------------------------------------------------------------------------- #

class TestRunBookkeeping:
    def test_llm_stats_count_initial_and_plan_calls(self, facts, metadata, tmp_path):
        """A clean run is exactly 2 LLM calls: brainstorm + plan."""
        mock = MockLLM(
            concepts=[{
                "title": "Grounded", "hook": "h",
                "thesis": "a specific grounded claim about the saloon",
                "why_interesting": "w", "required_evidence": ["revolver"],
                "visual_opportunity": "close-up", "format": "f",
            }],
            plan={
                "concept": {"title": "Grounded", "hook": "h", "thesis": "s"},
                "format": {"type": "short_video_essay", "duration_sec": 90},
                "editorial_direction": {"pacing": "p", "visual_style": "v",
                                        "audio_style": "a", "editing_style": "e"},
            },
        )
        director = MovieGroundedDirector(mock, memory_dir=tmp_path / "mem")
        res = director.develop(metadata, facts, num_concepts=1, min_coverage=0.4)
        stats = res["llm_stats"]
        assert stats["llm_calls"] == 2
        assert stats["regeneration_rounds"] == 0
        assert stats["substitutes_generated"] == 0

    def test_llm_stats_count_regeneration(self, facts, metadata, tmp_path):
        """Rejections trigger a regeneration round that is honestly counted."""
        mock = MockLLM(
            concepts=[{
                "title": "Bad", "hook": "h", "thesis": "a specific claim about "
                "alien mind control", "why_interesting": "w",
                "required_evidence": ["flying saucer", "telepathy beam"],
                "visual_opportunity": "x", "format": "f",
            }],
            regenerate_to=[{
                "title": "Good", "hook": "h",
                "thesis": "a specific claim grounded in the saloon scene",
                "why_interesting": "w", "required_evidence": ["revolver"],
                "visual_opportunity": "close-up", "format": "f",
            }],
            plan={
                "concept": {"title": "Good", "hook": "h", "thesis": "s"},
                "format": {"type": "short_video_essay", "duration_sec": 90},
                "editorial_direction": {"pacing": "p", "visual_style": "v",
                                        "audio_style": "a", "editing_style": "e"},
            },
        )
        director = MovieGroundedDirector(mock, memory_dir=tmp_path / "mem")
        res = director.develop(metadata, facts, num_concepts=1, min_coverage=0.5)
        stats = res["llm_stats"]
        assert stats["regeneration_rounds"] == 1
        assert stats["substitutes_generated"] == 1
        assert stats["llm_calls"] == 3  # brainstorm + regenerate + plan


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
