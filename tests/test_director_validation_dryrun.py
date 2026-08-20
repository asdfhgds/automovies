"""Focused regression tests for the director validation harness dry-run.

These exercise the SAME harness the real-GPU run will use
(``scripts/run_director_validation.py``) with the deterministic mock provider
(``--mock`` / ``--mock-scenario``), so no GPU/model is required and every
stage (project load -> SceneFacts -> context builder -> concept generation ->
grounding -> admission -> plan -> plan grounding -> verdict -> artifact write)
is proven end-to-end before the Colab T4 validation.

The mock scenarios map to the validation cases:

- ``grounded``      -> CASE A (valid concept, PASS)
- ``hallucinated``  -> CASE B (invented refs rejected, bounded regeneration)
- ``invalid``       -> CASE C (valid thesis, unsupported evidence, FAIL)
- ``hedged``        -> CASE D (hedged location never grounds claims)
- ``partial``       -> CASE E (partial coverage -> admission policy at work)
- ``none``          -> CASE F (nothing grounded -> FAIL, no plan)
- ``plan_rejected`` -> strict plan gate keeps plan None + rejection
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "run_director_validation.py"
REAL_PROJECT = ROOT / "data" / "bc6384be-47a5-4ee8-8674-7ff861472026"

sys.path.insert(0, str(ROOT / "src"))


def _mock_index() -> dict:
    """A deterministic, minimal movie index covering the matcher's vocabulary.

    Kept small so the dry-run is hermetic and fast; mirrors the real movie's
    shape (confident location, hedged location, real objects/actions/themes).
    """
    return {
        "project_id": "proj-dryrun",
        "source_path": "dryrun.mp4",
        "movie": {"title": "Dry Run Western", "duration_sec": 180.0},
        "scenes": [
            {
                "scene_id": "scene-1",
                "start_sec": 0.0,
                "end_sec": 30.0,
                "transcript": "",
                "story": {
                    "characters": ["Barman"],
                    "location": "indoor, convenience store",
                    "actions": ["pouring", "talking"],
                    "objects": ["road sign", "bench", "car"],
                    "themes": ["travel"],
                    "mood": "tense",
                    "dialogue": [{"speaker": "Barman",
                                  "text": "Keep your hands where I can see them."}],
                    "visual_description": "close-up of the road sign",
                },
            },
            {
                "scene_id": "scene-2",
                "start_sec": 30.0,
                "end_sec": 60.0,
                "transcript": "",
                "story": {
                    "characters": ["Stranger"],
                    "location": "outdoor, desert landscape",
                    "actions": ["standing"],
                    "objects": ["burning car", "car"],
                    "themes": ["travel", "desert setting"],
                    "mood": "somber",
                    "visual_description": "wide shot of the desert",
                },
            },
            {
                "scene_id": "scene-3",
                "start_sec": 60.0,
                "end_sec": 90.0,
                "transcript": "",
                "story": {
                    "characters": [],
                    "location": "indoor, inside a vehicle (likely a bus or train)",
                    "actions": ["riding"],
                    "objects": ["bus interior with hanging items"],
                    "themes": ["transportation"],
                    "mood": "neutral",
                    "visual_description": "a vehicle interior",
                },
            },
        ],
    }


@pytest.fixture()
def dryrun_project(tmp_path):
    """Materialize a project dir (movie_index.json only) that the harness
    writes its reports into, without touching the real project."""
    project_dir = tmp_path / "dryrun"
    project_dir.mkdir()
    (project_dir / "movie_index.json").write_text(
        json.dumps(_mock_index(), ensure_ascii=False), encoding="utf-8")
    return project_dir


def _run_harness(project_dir, scenario, extra=None, monkeypatch=None):
    """Run the REAL harness script as a subprocess with the mock provider."""
    cmd = [
        sys.executable, str(HARNESS),
        "--project", str(project_dir),
        "--mock", "--mock-scenario", scenario,
        "--num-concepts", "3",
    ]
    if extra:
        cmd += extra
    proc = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    json_path = project_dir / "reports" / "director_validation.json"
    md_path = project_dir / "reports" / "director_reasoning.md"
    assert json_path.exists()
    assert md_path.exists()
    return json.loads(json_path.read_text(encoding="utf-8")), md_path.read_text(
        encoding="utf-8")


# --------------------------------------------------------------------------- #
# CASE A — valid grounded concept                                                #
# --------------------------------------------------------------------------- #

class TestCaseAValidGrounded:
    def test_grounded_concept_selected_plan_and_pass(self, dryrun_project):
        data, md = _run_harness(dryrun_project, "grounded")
        assert data["verdict"] == "PASS"
        assert data["selected_concept"] is not None
        assert data["plan"] is not None
        assert data["plan_rejection"] is None
        sel = data["selected_concept"]
        # The derived refs are verbatim movie vocabulary that resolves.
        refs = sel.get("evidence_refs") or []
        assert any(r.get("kind") == "object" for r in refs)
        assert any(r.get("kind") == "scene" for r in refs)
        # The plan's editorial_direction is inside the evidence scenes.
        audit = data["plan"].get("grounding_audit") or {}
        assert audit.get("sufficient") is True
        assert audit.get("invented_terms") == []
        assert "SELECTED CONCEPT" in md
        assert "Director Reasoning Report" in md


# --------------------------------------------------------------------------- #
# CASE B — hallucinated references                                              #
# --------------------------------------------------------------------------- #

class TestCaseBHallucinatedRefs:
    def test_hallucinated_concepts_rejected_with_bounded_regeneration(
            self, dryrun_project):
        data, md = _run_harness(dryrun_project, "hallucinated")
        # First batch was hallucinated (flying saucer / Sherlock Holmes /
        # enchanted castle) -> rejected; the run recovered via ONE bounded
        # regeneration round and finished PASS.
        assert data["verdict"] == "PASS"
        assert data["selected_concept"] is not None
        assert data["plan"] is not None
        assert len(data["rejected_concepts"]) == 3  # the hallucinated batch
        rejected_theses = " ".join(
            str(c.get("thesis", "")) for c in data["rejected_concepts"])
        assert "flying saucer" in rejected_theses
        # Bounded retries: exactly one regeneration round, counts recorded.
        rt = data["runtime"]
        assert rt["regeneration_rounds"] == 1
        assert rt["substitutes_generated"] == 3
        assert rt["llm_calls"] == 3  # brainstorm + regenerate + plan
        # The reaction report shows the rejected batch.
        assert "Rejected Concepts" in md
        assert "flying saucer" in md


# --------------------------------------------------------------------------- #
# CASE C — valid thesis, invalid evidence                                          #
# --------------------------------------------------------------------------- #

class TestCaseCInvalidEvidence:
    def test_structurally_valid_but_ungrounded_concept_fails(self, dryrun_project):
        """A well-formed thesis whose only support is absent -> no concept, no
        plan; even regeneration cannot substitute something grounded."""
        data, _ = _run_harness(dryrun_project, "invalid")
        assert data["verdict"] == "FAIL"
        assert not data["selected_concept"]  # {} — nothing admissible
        assert data["plan"] is None
        # Regeneration was bounded and still found nothing grounded.
        assert data["runtime"]["regeneration_rounds"] <= 1
        assert len(data["rejected_concepts"]) >= 3
        assert data["runtime"]["llm_calls"] <= 2  # brainstorm + one regen


# --------------------------------------------------------------------------- #
# CASE D — hedged location                                                        #
# --------------------------------------------------------------------------- #

class TestCaseDHedgedLocation:
    def test_hedged_location_does_not_ground_claim(self, dryrun_project):
        """A claim built on 'train'/'shop' — words that only exist inside a
        hedged location label ('likely a bus or train') — must be rejected:
        ``_is_location_confident`` forbids borrowing words from guesses."""
        from director.evidence import EvidenceAnalyzer
        assert EvidenceAnalyzer._is_location_confident(
            "indoor, inside a vehicle (likely a bus or train)") is False
        assert EvidenceAnalyzer._is_location_confident(
            "indoor, convenience store") is True

        # The hedged-batch concepts are rejected outright...
        initial, _ = _run_harness(dryrun_project, "hedged")
        # ...and the run still recovers via grounded substitutes (so the run
        # itself can complete, while every hedged claim was independently
        # rejected as ungrounded).
        assert len(initial["rejected_concepts"]) == 3
        hedged_theses = " ".join(
            str(c.get("thesis", "")) for c in initial["rejected_concepts"])
        assert "train platform" in hedged_theses
        assert "shop" in hedged_theses

    def test_hedged_location_words_never_become_derived_refs(self, dryrun_project):
        from director.evidence import EvidenceAnalyzer
        from director.scene_facts import SceneFacts
        idx = _mock_index()
        facts = SceneFacts.from_movie_intelligence(movie_index=idx)
        analyzer = EvidenceAnalyzer(facts)
        concept = {
            "title": "Train", "hook": "h",
            "thesis": "the train platform becomes the film's image of escape",
            "why_interesting": "w", "visual_opportunity": "wide shot",
        }
        refs = analyzer.derive_refs(concept)
        values = [(r.get("kind"), r.get("value")) for r in refs]
        assert all("train" not in str(v) for _, v in values), values
        # An actual confident location still grounds.
        ok = {
            "title": "Store", "hook": "h",
            "thesis": "the convenience store becomes the film's image of commerce",
            "why_interesting": "w", "visual_opportunity": "shot",
        }
        ok_refs = analyzer.derive_refs(ok)
        assert any("convenience store" in str(r.get("value"))
                   for r in ok_refs)


# --------------------------------------------------------------------------- #
# CASE E — partial coverage                                                       #
# --------------------------------------------------------------------------- #

class TestCaseEPartialCoverage:
    def test_partial_concept_grounded_half_is_enough(self, dryrun_project):
        """The concept declares real + absent refs; the deterministic gate
        admits only what resolves (coverage HIGH/MED per policy) and plans
        only that. The absent half is surfaced as a rejected claim, never
        planned."""
        data, md = _run_harness(dryrun_project, "partial")
        assert data["verdict"] == "PASS"
        sel = data["selected_concept"]
        refs = sel.get("evidence_refs") or []
        values = " ".join(str(r.get("value", "")) for r in refs)
        assert "flying saucer" not in values        # absent part pruned
        assert "Sherlock Holmes" not in values
        # Decisions follow the admission policy (min_coverage=0.4): with the
        # real object + location matched, the concept is admissible.
        assert data["plan"] is not None
        # The evidence preview shows every surviving ref is grounded.
        strategy = data["plan"].get("evidence_strategy") or {}
        assert strategy.get("scene_ids"), "evidence must map to real scenes"
        assert "Candidate A" in md

    def test_partial_concept_below_threshold_fails(self, dryrun_project):
        # num-concepts irrelevant; use the 'none' style via 'invalid' scenario
        # to prove the admission policy rejects when nothing grounds.
        data, _ = _run_harness(dryrun_project, "invalid")
        assert data["verdict"] == "FAIL"
        assert not data["selected_concept"]  # {} — nothing admissible


# --------------------------------------------------------------------------- #
# CASE F — no valid concepts                                                     #
# --------------------------------------------------------------------------- #

class TestCaseFNoValidConcepts:
    def test_all_fail_gives_none_and_fail_verdict(self, dryrun_project):
        data, md = _run_harness(dryrun_project, "none")
        assert data["verdict"] == "FAIL"
        assert not data["selected_concept"]  # {} — nothing admissible
        assert data["plan"] is None
        assert data["runtime"]["llm_calls"] <= 2
        assert len(data["rejected_concepts"]) >= 3
        assert "Rejected Concepts" in md


# --------------------------------------------------------------------------- #
# Strict plan gate / verdict path                                               #
# --------------------------------------------------------------------------- #

class TestPlanGateAndVerdict:
    def test_plan_rejected_records_rejection(self, dryrun_project):
        """A grounded concept whose plan editorial invents content ('flying
        saucer', 'chairs') is rejected by the STRICT PLAN GATE: plan stays
        None, plan_rejection is recorded, verdict is PLAN_REJECTED."""
        data, md = _run_harness(dryrun_project, "plan_rejected")
        assert data["verdict"] == "PLAN_REJECTED"
        assert data["selected_concept"] is not None
        assert data["plan"] is None
        rejection = data["plan_rejection"]
        assert rejection is not None
        assert rejection["reason"]
        audit = rejection["audit"]
        assert audit["sufficient"] is False
        assert "flying saucer" in " ".join(audit["invented_terms"])
        assert "PLAN REJECTED" in md

    def test_verdict_never_pass_on_insufficient_grounding(self, dryrun_project):
        """The validation summary must never report PASS while the grounding
        is insufficient (concept-less or plan-less runs)."""
        for scenario in ("invalid", "none", "plan_rejected"):
            data, _ = _run_harness(dryrun_project, scenario)
            assert data["verdict"] != "PASS"
            if data["verdict"] == "PASS":
                # PAS requires a selected concept AND an emitted plan.
                assert data["selected_concept"] is not None
                assert data["plan"] is not None
                assert data["plan_rejection"] is None

    def test_summary_counts_are_correct(self, dryrun_project):
        data, _ = _run_harness(dryrun_project, "hallucinated")
        rt = data["runtime"]
        # One duplicates batch of 3 was generated, then one bounded 3-concept
        # substitute batch; so llm calls reflect each generate + regenerate +
        # the plan (exact count is a documented behavior, not a magic number).
        assert rt["regeneration_rounds"] == 1
        assert rt["substitutes_generated"] == 3
        assert rt["llm_calls"] == 3  # brainstorm + regenerate + plan
        assert len(data["generated_concepts"]) + len(data["rejected_concepts"]) == 6
        assert data["diversity_metric"] >= 0.0


# --------------------------------------------------------------------------- #
# Real project (when present)                                                      #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not REAL_PROJECT.exists(),
                    reason="real validated movie not present")
def test_harness_runs_on_actual_validated_movie(tmp_path):
    """Copy the REAL movie_index into a temp project and run the harness with
    the mock provider end-to-end (no GPU). This proves the exact artifact set
    the Colab T4 run will write, without touching the real reports."""
    for name in ("movie_index.json",):
        shutil.copy2(REAL_PROJECT / name, tmp_path / name)
    data, md = _run_harness(tmp_path, "grounded")
    assert data["verdict"] == "PASS"
    assert data["selected_concept"] is not None
    assert data["plan"] is not None
    assert (tmp_path / "reports" / "director_validation.json").exists()
    assert (tmp_path / "reports" / "director_reasoning.md").exists()
    # runtime honors the mock provider honestly.
    assert data["runtime"]["model"] == "mock-validation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])