"""Editorial Decision List: schema roundtrip, validation, and planner variety."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tests.editorial_fixtures as f
from editorial.decision import (
    AudioIntent,
    EditingIntent,
    EditorialDecision,
    EditorialDecisionList,
    EditorialEvidence,
    NarrationBlock,
    NarrationDelivery,
    Pacing,
    VisualStrategy,
    compile_editorial_plan,
    load_decision_list,
    save_decision_list,
    validate_decision_list,
)
from editorial.director import HeuristicEditorialPlanner, QwenEditorialPlanner
from editorial.retrieval import EvidenceRetriever


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    return f.seed_project(tmp_path / "p")


def _heuristic_decisions(project_dir: Path) -> EditorialDecisionList:
    planner = HeuristicEditorialPlanner()
    mi = f.make_movie_index()
    retriever = EvidenceRetriever.from_project_dicts(mi)
    return planner.create_decisions(
        mi, f.DIRECTOR_PLAN, retriever,
        f.DIRECTOR_PLAN.get("creative_task", ""),
        target_sec=60.0,
        source_duration=30.0,
    )


def test_decision_list_roundtrip(project_dir: Path):
    dl = _heuristic_decisions(project_dir)
    assert validate_decision_list(dl) == []

    restored = EditorialDecisionList.from_dict(dl.to_dict())
    assert restored.title == dl.title
    assert restored.thesis == dl.thesis
    assert len(restored.decisions) == len(dl.decisions)
    for a, b in zip(restored.decisions, dl.decisions):
        assert a.segment_id == b.segment_id
        assert a.purpose == b.purpose
        assert a.evidence == b.evidence
        assert a.narration.text == b.narration.text
        assert a.visual_strategy == b.visual_strategy
        assert a.audio == b.audio
        assert a.editing == b.editing


def test_validate_decision_list_accepts_real_list(project_dir: Path):
    dl = _heuristic_decisions(project_dir)
    errors = validate_decision_list(dl)
    assert errors == []


def test_validate_decision_list_rejects_invalid_window(project_dir: Path):
    dl = _heuristic_decisions(project_dir)
    dl.decisions[0].evidence[0] = EditorialEvidence(
        scene_id="scene-1", start_sec=3.0, end_sec=1.0, reason="reversed window")
    errors = validate_decision_list(dl)
    assert any("window" in e for e in errors)


def test_validate_decision_list_rejects_missing_narration(project_dir: Path):
    dl = _heuristic_decisions(project_dir)
    dl.decisions[0].narration = NarrationBlock(text="", delivery=NarrationDelivery())
    errors = validate_decision_list(dl)
    assert errors, "expected validation errors for an empty narration block"


def test_heuristic_planner_produces_varied_decisions(project_dir: Path):
    dl = _heuristic_decisions(project_dir)
    assert len(dl.decisions) >= 3
    purposes = {d.purpose for d in dl.decisions}
    assert len(purposes) >= 2, "edit must vary its purposes, not repeat one beat"
    strategies = {d.visual_strategy.type for d in dl.decisions}
    assert len(strategies) >= 2, "visual strategies must genuinely differ"

    scenes: dict = {}
    for d in dl.decisions:
        for e in d.evidence:
            scenes.setdefault(e.scene_id, 0)
            scenes[e.scene_id] += 1
    assert scenes, "every decision must cite real, anchored evidence"
    assert all(0.0 <= e.end_sec - e.start_sec <= 6.0
               for d in dl.decisions for e in d.evidence)


def test_heuristic_planner_varies_director_signals(project_dir: Path):
    """The whole edit must be intentionally directed: narration performance,
    audio intent, pacing and transitions all differ across the arc."""
    dl = _heuristic_decisions(project_dir)
    assert len({d.narration.delivery.tone for d in dl.decisions}) >= 2
    assert len({d.narration.delivery.emotion for d in dl.decisions}) >= 2
    assert len({d.narration.delivery.energy for d in dl.decisions}) >= 2
    assert len({d.pacing.rhythm for d in dl.decisions}) >= 2
    assert len({d.audio.music for d in dl.decisions}) >= 2
    assert len({d.editing.transition for d in dl.decisions}) >= 2
    # Voice direction survived into the narration blocks (TTS consumes it).
    assert all(d.narration.delivery.tone for d in dl.decisions)


def test_heuristic_planner_justifies_deliberate_reuse(project_dir: Path):
    dl = _heuristic_decisions(project_dir)
    reused = [e.scene_id for d in dl.decisions for e in d.evidence]
    dupes = {s for s in reused if reused.count(s) > 1}
    if dupes:  # the director must explain motif returns, never repeat blindly
        for sid in dupes:
            assert sid in dl.scene_reuse_justification, (
                f"reuse of {sid} must be a deliberate, justified motif return")


def test_qwen_planner_works_with_stub_llm(project_dir: Path, monkeypatch):
    monkeypatch.setenv("REQUIRE_REAL_LLM", "true")

    def stub_llm(prompt: str) -> str:
        return json.dumps({
            "title": "Chance as Fate",
            "hook": {"text": "A coin decides. Or does it?",
                     "visual_strategy": "wide"},
            "scene_reuse_justification": {},
            "decisions": [
                {
                    "segment_id": "seg_00",
                    "purpose": "hook",
                    "narrative_beat": "open the argument",
                    "evidence": [{"scene_id": "scene-1", "start_sec": 0.2,
                                  "end_sec": 4.0, "reason": "the coin spins"}],
                    "narration": {"text": "A coin decides. Or does it?",
                                  "delivery": {"tone": "inviting",
                                               "emotion": "curious",
                                               "energy": 0.6, "pace": 0.95}},
                    "visual_strategy": {"type": "wide", "description": "open"},
                    "pacing": {"duration_sec": 4.0, "rhythm": "slow"},
                    "audio": {"movie_audio": "duck", "music": "low",
                              "narration": "dominant", "silence": ""},
                    "editing": {"transition": "cut", "speed": 1.0,
                                "hold": False, "crop_zoom": 1.0}},
                {
                    "segment_id": "seg_01",
                    "purpose": "contrast",
                    "narrative_beat": "belief vs reality",
                    "evidence": [{"scene_id": "scene-2", "start_sec": 10.5,
                                  "end_sec": 14.0, "reason": "rosa laughs"}],
                    "narration": {"text": "Rosa shows him the trick.",
                                  "delivery": {"tone": "wry",
                                               "emotion": "amused",
                                               "energy": 0.5, "pace": 1.05}},
                    "visual_strategy": {"type": "cross_cut",
                                        "description": "cut worlds"},
                    "pacing": {"duration_sec": 4.0, "rhythm": "fast"},
                    "audio": {"movie_audio": "retain", "music": "rise",
                              "narration": "dominant", "silence": ""},
                    "editing": {"transition": "cut", "speed": 1.05,
                                "hold": False, "crop_zoom": 1.0}},
                {
                    "segment_id": "seg_02",
                    "purpose": "conclusion",
                    "narrative_beat": "close the argument",
                    "evidence": [{"scene_id": "scene-3", "start_sec": 18.5,
                                  "end_sec": 22.0, "reason": "pattern clear"}],
                    "narration": {"text": "Chance only looks like choice.",
                                  "delivery": {"tone": "quiet",
                                               "emotion": "resolute",
                                               "energy": 0.4, "pace": 0.85}},
                    "visual_strategy": {"type": "motif_return",
                                        "description": "back to the coin"},
                    "pacing": {"duration_sec": 4.0, "rhythm": "slow"},
                    "audio": {"movie_audio": "duck", "music": "resolve",
                              "narration": "dominant", "silence": "final beat"},
                    "editing": {"transition": "fade", "speed": 1.0,
                                "hold": True, "crop_zoom": 1.0}},
            ],
        })

    planner = QwenEditorialPlanner(llm=stub_llm, strict=True)
    dl = planner.create_decisions(
        f.make_movie_index(), f.DIRECTOR_PLAN,
        EvidenceRetriever.from_project_dicts(f.make_movie_index()),
        "edit the essay", target_sec=60.0, source_duration=30.0)
    assert validate_decision_list(dl) == []
    assert [d.purpose for d in dl.decisions] == ["hook", "contrast", "conclusion"]
    # windows must sit inside the real scene footprints
    scenes = {s["scene_id"]: s for s in f.SCENES}
    for d in dl.decisions:
        for e in d.evidence:
            sc = scenes[e.scene_id]
            assert sc["start_sec"] <= e.start_sec < e.end_sec <= sc["end_sec"]


def test_qwen_planner_persists_decision_list(project_dir: Path, monkeypatch):
    monkeypatch.setenv("REQUIRE_REAL_LLM", "true")

    def stub_llm(prompt: str) -> str:
        return json.dumps({
            "decisions": [
                {"segment_id": "s1", "purpose": "hook", "narrative_beat": "b",
                 "evidence": [{"scene_id": "scene-1", "start_sec": 0.0,
                               "end_sec": 4.0, "reason": "r"}],
                 "narration": {"text": "Watch closely.", "delivery": {}},
                 "visual_strategy": {"type": "wide"},
                 "pacing": {"duration_sec": 4.0},
                 "audio": {}, "editing": {}},
                {"segment_id": "s2", "purpose": "conclusion", "narrative_beat": "b",
                 "evidence": [{"scene_id": "scene-3", "start_sec": 18.0,
                               "end_sec": 22.0, "reason": "r"}],
                 "narration": {"text": "Now you see.", "delivery": {}},
                 "visual_strategy": {"type": "motif_return"},
                 "pacing": {"duration_sec": 4.0},
                 "audio": {}, "editing": {}},
            ]
        })

    save_decision_list(project_dir, QwenEditorialPlanner(llm=stub_llm).create_decisions(
        f.make_movie_index(), f.DIRECTOR_PLAN,
        EvidenceRetriever.from_project_dicts(f.make_movie_index()),
        "t", target_sec=60.0, source_duration=30.0))
    path = project_dir / "editorial_decisions.json"
    assert path.exists()
    loaded = load_decision_list(project_dir)
    assert [d.purpose for d in loaded.decisions] == ["hook", "conclusion"]


def test_qwen_planner_refuses_hallucinated_scenes(monkeypatch):
    monkeypatch.setenv("REQUIRE_REAL_LLM", "true")

    def stub_llm(prompt: str) -> str:
        return json.dumps({
            "decisions": [
                {"segment_id": "s1", "purpose": "hook",
                 "evidence": [{"scene_id": "scene-999", "start_sec": 0.0,
                               "end_sec": 4.0, "reason": "not in movie"}],
                 "narration": {"text": "x", "delivery": {}},
                 "visual_strategy": {}, "pacing": {}, "audio": {}, "editing": {}},
            ]
        })

    planner = QwenEditorialPlanner(llm=stub_llm, strict=True)
    with pytest.raises(ValueError):
        planner.create_decisions(
            f.make_movie_index(), f.DIRECTOR_PLAN,
            EvidenceRetriever.from_project_dicts(f.make_movie_index()),
            "t", target_sec=60.0, source_duration=30.0)


def test_compile_editorial_plan_from_decisions(project_dir: Path):
    dl = _heuristic_decisions(project_dir)
    plan = compile_editorial_plan(dl)
    assert plan is not None
    assert len(plan.segments) == len(dl.decisions)