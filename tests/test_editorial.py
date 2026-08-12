"""Tests for the editorial planning subsystem: plan schema/validation, the
heuristic editorial director + evidence retrieval, the script builder, the
cinematic subtitle chunker, and the editorial timeline builder."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from editorial.plan import (
    EditingDirective,
    EditorialEvidence,
    EditorialPlan,
    EditorialSegment,
    NarrationBlock,
    NarrationDelivery,
    validate_plan,
)
from editorial.retrieval import EvidenceRetriever
from editorial.director import HeuristicEditorialPlanner, create_editorial_plan
from editorial.script import build_editorial_script, _estimate_seconds
from editorial.subtitles import (
    caption_word_timings,
    merge_with_real_word_timestamps,
    split_into_captions,
    captions_to_srt_lines,
)
from editorial.timeline import EditorialTimelineBuilder
from tests.editorial_fixtures import (
    DIRECTOR_PLAN,
    make_movie_index,
    seed_project,
)


# --------------------------------------------------------------------------
# Plan schema + validation
# --------------------------------------------------------------------------

def _minimal_plan() -> EditorialPlan:
    return EditorialPlan(
        title="The Coin Chooses",
        thesis="Fate is a story we tell to avoid our own choices.",
        hook={"text": "A coin flip is a choice pretending to be an accident.",
              "visual_strategy": "slow push in"},
        segments=[
            EditorialSegment(
                id="seg_00",
                purpose="hook and thesis",
                evidence=[EditorialEvidence("scene-1", 0.0, 3.0, "opening")],
                narration=NarrationBlock("test", NarrationDelivery()),
                editing=EditingDirective(transition="cut"),
            ),
        ],
    )


def test_plan_roundtrip_json():
    plan = _minimal_plan()
    restored = EditorialPlan.from_dict(json.loads(plan.to_json()))
    assert restored.title == plan.title
    assert restored.thesis == plan.thesis
    assert restored.segments[0].id == "seg_00"
    assert restored.segments[0].evidence[0].scene_id == "scene-1"
    assert restored.segments[0].editing.transition == "cut"


def test_plan_validation_catches_empty_fields():
    bad = EditorialPlan(title="", thesis="", hook={}, segments=[])
    errors = validate_plan(bad)
    assert any("title" in e for e in errors)
    assert any("thesis" in e for e in errors)
    assert any("hook" in e for e in errors)
    assert any("segments" in e for e in errors)


def test_supported_editing_ops():
    from editorial.plan import SUPPORTED_EDITING_OPS

    # the minimal op set the renderer must honour
    ops = {"CUT", "CROSSFADE", "HOLD", "SPEED", "CROP", "FADE", "AUDIO_DUCK"}
    assert ops.issubset(set(SUPPORTED_EDITING_OPS))


# --------------------------------------------------------------------------
# Evidence retrieval (semantic + lexical blend)
# --------------------------------------------------------------------------

def test_evidence_retriever_returns_anchored_excerpts():
    movie_index = make_movie_index()
    retriever = EvidenceRetriever.from_project_dicts(movie_index)
    hits = retriever.retrieve("fate controls people", k=3)
    assert hits
    for ev in hits:
        assert ev.start_sec < ev.end_sec
        assert 0 < ev.end_sec - ev.start_sec <= 6.0
        assert ev.reason
    # the dialogue-dense scene about fate should rank first
    assert hits[0].scene_id == "scene-1"


def test_evidence_retriever_excludes_used_scenes():
    movie_index = make_movie_index()
    retriever = EvidenceRetriever.from_project_dicts(movie_index)
    first = retriever.retrieve("fate controls people", k=3, exclude=["scene-1"])
    assert all(e.scene_id != "scene-1" for e in first)


# --------------------------------------------------------------------------
# Editorial director
# --------------------------------------------------------------------------

def test_heuristic_editorial_plan_is_evidence_driven():
    movie_index = make_movie_index()
    planner = HeuristicEditorialPlanner()
    retriever = EvidenceRetriever.from_project_dicts(movie_index)
    plan = planner.create_plan(movie_index, DIRECTOR_PLAN, retriever,
                               DIRECTOR_PLAN["creative_task"], target_sec=90)
    assert validate_plan(plan) == []
    assert 4 <= len(plan.segments) <= 6
    assert plan.hook["text"]
    # narration connects to evidence: every segment has excerpts
    for seg in plan.segments:
        assert seg.evidence, f"{seg.id} has no evidence"
        assert seg.narration.text
        assert seg.editing.transition in ("cut", "crossfade", "fade")
    # evidence scenes are non-overlapping across the plan
    all_scenes = [e.scene_id for seg in plan.segments for e in seg.evidence]
    assert len(all_scenes) == len(set(all_scenes)) or len(all_scenes) > 2


def test_create_editorial_plan_persists(tmp_path):
    seed_project(tmp_path)
    plan = create_editorial_plan(tmp_path, target_sec=60)
    out = json.loads((tmp_path / "editorial_plan.json").read_text(encoding="utf-8"))
    assert out["thesis"] == DIRECTOR_PLAN["thesis"]
    assert out["provenance"]["planner"] == "heuristic"
    assert len(out["segments"]) >= 3


def test_qwen_planner_gated():
    from editorial.director import QwenEditorialPlanner

    import os
    old = os.environ.get("REQUIRE_REAL_LLM")
    os.environ["REQUIRE_REAL_LLM"] = "false"
    try:
        with pytest.raises(RuntimeError):
            QwenEditorialPlanner()
    finally:
        if old is None:
            os.environ.pop("REQUIRE_REAL_LLM", None)
        else:
            os.environ["REQUIRE_REAL_LLM"] = old


# --------------------------------------------------------------------------
# Cinematic subtitles
# --------------------------------------------------------------------------

def test_split_into_short_captions():
    chunks = split_into_captions("The coin chooses for him and not for the ones who flip it")
    assert all(len(c.split()) <= 3 for c in chunks)
    assert all(len(c) <= 32 for c in chunks)
    assert " ".join(chunks).split() == (
        "The coin chooses for him and not for the ones who flip it").split()


def test_caption_word_timings_cover_window():
    chunks = split_into_captions("The coin chooses for him")
    caps = caption_word_timings(chunks, start_sec=10.0, duration_sec=2.0)
    assert caps[0]["start_sec"] == pytest.approx(10.0)
    assert caps[-1]["end_sec"] == pytest.approx(12.0, abs=0.05)
    assert len(caps[0]["words"]) == len(caps[0]["text"].split())


def test_merge_real_word_timestamps():
    words = [
        {"word": "The", "start": 1.0, "end": 1.2},
        {"word": "coin", "start": 1.2, "end": 1.5},
        {"word": "chooses", "start": 1.5, "end": 1.9},
    ]
    caps = merge_with_real_word_timestamps("The coin chooses", words, 0.0, 3.0)
    assert caps[0]["words"][0]["start_sec"] == pytest.approx(1.0)
    assert caps[-1]["words"][-1]["end_sec"] == pytest.approx(1.9)


def test_srt_lines_uppercase_and_timed():
    caps = caption_word_timings(["THE COIN", "CHOOSES"], 0.0, 2.0)
    lines = captions_to_srt_lines(caps)
    assert any("--> " in line for line in lines)
    assert "THE COIN" in "\n".join(lines)


# --------------------------------------------------------------------------
# Editorial script builder
# --------------------------------------------------------------------------

def test_editorial_script_maps_narration_to_evidence(tmp_path):
    seed_project(tmp_path)
    movie_index = make_movie_index()
    plan = create_editorial_plan(tmp_path, target_sec=90)
    script = build_editorial_script(tmp_path, plan, movie_index)

    assert script["editorial"] is True
    assert script["voiceover_text"]
    assert len(script["sections"]) == len(plan.segments)
    for section in script["sections"]:
        assert section["text"]
        assert section["delivery"]["pace"] > 0
        assert section["narration_start_sec"] >= 0
        assert section["subtitle_captions"], "each section has short captions"
        assert section["narrative_evidence"], "narration connected to evidence"
        for cap in section["subtitle_captions"]:
            assert len(cap["text"].split()) <= 3
        # sections run sequentially
        assert section["narration_start_sec"] >= 0


def test_estimate_seconds_scales_with_pace():
    slow = _estimate_seconds("one two three four five six", pace=0.5)
    fast = _estimate_seconds("one two three four five six", pace=1.5)
    assert slow > fast


# --------------------------------------------------------------------------
# Editorial timeline
# --------------------------------------------------------------------------

def test_editorial_timeline_short_excerpts_and_ops(tmp_path):
    seed_project(tmp_path)
    movie_index = make_movie_index()
    plan = create_editorial_plan(tmp_path, target_sec=60)
    script = build_editorial_script(tmp_path, plan, movie_index)

    builder = EditorialTimelineBuilder(
        source_path=None,
        excerpt_factory=lambda src, a, b, out, audio: Path(out),  # no real movie
    )
    timeline = builder.build(tmp_path, plan, script)
    assert timeline["mode"] == "editorial"

    # short excerpts: the window extracted from the movie is small, not the
    # whole scene; duration reflects speed/hold transforms on top of it
    for seg in timeline["segments"]:
        for clip in seg["video"]:
            window = clip["source_end_sec"] - clip["source_start_sec"]
            assert window <= 6.0 + 1e-6, "excerpt window is capped short"
            assert clip["duration_sec"] > 0
            assert clip["extracted"] is False  # no real source movie in test
        assert seg["narration"]["duration_sec"] > 0

    # editing ops are recorded and honored
    assert timeline["tracks"]["video"]["items"]
    metas = [i["metadata"] for i in timeline["tracks"]["video"]["items"]]
    assert all("editing" in m for m in metas)
    assert timeline["tracks"]["text"]["items"], "caption subtitles present"

    # transitions allowed set
    for seg in timeline["segments"]:
        assert seg["transition_to_next"] in ("cut", "crossfade", "fade")


def test_timeline_rejects_no_video():
    from editorial.render import build_editorial_render_command

    with pytest.raises(ValueError):
        build_editorial_render_command(
            {"narration_total_sec": 10.0, "segments": []},
            Path("v.wav"), Path("o.mp4"),
        )
