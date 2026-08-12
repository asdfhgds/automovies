"""Tests for the movie understanding layer (scene enrichment / semantic index /
characters / events / analyzer persistence)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_understanding.analyzer import MovieAnalyzer
from movie_understanding.semantic_index import SemanticIndex
from movie_understanding.scene_analyzer import HeuristicSceneEnricher


def _segments():
    return [
        {"id": "seg_000", "start_sec": 0.0,  "end_sec": 2.0,
         "text": "The coin spins and Sam watches it fall."},
        {"id": "seg_001", "start_sec": 3.0,  "end_sec": 5.0,
         "text": "Sam believes fate controls everything."},
        {"id": "seg_002", "start_sec": 12.0, "end_sec": 14.0,
         "text": "But Rosa laughs and shows him the trick."},
    ]


def _scenes():
    return [
        {"scene_id": "scene-1", "start_sec": 0.0, "end_sec": 6.0,
         "duration": 6.0, "transcript": "The coin spins and Sam watches it fall. "
                                        "Sam believes fate controls everything."},
        {"scene_id": "scene-2", "start_sec": 10.0, "end_sec": 16.0,
         "duration": 6.0, "transcript": "Rosa laughs and shows him the trick."},
    ]


def test_heuristic_enrichment_populates_story():
    enricher = HeuristicSceneEnricher()
    enriched = enricher.enrich(_scenes()[0], _segments())
    story = enriched["story"]
    assert enriched["scene_id"] == "scene-1"
    assert isinstance(story["summary"], str) and story["summary"]
    assert "fate" in story["topics"] or "coin" in story["topics"]
    assert len(story["dialogue"]) == 2
    assert all(d["text"] for d in story["dialogue"])
    # characters: Sam mentioned twice -> candidate
    assert "Sam" in story["characters"]
    # honestly-unavailable fields are None, not invented
    assert story["location"] is None
    assert story["actions"] is None
    assert story["visual_description"] is None
    assert story["provenance"]["characters"] == "transcript_capitalized_names"


def test_emotional_tone_heuristic():
    assert HeuristicSceneEnricher().enrich(
        {"scene_id": "s", "start_sec": 0, "end_sec": 1,
         "transcript": "fear danger chase the shadow"},
        [],
    )["story"]["emotional_tone"] == "tension"


def test_semantic_search_ranks_relevant_scene_first():
    index = SemanticIndex().build(
        [HeuristicSceneEnricher().enrich(s, _segments()) for s in _scenes()]
    )
    results = index.search("fate and destiny control", k=5)
    assert results and results[0]["scene_id"] == "scene-1"
    assert all(r["scene_id"] in ("scene-1", "scene-2") for r in results)


def test_semantic_search_fallback_on_non_ascii_query():
    # queries with no recognizable tokens return empty rather than crashing
    index = SemanticIndex().build([])
    assert index.search("") == []


def test_character_index_builds_and_requires_two_mentions():
    from movie_understanding.character_analyzer import build_character_index

    chars = build_character_index(_scenes(), _segments())
    names = [c["name"] for c in chars]
    assert "Sam" in names          # 2 mentions
    assert "Rosa" not in names     # 1 mention only
    assert any(c["name"] == "Sam" and len(c["scene_ids"]) == 1 for c in chars)


def test_event_index_splits_on_gap_and_scene():
    from movie_understanding.event_index import build_event_index

    events = build_event_index(_scenes(), _segments(), gap_threshold_sec=3.0)
    # seg_000+seg_001 same scene but 3.0 gap between 2.0 and 3.0 -> 1.0 gap,
    # below threshold so they fuse; seg_002 is a different scene.
    assert len(events) == 2
    assert events[0]["scene_id"] == "scene-1"
    assert events[-1]["scene_id"] == "scene-2"
    assert events[0]["event_id"].startswith("event_")


def test_movie_analyzer_persists_artifacts(tmp_path):
    (tmp_path / "scenes").mkdir(parents=True)
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "scenes" / "scene_index.json").write_text(
        json.dumps(_scenes()), encoding="utf-8")
    (tmp_path / "transcripts" / "transcript.json").write_text(
        json.dumps({"segments": _segments()}), encoding="utf-8")
    (tmp_path / "project_meta.json").write_text(
        json.dumps({"project_id": "p1", "title": "Coin", "source_path": "x.mp4"}),
        encoding="utf-8")

    idx = MovieAnalyzer().analyze(tmp_path)
    assert (tmp_path / "movie_index.json").exists()
    assert (tmp_path / "semantic_index.json").exists()
    assert idx["project_id"] == "p1"
    assert len(idx["scenes"]) == 2
    assert idx["scenes"][0]["story"]["topics"]
    assert idx["provenance"]["scene_enricher"] == "heuristic"


def test_movie_analyzer_tolerates_missing_artifacts(tmp_path):
    idx = MovieAnalyzer().analyze(tmp_path)
    assert idx["scenes"] == []
    assert idx["characters"] == []
    assert idx["events"] == []