"""Tests for the movie intelligence artifacts (scene_index_v2.json,
movie_memory/ bundle, movie_understanding_report.md) and for the semantic
index consuming vision fields."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_understanding import artifacts, movie_memory
from movie_understanding.analyzer import MovieAnalyzer
from movie_understanding.semantic_index import SemanticIndex


def _scenes():
    return [
        {"scene_id": "scene-1", "start_sec": 0.0, "end_sec": 6.0,
         "duration": 6.0, "transcript": "Sam spins the coin on the bar."},
        {"scene_id": "scene-2", "start_sec": 10.0, "end_sec": 16.0,
         "duration": 6.0, "transcript": "Rosa shows him the trick."},
    ]


def _segments():
    return [
        {"id": "seg_000", "start_sec": 0.0, "end_sec": 2.0,
         "text": "The coin spins and Sam watches it fall."},
        {"id": "seg_001", "start_sec": 3.0, "end_sec": 5.0,
         "text": "Sam believes fate controls everything."},
    ]


def _project(tmp_path):
    (tmp_path / "scenes").mkdir(parents=True)
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "scenes" / "scene_index.json").write_text(
        json.dumps(_scenes()), encoding="utf-8")
    (tmp_path / "transcripts" / "transcript.json").write_text(
        json.dumps({"segments": _segments()}), encoding="utf-8")
    (tmp_path / "project_meta.json").write_text(
        json.dumps({"project_id": "p1", "title": "Coin", "source_path": "x.mp4"}),
        encoding="utf-8")
    return tmp_path


def _vision_scene():
    return {
        "scene_id": "scene-1",
        "start_sec": 0.0,
        "end_sec": 6.0,
        "duration_sec": 6.0,
        "transcript": "Sam spins the coin on the bar.",
        "shot_ids": ["shot-1"],
        "shot_count": 1,
        "key_frames": ["/tmp/k1.jpg"],
        "key_frame_times_sec": [1.2345678],
        "analysis": {
            "transcript": {
                "summary": "Sam spins a coin at a bar.",
                "topics": ["coin", "fate"],
                "dialogue": [{"speaker": "Sam", "text": "fate controls everything",
                              "start_sec": 3.0, "end_sec": 5.0}],
                "characters": ["Sam"],
                "emotional_tone": "mystery",
                "provenance": {
                    "summary": "transcript", "topics": "transcript_frequency",
                    "dialogue": "transcript_alignment",
                    "characters": "diarization_speaker_labels",
                    "emotional_tone": "transcript_lexicon",
                },
            },
            "visual": {
                "location": "dim bar at night",
                "actions": ["spins coin"],
                "objects": ["coin", "counter", "shot glass"],
                "visual_description": "A coin spins on a wooden bar under warm light.",
                "visual_events": ["coin flip at ~1s", "Sam leans in at ~3s"],
                "emotional_cues": ["furrowed brow", "hushed tone"],
                "themes": ["fate", "control"],
                "mood": "tense",
                "cinematography": "close-up, shallow depth of field",
                "confidence": 0.9,
                "provenance": {
                    "location": "qwen3vl", "actions": "qwen3vl",
                    "objects": "qwen3vl", "visual_description": "qwen3vl",
                    "visual_events": "qwen3vl", "emotional_cues": "qwen3vl",
                    "themes": "qwen3vl", "mood": "qwen3vl",
                    "cinematography": "qwen3vl", "confidence": "qwen3vl",
                },
            },
        },
        "story": {
            "summary": "Sam spins a coin at a bar.",
            "topics": ["coin", "fate"],
            "dialogue": [{"text": "fate controls everything", "start_sec": 3.0, "end_sec": 5.0}],
            "characters": ["Sam"],
            "location": "dim bar at night",
            "actions": ["spins coin"],
            "objects": ["coin", "counter", "shot glass"],
            "visual_description": "A coin spins on a wooden bar under warm light.",
            "visual_events": ["coin flip at ~1s", "Sam leans in at ~3s"],
            "emotional_cues": ["furrowed brow", "hushed tone"],
            "emotional_tone": "mystery",
            "themes": ["fate", "control"],
            "mood": "tense",
            "cinematography": "close-up, shallow depth of field",
            "confidence": 0.9,
            "provenance": {
                "summary": "transcript", "topics": "transcript_frequency",
                "dialogue": "transcript_alignment",
                "characters": "diarization_speaker_labels",
                "location": "qwen3vl", "actions": "qwen3vl",
                "objects": "qwen3vl", "visual_description": "qwen3vl",
                "visual_events": "qwen3vl", "emotional_cues": "qwen3vl",
                "emotional_tone": "transcript_lexicon",
                "themes": "qwen3vl", "mood": "qwen3vl",
                "cinematography": "qwen3vl", "confidence": "qwen3vl",
            },
        },
    }


def _shot():
    return {"shot_id": "shot-1", "start_sec": 0.0, "end_sec": 6.0,
            "transcript": "Sam spins the coin on the bar."}


# ---------------------------------------------------------------------------
# scene_index_v2.json
# ---------------------------------------------------------------------------


def test_write_scene_index_v2(tmp_path):
    movie_index = {"project_id": "p1", "movie": {"title": "Coin", "duration_sec": 16.0},
                   "provenance": {"scene_enricher": "qwen3vl"},
                   "shots": [_shot()],
                   "scenes": [_vision_scene()]}
    path = artifacts.write_scene_index_v2(tmp_path, movie_index)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 3
    assert data["shots"][0]["shot_id"] == "shot-1"
    scene = data["scenes"][0]
    assert scene["scene_id"] == "scene-1"
    assert scene["shot_ids"] == ["shot-1"]
    assert scene["key_frame_times_sec"] == [1.2345678]
    card = scene["story"]
    assert card["location"] == "dim bar at night"
    assert card["objects"] == ["coin", "counter", "shot glass"]
    assert card["visual_events"][0].startswith("coin flip")
    assert card["confidence"] == 0.9
    assert card["provenance"]["location"] == "qwen3vl"
    # analysis halves are persisted too
    assert data["scenes"][0]["analysis"]["transcript"]["characters"] == ["Sam"]
    assert data["scenes"][0]["analysis"]["visual"]["location"] == "dim bar at night"


# ---------------------------------------------------------------------------
# movie_memory/ bundle
# ---------------------------------------------------------------------------


def test_write_movie_memory_bundle(tmp_path):
    movie_index = {"project_id": "p1", "movie": {"title": "Coin", "duration_sec": 16.0},
                   "provenance": {"scene_enricher": "qwen3vl"},
                   "scenes": [_vision_scene()],
                   "characters": [{"name": "Sam"}],
                   "events": [{"event_id": "event_000"}]}
    mem_dir = artifacts.write_movie_memory_bundle(tmp_path, movie_index)
    assert mem_dir.is_dir()
    assert (mem_dir / "movie_index.json").exists()
    assert (mem_dir / "scene_index_v2.json").exists()
    assert (mem_dir / "characters.json").exists()
    assert (mem_dir / "events.json").exists()
    manifest = json.loads((mem_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scene_enricher"] == "qwen3vl"
    assert manifest["scene_index_version"] == 3


def test_write_movie_memory_bundle_with_semantic(tmp_path):
    movie_index = {"project_id": "p1", "movie": {}, "provenance": {},
                   "scenes": [_vision_scene()], "characters": [], "events": []}
    semantic = SemanticIndex().build(movie_index["scenes"]).to_dict()
    movie_memory.save_semantic_index(tmp_path, semantic)
    mem_dir = artifacts.write_movie_memory_bundle(tmp_path, movie_index)
    assert (mem_dir / "semantic_index.json").exists()


# ---------------------------------------------------------------------------
# movie_understanding_report.md
# ---------------------------------------------------------------------------


def test_write_movie_understanding_report(tmp_path):
    movie_index = {"project_id": "p1", "movie": {"title": "Coin", "duration_sec": 16.0},
                   "provenance": {"scene_enricher": "qwen3vl"},
                   "shots": [_shot()],
                   "scenes": [_vision_scene()]}
    movie_memory.save_movie_index(tmp_path, movie_index)
    path = artifacts.write_movie_understanding_report(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "Movie Understanding Report" in text
    assert "Narrative scenes" in text
    assert "dim bar at night" in text
    assert "coin, counter, shot glass" in text
    assert "Confidence**: 0.9" in text


# ---------------------------------------------------------------------------
# Analyzer emits the new artifacts end-to-end
# ---------------------------------------------------------------------------


def test_movie_analyzer_emits_scene_v2_and_memory(tmp_path):
    proj = _project(tmp_path)
    MovieAnalyzer().analyze(proj)
    assert (proj / "scene_index_v2.json").exists()
    assert (proj / "movie_memory").is_dir()
    assert (proj / "movie_memory" / "scene_index_v2.json").exists()
    assert (proj / "movie_memory" / "semantic_index.json").exists()
    v2 = json.loads((proj / "scene_index_v2.json").read_text(encoding="utf-8"))
    assert v2["version"] == 3
    assert len(v2["scenes"]) == 2
    assert v2["scenes"][0]["story"]["location"] is None  # heuristic -> honest None
    assert v2["scenes"][0]["analysis"]["transcript"]["characters"] == []
    assert v2["scenes"][0]["analysis"]["visual"]["location"] is None
    assert v2["scenes"][0]["shot_count"] == 1
    assert v2["scenes"][0]["shot_ids"] == ["shot-1"]
    assert [s["shot_id"] for s in v2["shots"]] == ["shot-1", "shot-2"]


# ---------------------------------------------------------------------------
# Semantic index consumes vision fields
# ---------------------------------------------------------------------------


def test_semantic_index_retrieves_by_vision_field():
    index = SemanticIndex().build([_vision_scene()])
    hits = index.search("where an important object is emphasized close-up", k=3)
    # "coin" object + "close-up" cinematography both surface in the corpus
    assert hits and hits[0]["scene_id"] == "scene-1"


def test_semantic_index_retrieves_by_visual_event():
    index = SemanticIndex().build([_vision_scene()])
    hits = index.search("coin flip event", k=3)
    assert hits and hits[0]["scene_id"] == "scene-1"


def test_semantic_index_to_dict_keeps_vision_fields():
    index = SemanticIndex().build([_vision_scene()])
    d = index.to_dict()
    assert d["scenes"][0]["objects"] == ["coin", "counter", "shot glass"]
    assert d["scenes"][0]["cinematography"] == "close-up, shallow depth of field"


# ---------------------------------------------------------------------------
# Retrieval evaluation harness (scripts/evaluate_retrieval.py)
# ---------------------------------------------------------------------------


def _write_project_for_eval(tmp_path):
    scenes = [
        _vision_scene(),
        {
            "scene_id": "scene-2",
            "start_sec": 10.0,
            "end_sec": 16.0,
            "duration_sec": 6.0,
            "transcript": "Rosa shows Sam the coin trick.",
            "story": {
                "summary": "Rosa demonstrates the coin trick.",
                "topics": ["trick"],
                "dialogue": [],
                "characters": ["Rosa", "Sam"],
                "location": "street at noon",
                "actions": ["shows trick"],
                "objects": ["coin"],
                "visual_description": "Two people outside in daylight.",
                "visual_events": ["trick revealed at ~2s"],
                "emotional_cues": ["laughing"],
                "themes": ["revelation"],
                "mood": "bright",
                "cinematography": "medium two-shot",
                "confidence": 0.8,
                "provenance": {},
            },
        },
    ]
    movie_index = {"project_id": "p1", "movie": {"title": "Coin", "duration_sec": 16.0},
                   "provenance": {"scene_enricher": "qwen3vl"},
                   "scenes": scenes}
    movie_memory.save_movie_index(tmp_path, movie_index)
    movie_memory.save_semantic_index(tmp_path, SemanticIndex().build(scenes).to_dict())
    return tmp_path


def test_eval_harness_writes_reports(tmp_path):
    import subprocess
    import sys

    proj = _write_project_for_eval(tmp_path)
    script = str(Path(__file__).resolve().parents[1] / "scripts" / "evaluate_retrieval.py")
    res = subprocess.run(
        [sys.executable, script, "--project", str(proj), "--k", "3"],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    json_recs = json.loads(
        (proj / "reports" / "retrieval_evaluation.json").read_text(encoding="utf-8"))
    assert "queries" in json_recs and len(json_recs["queries"]) > 0
    first = json_recs["queries"][0]
    for key in ("query", "top_scene_ids", "timestamps", "scores",
                "model_reason", "human_assessment", "human_notes"):
        assert key in first

    md = (proj / "reports" / "retrieval_evaluation.md").read_text(encoding="utf-8")
    assert "GOOD" in md and "Retrieval Evaluation" in md


def test_eval_harness_custom_queries(tmp_path):
    import subprocess
    import sys

    proj = _write_project_for_eval(tmp_path)
    qfile = tmp_path / "queries.json"
    qfile.write_text(json.dumps(["coin object emphasized"]), encoding="utf-8")
    script = str(Path(__file__).resolve().parents[1] / "scripts" / "evaluate_retrieval.py")
    res = subprocess.run(
        [sys.executable, script, "--project", str(proj), "--queries", str(qfile)],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    recs = json.loads(
        (proj / "reports" / "retrieval_evaluation.json").read_text(encoding="utf-8"))
    assert len(recs["queries"]) == 1
    assert recs["queries"][0]["query"] == "coin object emphasized"
