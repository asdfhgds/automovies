"""Shared fixtures for editorial tests: a small deterministic movie index,
director plan, and helper to fabricate excerpt clips into a project dir."""
import json
from pathlib import Path

SEGMENTS = [
    {"id": "seg_000", "start_sec": 0.0, "end_sec": 2.0,
     "text": "The coin spins and Sam watches it fall."},
    {"id": "seg_001", "start_sec": 3.0, "end_sec": 5.0,
     "text": "Sam believes fate controls everything."},
    {"id": "seg_002", "start_sec": 12.0, "end_sec": 14.0,
     "text": "But Rosa laughs and shows him the trick."},
    {"id": "seg_003", "start_sec": 20.0, "end_sec": 22.0,
     "text": "Chance only looks like choice."},
]

SCENES = [
    {"scene_id": "scene-1", "start_sec": 0.0, "end_sec": 6.0, "duration": 6.0,
     "transcript": "The coin spins and Sam watches it fall. Sam believes fate "
                   "controls everything."},
    {"scene_id": "scene-2", "start_sec": 10.0, "end_sec": 16.0, "duration": 6.0,
     "transcript": "But Rosa laughs and shows him the trick."},
    {"scene_id": "scene-3", "start_sec": 18.0, "end_sec": 24.0, "duration": 6.0,
     "transcript": "Chance only looks like choice. Sam sees the pattern."},
]


def make_movie_index() -> dict:
    """Build a movie_index dict using the real heuristic enrichment."""
    from movie_understanding.analyzer import MovieAnalyzer
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "scenes").mkdir()
        (tmp / "transcripts").mkdir()
        (tmp / "scenes" / "scene_index.json").write_text(
            json.dumps(SCENES), encoding="utf-8")
        (tmp / "transcripts" / "transcript.json").write_text(
            json.dumps({"segments": SEGMENTS}), encoding="utf-8")
        (tmp / "project_meta.json").write_text(
            json.dumps({"project_id": "t", "title": "Coinflip",
                        "source_path": "movie.mp4"}), encoding="utf-8")
        return MovieAnalyzer().analyze(tmp)


DIRECTOR_PLAN = {
    "project_id": "t",
    "title": "The Coin Chooses",
    "thesis": "The film argues that the belief in fate is a way to escape "
              "responsibility for the choices people have actually made.",
    "hook": "A coin flip is a choice pretending to be an accident.",
    "director_provider": "mock",
    "creative_task": "Find a subtle philosophical idea and build a 60-120s visual essay.",
}


def seed_project(project_dir: Path, clip_count: int = 4) -> Path:
    """Write movie_index.json + director_plan.json + voice.wav + fabricate
    excerpt clip files (empty placeholders are fine for command-build tests;
    really rendering requires real clips, covered by the e2e test)."""
    project_dir = Path(project_dir)
    (project_dir / "scenes").mkdir(parents=True, exist_ok=True)
    (project_dir / "transcripts").mkdir()
    (project_dir / "audio").mkdir()
    (project_dir / "renders").mkdir()
    (project_dir / "assets").mkdir()

    (project_dir / "scenes" / "scene_index.json").write_text(
        json.dumps(SCENES), encoding="utf-8")
    (project_dir / "transcripts" / "transcript.json").write_text(
        json.dumps({"segments": SEGMENTS}), encoding="utf-8")
    (project_dir / "project_meta.json").write_text(
        json.dumps({"project_id": "t", "title": "Coinflip",
                    "source_path": "movie.mp4"}), encoding="utf-8")

    idx = make_movie_index()
    (project_dir / "movie_index.json").write_text(
        json.dumps(idx), encoding="utf-8")
    (project_dir / "director_plan.json").write_text(
        json.dumps(DIRECTOR_PLAN), encoding="utf-8")

    (project_dir / "audio" / "voice.wav").write_bytes(b"RIFF-fake-wav")
    return project_dir