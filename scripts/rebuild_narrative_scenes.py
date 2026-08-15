#!/usr/bin/env python3
"""Re-run the repaired MovieAnalyzer for a project built before narrative scenes.

The repair model (see ``src/movie_understanding/analyzer.py``) changed how a
movie is represented:

* ``scenes/scene_index.json`` now holds raw PySceneDetect shots and the
  analyzer deterministically groups them into narrative scenes;
* ``transcripts/transcript.json`` supplies dialogue segments whose ``speaker``
  fields drive character extraction (the old transcript-capitalized-words
  heuristic is gone);
* every enriched scene carries ``analysis.transcript`` / ``analysis.visual``,
  exact (never rounded) temporal coordinates, and ``key_frame_times_sec``.

Projects analysed before the repair only keep ``movie_index.json`` (whose
scenes are the raw shots and whose per-scene ``story.dialogue`` is the
transcript).  This script reconstructs the two raw inputs from that file and
re-runs the analyzer so the project carries the narrative-scene artifacts.

Usage::

    python scripts/rebuild_narrative_scenes.py \
        --project data/5398e39c-d35b-481a-b580-42d7224732eb \
        [--no-keyframes] [--max-frames 3] [--out BACKUP_DIR]
"""
import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from movie_understanding.analyzer import MovieAnalyzer  # noqa: E402
from movie_understanding import movie_memory  # noqa: E402


def reconstruct_inputs(project_dir: Path) -> None:
    """Write scenes/scene_index.json + transcripts/transcript.json from movie_index.json."""
    movie_index = movie_memory.load_json(project_dir, "movie_index.json")
    if movie_index is None:
        sys.exit(f"No movie_index.json found under {project_dir}")

    scenes = movie_index.get("scenes") or []
    if not scenes:
        sys.exit(f"{project_dir}/movie_index.json has no scenes")

    raw_shots = [{
        "scene_id": s.get("scene_id") or f"scene-{i}",
        "start_sec": s.get("start_sec", 0.0),
        "end_sec": s.get("end_sec", 0.0),
        "duration_sec": s.get("duration_sec", 0.0),
        "transcript": s.get("transcript", ""),
    } for i, s in enumerate(scenes, start=1)]

    dialogue = []
    for s in scenes:
        for d in (s.get("story", {}).get("dialogue") or []):
            dialogue.append({
                "speaker": d.get("speaker"),
                "text": d.get("text", ""),
                "start_sec": d.get("start_sec", 0.0),
                "end_sec": d.get("end_sec", 0.0),
            })
    dialogue.sort(key=lambda d: (float(d["start_sec"]), float(d["end_sec"])))
    segments = [{
        "id": f"seg_{i:03d}",
        "start_sec": d["start_sec"],
        "end_sec": d["end_sec"],
        "text": d["text"],
        "speaker": d["speaker"],
    } for i, d in enumerate(dialogue)]

    movie_memory.save_json(project_dir, "scenes/scene_index.json", raw_shots)
    movie_memory.save_json(project_dir, "transcripts/transcript.json",
                           {"segments": segments})
    print(f"reconstructed {len(raw_shots)} raw shots -> "
          f"{project_dir}/scenes/scene_index.json")
    print(f"reconstructed {len(segments)} transcript segments -> "
          f"{project_dir}/transcripts/transcript.json")


def write_report(project_dir: Path, movie_index: dict, temporal_probe: dict) -> None:
    scenes = movie_index.get("scenes", [])
    lines = [
        "# Movie Understanding Report (rebuilt narrative scenes)",
        "",
        f"- project: {movie_index.get('project_id')}",
        f"- title: {movie_index.get('movie', {}).get('title')}",
        f"- duration_sec: {movie_index.get('movie', {}).get('duration_sec')}",
        f"- narrative scenes: {len(scenes)}",
        f"- raw shots: {len(movie_index.get('shots', []))}",
        f"- characters: {', '.join(c.get('name', '') for c in movie_index.get('characters', [])) or 'none'}",
        f"- provenance: {json.dumps(movie_index.get('provenance', {}), indent=2)}",
        "",
        "## Narrative scene cards",
        "",
    ]
    for s in scenes:
        analysis = s.get("analysis", {})
        lines += [
            f"### {s.get('scene_id')}  ({s.get('start_sec')}-{s.get('end_sec')}s, "
            f"{s.get('shot_count')} shots)",
            "",
            f"- shots: {', '.join(s.get('shot_ids', []))}",
            f"- characters: {', '.join(s.get('story', {}).get('characters', [])) or 'none'}",
            f"- location: {s.get('story', {}).get('location')}",
            f"- beats: {', '.join(str(b) for b in s.get('story', {}).get('beats', [])) or 'none'}",
            f"- mood: {s.get('story', {}).get('mood')}",
            f"- tone: {s.get('story', {}).get('emotional_tone')}",
            f"- keyframes: {len(s.get('key_frames', []))} "
            f"(times {s.get('key_frame_times_sec')})",
            f"- analysis.transcript characters: "
            f"{analysis.get('transcript', {}).get('characters')}",
            f"- analysis.visual location: {analysis.get('visual', {}).get('location')}",
            f"- transcript: {s.get('transcript', '') or '—'}",
            "",
        ]
    if temporal_probe:
        lines += [
            "## Temporal probe",
            "",
            "```json",
            json.dumps(temporal_probe, indent=2),
            "```",
            "",
        ]
    report_path = project_dir / "reports" / "movie_understanding_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report_path}")


def probe_temporal(movie_index: dict) -> dict:
    """Sanity-check that every reported coordinate is exact (never rounded)."""
    issues = []
    for s in movie_index.get("scenes", []):
        for key in ("start_sec", "end_sec", "duration_sec"):
            if s.get(key) is None or float(s[key]) == round(float(s[key])):
                if not isinstance(s.get(key), float):
                    issues.append(f"{s.get('scene_id')}.{key} = {s.get(key)}")
    for e in movie_index.get("events", []):
        if not isinstance(e.get("start_sec"), float) or not isinstance(e.get("end_sec"), float):
            issues.append(f"event.start_sec/end_sec not float: {e.get('event_id')}")
    return {
        "recorded": True,
        "exact_float_coordinates": len(issues) == 0,
        "issues": issues[:5],
        "scene_count": len(movie_index.get("scenes", [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--no-keyframes", action="store_true",
                        help="skip keyframe extraction (no source video available)")
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None,
                        help="optional backup dir for the old movie_index.json")
    args = parser.parse_args()

    project_dir = args.project.resolve()
    if not project_dir.is_dir():
        sys.exit(f"project dir not found: {project_dir}")

    if args.out is not None:
        backup = Path(args.out)
        backup.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for name in ("movie_index.json", "semantic_index.json", "characters.json",
                     "events.json", "scene_index_v2.json"):
            src = project_dir / name
            if src.is_file():
                shutil.copy2(src, backup / f"{stamp}-{name}")
        print(f"backed up old artifacts -> {backup}")

    reconstruct_inputs(project_dir)

    analyzer = MovieAnalyzer(attach_keyframes=not args.no_keyframes,
                             max_frames=args.max_frames)
    movie_index = analyzer.analyze(project_dir)

    probe = probe_temporal(movie_index)
    write_report(project_dir, movie_index, probe)

    print()
    print(f"narrative scenes: {len(movie_index['scenes'])}")
    print(f"characters: {[c['name'] for c in movie_index['characters']]}")
    print(f"events: {len(movie_index['events'])}")
    print(f"temporal probe exact-float: {probe['exact_float_coordinates']}")
    print(f"movie_index.json written to {project_dir / 'movie_index.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
