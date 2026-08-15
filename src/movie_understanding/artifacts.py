"""Movie intelligence artifacts — the validated scene knowledge the director
consumes.

After ``MovieAnalyzer`` builds the in-memory index this module persists four
things a downstream Creative Director needs:

- ``scene_index_v2.json`` — a versioned, enriched scene index. v3 carries the
  repaired model: the raw ``shots`` collection, deterministic ``scenes``
  (narrative groupings with their ``shot_ids``), per-scene exact
  ``key_frame_times_sec``, the split ``analysis.transcript`` /
  ``analysis.visual`` cards AND the merged per-field-provenance ``story`` view.
- ``movie_memory/`` — a self-contained directory bundle of the movie
  intelligence layer (movie index, scene index v2, semantic index, characters,
  events) so later stages can load one folder.
- ``reports/movie_understanding_report.md`` — a human-readable report of the
  same knowledge, used to audit what the system actually understands.

Nothing here invents content: it copies the fields already produced by the
enricher (heuristic or Qwen3-VL) verbatim and records provenance.
"""
from pathlib import Path
from typing import Optional

from movie_understanding import movie_memory

SCENE_INDEX_V2 = "scene_index_v2.json"
SCENE_INDEX_VERSION = 3
MOVIE_MEMORY_DIR = "movie_memory"
REPORT_PATH = "reports/movie_understanding_report.md"


def _story_card(scene: dict) -> dict:
    story = scene.get("story") or {}
    return {
        "summary": story.get("summary"),
        "topics": story.get("topics") or [],
        "dialogue": story.get("dialogue") or [],
        "characters": story.get("characters") or [],
        "location": story.get("location"),
        "actions": story.get("actions") or [],
        "objects": story.get("objects") or [],
        "visual_description": story.get("visual_description"),
        "visual_events": story.get("visual_events") or [],
        "emotional_cues": story.get("emotional_cues") or [],
        "emotional_tone": story.get("emotional_tone"),
        "themes": story.get("themes") or [],
        "mood": story.get("mood"),
        "cinematography": story.get("cinematography"),
        "confidence": story.get("confidence"),
        "provenance": story.get("provenance") or {},
    }


def _analysis_card(scene: dict) -> dict:
    """Split transcript vs visual analysis halves (verbatim, with provenance)."""
    analysis = scene.get("analysis") or {}
    return {
        "transcript": {
            "summary": analysis.get("transcript", {}).get("summary"),
            "topics": analysis.get("transcript", {}).get("topics") or [],
            "dialogue": analysis.get("transcript", {}).get("dialogue") or [],
            "characters": analysis.get("transcript", {}).get("characters") or [],
            "emotional_tone": analysis.get("transcript", {}).get("emotional_tone"),
            "provenance": analysis.get("transcript", {}).get("provenance") or {},
        },
        "visual": {
            "location": analysis.get("visual", {}).get("location"),
            "actions": analysis.get("visual", {}).get("actions") or [],
            "objects": analysis.get("visual", {}).get("objects") or [],
            "visual_description": analysis.get("visual", {}).get("visual_description"),
            "visual_events": analysis.get("visual", {}).get("visual_events") or [],
            "emotional_cues": analysis.get("visual", {}).get("emotional_cues") or [],
            "themes": analysis.get("visual", {}).get("themes") or [],
            "mood": analysis.get("visual", {}).get("mood"),
            "cinematography": analysis.get("visual", {}).get("cinematography"),
            "confidence": analysis.get("visual", {}).get("confidence"),
            "provenance": analysis.get("visual", {}).get("provenance") or {},
        },
    }


def _scene_index_v2(movie_index: dict) -> dict:
    shots = movie_index.get("shots", [])
    scenes = []
    for scene in movie_index.get("scenes", []):
        scenes.append({
            "scene_id": scene.get("scene_id"),
            "start_sec": scene.get("start_sec"),
            "end_sec": scene.get("end_sec"),
            "duration_sec": scene.get("duration_sec"),
            "transcript": scene.get("transcript"),
            "shot_ids": scene.get("shot_ids") or [],
            "shot_count": scene.get("shot_count"),
            "key_frames": scene.get("key_frames") or [],
            "key_frame_times_sec": scene.get("key_frame_times_sec") or [],
            "analysis": _analysis_card(scene),
            "story": _story_card(scene),
        })
    return {
        "version": SCENE_INDEX_VERSION,
        "project_id": movie_index.get("project_id"),
        "movie": movie_index.get("movie", {}),
        "shots": [
            {
                "shot_id": s.get("shot_id"),
                "start_sec": s.get("start_sec"),
                "end_sec": s.get("end_sec"),
                "transcript": s.get("transcript"),
            }
            for s in shots
        ],
        "provenance": movie_index.get("provenance", {}),
        "scenes": scenes,
    }


def write_scene_index_v2(project_dir: Path, movie_index: dict) -> Path:
    """Persist the versioned enriched scene index to ``scene_index_v2.json``."""
    data = _scene_index_v2(movie_index)
    return movie_memory.save_json(Path(project_dir), SCENE_INDEX_V2, data)


def write_movie_memory_bundle(project_dir: Path, movie_index: dict) -> Path:
    """Write the ``movie_memory/`` bundle (index, scene v2, semantic, chars).

    Returns the directory path.
    """
    project_dir = Path(project_dir)
    mem_dir = project_dir / MOVIE_MEMORY_DIR
    mem_dir.mkdir(parents=True, exist_ok=True)

    movie_memory.save_json(mem_dir, "movie_index.json", movie_index)
    movie_memory.save_json(mem_dir, SCENE_INDEX_V2, _scene_index_v2(movie_index))
    semantic = movie_memory.load_json(project_dir, "semantic_index.json", {})
    if semantic:
        movie_memory.save_json(mem_dir, "semantic_index.json", semantic)
    movie_memory.save_json(mem_dir, "characters.json", movie_index.get("characters", []))
    movie_memory.save_json(mem_dir, "events.json", movie_index.get("events", []))
    movie_memory.save_json(
        mem_dir,
        "manifest.json",
        {
            "scene_index_version": SCENE_INDEX_VERSION,
            "scene_enricher": movie_index.get("provenance", {}).get("scene_enricher"),
            "grouping": movie_index.get("provenance", {}).get("grouping"),
            "keyframes": bool(movie_index.get("provenance", {}).get("keyframes")),
        },
    )
    return mem_dir


def _fmt_line(text: Optional[str], limit: int = 160) -> str:
    if not text:
        return "—"
    text = str(text).strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _fmt_list(items, limit: int = 5) -> str:
    if not items:
        return "—"
    return ", ".join(str(i) for i in items[:limit])


def build_movie_understanding_report(project_dir: Path) -> str:
    """Render the human-readable movie understanding report (markdown)."""
    project_dir = Path(project_dir)
    idx = movie_memory.load_json(project_dir, "movie_index.json", {})
    scenes = idx.get("scenes", [])
    prov = idx.get("provenance", {})
    movie = idx.get("movie", {})

    lines = [
        "# Movie Understanding Report",
        "",
        f"- **Project**: {idx.get('project_id', '?')}",
        f"- **Title**: {movie.get('title', '?')}",
        f"- **Duration (s)**: {movie.get('duration_sec', '?')}",
        f"- **Narrative scenes**: {len(scenes)}",
        f"- **Raw shots**: {len(idx.get('shots', []))}",
        f"- **Grouping**: `{prov.get('grouping', {}).get('method', '?')}`",
        f"- **Scene enricher**: {prov.get('scene_enricher', '?')}",
        f"- **Semantic method**: {prov.get('semantic_method', '?')}",
        f"- **Keyframes attached**: {prov.get('keyframes', False)}",
        f"- **Word-level timestamps**: {prov.get('word_level_timestamps', False)}",
        "",
    ]

    vision_filled = 0
    for i, scene in enumerate(scenes, 1):
        s = scene.get("story") or {}
        analysis = scene.get("analysis") or {}
        sid = scene.get("scene_id", f"scene-{i}")
        lines += [
            f"## {i}. {sid}",
            "",
            f"- **Time**: {_fmt_line(scene.get('start_sec'))} - {_fmt_line(scene.get('end_sec'))} s"
            f" (duration {_fmt_line(scene.get('duration_sec'))})",
            f"- **Shots**: {_fmt_list(scene.get('shot_ids'))}"
            + ("" if scene.get("shot_count") is None
               else f" (n={scene.get('shot_count')})"),
            f"- **Transcript**: {_fmt_line(scene.get('transcript'))}",
            f"- **Characters**: {_fmt_list(s.get('characters'))}",
            f"- **Location**: {_fmt_line(s.get('location'))}",
            f"- **Actions**: {_fmt_list(s.get('actions'))}",
            f"- **Objects**: {_fmt_list(s.get('objects'))}",
            f"- **Visual description**: {_fmt_line(s.get('visual_description'))}",
            f"- **Visual events**: {_fmt_list(s.get('visual_events'))}",
            f"- **Emotional cues**: {_fmt_list(s.get('emotional_cues'))}",
            f"- **Emotional tone**: {_fmt_line(s.get('emotional_tone'))}",
            f"- **Themes**: {_fmt_list(s.get('themes'))}",
            f"- **Mood**: {_fmt_line(s.get('mood'))}",
            f"- **Cinematography**: {_fmt_line(s.get('cinematography'))}",
            f"- **Confidence**: {_fmt_line(s.get('confidence'))}",
            "",
        ]
        visual_prov = (analysis.get("visual") or {}).get("provenance") or {}
        if visual_prov.get("location") == prov.get("scene_enricher"):
            vision_filled += 1

    lines += [
        "## Coverage",
        "",
        f"- Narrative scenes with vision-enriched `analysis.visual` (per-field "
        f"provenance == `{prov.get('scene_enricher', '?')}`): {vision_filled}/{len(scenes)}",
        "",
    ]
    return "\n".join(lines)


def write_movie_understanding_report(project_dir: Path) -> Path:
    """Write the report to ``reports/movie_understanding_report.md``."""
    project_dir = Path(project_dir)
    text = build_movie_understanding_report(project_dir)
    path = project_dir / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
