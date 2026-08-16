"""MovieAnalyzer — transforms raw artifacts into a structured movie index.

Pipeline stage input:  ``transcripts/transcript.json`` + ``scenes/scene_index.json``
Pipeline stage output: ``movie_index.json`` + ``semantic_index.json``

Repair model:

- ``scenes/scene_index.json`` holds the raw PySceneDetect **shots**; the
  analyzer deterministically groups them into **narrative scenes** (see
  :mod:`movie_understanding.grouping`). ``movie_index.shots`` keeps the raw
  shot list; ``movie_index.scenes`` holds the narrative scenes (each carrying
  its ``shot_ids`` / inline ``shots``).
- every enriched scene carries ``key_frames`` AND ``key_frame_times_sec`` (the
  exact absolute capture coordinates) when keyframe extraction runs, plus
  ``analysis.transcript`` / ``analysis.visual`` alongside the merged ``story``.
- temporal coordinates are exact floats, never rounded.

``movie_index.json``::

    {
      "project_id", "source_path", "movie": {"title", "duration_sec"},
      "shots": [...], "scenes": [ {scene_id, start_sec, end_sec, transcript,
                                    shot_ids, shots, key_frames,
                                    key_frame_times_sec, analysis, story} ],
      "characters": [...], "events": [...],
      "provenance": {"scene_enricher", "grouping", "semantic_method",
                     "word_level_timestamps", "keyframes"}
    }
"""
from pathlib import Path
from typing import Dict, List, Optional

from movie_understanding import movie_memory
from movie_understanding.character_analyzer import build_character_index
from movie_understanding.enrich_factory import create_scene_enricher_from_env
from movie_understanding.event_index import build_event_index
from movie_understanding.grouping import (
    DEFAULT_GAP_THRESHOLD_SEC,
    describe_grouping,
    group_shots_into_narrative_scenes,
    shots_from_scene_index,
)
from movie_understanding.scene_analyzer import SceneEnricher
from movie_understanding.semantic_index import SemanticIndex

DEFAULT_GROUP_MAX_SCENE_SEC = 30.0
DEFAULT_GROUP_MAX_SHOTS = 12


def _load_scene_index(project_dir: Path) -> List[dict]:
    scene_index = movie_memory.load_json(project_dir, "scenes/scene_index.json")
    if scene_index is None:
        scene_index = movie_memory.load_json(project_dir, "scenes/scene_cards.json")
    if scene_index is None:
        return []
    if isinstance(scene_index, dict):
        scene_index = scene_index.get("scenes", []) or scene_index.get("shots", [])
    return scene_index if isinstance(scene_index, list) else []


def _load_transcript_segments(project_dir: Path) -> List[dict]:
    transcript = movie_memory.load_json(project_dir, "transcripts/transcript.json", {})
    return transcript.get("segments", []) if isinstance(transcript, dict) else []


class MovieAnalyzer:
    """Builds the structured movie index from existing pipeline artifacts."""

    def __init__(self, scene_enricher: Optional[SceneEnricher] = None,
                 embedder=None, attach_keyframes: bool = False,
                 max_frames: int = 1,
                 group_max_scene_sec: float = DEFAULT_GROUP_MAX_SCENE_SEC,
                 group_max_shots: int = DEFAULT_GROUP_MAX_SHOTS,
                 group_gap_threshold_sec: Optional[float] = DEFAULT_GAP_THRESHOLD_SEC):
        self.scene_enricher = scene_enricher or create_scene_enricher_from_env()
        self.embedder = embedder
        self.attach_keyframes = attach_keyframes
        self.max_frames = max_frames
        self.group_max_scene_sec = group_max_scene_sec
        self.group_max_shots = group_max_shots
        self.group_gap_threshold_sec = group_gap_threshold_sec

    def analyze(self, project_dir: Path) -> dict:
        project_dir = Path(project_dir)
        meta = movie_memory.load_json(project_dir, "project_meta.json", {})
        raw_scenes = _load_scene_index(project_dir)
        segments = _load_transcript_segments(project_dir)

        pre_grouped = [s for s in raw_scenes
                       if isinstance(s, dict) and (s.get("shot_ids") or s.get("shots"))]
        if len(pre_grouped) == len(raw_scenes) and pre_grouped:
            # Idempotent re-run: the input is already a set of narrative scenes
            # (e.g. a previous repaired movie_index). Use them and rebuild the
            # raw shot collection from their inline members.
            scenes = [dict(s) for s in pre_grouped]
            shots = _collect_shots(scenes)
        else:
            shots = shots_from_scene_index(raw_scenes)
            scenes = group_shots_into_narrative_scenes(
                shots,
                max_scene_sec=self.group_max_scene_sec,
                max_shots=self.group_max_shots,
                gap_threshold_sec=self.group_gap_threshold_sec,
            )

        if self.attach_keyframes and scenes:
            from movie_understanding.keyframes import extract_all_scene_keyframes_with_times
            from movie_understanding.vision_enricher import attach_keyframes_to_scenes
            kf_dir = project_dir / "scenes" / "keyframes"
            attach_keyframes_to_scenes(
                scenes, meta.get("source_path"), kf_dir,
                max_frames=self.max_frames,
            )

        enriched = [
            self.scene_enricher.enrich(scene, segments)
            for scene in scenes
        ]

        semantic = SemanticIndex(self.embedder)
        semantic.build(enriched)

        characters = build_character_index(scenes, segments)
        events = build_event_index(scenes, segments)
        duration = max(
            [float(s.get("end_sec", 0.0)) for s in scenes] + [0.0]
        ) if scenes else 0.0

        movie_index = {
            "project_id": meta.get("project_id") or project_dir.name,
            "source_path": meta.get("source_path"),
            "movie": {
                "title": meta.get("title") or project_dir.name,
                "duration_sec": duration,
            },
            "shots": shots,
            "scenes": enriched,
            "characters": characters,
            "events": events,
            "scene_character_map": _scene_character_map(enriched),
            "provenance": {
                "scene_enricher": self.scene_enricher.name,
                "grouping": describe_grouping(
                    max_scene_sec=self.group_max_scene_sec,
                    max_shots=self.group_max_shots,
                    gap_threshold_sec=self.group_gap_threshold_sec,
                ),
                "semantic_method": "tfidf" if self.embedder is None else "embedder",
                "word_level_timestamps": _has_word_timestamps(segments),
                "keyframes": bool(self.attach_keyframes),
            },
        }

        movie_memory.save_movie_index(project_dir, movie_index)
        movie_memory.save_semantic_index(project_dir, semantic.to_dict())
        _write_artifacts(project_dir, movie_index)

        # Hand VRAM back to the next stage (director LLM, TTS, editorial). The
        # vision model rides a class-level cache that otherwise stays resident
        # and OOMs a 16GB T4 when a later stage loads its own model.
        _release_enricher(self.scene_enricher)
        return movie_index


def _write_artifacts(project_dir: Path, movie_index: dict) -> None:
    """Persist the derived scene-index v2 + movie_memory bundle.

    Runs after the canonical movie_index.json / semantic_index.json so the
    project always carries the versioned, director-facing artifacts too.
    """
    from movie_understanding.artifacts import (
        write_movie_memory_bundle,
        write_scene_index_v2,
    )

    write_scene_index_v2(project_dir, movie_index)
    write_movie_memory_bundle(project_dir, movie_index)


def _scene_character_map(enriched: List[dict]) -> Dict[str, List[str]]:
    return {
        e["scene_id"]: e["story"]["characters"]
        for e in enriched
    }


def _collect_shots(scenes: List[dict]) -> List[dict]:
    """Rebuild the raw shot collection from pre-grouped narrative scenes."""
    out: List[dict] = []
    seen: set = set()
    for scene in scenes:
        by_id = {str(s.get("shot_id")): s for s in scene.get("shots") or []}
        for sid in scene.get("shot_ids", []):
            if sid in seen:
                continue
            shot = by_id.get(str(sid))
            if shot is None:
                continue
            seen.add(sid)
            out.append(shot)
    return out


def _has_word_timestamps(segments: List[dict]) -> bool:
    return any(seg.get("words") for seg in (segments or []))


def _release_enricher(enricher: Optional[SceneEnricher]) -> None:
    """Release GPU memory held by a scene enricher, if it supports it."""
    release = getattr(enricher, "release", None)
    if callable(release):
        try:
            release()
        except Exception:
            pass


def build_movie_index(project_dir: Path, scene_enricher: Optional[SceneEnricher] = None,
                      embedder=None, attach_keyframes: bool = False,
                      max_frames: int = 1) -> dict:
    """Convenience entry point (mirrors other pipeline stage entry points)."""
    return MovieAnalyzer(
        scene_enricher=scene_enricher,
        embedder=embedder,
        attach_keyframes=attach_keyframes,
        max_frames=max_frames,
    ).analyze(project_dir)