"""Scene facts: a uniform, fact-only view over the Movie Intelligence layer.

The Creative Director must reason from what is *actually present* in the movie.
This module normalizes the several on-disk representations the project can write
(`movie_index.json` with ``scenes[].story`` enrichments, a raw list of enriched
scene dicts, or a `scene_index.json`) into one list of ``SceneFact`` records that
carry only vision/transcript-derived facts.

It is deliberately free of any LLM call. Everything here is deterministic:

- a glance at a scene's facts needed by the context builder,
- the set of characters / locations / objects that are KNOWN to exist, so the
  director can be told "these are the only names you may use",
- a hallucination guard: given a claimed name / location / object, tell the
  caller whether it is grounded in the index or not.
"""
from typing import Any, Dict, Iterable, List, Optional

# Fields each SceneFact exposes. ``story`` is the enriched Qwen3-VL card when
# available; plain `scene_index`-style dicts degrade gracefully to transcript.
_STORY_FIELDS = (
    "summary",
    "topics",
    "dialogue",
    "characters",
    "location",
    "actions",
    "objects",
    "visual_description",
    "visual_events",
    "emotional_cues",
    "themes",
    "mood",
    "cinematography",
)


def _as_list(value: Optional[Any]) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)]


class SceneFact:
    """A single narrative scene with flat, director-facing facts."""

    __slots__ = (
        "scene_id", "start_sec", "end_sec", "transcript",
        "characters", "location", "actions", "objects",
        "visual_description", "visual_events", "emotional_cues",
        "themes", "mood", "cinematography", "dialogue",
    )

    def __init__(self, scene: Dict[str, Any], index: int = 0):
        story = scene.get("story") or {}
        analysis = scene.get("analysis") or {}
        visual = analysis.get("visual") or {}

        self.scene_id = str(scene.get("scene_id") or f"scene-{index}")
        self.start_sec = scene.get("start_sec")
        self.end_sec = scene.get("end_sec")
        self.transcript = scene.get("transcript") or ""

        # Prefer the merged story card; fall back to the split visual analysis.
        self.characters = _as_list(story.get("characters"))
        self.location = story.get("location") or visual.get("location")
        self.actions = _as_list(story.get("actions")) or _as_list(visual.get("actions"))
        self.objects = _as_list(story.get("objects")) or _as_list(visual.get("objects"))
        self.visual_description = (
            story.get("visual_description") or visual.get("visual_description")
        )
        self.visual_events = (
            _as_list(story.get("visual_events")) or _as_list(visual.get("visual_events"))
        )
        self.emotional_cues = (
            _as_list(story.get("emotional_cues")) or _as_list(visual.get("emotional_cues"))
        )
        self.themes = _as_list(story.get("themes")) or _as_list(visual.get("themes"))
        self.mood = story.get("mood") or visual.get("mood")
        self.cinematography = (
            story.get("cinematography") or visual.get("cinematography")
        )
        dialogue = story.get("dialogue") or []
        self.dialogue = [d for d in dialogue if isinstance(d, dict)]

    @property
    def dialogue_text(self) -> str:
        """Single string of all on-screen dialogue for this scene."""
        return " ".join(str(d.get("text", "")) for d in self.dialogue)

    @property
    def has_dialogue(self) -> bool:
        return bool(getattr(self, "dialogue_text", "").strip())

    def fact_text(self) -> str:
        """All facts as one lower-cased blob (for lexical evidence matching)."""
        parts = [
            self.transcript,
            self.dialogue_text,
            self.location or "",
            " ".join(self.characters),
            " ".join(self.actions),
            " ".join(self.objects),
            " ".join(self.visual_events),
            " ".join(self.themes),
            " ".join(self.emotional_cues),
            self.mood or "",
            self.cinematography or "",
            self.visual_description or "",
        ]
        return " ".join(parts)


class SceneFacts:
    """A normalized, fact-only scene index for the director."""

    def __init__(self, scenes: List[SceneFact]):
        self.scenes = scenes

    @classmethod
    def from_movie_intelligence(
        cls,
        movie_index: Optional[Dict[str, Any]] = None,
        scenes: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> "SceneFacts":
        """Load movie intelligence from any supported representation.

        ``movie_index`` may be the full `movie_index.json` (with a ``scenes``
        key) or (via ``scenes=``) a bare list of enriched scene dicts.
        """
        if scenes is None:
            scenes = (movie_index or {}).get("scenes", []) or []
        facts = [
            SceneFact(s, index=i)
            for i, s in enumerate(scenes)
            if isinstance(s, dict)
        ]
        return cls(facts)

    @classmethod
    def from_project_dir(cls, project_dir, bundle_dir: Optional[str] = None) -> "SceneFacts":
        """Load the movie intelligence bundle for a project directory.

        Checks the ``movie_memory/`` bundle first (it is the self-contained
        artifact the analyzer writes), then the project root.
        """
        import json
        from pathlib import Path

        project_dir = Path(project_dir)
        candidates = []
        if bundle_dir:
            candidates.append(project_dir / bundle_dir)
        candidates.append(project_dir / "movie_memory")
        candidates.append(project_dir)

        for base in candidates:
            p = base / "movie_index.json"
            if p.exists():
                idx = json.loads(p.read_text(encoding="utf-8"))
                f = cls.from_movie_intelligence(movie_index=idx)
                if f.scenes:
                    return f
            p = base / "scene_index.json"
            if p.exists():
                scenes = json.loads(p.read_text(encoding="utf-8"))
                f = cls.from_movie_intelligence(scenes=scenes)
                if f.scenes:
                    return f
        return cls([])

    # -- Accessors ---------------------------------------------------------

    def by_id(self, scene_id: str) -> Optional[SceneFact]:
        for s in self.scenes:
            if s.scene_id == scene_id:
                return s
        return None

    def __len__(self) -> int:
        return len(self.scenes)

    def __iter__(self):
        return iter(self.scenes)

    # -- Ground-truth vocabulary (for hallucination prevention) ------------

    def known_characters(self) -> List[str]:
        out, seen = [], set()
        for s in self.scenes:
            for c in s.characters:
                key = str(c).strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    out.append(str(c).strip())
        return out

    def known_locations(self) -> List[str]:
        out, seen = [], set()
        for s in self.scenes:
            for loc in _as_list(s.location):
                key = str(loc).strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    out.append(str(loc).strip())
        return out

    def known_objects(self) -> List[str]:
        out, seen = [], set()
        for s in self.scenes:
            for o in s.objects:
                key = str(o).strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    out.append(str(o).strip())
        return out

    def all_facts_text(self) -> str:
        return " ".join(s.fact_text() for s in self.scenes)

    # -- Hallucination guard ------------------------------------------------

    def is_known_character(self, name: str) -> bool:
        target = str(name).strip().lower()
        return any(target in str(c).lower() for c in self.known_characters())

    def is_known_object(self, term: str) -> bool:
        target = str(term).strip().lower()
        return any(target in str(o).lower() for o in self.known_objects())

    def is_grounded(self, term: str) -> bool:
        """True if ``term`` appears anywhere in the movie's actual facts."""
        if not term:
            return True  # vacuous
        return str(term).strip().lower() in str(self.all_facts_text()).lower()

    def used_scene_ids(self) -> List[str]:
        return [s.scene_id for s in self.scenes]
