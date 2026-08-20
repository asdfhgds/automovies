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
import re
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


# -- Canonical vocabulary helpers --------------------------------------------

_EN_ARTICLES = ("a ", "an ", "the ")

# Significant-token filtering for exact (token-level) grounding. A claimed
# value only counts as present if every one of these tokens literally appears
# in the scene's facts — never via arbitrary substring containment.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "by", "for",
    "with", "from", "as", "is", "are", "was", "were", "be", "been", "it", "its",
    "this", "that", "these", "those", "his", "her", "their", "our", "your",
    "you", "they", "we", "he", "she", "him", "them", "his", "her", "not",
    "then", "than", "when", "while", "into", "onto", "about", "through",
    "since", "out", "over", "under", "between", "after", "before", "also",
    "just", "very", "still", "who", "which", "what", "how", "there", "here",
})


def strip_articles(text: str) -> str:
    low = text.lower()
    for art in _EN_ARTICLES:
        if low.startswith(art):
            return text[len(art):]
    return text


def normalize_entity(text: str) -> str:
    """One canonical form for entity matching: lowercase, article-stripped,
    whitespace-collapsed, edge punctuation removed."""
    return " ".join((text or "").lower().split()).strip(" .,;:!?\"'()")


def significant_tokens(text: str) -> List[str]:
    """Content tokens of ``text`` (no stopwords, len > 2) — the tokens used for
    exact, non-substring grounding checks."""
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if len(t) > 2 and t not in _STOPWORDS]


def _scene_id_normalized(value: str) -> str:
    """Normalize a scene id for comparison (scene-10 / scene10 / scene 10)."""
    return re.sub(r"[\s\-_]+", "", (value or "").lower().strip())


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

    def known_actions(self) -> List[str]:
        out, seen = [], set()
        for s in self.scenes:
            for a in s.actions:
                key = str(a).strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    out.append(str(a).strip())
        return out

    def known_themes(self) -> List[str]:
        out, seen = [], set()
        for s in self.scenes:
            for t in s.themes:
                key = str(t).strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    out.append(str(t).strip())
        return out

    def known_moods(self) -> List[str]:
        out, seen = [], set()
        for s in self.scenes:
            for m in _as_list(s.mood):
                key = str(m).strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    out.append(str(m).strip())
        return out

    def known_dialogue(self) -> List[str]:
        out, seen = [], set()
        for s in self.scenes:
            for d in s.dialogue:
                text = str(d.get("text", "")).strip()
                if text and text.lower() not in seen:
                    seen.add(text.lower())
                    out.append(text)
        return out

    # -- Canonical vocabulary (for exact, alias-based grounding) --------------

    def _entity_vocabulary(self, items: List[str], owner_getter) -> List[Dict[str, Any]]:
        """Build a canonical entity vocabulary.

        Each entry: ``{"display", "canonical", "aliases", "scenes"}`` where
        ``aliases`` are the canonical form plus its significant content tokens
        (so "counter" resolves to the canonical "counter with various items"),
        and ``scenes`` lists every scene a viewer sees that entity in.
        """
        vocab: List[Dict[str, Any]] = []
        seen_canon = set()
        for item in items:
            display = str(item).strip()
            canonical = normalize_entity(strip_articles(display)) or display.lower()
            if canonical in seen_canon:
                continue
            seen_canon.add(canonical)
            aliases = [a for a in {canonical} | set(significant_tokens(display)) if a]
            vocab.append({
                "display": display,
                "canonical": canonical,
                "aliases": sorted(aliases),
                "scenes": [],
            })
        for scene in self.scenes:
            for owner in owner_getter(scene):
                canon = normalize_entity(strip_articles(str(owner)))
                for entry in vocab:
                    if entry["canonical"] == canon and scene.scene_id not in entry["scenes"]:
                        entry["scenes"].append(scene.scene_id)
        return vocab

    def character_vocabulary(self) -> List[Dict[str, Any]]:
        """Canonical character vocabulary with scene membership."""
        return self._entity_vocabulary(self.known_characters(), lambda s: s.characters)

    def location_vocabulary(self) -> List[Dict[str, Any]]:
        """Canonical location vocabulary with scene membership."""
        return self._entity_vocabulary(self.known_locations(), lambda s: _as_list(s.location))

    def object_vocabulary(self) -> List[Dict[str, Any]]:
        """Canonical object vocabulary with scene membership."""
        return self._entity_vocabulary(self.known_objects(), lambda s: s.objects)

    def all_facts_text(self) -> str:
        return " ".join(s.fact_text() for s in self.scenes)

    # -- Hallucination guard ------------------------------------------------

    @staticmethod
    def _all_significant_tokens_present(target: str, corpus: str) -> bool:
        tokens = significant_tokens(target)
        if not tokens:
            return False
        body = set(significant_tokens(corpus))
        return all(t in body for t in tokens)

    def is_known_character(self, name: str) -> bool:
        target = str(name).strip()
        canon = normalize_entity(strip_articles(target))
        vocab = self.character_vocabulary()
        if any(canon in entry["aliases"] for entry in vocab):
            return True
        return self._all_significant_tokens_present(
            target, " ".join(self.known_characters()))

    def is_known_object(self, term: str) -> bool:
        target = str(term).strip()
        canon = normalize_entity(strip_articles(target))
        vocab = self.object_vocabulary()
        if any(canon in entry["aliases"] for entry in vocab):
            return True
        return self._all_significant_tokens_present(
            target, " ".join(self.known_objects()))

    def is_known_location(self, term: str) -> bool:
        target = str(term).strip()
        canon = normalize_entity(strip_articles(target))
        vocab = self.location_vocabulary()
        if any(canon in entry["aliases"] for entry in vocab):
            return True
        return self._all_significant_tokens_present(
            target, " ".join(self.known_locations()))

    def is_grounded(self, term: str) -> bool:
        """True if every significant token of ``term`` literally appears in the
        movie's actual facts (exact token containment, never substring)."""
        if not term:
            return True  # vacuous
        return self._all_significant_tokens_present(term, self.all_facts_text())

    def used_scene_ids(self) -> List[str]:
        return [s.scene_id for s in self.scenes]
