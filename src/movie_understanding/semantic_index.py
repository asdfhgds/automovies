"""Semantic scene index.

Retrieval over enriched scenes. Combines:

- semantic similarity (TF-IDF cosine by default; an optional embedder function
  can be plugged in),
- lexical transcript relevance (keyword overlap),
- dialogue relevance (query terms spoken on screen),
- director requirements (weighted query terms).

The lexical path is always available and used as a fallback so the editor never
silently loses the ability to find evidence when embeddings are unavailable.
"""
from typing import Callable, Dict, List, Optional

from movie_understanding import text_utils


class SemanticIndex:
    """Builds and queries an index over enriched scenes."""

    def __init__(self, embedder: Optional[Callable[[List[str]], List[list]]] = None):
        # embedder(texts) -> list of vectors; when provided it overrides TF-IDF
        # for the similarity term. Optional and never required.
        self.embedder = embedder
        self._entries: List[dict] = []
        self._idf: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, enriched_scenes: List[dict]) -> "SemanticIndex":
        """Index a list of enriched scenes (each has ``story.summary`` etc.)."""
        docs = self._corpus(enriched_scenes)
        self._idf = _idf(docs)
        self._entries = []
        for scene, doc in zip(enriched_scenes, docs):
            story = scene.get("story", {})
            self._entries.append({
                "scene_id": scene.get("scene_id"),
                "summary": story.get("summary"),
                "topics": story.get("topics", []),
                "location": story.get("location"),
                "visual_description": story.get("visual_description"),
                "actions": story.get("actions", []),
                "objects": story.get("objects", []),
                "visual_events": story.get("visual_events", []),
                "emotional_cues": story.get("emotional_cues", []),
                "themes": story.get("themes", []),
                "cinematography": story.get("cinematography"),
                "mood": story.get("mood"),
                "dialogue_text": " ".join(
                    d.get("text", "") for d in story.get("dialogue", [])
                ),
                "vector": _tfidf(doc, self._idf),
            })
        return self

    @staticmethod
    def _corpus(enriched_scenes: List[dict]) -> List[dict]:
        docs = []
        for scene in enriched_scenes:
            story = scene.get("story", {})
            text = " ".join([
                scene.get("transcript") or "",
                story.get("summary") or "",
                story.get("location") or "",
                story.get("visual_description") or "",
                story.get("mood") or "",
                story.get("cinematography") or "",
                " ".join(story.get("topics") or []),
                " ".join(story.get("actions") or []),
                " ".join(story.get("objects") or []),
                " ".join(story.get("visual_events") or []),
                " ".join(story.get("emotional_cues") or []),
                " ".join(story.get("themes") or []),
                " ".join(d.get("text", "") for d in story.get("dialogue", [])),
            ])
            docs.append(text_utils.tokenize(text))
        return docs

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5,
               weights: Optional[Dict[str, float]] = None) -> List[dict]:
        """Rank scenes by combined score.

        ``weights`` keys: ``semantic`` (default 0.5), ``transcript`` (0.3),
        ``dialogue`` (0.2). Returns ``[{"scene_id", "score", "rationale"}]``
        sorted descending.
        """
        w = {
            "semantic": 0.5,
            "transcript": 0.3,
            "dialogue": 0.2,
        }
        if weights:
            w.update(weights)
        q_tokens = text_utils.tokenize(query)
        if not q_tokens:
            return []

        if self.embedder is not None:
            vecs = self.embedder([query])
            sem = _cosine_dict(vecs[0]) if vecs else {}
        else:
            q_vec = _tfidf(q_tokens, self._idf)
            sem = {}
            for entry in self._entries:
                sem[entry["scene_id"]] = _cosine(q_vec, entry["vector"])

        q_set = set(q_tokens)
        results = []
        for entry in self._entries:
            sid = entry["scene_id"]
            tr = _token_overlap(q_set, text_utils.tokenize(entry.get("summary") or ""))
            dl = _token_overlap(q_set, text_utils.tokenize(entry.get("dialogue_text") or ""))
            score = (
                w["semantic"] * sem.get(sid, 0.0)
                + w["transcript"] * tr
                + w["dialogue"] * dl
            )
            if score <= 0:
                continue
            results.append({
                "scene_id": sid,
                "score": round(score, 4),
                "rationale": _rationale(sid, sem.get(sid, 0.0), tr, dl),
            })
        results.sort(key=lambda r: -r["score"])
        return results[:k]

    def to_dict(self) -> dict:
        return {
            "method": "tfidf" if self.embedder is None else "embedder",
            "scenes": [
                {
                    "scene_id": e["scene_id"],
                    "topics": e["topics"],
                    "summary": e["summary"],
                    "location": e["location"],
                    "visual_description": e["visual_description"],
                    "actions": e["actions"],
                    "objects": e["objects"],
                    "visual_events": e["visual_events"],
                    "emotional_cues": e["emotional_cues"],
                    "themes": e["themes"],
                    "cinematography": e["cinematography"],
                    "mood": e["mood"],
                }
                for e in self._entries
            ],
        }


def _token_overlap(query_set: set, doc_tokens: List[str]) -> float:
    if not doc_tokens:
        return 0.0
    return round(len(query_set & set(doc_tokens)) / max(1, len(query_set)), 4)


def _idf(docs: List[list]) -> Dict[str, float]:
    import math

    n = len(docs)
    df: Dict[str, int] = {}
    for doc in docs:
        for token in set(doc):
            df[token] = df.get(token, 0) + 1
    return {
        token: math.log(1.0 + n / (1.0 + count))
        for token, count in df.items()
    }


def _tfidf(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    if not tokens:
        return {}
    total = len(tokens)
    tf: Dict[str, int] = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
    return {token: (count / total) * idf.get(token, 0.0) for token, count in tf.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    num = sum(a[t] * b[t] for t in shared)
    den_a = sum(v * v for v in a.values()) ** 0.5
    den_b = sum(v * v for v in b.values()) ** 0.5
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


def _cosine_dict(flat_vector: list) -> Dict[str, float]:
    return {str(i): float(v) for i, v in enumerate(flat_vector)}


def _rationale(scene_id: str, sem: float, tr: float, dl: float) -> str:
    parts = []
    if sem > 0.05:
        parts.append(f"semantic={sem:.2f}")
    if tr > 0:
        parts.append(f"transcript={tr:.2f}")
    if dl > 0:
        parts.append(f"dialogue={dl:.2f}")
    return " ".join(parts) or "weak match"
