"""Semantic scene index.

Retrieval over enriched scenes. Combines:

- semantic similarity (TF-IDF cosine by default; an optional dense embedder can
  be plugged in at ``build()`` time — see ``embedding_retriever.py``),
- lexical transcript relevance (keyword overlap),
- dialogue relevance (query terms spoken on screen).

The lexical path is always available and used as the default so the editor
never silently loses the ability to find evidence when embeddings are
unavailable.
"""
from typing import Callable, Dict, List, Optional

from movie_understanding import text_utils


class SemanticIndex:
    """Builds and queries an index over enriched scenes."""

    def __init__(
        self,
        embedder: Optional[Callable[[List[str]], List[list]]] = None,
    ):
        # embedder(texts) -> list of dense vectors; when provided, dense
        # cosine similarity overrides TF-IDF for the semantic term. Optional.
        self.embedder = embedder
        self._doc_vectors: Optional[List[list]] = None
        self._entries: List[dict] = []
        self._idf: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, enriched_scenes: List[dict],
              embedder: Optional[Callable[[List[str]], List[list]]] = None
              ) -> "SemanticIndex":
        """Index a list of enriched scenes (each has ``story.summary`` etc.).

        When an ``embedder`` is supplied (or was passed to the constructor),
        dense vectors are computed for the whole corpus at build time and
        ``search()`` ranks with cosine similarity against the embedded query.
        TF-IDF remains the default and the always-available fallback.
        """
        if embedder is not None:
            self.embedder = embedder
        texts = self._corpus_texts(enriched_scenes)
        docs = [text_utils.tokenize(t) for t in texts]
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
        self._doc_vectors = None
        if self.embedder is not None:
            # [] marks "embedder supplied but no corpus"; None marks "tfidf".
            self._doc_vectors = list(self.embedder(texts)) if texts else []
        return self

    @staticmethod
    def _corpus_texts(enriched_scenes: List[dict]) -> List[str]:
        texts = []
        for scene in enriched_scenes:
            story = scene.get("story", {})
            texts.append(" ".join([
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
            ]))
        return texts

    @staticmethod
    def _corpus(enriched_scenes: List[dict]) -> List[list]:
        return [
            text_utils.tokenize(t)
            for t in SemanticIndex._corpus_texts(enriched_scenes)
        ]

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

        if self._doc_vectors is not None:
            qv = list(self.embedder([query])[0])
            sem = {}
            for entry, doc_vec in zip(self._entries, self._doc_vectors):
                sem[entry["scene_id"]] = _cosine_list(qv, doc_vec)
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
            "method": "tfidf" if self._doc_vectors is None else "embedder",
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


def _cosine_list(a: list, b: list) -> float:
    if not a or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    den_a = sum(x * x for x in a) ** 0.5 or 1.0
    den_b = sum(y * y for y in b) ** 0.5 or 1.0
    return num / (den_a * den_b)


def _rationale(scene_id: str, sem: float, tr: float, dl: float) -> str:
    parts = []
    if sem > 0.05:
        parts.append(f"semantic={sem:.2f}")
    if tr > 0:
        parts.append(f"transcript={tr:.2f}")
    if dl > 0:
        parts.append(f"dialogue={dl:.2f}")
    return " ".join(parts) or "weak match"
