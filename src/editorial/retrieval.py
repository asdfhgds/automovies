"""Evidence retrieval for the editorial director.

Turns the director's argument needs into concrete *movie excerpts*: a short
sub-window inside a scene. Retrieval blends semantic similarity (the movie's
semantic index) with lexical transcript relevance and scene metadata, then
chooses a dialogue-anchored excerpt window so the edit can show exactly the
moments the narration refers to.
"""
from typing import Dict, List, Optional

from editorial.plan import (
    MAX_EXCERPT_SEC,
    MAX_EXCERPTS_PER_SEGMENT,
    MIN_EXCERPT_SEC,
    EditorialEvidence,
)
from movie_understanding.semantic_index import SemanticIndex
from movie_understanding import text_utils


class EvidenceRetriever:
    def __init__(self, semantic_index: Optional[SemanticIndex] = None,
                 movie_index: Optional[dict] = None):
        self.semantic_index = semantic_index
        self.movie_index = movie_index or {}

    @staticmethod
    def from_project_dicts(movie_index: dict, semantic_index_dict: Optional[dict] = None) -> "EvidenceRetriever":
        """Rehydrate from persisted ``movie_index.json`` / ``semantic_index.json``."""
        index = SemanticIndex()
        index.build(movie_index.get("scenes", []))
        return EvidenceRetriever(semantic_index=index, movie_index=movie_index)

    def _scene(self, scene_id: str) -> Optional[dict]:
        for scene in self.movie_index.get("scenes", []):
            if scene.get("scene_id") == scene_id:
                return scene
        return None

    def retrieve(self, query: str, k: int = MAX_EXCERPTS_PER_SEGMENT,
                 exclude: Optional[List[str]] = None,
                 min_excerpt_sec: float = MIN_EXCERPT_SEC,
                 max_excerpt_sec: float = MAX_EXCERPT_SEC) -> List[EditorialEvidence]:
        """Return the top evidence excerpts for one argumentative query.

        Guarantees a result whenever the movie has any usable scenes: semantic
        ranking first, then a lexical overlap fallback, then a weakest-match
        pool. The editor never silently loses the ability to show evidence.
        """
        exclude = set(exclude or [])
        scored = self.semantic_index.search(query, k=len(self.movie_index.get("scenes", [])))
        picked = self._pick_from(scored, exclude, query, k, min_excerpt_sec, max_excerpt_sec)
        if picked:
            return picked

        # Lexical fallback: rank every scene by query-token overlap.
        q_set = set(text_utils.tokenize(query))
        lexical = []
        for scene in self.movie_index.get("scenes", []):
            sid = scene.get("scene_id")
            if sid in exclude:
                continue
            story = scene.get("story", {})
            text = " ".join([
                scene.get("transcript") or "",
                story.get("summary") or "",
                " ".join(story.get("topics") or []),
            ])
            tokens = set(text_utils.tokenize(text))
            overlap = len(q_set & tokens) / max(1, len(q_set))
            lexical.append({"scene_id": sid, "score": overlap})
        lexical.sort(key=lambda r: -r["score"])
        picked = self._pick_from(lexical, exclude, query, k, min_excerpt_sec, max_excerpt_sec,
                                 fallback_label="lexical fallback")
        if picked:
            return picked

        # Weakest-match pool: any remaining scene, so the editor always has
        # footage to cut against (director requirement #8: no empty segments).
        pool = [{"scene_id": s.get("scene_id"), "score": 0.0}
                for s in self.movie_index.get("scenes", [])
                if s.get("scene_id") not in exclude]
        return self._pick_from(pool, exclude, query, k, min_excerpt_sec, max_excerpt_sec,
                               fallback_label="weakest-match pool")

    def _pick_from(self, scored, exclude, query, k, min_sec, max_sec,
                   fallback_label: Optional[str] = None) -> List[EditorialEvidence]:
        picked = []
        for hit in scored:
            sid = hit["scene_id"]
            if sid in exclude:
                continue
            scene = self._scene(sid)
            if scene is None:
                continue
            window = self._excerpt_window(scene, min_sec, max_sec)
            if window is None:
                continue
            rationale = hit.get("rationale") or (fallback_label or "match")
            picked.append(EditorialEvidence(
                scene_id=sid,
                start_sec=window[0],
                end_sec=window[1],
                reason=f"{rationale} (for: {query[:80]})",
            ))
            if len(picked) >= k:
                break
        return picked

    def _excerpt_window(self, scene: dict, min_sec: float, max_sec: float):
        start = float(scene.get("start_sec", 0.0))
        end = float(scene.get("end_sec", 0.0))
        if end - start < min_sec:
            return None
        dialogue = scene.get("story", {}).get("dialogue", [])
        if dialogue:
            d_start = min(float(d["start_sec"]) for d in dialogue)
            # anchor the excerpt at the first line of dialogue in the scene
            w0 = max(start, d_start - 0.5)
            w1 = min(end, w0 + max_sec)
        else:
            # spread through the scene: take the middle block
            mid = (start + end) / 2.0
            w0 = max(start, mid - max_sec / 2.0)
            w1 = min(end, w0 + max_sec)
        if w1 - w0 < min_sec:
            return None
        return (round(w0, 3), round(w1, 3))


def blend_director_requirements(purpose: str, scene_summary: str) -> str:
    """Combine a segment purpose with scene content to form a retrieval query."""
    purpose_tokens = text_utils.tokenize(purpose)
    summary_tokens = text_utils.tokenize(scene_summary)
    return " ".join((purpose_tokens + summary_tokens)[:24])