"""Tests for the dense-embedding retrieval layer.

Covers the opt-in ``embedder`` path of ``SemanticIndex`` (build-time dense
vectors, cosine search), the eval harness ``--method embedding`` wiring, and
the honest failure mode when an embedder cannot be created. Everything here is
offline/deterministic: a tiny alias-based ``TokenEmbedder`` stands in for a
real embedding model, so no model downloads and no network.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from movie_understanding.embedding_retriever import (  # noqa: E402
    SentenceEmbedder,
    create_embedder_from_env,
)
from movie_understanding.semantic_index import SemanticIndex  # noqa: E402
from movie_understanding import text_utils  # noqa: E402

STUB = '''
"""Deterministic test embedder: synonym groups share a vector."""
import hashlib
import math
import sys

sys.path.insert(0, %(src)r)

from movie_understanding import text_utils

DIM = 64
# canonical -> members that map to the same vector
ALIASES = {
    "eatery": ["diner", "restaurant", "cafe", "eatery"],
    "dark": ["night", "evening", "late", "dark"],
    "noise": ["loud", "noisy", "crowd", "stadium"],
}


class TokenEmbedder:
    def __init__(self):
        self._vecs = {}

    def _vec_for(self, key):
        if key not in self._vecs:
            h = int.from_bytes(hashlib.md5(key.encode()).digest()[:8], "big")
            v = [math.sin(h * (i + 3)) for i in range(DIM)]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            self._vecs[key] = [x / n for x in v]
        return self._vecs[key]

    def _token_vec(self, tok):
        for canonical, members in ALIASES.items():
            if tok in members:
                return self._vec_for(canonical)
        return self._vec_for(tok)

    def __call__(self, texts):
        out = []
        for t in texts:
            v = [0.0] * DIM
            for tok in text_utils.tokenize(t):
                tv = self._token_vec(tok)
                v = [a + b for a, b in zip(v, tv)]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def factory():
    return TokenEmbedder()
'''


@pytest.fixture
def stub_module(tmp_path):
    (tmp_path / "embedder_stub.py").write_text(
        STUB % {"src": str(SRC)}, encoding="utf-8")
    return tmp_path


def _scenes():
    return [
        {
            "scene_id": "scene-1",
            "start_sec": 0.0,
            "end_sec": 6.0,
            "transcript": "Sam sits in a quiet diner.",
            "story": {
                "summary": "A man eats alone in a diner.",
                "topics": ["diner"],
                "dialogue": [],
                "characters": ["Sam"],
                "location": "diner at night",
                "actions": ["eating"],
                "objects": ["plate"],
                "visual_description": "A man at a counter under dim light.",
                "visual_events": ["man sits at ~1s"],
                "emotional_cues": ["alone"],
                "themes": ["loneliness"],
                "mood": "quiet",
                "cinematography": "close-up",
                "confidence": 0.9,
                "provenance": {},
            },
        },
        {
            "scene_id": "scene-2",
            "start_sec": 10.0,
            "end_sec": 16.0,
            "transcript": "Rosa sprints across an empty stadium.",
            "story": {
                "summary": "A woman runs laps on the track.",
                "topics": ["race"],
                "dialogue": [],
                "characters": ["Rosa"],
                "location": "sports arena under floodlights",
                "actions": ["sprinting"],
                "objects": [],
                "visual_description": "Floodlights over an empty stand.",
                "visual_events": ["Rosa crosses the line"],
                "emotional_cues": [],
                "themes": ["determination"],
                "mood": "loud",
                "cinematography": "wide shot",
                "confidence": 0.9,
                "provenance": {},
            },
        },
    ]


def _token_embedder():
    ns = {}
    exec(STUB % {"src": str(SRC)}, ns)
    return ns["factory"]()


def test_embedding_search_beats_tfidf_on_synonym_query():
    from movie_understanding.semantic_index import SemanticIndex

    # A paraphrased query with zero shared vocabulary with the corpus.
    query = "find a restaurant open in the evening"
    assert set(text_utils.tokenize(query)) & set(text_utils.tokenize(
        "Sam sits in a quiet diner A man eats alone in a diner diner at night")) == set()

    tfidf = SemanticIndex().build(_scenes())
    assert tfidf.search(query, k=3) == []  # TF-IDF: no shared tokens -> nothing

    embedder = _token_embedder()
    dense = SemanticIndex().build(_scenes(), embedder=embedder)
    hits = dense.search(query, k=3)
    assert hits and hits[0]["scene_id"] == "scene-1"


def test_embedding_build_tags_method_and_scores():
    dense = SemanticIndex().build(_scenes(), embedder=_token_embedder())
    assert dense.to_dict()["method"] == "embedder"
    assert SemanticIndex().build(_scenes()).to_dict()["method"] == "tfidf"
    assert dense.search("restaurant evening")[0]["score"] > 0


def test_embedding_handles_empty_corpus():
    dense = SemanticIndex().build([], embedder=_token_embedder())
    assert dense.to_dict()["method"] == "embedder"
    assert dense.search("anything") == []


def test_eval_harness_embedding_mode_writes_reports(stub_module):
    proj = stub_module
    movie_index = {"project_id": "p1", "movie": {"title": "Coin", "duration_sec": 16.0},
                   "provenance": {"scene_enricher": "qwen3vl"}, "scenes": _scenes()}
    (proj / "movie_index.json").write_text(
        json.dumps(movie_index, ensure_ascii=False), encoding="utf-8")

    script = str(Path(__file__).resolve().parents[1] / "scripts" / "evaluate_retrieval.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(proj), str(SRC)])
    res = subprocess.run(
        [sys.executable, script, "--project", str(proj), "--k", "3",
         "--method", "embedding", "--embedder", "embedder_stub:factory"],
        capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr

    recs = json.loads((proj / "reports" / "retrieval_evaluation.json")
                      .read_text(encoding="utf-8"))
    assert recs["method"] == "embedding"
    assert recs["queries"]
    md = (proj / "reports" / "retrieval_evaluation.md").read_text(encoding="utf-8")
    assert "Dense embeddings" in md


def test_eval_harness_embedding_fails_loudly_on_missing_embedder(stub_module):
    # Honest failure: the harness must NOT silently fall back to TF-IDF.
    proj = stub_module
    movie_index = {"project_id": "p1", "movie": {"title": "Coin", "duration_sec": 16.0},
                   "provenance": {"scene_enricher": "qwen3vl"}, "scenes": _scenes()}
    (proj / "movie_index.json").write_text(
        json.dumps(movie_index, ensure_ascii=False), encoding="utf-8")
    script = str(Path(__file__).resolve().parents[1] / "scripts" / "evaluate_retrieval.py")
    res = subprocess.run(
        [sys.executable, script, "--project", str(proj),
         "--method", "embedding", "--embedder", "no_such_module:factory"],
        capture_output=True, text=True)
    assert res.returncode == 2
    assert "unavailable" in res.stderr.lower()


def test_create_embedder_from_env_module_attr(stub_module, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_EMBEDDER", "embedder_stub:factory")
    monkeypatch.syspath_prepend(str(stub_module))
    embedder = create_embedder_from_env()
    assert callable(embedder)
    assert embedder(["a restaurant at night"])


def test_create_embedder_from_env_invalid_spec_raises(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_EMBEDDER", "totally-bogus-spec")
    with pytest.raises(ValueError):
        create_embedder_from_env()


def test_create_embedder_from_env_default_is_sentence_embedder(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_EMBEDDER", raising=False)
    embedder = create_embedder_from_env()
    assert isinstance(embedder, SentenceEmbedder)
    assert isinstance(SentenceEmbedder.is_available(), bool)