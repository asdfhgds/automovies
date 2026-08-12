"""Movie intelligence layer.

Transforms the raw movie into a structured representation the editorial
director can reason over: enriched scenes (summary / dialogue / characters /
tone / topics), characters, events, and a semantic index for evidence
retrieval. Everything is deterministic-first so it runs on CPU locally and in
Colab; LLM/vision enrichment slots in behind the provider interface later.
"""
from movie_understanding.analyzer import MovieAnalyzer
from movie_understanding.scene_analyzer import SceneEnricher, HeuristicSceneEnricher
from movie_understanding.semantic_index import SemanticIndex

__all__ = [
    "MovieAnalyzer",
    "SceneEnricher",
    "HeuristicSceneEnricher",
    "SemanticIndex",
]
