"""Shared text helpers for the movie understanding layer.

Deterministic, dependency-free. We intentionally avoid pulling in a vector
library for the default index; an optional embedder can be plugged into
:class:`~movie_understanding.semantic_index.SemanticIndex` separately.
"""
import re

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "in",
    "on", "at", "to", "for", "with", "from", "by", "as", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "not", "no", "yes", "you", "your", "i", "me", "my", "we", "us", "our",
    "he", "him", "his", "she", "her", "they", "them", "their", "it", "its",
    "this", "that", "these", "those", "there", "here", "who", "what", "when",
    "where", "why", "how", "which", "about", "just", "like", "into", "over",
    "upon", "out", "up", "down", "again", "more", "most", "then", "than",
}

_TOKEN_RE = re.compile(r"[a-zA-Z']+")
_CAPS_RE = re.compile(r"\b([A-Z][a-zA-Z]+)\b")


def tokenize(text: str) -> list:
    """Lowercased alphabetic tokens, stopwords removed."""
    if not text:
        return []
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if t.lower() not in _STOPWORDS and len(t) > 1
    ]


def sentencize(text: str) -> list:
    """Very simple sentence split on ``.``/``!``/``?`` (keeps abbreviations)."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def top_keywords(text: str, k: int = 5) -> list:
    """Most frequent non-stopword tokens, descending (ties by first occurrence)."""
    counts: dict = {}
    for tok in tokenize(text):
        counts[tok] = counts.get(tok, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], len(kv[0])))
    return [w for w, _ in ordered[:k]]


def candidate_names(text: str, min_mentions: int = 2) -> list:
    """Capitalized multi-mention tokens as candidate character names.

    Honest heuristic: no diarization / NER. It catches repeated proper nouns;
    single-mention names and pronouns are intentionally missed. Real speakers
    come from a future diarization or LLM pass.
    """
    if not text:
        return []
    counts: dict = {}
    for match in _CAPS_RE.finditer(text):
        name = match.group(1)
        if name.lower() in _STOPWORDS or len(name) <= 2:
            continue
        counts[name] = counts.get(name, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    return [name for name, n in ordered if n >= min_mentions]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))