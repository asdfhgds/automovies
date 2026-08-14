"""Dense-embedding retrieval for the Movie Intelligence layer.

Opt-in semantic retrieval bolted onto :class:`SemanticIndex`. When an embedder
is available (sentence-transformers on CPU/GPU, or any callable of the same
shape) ``SemanticIndex.build(..., embedder=...)`` stores dense vectors for the
whole scene corpus and ``search()`` ranks by cosine similarity against the
embedded query. TF-IDF stays the default and the always-available fallback.

Env knobs (mirroring ``enrich_factory`` conventions):

- ``RETRIEVAL_EMBEDDER`` — ``sentence-transformers`` (default) or
  ``module:attr`` where ``attr`` is a callable factory returning an embedder
  callable ``embed(texts) -> list[list[float]]``.
- ``RETRIEVAL_EMBEDDER_MODEL`` — default ``sentence-transformers/all-MiniLM-L6-v2``.
- ``RETRIEVAL_DEVICE`` — ``auto`` (default) / ``cpu`` / ``cuda``.

Models load lazily (first ``embed`` call, like the Qwen-VL enricher); creating
an embedder never downloads anything. If the package is missing the caller must
fail loudly (the eval harness does) rather than silently substituting TF-IDF.
"""
import importlib
import importlib.util
import os
from typing import Callable, List

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_KNOWN_NAMES = ("sentence-transformers", "sentencetransformers", "sentence_transformers")


class SentenceEmbedder:
    """Callable embedder backed by ``sentence-transformers`` (lazy-loaded)."""

    def __init__(self, model_name: str = None, device: str = None) -> None:
        self.model_name = model_name or os.environ.get(
            "RETRIEVAL_EMBEDDER_MODEL") or DEFAULT_MODEL
        self.device = device or os.environ.get("RETRIEVAL_DEVICE") or "auto"
        self._model = None

    @staticmethod
    def is_available() -> bool:
        return importlib.util.find_spec("sentence_transformers") is not None

    def ensure_loaded(self):
        if self._model is not None:
            return self._model
        if not self.is_available():
            raise ImportError(
                "sentence-transformers is not installed — run "
                "`python -m pip install sentence-transformers` (or on Colab: "
                "`bash scripts/colab_vision_setup.sh`)."
            )
        from sentence_transformers import SentenceTransformer  # lazy

        kwargs = {}
        if self.device in ("cpu", "cuda"):
            kwargs["device"] = self.device
        self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def __call__(self, texts: List[str]) -> List[list]:
        model = self.ensure_loaded()
        if not texts:
            return []
        vecs = model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True,
            batch_size=64)
        return [v.tolist() for v in vecs]


def create_embedder_from_env() -> Callable[[List[str]], List[list]]:
    """Build an embedder from ``RETRIEVAL_EMBEDDER`` (``module:attr`` factory).

    Never loads a model (lazy). Raises ``ImportError``/``ValueError`` with an
    actionable message when the requested backend cannot be created.
    """
    spec = (os.environ.get("RETRIEVAL_EMBEDDER") or "sentence-transformers").strip()
    if spec in _KNOWN_NAMES:
        return SentenceEmbedder()
    if ":" in spec:
        module_name, attr_name = spec.split(":", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, attr_name)
        embedder = factory()
        if not callable(embedder):
            raise ValueError(
                f"RETRIEVAL_EMBEDDER={spec!r}: {attr_name}() did not return a "
                "callable embedder")
        return embedder
    raise ValueError(
        f"RETRIEVAL_EMBEDDER={spec!r}: expected 'sentence-transformers' or a "
        "'module:attr' importable factory (e.g. 'embedder_stub:factory').")