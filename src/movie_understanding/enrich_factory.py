"""Factory for scene enrichment providers.

Picks the :class:`~movie_understanding.scene_analyzer.SceneEnricher` used by the
movie index build based on ``VISION_ENRICHER``:

- ``heuristic`` (default) — deterministic, transcript-only. Vision fields stay
  ``None`` (honest). Works everywhere.
- ``qwen3vl`` — real Qwen3-VL / Qwen2.5-VL vision-language enrichment. Requires
  a CUDA GPU + transformers. Falls back to heuristic when not available, unless
  ``REQUIRE_REAL_VISION=true`` (then a hard error).

Config via env (mirrors the LLM/TTS factory conventions):

- ``VISION_ENRICHER`` — ``heuristic`` | ``qwen3vl``
- ``VISION_MODEL`` — default ``Qwen/Qwen2.5-VL-7B-Instruct``
- ``VISION_DEVICE`` — ``auto`` | ``cuda`` | ``cpu``
- ``VISION_DTYPE`` — ``auto`` | ``float16`` | ``4bit`` (NF4, saves VRAM on T4)
- ``VISION_MAX_FRAMES`` — keyframes per scene to feed the model (default 1)
- ``REQUIRE_REAL_VISION`` — strict mode: refuse heuristic degradation
"""
import logging
import os

from utils.strict import require_real_vision, vision_strict_mode_enabled

logger = logging.getLogger(__name__)

DEFAULT_VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def get_vision_config_from_env() -> dict:
    """Resolve vision enrichment config (like get_director_config_from_env)."""
    config = {"enricher": "heuristic"}
    if os.getenv("VISION_ENRICHER") in ("heuristic", "qwen3vl"):
        config["enricher"] = os.getenv("VISION_ENRICHER")
    if os.getenv("VISION_MODEL"):
        config["model"] = os.getenv("VISION_MODEL")
    if os.getenv("VISION_DEVICE"):
        config["device"] = os.getenv("VISION_DEVICE")
    if os.getenv("VISION_DTYPE"):
        config["dtype"] = os.getenv("VISION_DTYPE")
    if os.getenv("VISION_MAX_FRAMES"):
        try:
            config["max_frames"] = int(os.getenv("VISION_MAX_FRAMES"))
        except ValueError:
            pass
    return config


def create_scene_enricher_from_env():
    """Instantiate the scene enricher for the current env config.

    Returns the heuristic enricher by default. With ``VISION_ENRICHER=qwen3vl``
    returns a Qwen3VLEnricher (which degrades to unset vision fields when a GPU
    is absent, unless strict mode raises instead).
    """
    config = get_vision_config_from_env()
    enricher_name = config.get("enricher", "heuristic")
    strict = vision_strict_mode_enabled()

    if enricher_name == "qwen3vl":
        try:
            from movie_understanding.vision_enricher import Qwen3VLEnricher

            provider = Qwen3VLEnricher(
                model=config.get("model", DEFAULT_VISION_MODEL),
                device=config.get("device", "auto"),
                dtype=config.get("dtype", "auto"),
                max_frames=config.get("max_frames", 1),
                strict=strict,
            )
            logger.info(
                f"Scene enricher: qwen3vl ({config.get('model', DEFAULT_VISION_MODEL)})"
            )
            return require_real_vision(provider, "VisionSceneEnricher")
        except Exception as e:
            if strict:
                raise RuntimeError(
                    f"REQUIRE_REAL_VISION=true but Qwen3VLEnricher creation failed: {e}"
                ) from e
            logger.warning(f"Qwen3VLEnricher unavailable ({e}); using heuristic")
            from movie_understanding.scene_analyzer import HeuristicSceneEnricher
            return HeuristicSceneEnricher()

    if strict:
        raise RuntimeError(
            f"REQUIRE_REAL_VISION=true but VISION_ENRICHER={enricher_name} "
            "(heuristic). Vision scene enrichment must use qwen3vl."
        )

    from movie_understanding.scene_analyzer import HeuristicSceneEnricher
    return HeuristicSceneEnricher()