"""Strict validation modes.

Strict LLM mode is enabled with REQUIRE_REAL_LLM=true (typically together with
STUDIO_PROFILE=colab-gpu). When active, the pipeline:

- MUST execute Qwen (director and script) on CUDA.
- MUST NOT silently fall back to MockLLMProvider, the deterministic director, or
  the deterministic script writer.

Strict TTS mode is enabled with REQUIRE_REAL_TTS=true. When active, the pipeline
MUST synthesize narration with a real TTS provider (kokoro / chatterbox /
qwen3_tts) and MUST NOT silently fall back to the mock TTS or any pyttsx3-style
stub.

If any of those requirements is violated the pipeline raises a clear error so a
real production run can never "pass" on mocks by accident.
"""
import os

STRICT_ENV = "REQUIRE_REAL_LLM"
STRICT_TTS_ENV = "REQUIRE_REAL_TTS"


def strict_mode_enabled() -> bool:
    """Return True when strict GPU validation mode is requested."""
    return os.getenv(STRICT_ENV, "false").lower() == "true"


def tts_strict_mode_enabled() -> bool:
    """Return True when strict real-TTS production mode is requested."""
    return os.getenv(STRICT_TTS_ENV, "false").lower() == "true"


def require_cuda():
    """Raise immediately if strict mode requires CUDA but none is available."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "REQUIRE_REAL_LLM=true but CUDA is not available. "
            "GPU validation requires a CUDA GPU. Refusing to run with mocks."
        )
    return torch.cuda.get_device_name(0)


def require_real_provider(provider, role: str):
    """Reject None or Mock providers under strict mode.

    Returns the provider unchanged when it is a real provider.
    """
    if provider is None:
        raise RuntimeError(
            f"{role} provider failed to load while REQUIRE_REAL_LLM=true. "
            "GPU validation aborted (no provider)."
        )

    try:
        from director.providers.mock_llm import MockLLMProvider

        if isinstance(provider, MockLLMProvider):
            raise RuntimeError(
                f"{role} resolved to MockLLMProvider while REQUIRE_REAL_LLM=true. "
                "Mock must not be used during GPU validation. Aborting."
            )
    except ImportError:
        pass

    return provider


def require_real_tts(provider, role: str = "TTS"):
    """Reject None or mock TTS providers under REQUIRE_REAL_TTS=true.

    Returns the provider unchanged when it is a real provider.
    """
    if not tts_strict_mode_enabled():
        return provider

    if provider is None:
        raise RuntimeError(
            f"{role} provider failed to load while REQUIRE_REAL_TTS=true. "
            "Production TTS aborted (no provider)."
        )

    name = getattr(provider, "name", "").lower()
    if not name or name in ("mock", "unknown"):
        raise RuntimeError(
            f"{role} resolved to {type(provider).__name__} while REQUIRE_REAL_TTS=true. "
            "Real-movie narration must use a real TTS model, not mock audio. Aborting."
        )
    return provider