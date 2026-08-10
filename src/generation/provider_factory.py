"""Provider factory - dynamically loads providers based on configuration."""
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def get_script_provider(config: Dict[str, Any]) -> Optional[Any]:
    """Get script provider based on configuration."""
    provider_type = config.get("provider", "mock").lower()
    
    try:
        if provider_type == "mock":
            from .mock import MockScriptProvider
            return MockScriptProvider()
        else:
            logger.warning(f"Unknown script provider: {provider_type}, falling back to mock")
            from .mock import MockScriptProvider
            return MockScriptProvider()
    except Exception as e:
        logger.error(f"Failed to load script provider {provider_type}: {e}, falling back to mock")
        try:
            from .mock import MockScriptProvider
            return MockScriptProvider()
        except Exception as e2:
            logger.error(f"Failed to load mock script provider: {e2}")
            return None


def get_tts_provider(config: Dict[str, Any]) -> Optional[Any]:
    """Get TTS provider based on configuration.

    Honors ``REQUIRE_REAL_TTS=true``: when strict production TTS mode is on, a
    mock provider or a failed real-provider load raises instead of silently
    falling back, so a real-movie run can never produce mock audio by accident.
    """
    provider_type = os.environ.get("TTS_PROVIDER", config.get("provider", "mock")).lower()
    strict = os.getenv("REQUIRE_REAL_TTS", "false").lower() == "true"

    def _mock(message: str):
        if strict:
            raise RuntimeError(
                f"REQUIRE_REAL_TTS=true but TTS_PROVIDER={provider_type!r} cannot run: {message}"
            )
        logger.warning(message)
        from .mock import MockTTSProvider
        return MockTTSProvider()

    real_map = {
        "kokoro": ("kokoro", "KokoroTTSProvider"),
        "chatterbox": ("chatterbox", "ChatterboxTTSProvider"),
        "qwen3_tts": ("qwen_tts", "Qwen3TTSProvider"),
        "qwen3-tts": ("qwen_tts", "Qwen3TTSProvider"),
    }

    try:
        if provider_type == "mock":
            if strict:
                raise RuntimeError(
                    "REQUIRE_REAL_TTS=true but TTS_PROVIDER=mock. "
                    "Production TTS refuses silent mock audio."
                )
            from .mock import MockTTSProvider
            return MockTTSProvider()
        if provider_type in real_map:
            module_name, cls_name = real_map[provider_type]
            try:
                mod = __import__(f"generation.{module_name}", fromlist=[cls_name])
                cls = getattr(mod, cls_name)
                provider = cls(config)
                if strict and not provider.is_available():
                    raise RuntimeError(
                        f"REQUIRE_REAL_TTS=true but {provider_type} is not installed."
                    )
                return provider
            except ImportError as e:
                return _mock(f"{provider_type} package not installed ({e})")
            except RuntimeError as e:
                raise RuntimeError(
                    f"REQUIRE_REAL_TTS=true but {provider_type} failed: {e}"
                ) from e
            except Exception as e:
                return _mock(f"{provider_type} failed to initialize: {e}")
        else:
            return _mock(f"Unknown TTS provider: {provider_type}")
    except Exception as e:
        if strict:
            raise RuntimeError(f"TTS provider resolution failed: {e}") from e
        logger.error(f"Failed to load TTS provider {provider_type}: {e}, falling back to mock")
        try:
            from .mock import MockTTSProvider
            return MockTTSProvider()
        except Exception as e2:
            logger.error(f"Failed to load mock TTS provider: {e2}")
            return None


def available_tts_providers() -> Dict[str, Any]:
    """Return TTS provider availability for doctor/benchmark tooling."""
    out = {}
    for name, (module_name, cls_name) in {
        "kokoro": ("kokoro", "KokoroTTSProvider"),
        "chatterbox": ("chatterbox", "ChatterboxTTSProvider"),
        "qwen3_tts": ("qwen_tts", "Qwen3TTSProvider"),
    }.items():
        try:
            mod = __import__(f"generation.{module_name}", fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            out[name] = {"available": bool(cls.is_available())}
        except Exception as e:
            out[name] = {"available": False, "error": str(e)}
    return out


def get_image_provider(config: Dict[str, Any]) -> Optional[Any]:
    """Get image provider based on configuration."""
    provider_type = os.environ.get("IMAGE_PROVIDER", config.get("provider", "mock")).lower()
    
    try:
        if provider_type == "mock":
            from .mock import MockImageProvider
            return MockImageProvider()
        elif provider_type == "comfyui":
            try:
                from .comfyui import ComfyUIImageProvider
                return ComfyUIImageProvider(config)
            except ImportError:
                logger.warning("ComfyUI not available, falling back to mock")
                from .mock import MockImageProvider
                return MockImageProvider()
        else:
            logger.warning(f"Unknown image provider: {provider_type}, falling back to mock")
            from .mock import MockImageProvider
            return MockImageProvider()
    except Exception as e:
        logger.error(f"Failed to load image provider {provider_type}: {e}, falling back to mock")
        try:
            from .mock import MockImageProvider
            return MockImageProvider()
        except Exception as e2:
            logger.error(f"Failed to load mock image provider: {e2}")
            return None


def get_video_provider(config: Dict[str, Any]) -> Optional[Any]:
    """Get video provider based on configuration."""
    provider_type = os.environ.get("VIDEO_PROVIDER", config.get("provider", "mock")).lower()
    
    try:
        if provider_type == "mock":
            from .mock import MockVideoProvider
            return MockVideoProvider()
        else:
            logger.warning(f"Unknown video provider: {provider_type}, falling back to mock")
            from .mock import MockVideoProvider
            return MockVideoProvider()
    except Exception as e:
        logger.error(f"Failed to load video provider {provider_type}: {e}, falling back to mock")
        try:
            from .mock import MockVideoProvider
            return MockVideoProvider()
        except Exception as e2:
            logger.error(f"Failed to load mock video provider: {e2}")
            return None
