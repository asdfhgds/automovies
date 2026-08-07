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
    """Get TTS provider based on configuration."""
    provider_type = os.environ.get("TTS_PROVIDER", config.get("provider", "mock")).lower()
    
    try:
        if provider_type == "mock":
            from .mock import MockTTSProvider
            return MockTTSProvider()
        elif provider_type == "kokoro":
            try:
                from .kokoro import KokoroTTSProvider
                return KokoroTTSProvider(config)
            except ImportError:
                logger.warning("Kokoro TTS not available, falling back to mock")
                from .mock import MockTTSProvider
                return MockTTSProvider()
        elif provider_type in ["qwen3_tts", "qwen3-tts"]:
            try:
                from .qwen_tts import Qwen3TTSProvider
                return Qwen3TTSProvider(config)
            except ImportError:
                logger.warning("Qwen3 TTS not available, falling back to mock")
                from .mock import MockTTSProvider
                return MockTTSProvider()
        else:
            logger.warning(f"Unknown TTS provider: {provider_type}, falling back to mock")
            from .mock import MockTTSProvider
            return MockTTSProvider()
    except Exception as e:
        logger.error(f"Failed to load TTS provider {provider_type}: {e}, falling back to mock")
        try:
            from .mock import MockTTSProvider
            return MockTTSProvider()
        except Exception as e2:
            logger.error(f"Failed to load mock TTS provider: {e2}")
            return None


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
