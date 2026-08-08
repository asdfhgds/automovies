"""Factory for instantiating LLM providers based on configuration."""
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_llm_provider_from_config(config: Dict[str, Any]):
    """
    Instantiate LLM provider based on configuration.
    
    Configuration should have:
    - provider: "qwen", "mock", "openai", etc.
    - model: model identifier
    - device: "auto", "cuda", "cpu"
    - other provider-specific settings
    
    Args:
        config: Director configuration dict
        
    Returns:
        LLMProvider instance, or None if provider not available
    """
    provider_name = config.get("provider", "mock").lower()
    
    logger.info(f"Loading LLM provider: {provider_name}")
    
    if provider_name == "mock":
        try:
            from src.director.providers.mock_llm import MockLLMProvider
            logger.info("Using MockLLMProvider")
            return MockLLMProvider()
        except ImportError as e:
            logger.error(f"Failed to load MockLLMProvider: {e}")
            return None
    
    elif provider_name == "qwen":
        try:
            from src.director.providers.qwen import QwenProvider
            
            # Extract Qwen-specific config
            model = config.get("model", "Qwen/Qwen3-30B-A3B")
            device = config.get("device", "auto")
            dtype = config.get("dtype", "auto")
            thinking = config.get("thinking", True)
            temperature = config.get("temperature", 0.8)
            top_p = config.get("top_p", 0.9)
            max_new_tokens = config.get("max_new_tokens", 2048)
            timeout_sec = config.get("timeout_sec", 180)
            
            logger.info(f"Loading Qwen provider: model={model}, device={device}")
            
            provider = QwenProvider(
                model=model,
                device=device,
                dtype=dtype,
                thinking=thinking,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                timeout_sec=timeout_sec,
            )
            
            logger.info("QwenProvider instantiated (model will load lazily)")
            return provider
            
        except ImportError as e:
            logger.error(f"Qwen dependencies not available: {e}")
            logger.info("Falling back to MockLLMProvider")
            try:
                from src.director.providers.mock_llm import MockLLMProvider
                return MockLLMProvider()
            except ImportError:
                logger.error("MockLLMProvider also unavailable")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize QwenProvider: {e}")
            return None
    
    else:
        logger.warning(f"Unknown provider: {provider_name}")
        return None


def get_director_config_from_env(app_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get director configuration from app config and environment overrides.
    
    Args:
        app_config: Application config dict (e.g., from app.yaml)
        
    Returns:
        Director configuration dict
    """
    # Start with app config
    config = {}
    if app_config and "director" in app_config:
        config = app_config["director"].copy()
    
    # Environment overrides
    if os.getenv("DIRECTOR_PROVIDER"):
        config["provider"] = os.getenv("DIRECTOR_PROVIDER")
    
    if os.getenv("DIRECTOR_MODEL"):
        config["model"] = os.getenv("DIRECTOR_MODEL")
    
    if os.getenv("DIRECTOR_DEVICE"):
        config["device"] = os.getenv("DIRECTOR_DEVICE")
    
    if os.getenv("DIRECTOR_TEMPERATURE"):
        try:
            config["temperature"] = float(os.getenv("DIRECTOR_TEMPERATURE"))
        except ValueError:
            pass
    
    # Default to mock if not configured
    if not config:
        config = {
            "provider": "mock",
            "enabled": True,
        }
    
    logger.debug(f"Director config resolved: {config}")
    return config
