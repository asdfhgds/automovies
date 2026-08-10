"""Factory for instantiating LLM providers based on configuration."""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from utils.strict import strict_mode_enabled, require_cuda

logger = logging.getLogger(__name__)


def get_llm_provider_from_config(config: Dict[str, Any]):
    """
    Instantiate LLM provider based on configuration.

    Configuration should have:
    - provider: "qwen", "mock", "openai", etc.
    - model: model identifier
    - device: "auto", "cuda", "cpu"
    - other provider-specific settings

    In strict mode (REQUIRE_REAL_LLM=true) the provider MUST be Qwen and MUST
    run on CUDA. Failures raise instead of silently falling back to mock.

    Args:
        config: Director configuration dict

    Returns:
        LLMProvider instance, or None if provider not available (non-strict).
    """
    provider_name = config.get("provider", "mock").lower()
    strict = strict_mode_enabled()

    logger.info(f"Loading LLM provider: {provider_name} (strict={strict})")

    if strict and provider_name != "qwen":
        raise RuntimeError(
            f"REQUIRE_REAL_LLM=true requires DIRECTOR_PROVIDER=qwen, got "
            f"'{provider_name}'. Refusing to run GPU validation with a mock/deterministic director."
        )

    if provider_name == "mock":
        if strict:
            raise RuntimeError("REQUIRE_REAL_LLM=true forbids the mock LLM provider.")
        try:
            from director.providers.mock_llm import MockLLMProvider
            logger.info("Using MockLLMProvider")
            return MockLLMProvider()
        except ImportError as e:
            logger.error(f"Failed to load MockLLMProvider: {e}")
            return None

    elif provider_name == "qwen":
        try:
            from director.providers.qwen import QwenProvider

            # Extract Qwen-specific config
            model = config.get("model", "Qwen/Qwen3-4B-Instruct-2507")
            device = config.get("device", "auto")
            dtype = config.get("dtype", "auto")
            thinking = config.get("thinking", False)
            temperature = config.get("temperature", 0.8)
            top_p = config.get("top_p", 0.9)
            max_new_tokens = config.get("max_new_tokens", 2048)
            timeout_sec = config.get("timeout_sec", 180)

            if strict and device in ("auto", "cuda"):
                gpu = require_cuda()
                logger.info(f"Strict mode: CUDA OK ({gpu})")
                device = "cuda"

            if device == "cuda":
                import torch

                if not torch.cuda.is_available():
                    if strict:
                        raise RuntimeError(
                            "DIRECTOR_DEVICE=cuda but CUDA is not available "
                            "(REQUIRE_REAL_LLM=true)."
                        )

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
            if strict:
                raise RuntimeError(
                    f"REQUIRE_REAL_LLM=true but Qwen dependencies are missing: {e}"
                ) from e
            try:
                from director.providers.mock_llm import MockLLMProvider
                logger.info("Falling back to MockLLMProvider")
                return MockLLMProvider()
            except ImportError:
                logger.error("MockLLMProvider also unavailable")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize QwenProvider: {e}")
            if strict:
                raise
            return None

    else:
        logger.warning(f"Unknown provider: {provider_name}")
        if strict:
            raise RuntimeError(
                f"REQUIRE_REAL_LLM=true but unknown director provider '{provider_name}'."
            )
        return None


def get_director_config_from_env(app_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get director configuration from app config and environment overrides.

    Args:
        app_config: Application config dict (e.g., from app.yaml)

    Returns:
        Director configuration dict
    """
    # Start with app config (load configs/app.yaml when not supplied)
    config = {}
    if app_config and "director" in app_config:
        config = app_config["director"].copy()
    elif app_config is None:
        try:
            import yaml
        except Exception:
            yaml = None
        if yaml is not None:
            p = Path(__file__).resolve().parent.parent.parent / "configs" / "app.yaml"
            if p.exists():
                try:
                    loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
                    if loaded and isinstance(loaded, dict) and "director" in loaded:
                        config = loaded["director"].copy()
                except Exception:
                    pass

    # Environment overrides
    if os.getenv("DIRECTOR_PROVIDER"):
        config["provider"] = os.getenv("DIRECTOR_PROVIDER")

    if os.getenv("DIRECTOR_MODEL"):
        config["model"] = os.getenv("DIRECTOR_MODEL")

    if os.getenv("DIRECTOR_DEVICE"):
        config["device"] = os.getenv("DIRECTOR_DEVICE")

    if os.getenv("DIRECTOR_DTYPE"):
        config["dtype"] = os.getenv("DIRECTOR_DTYPE")

    if os.getenv("DIRECTOR_TEMPERATURE"):
        try:
            config["temperature"] = float(os.getenv("DIRECTOR_TEMPERATURE"))
        except ValueError:
            pass

    # Default to mock if not configured (only meaningful outside strict mode)
    if not config:
        config = {
            "provider": "mock",
            "enabled": True,
        }

    logger.debug(f"Director config resolved: {config}")
    return config