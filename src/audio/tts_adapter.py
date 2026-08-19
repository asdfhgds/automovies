"""TTS adapter: synthesize the project's narration through a configured provider.

Resolution order for the provider config:
1. Environment overrides (TTS_PROVIDER, TTS_VOICE, TTS_DEVICE, ...)
2. ``configs/app.yaml`` -> ``tts`` block
3. Fallback: mock provider (unless REQUIRE_REAL_TTS=true, which refuses mock audio)

The selected provider is also recorded in ``audio/tts_meta.json`` together with
the model, device, generation time, duration, sample rate, and the narration
properties actually applied, so the benchmark/QC stages can audit it.
"""
import json
import os
import time
from pathlib import Path
from typing import Optional

from utils.strict import require_real_tts, tts_strict_mode_enabled


def _load_yaml():
    try:
        import yaml
    except Exception:
        return {}
    for path in ("configs/app.yaml", "configs/profiles.yaml"):
        p = Path(__file__).resolve().parent.parent.parent / path
        if p.exists():
            try:
                return yaml.safe_load(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {}


def _tts_config() -> dict:
    config = _load_yaml()
    tts = {}
    for section in ("app", "profiles"):
        if section in config and isinstance(config[section], dict):
            tts = config[section].get("tts", tts)
    # env overrides on top
    tts = dict(tts)
    tts["provider"] = os.getenv("TTS_PROVIDER", tts.get("provider", "mock"))
    tts["voice"] = os.getenv("TTS_VOICE", tts.get("voice", "default"))
    tts["language"] = os.getenv("TTS_LANGUAGE", tts.get("language", "en"))
    tts["device"] = os.getenv("TTS_DEVICE", tts.get("device", "auto"))
    return tts


def load_tts_provider(config: Optional[dict] = None) -> object:
    """Build the TTS provider for the active config, honoring strict mode."""
    from generation.provider_factory import get_tts_provider

    config = config or _tts_config()
    provider = get_tts_provider(config)
    if tts_strict_mode_enabled():
        provider = require_real_tts(provider, "TTS")
    return provider


def synthesize_voice(project_dir: Path, script_path: str = None):
    project_dir = Path(project_dir)
    script_file = Path(script_path) if script_path else project_dir / "script.json"
    if not script_file.exists():
        raise FileNotFoundError(f"Script not found: {script_file}")
    script = json.loads(script_file.read_text(encoding="utf-8"))

    # Narration Extractor (TTS input contract): the provider MUST receive only
    # plain narration. This step validates every section's narration text and
    # rejects empty/JSON/artefact-laden/debug text (fail closed).
    from audio.narration_contract import (
        NarrationSanitizationError,
        build_tts_inputs,
        joint_text,
        sanitize_narration,
        write_tts_input_manifest,
    )
    try:
        inputs = build_tts_inputs(script)
    except NarrationSanitizationError as e:
        raise ValueError(
            f"TTS FAIL-CLOSED: narration sanitization rejected the script. {e}"
        ) from e
    text = joint_text(inputs)
    if not text.strip():
        raise ValueError("TTS FAIL-CLOSED: narration extractor produced empty text")
    # final belt-and-braces: never synthesize raw director/script blobs.
    sanitize_narration(text, "voiceover", raise_on_error=True)
    write_tts_input_manifest(project_dir, inputs)

    config = _tts_config()
    provider = load_tts_provider(config)

    out_dir = project_dir / 'audio'
    out_dir.mkdir(parents=True, exist_ok=True)
    voice_path = out_dir / 'voice.wav'

    narration = script.get("narration_properties") or {}
    from script.narration import finalize_narration_properties

    narration = finalize_narration_properties(narration)

    t0 = time.monotonic()
    result = provider.synthesize(
        text,
        voice=config.get("voice", "default"),
        language=config.get("language", "en"),
        emotion=narration.get("emotion", "neutral"),
        speaking_rate=float(narration.get("pace", 1.0)),
        pitch=1.0,
        output_path=voice_path,
        narration=narration,
    )
    generation_time = time.monotonic() - t0

    meta = {
        "voice_path": str(voice_path),
        "voice_provider": result.get("provider", getattr(provider, "name", "unknown")),
        "voice_model": result.get("model", "unknown"),
        "voice_device": result.get("device", "unknown"),
        "generation_time_sec": round(generation_time, 3),
        "duration_sec": result.get("duration_sec"),
        "sample_rate": result.get("sample_rate"),
        "voice": result.get("voice"),
        "supported": result.get("supported", {}),
        "narration_properties": narration,
        "mock": bool(result.get("mock", False)),
        "text": text,
        "tts_input_contract": {
            "source": "narration_extractor",
            "sections": [i.section_id for i in inputs],
            "schema": "tts_input_contract_v1",
            "manifest": str(out_dir / "narration_inputs.json"),
        },
    }
    meta_path = out_dir / 'tts_meta.json'
    with meta_path.open('w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Synthesized voice -> {voice_path} ({meta['voice_provider']})")
    return voice_path
