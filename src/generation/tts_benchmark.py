"""TTS benchmarking: synthesize the same narration with every available provider.

Writes ``reports/tts_benchmark.json`` with one entry per provider:
provider, model, device, generation_time_sec, duration_sec, sample_rate, status,
and error (on failure). Providers whose package is not installed are recorded
with ``status: "unavailable"`` instead of being skipped silently, so the report
shows exactly what was and was not benchmarked.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from generation.mock import MockTTSProvider
from generation.provider_factory import available_tts_providers

_BENCHMARK_TEXT = (
    "Welcome to this deep dive. Every frame we are about to examine was chosen "
    "for a reason. Let us look closer at what the director is actually doing, "
    "and why it matters."
)


def _provider_instance(name: str, config: dict):
    if name == "mock":
        return MockTTSProvider()
    from generation.provider_factory import get_tts_provider

    return get_tts_provider({"provider": name, **config})


def benchmark_tts(
    text: str = _BENCHMARK_TEXT,
    providers: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    include_mock: bool = True,
    narration: Optional[Dict[str, Any]] = None,
    allow_cpu: bool = False,
) -> Dict[str, Any]:
    """Benchmark each TTS provider on the same narration text.

    Args:
        text: Narration text to synthesize.
        providers: Provider names to benchmark (defaults to all detected ones).
        output_dir: Directory to write per-provider wavs + the JSON report.
        include_mock: Include the mock baseline for comparison (default True).
        narration: Director narration properties to pass through.
        allow_cpu: Allow real TTS models to run on CPU. Defaults to False so
            local runs never burn CPU-time synthesizing real models; those are
            recorded with ``status: "cpu_skipped"``. Set TTS_DEVICE=cuda (or
            pass device in provider config) on a GPU box to benchmark for real.
    """
    output_dir = Path(output_dir) if output_dir else Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    availability = available_tts_providers()
    candidates = providers or (["mock"] if include_mock else []) + list(availability.keys())
    if include_mock and "mock" not in candidates:
        candidates = ["mock"] + candidates
    # dedupe preserving order
    candidates = list(dict.fromkeys(candidates))

    config = {
        "device": None,
        "voice": "default",
        "language": "en",
    }
    results: List[Dict[str, Any]] = []
    for name in candidates:
        entry = {
            "provider": name,
            "model": None,
            "device": None,
            "generation_time_sec": None,
            "duration_sec": None,
            "sample_rate": None,
            "status": "ok",
            "error": None,
            "output_path": None,
            "mock": name == "mock",
        }
        if name != "mock" and not availability.get(name, {}).get("available", False):
            entry["status"] = "unavailable"
            entry["error"] = availability.get(name, {}).get("error", "package not installed")
            results.append(entry)
            continue
        if name != "mock":
            provider = _provider_instance(name, config)
            entry["model"] = getattr(provider, "model", None) or getattr(provider, "model_name", None)
            entry["device"] = getattr(provider, "device", None)
            if not allow_cpu and not str(entry["device"] or "").startswith("cuda"):
                entry["status"] = "cpu_skipped"
                entry["error"] = (
                    f"real TTS skipped on device={entry['device']}; "
                    "set TTS_DEVICE=cuda on a GPU box (CPU synthesis disabled)"
                )
                results.append(entry)
                continue
        try:
            provider = _provider_instance(name, config)
            out = output_dir / f"benchmark_{name}.wav"
            t0 = time.monotonic()
            result = provider.synthesize(
                text,
                voice=config.get("voice", "default"),
                language=config.get("language", "en"),
                emotion=(narration or {}).get("emotion", "neutral"),
                speaking_rate=float((narration or {}).get("pace", 1.0)),
                pitch=1.0,
                output_path=out,
                narration=narration,
            )
            entry.update({
                "model": result.get("model"),
                "device": result.get("device"),
                "generation_time_sec": round(time.monotonic() - t0, 3),
                "duration_sec": result.get("duration_sec"),
                "sample_rate": result.get("sample_rate"),
                "output_path": str(result.get("audio_path")),
                "supported": result.get("supported", {}),
            })
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = str(e)
        results.append(entry)

    report = {
        "text_synopsized": text[:120],
        "narration_properties": narration,
        "device_hint": "cuda" if _cuda_available() else "cpu",
        "results": results,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    report_path = output_dir / "tts_benchmark.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TTS benchmark written -> {report_path}")
    return report


def load_benchmark_report(output_dir: Path) -> Optional[Dict[str, Any]]:
    report_path = Path(output_dir) / "tts_benchmark.json"
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def _cuda_available() -> bool:
    # Avoid importing torch (which is slow) just for a metadata hint. Only
    # report CUDA when torch is already loaded in this process.
    import sys

    if "torch" not in sys.modules:
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
