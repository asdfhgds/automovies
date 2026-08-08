"""Transcription adapter: choose a transcription backend (whisperx preferred, fallback to stub).

Implements transcribe(project_dir: Path, source_path: Optional[str]) -> Path to transcript.json
"""
from pathlib import Path
from typing import Optional


def transcribe(project_dir: Path, source_path: Optional[str] = None):
    """Dispatch to a real transcription backend when available.

    If whisperx is installed and importable as a local adapter, use it.
    Otherwise, fall back to the lightweight stub included for testing.
    """
    # Lazy import backends to avoid heavy imports at module import time
    try:
        from . import whisperx_adapter  # type: ignore
        print("Using whisperx_adapter for transcription")
        try:
            out = whisperx_adapter.transcribe(project_dir, source_path)
            # validate normalized format: require 'segments' key in JSON
            import json
            from pathlib import Path
            p = Path(out)
            if p.exists():
                try:
                    jd = json.loads(p.read_text(encoding='utf-8'))
                    if isinstance(jd, dict) and 'segments' in jd:
                        return out
                    else:
                        print('whisperx_adapter returned non-normalized transcript — falling back to stub')
                except Exception:
                    print('Failed to parse whisperx_adapter output — falling back to stub')
        except Exception as e:
            print(f"whisperx_adapter failed: {e}")
        # fallback to stub
        from . import whisper_stub  # type: ignore
        print("Falling back to whisper_stub")
        return whisper_stub.transcribe(project_dir, source_path)
    except Exception as e:
        # final fallback: try stub directly
        try:
            from . import whisper_stub  # type: ignore
            print("whisperx_adapter import failed — using whisper_stub")
            return whisper_stub.transcribe(project_dir, source_path)
        except Exception as ex:
            raise RuntimeError(f"No transcription backend available: {e}; {ex}")
