#!/usr/bin/env bash
# colab_tts_setup.sh
# Installs open-source TTS deps for the Real-Movie + Real-TTS GPU notebook.
# Idempotent: safe to run on a fresh Colab runtime.
#
# Usage (from repo root, inside Colab):
#   bash scripts/colab_setup.sh     # base deps (run first)
#   bash scripts/colab_tts_setup.sh # TTS deps (this file)

set -euo pipefail

echo "== [1/5] System phonemizer (espeak-ng, required by Kokoro) =="
apt-get update -y -qq >/dev/null
apt-get install -y -qq espeak-ng >/dev/null 2>&1 || echo "   espeak-ng install failed (Kokoro may still work with misaki)"

echo "== [2/5] Kokoro (hexgrad/Kokoro-82M) =="
python -m pip install -q "kokoro>=0.9" soundfile || echo "   kokoro install failed"

echo "== [3/5] Chatterbox (resemble-ai/chatterbox, optional voice cloning) =="
python -m pip install -q chatterbox-tts || echo "   chatterbox-tts install failed (optional)"

echo "== [4/5] Qwen3-TTS (optional, QwenLM/qwen3-tts) =="
python -m pip install -q "qwen3_tts" || echo "   qwen3_tts install failed (optional)"

echo "== [5/5] Verify =="
python - <<'PY'
import importlib.util, sys
print("  python:", sys.version.split()[0])
for name in ("kokoro", "chatterbox", "qwen3_tts", "soundfile"):
    print(f"  {name}: {'FOUND' if importlib.util.find_spec(name) else 'missing'}")
PY

echo "Colab TTS setup complete. Set TTS_DEVICE=cuda before running the pipeline."
