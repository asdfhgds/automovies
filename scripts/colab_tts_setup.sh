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

echo "== [4/5] Qwen3-TTS (optional, not yet on PyPI) =="
python -m pip install -q "qwen3_tts" 2>/dev/null || echo "   qwen3_tts not on PyPI (optional; skipped)"

echo "== [5/5] Ensure torch/torchvision consistent + verify =="
# transformers<5 imports torchvision, which crashes hard (operator
# torchvision::nms does not exist) when torch and torchvision were installed
# from different PyTorch indices. Reinstall a matching trio if needed.
if ! python - <<'PY'
import torch  # noqa: F401
import torchvision  # noqa: F401
PY
then
  echo "   torch/torchvision mismatch -> reinstalling consistent cu124 trio"
  python -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
  python -m pip install --force-reinstall --no-deps \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

python - <<'PY'
import importlib.util, sys
print("  python:", sys.version.split()[0])
for name in ("kokoro", "chatterbox", "qwen3_tts", "soundfile"):
    print(f"  {name}: {'FOUND' if importlib.util.find_spec(name) else 'missing'}")
try:
    from transformers import Qwen3ForCausalLM  # noqa: F401
    import transformers
    print("  transformers:", transformers.__version__)
    print("  Qwen3ForCausalLM: OK")
except Exception as e:
    import transformers
    print(f"  Qwen3ForCausalLM MISSING (transformers {transformers.__version__}): {e}")
PY
# TTS packages (Kokoro/Chatterbox) may pull a transformers >=5, which removed
# Qwen3ForCausalLM. Restore the pinned <5 release the Qwen director needs BEFORE
# any fatal check, so this step heals its own environment.
if ! python - <<'PY'
try:
    from transformers import Qwen3ForCausalLM  # noqa: F401
    print("  Qwen3ForCausalLM: OK")
except Exception:
    raise SystemExit(3)
PY
then
  echo "   Qwen3 missing (transformers too new) -> restoring transformers <5"
  python -m pip install -q -U "transformers>=4.52,<5" accelerate sentencepiece protobuf
  python - <<'PY'
from transformers import Qwen3ForCausalLM  # noqa: F401
import transformers
print("   transformers now:", transformers.__version__, "(Qwen3 OK)")
PY
fi

echo "Colab TTS setup complete. Set TTS_DEVICE=cuda before running the pipeline."
