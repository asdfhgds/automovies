#!/usr/bin/env bash
# colab_setup.sh
# One-shot dependency setup for the Real Qwen GPU validation notebook.
# Idempotent: safe to run on a fresh Colab runtime.
#
# Usage (from repo root, inside Colab):
#   bash scripts/colab_setup.sh

set -euo pipefail

echo "== [1/6] System packages (ffmpeg, git) =="
apt-get update -y -qq >/dev/null
apt-get install -y -qq ffmpeg git >/dev/null

echo "== [2/6] Upgrade pip =="
python -m pip install --upgrade pip -q

echo "== [3/6] PyTorch with CUDA =="
# Colab normally ships a CUDA-enabled torch on GPU runtimes. If it is missing
# (e.g. a CPU-only runtime, or a stuck +cpu wheel), (re)install a CUDA build and
# verify it, failing loudly if the runtime has no GPU at all.
if python -c "import torch, torch.cuda; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "   Using CUDA-enabled PyTorch: $(python -c 'import torch;print(torch.__version__)')"
else
  echo "   CUDA torch missing -> reinstalling CUDA wheels"
  python -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
  python -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
  python - <<'PY'
import torch
print("   torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA still unavailable. This Colab runtime has no GPU visible. "
        "Set Runtime -> Change runtime type -> T4 GPU, then re-run."
    )
PY
fi

echo "== [4/6] Transformers + Accelerate (Qwen) =="
python -m pip install -q "transformers>=4.52,<5" accelerate sentencepiece protobuf

# Some Colab runtimes preinstall a transformers that predates Qwen3 support.
# Verify Qwen3ForCausalLM is importable and upgrade to the newest if not.
if ! python - <<'PY'
try:
    from transformers import Qwen3ForCausalLM  # noqa: F401
except Exception as e:
    raise SystemExit(f"Qwen3 unsupported in installed transformers: {e}")
PY
then
  echo "   Qwen3ForCausalLM missing -> upgrading transformers to latest"
  python -m pip install -q -U "transformers" accelerate sentencepiece protobuf
  python - <<'PY'
from transformers import Qwen3ForCausalLM  # noqa: F401
import transformers
print("   transformers now:", transformers.__version__, "(Qwen3 OK)")
PY
fi

echo "== [4b/6] bitsandbytes (optional 4-bit Qwen when VRAM is tight) =="
python -m pip install -q bitsandbytes || echo "   bitsandbytes install failed (4-bit mode unavailable; fp16 still works)"

echo "== [5/6] Whisper + PySceneDetect (understanding) =="
python -m pip install -q openai-whisper scenedetect opencv-python-headless || echo "   (optional understanding deps failed)"

echo "== [5b/6] edge-tts (spoken validation clip so whisper has real content) =="
python -m pip install -q edge-tts || echo "   edge-tts unavailable; will use a silent testsrc clip"

echo "== [6/6] Verify =="
python - <<'PY'
import shutil, sys, torch
print("  ffmpeg:", shutil.which("ffmpeg"))
print("  torch:", torch.__version__)
print("  cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  gpu:", torch.cuda.get_device_name(0))
    print("  vram GB:", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))
import transformers
print("  transformers:", transformers.__version__)
try:
    import accelerate
    print("  accelerate:", accelerate.__version__)
except Exception as e:
    print("  accelerate: MISSING (%s)" % e)
PY

echo "Colab setup complete."