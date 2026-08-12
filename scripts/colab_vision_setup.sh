#!/usr/bin/env bash
# colab_vision_setup.sh
# Dependency setup for the Qwen3-VL vision scene enrichment validation notebook.
# Idempotent: safe to run on a fresh Colab runtime. Vision-language models need
# a recent Transformers that carries the Qwen2.5-VL / Qwen3-VL support.
#
# Usage (from repo root, inside Colab):
#   bash scripts/colab_vision_setup.sh

set -euo pipefail

echo "== [1/4] Ensure a Transformers build with Qwen VL support =="
# The base colab_setup.sh pins to a transformers that supports Qwen3 chunked
# thinking; Qwen2.5-VL / Qwen3-VL are exposed via AutoProcessor + AutoModel as
# long as the installed transformers is recent enough. Upgrade when a feature
# (e.g. Qwen2_5_VLForConditionalGeneration) is missing.
python - <<'PY'
try:
    import transformers
    from transformers import AutoProcessor, AutoModel  # noqa: F401
    print("   transformers:", transformers.__version__, "(vl-capable AutoProcessor/AutoModel present)")
except Exception as e:
    raise SystemExit(f"vision transformers check failed: {e}")
PY
if ! python - <<'PY'
try:
    from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
except Exception:
    raise SystemExit("missing")
PY
then
  echo "   Qwen2_5_VLForConditionalGeneration missing -> upgrading transformers"
  python -m pip install -q -U "transformers>=4.57" accelerate sentencepiece protobuf
  python - <<'PY'
from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
import transformers
print("   transformers now:", transformers.__version__, "(Qwen2.5-VL OK)")
PY
fi

echo "== [2/4] bitsandbytes (optional 4-bit Qwen-VL when VRAM is tight) =="
python -m pip install -q bitsandbytes || echo "   bitsandbytes install failed (4-bit mode unavailable; fp16 still works)"

echo "== [3/4] Pillow + gdown (keyframes + Drive movie download) =="
python -m pip install -q pillow gdown || echo "   (optional deps failed)"

echo "== [4/4] Verify =="
python - <<'PY'
import shutil, torch, transformers
from PIL import Image  # noqa: F401
print("  ffmpeg:", shutil.which("ffmpeg"))
print("  torch:", torch.__version__)
print("  cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  gpu:", torch.cuda.get_device_name(0))
    print("  vram GB:", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))
print("  transformers:", transformers.__version__)
try:
    from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
    print("  Qwen2_5_VLForConditionalGeneration: available")
except Exception as e:
    print("  Qwen2_5_VLForConditionalGeneration: MISSING (%s)" % e)
PY

echo "Vision setup complete."