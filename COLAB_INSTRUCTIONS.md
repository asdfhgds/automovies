# GPU Validation with Real Qwen (Colab)

There are two ways to run real-LLM validation on Colab:

1. **Notebook (recommended):** `notebooks/colab_qwen_validation.ipynb` — a 14-cell
   notebook that drives the *same* code paths the orchestrator uses in strict mode
   (CUDA Qwen director + script writer, `provider_manifest.json`).
   Open it via:
   `https://colab.research.google.com/github/<OWNER>/<REPO>/blob/main/notebooks/colab_qwen_validation.ipynb`
2. **Manual**, as described below.

## Manual quick start

1) Open a new Google Colab notebook. Set Runtime > Change runtime type > GPU (T4+).

2) Run the idempotent setup script (installs ffmpeg, CUDA torch, transformers,
   accelerate, whisper, pyscenedetect):

```python
# in a Colab cell, after cloning the repo:
!bash scripts/colab_setup.sh
```

3) Clone your repository (replace REPO_URL):

```
!git clone <REPO_URL> repo
%cd repo
```

4) Run doctor to confirm GPU + deps:

```
!python src/main.py doctor
```

5) Generate tiny test video and run pipeline:

```
!python tests/fixtures/generate_test_fixture.py tests/fixtures/test_speech.mp4 "Short GPU test"
!python src/main.py init --title "Colab GPU Test" --source tests/fixtures/test_speech.mp4
# note printed project id, then run the pipeline:
!python src/main.py run --project-id <project-id>
```

6) Inspect outputs in data/<project-id>/
- transcripts/transcript.json
- scenes/scene_index.json (if PySceneDetect produced it)
- scenes/scene_ranking.json
- scenes/selected_scene.json
- assets/scenes/<scene_id>.mp4
- **provider_manifest.json** — records which providers actually executed

7) Use ffprobe to confirm clip properties:

```
!ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 data/<project-id>/assets/scenes/<scene_id>.mp4
```


## Real LLM Validation (Qwen with Creative Director)

The pipeline now supports real LLM-powered creative direction using Qwen models.

### Prerequisites

Qwen requires PyTorch and Transformers:

```
!pip install transformers accelerate
# For optimal performance:
!pip install flash-attn  # If your GPU supports it
```

### Model Size Considerations

**Colab T4 (16GB VRAM):**
- Qwen3-7B-A0.5B: ✅ Fits easily (default)
- Qwen3-30B-A3B: ⚠️ May OOM without quantization
- Qwen3-235B-A22B: ❌ Requires quantization or larger GPU

**Colab A100 (40GB VRAM):**
- Qwen3-30B-A3B: ✅ Fits comfortably
- Qwen3-235B-A22B: ✅ Fits with careful settings

### Setup for Real LLM

```python
# In Colab, set up environment:
!bash scripts/colab_setup.sh

# For the default T4 model:
%env DIRECTOR_PROVIDER=qwen
%env DIRECTOR_MODEL=Qwen/Qwen3-7B-A0.5B
%env DIRECTOR_DEVICE=cuda
%env DIRECTOR_TEMPERATURE=0.8
```

### Strict GPU Mode (REQUIRE_REAL_LLM=true)

`REQUIRE_REAL_LLM=true` enables **strict GPU validation** in the orchestrator. It
is the only way to *prove* a real LLM ran:

- `require_cuda()` fails hard on boxes without CUDA (no CPU/mock fallback).
- The director and script stages refuse `mock`/deterministic providers.
- A `provider_manifest.json` records provider, model, device, and load/generation
  timings.

```bash
export REQUIRE_REAL_LLM=true
export STUDIO_PROFILE=colab-gpu
export DIRECTOR_PROVIDER=qwen
export DIRECTOR_MODEL=Qwen/Qwen3-7B-A0.5B
export SCRIPT_PROVIDER=qwen
export SCRIPT_MODEL=Qwen/Qwen3-7B-A0.5B
export DIRECTOR_DEVICE=cuda
```

In non-strict mode (`REQUIRE_REAL_LLM=false`), the pipeline uses the deterministic
director/script when Qwen is unavailable, exactly as before.

### Running Pipeline with Real Qwen

```bash
# Enable creative director with real Qwen (non-strict mode)
export CREATIVE_DIRECTOR_ENABLED=true
export DIRECTOR_PROVIDER=qwen
export DIRECTOR_MODEL=Qwen/Qwen3-7B-A0.5B
export DIRECTOR_DEVICE=cuda

python src/main.py init --title "Qwen LLM Test" --source tests/fixtures/test_speech.mp4
python src/main.py run --project-id <project-id>
```

### Running Tests

```bash
# Run unit tests (no model download)
pytest tests/test_qwen_provider.py -v

# Run creative director tests (mock LLM, fast)
pytest tests/test_creative_director.py -v

# Run real LLM integration tests (requires Qwen model)
pytest tests/test_qwen_integration.py -m llm_integration -v

# Mark slow tests
pytest -m slow -v  # Will run real model tests if available
```

### Inspecting Generated Concepts

After running the pipeline, check the generated concepts in memory:

```bash
cat data/<project-id>/memory/concepts.jsonl | python -m json.tool
```

Each line is a JSON object with:
- `title`: Concept title
- `thesis`: Core argument
- `hook`: Engaging opening
- `scores`: 6-dimensional critique
- `selected`: Whether this concept was selected for production

### Director Configuration

All settings can be configured in `configs/app.yaml`:

```yaml
director:
  enabled: true
  provider: qwen  # or "mock"
  model: Qwen/Qwen3-7B-A0.5B  # T4 default; 30B-A3B for A100
  device: auto  # or "cuda", "cpu"
  dtype: auto   # or "float16", "float32", "bfloat16"
  thinking: false
  temperature: 0.8
  top_p: 0.9
  max_new_tokens: 2048
  timeout_sec: 180

script:
  provider: qwen
  model: Qwen/Qwen3-7B-A0.5B
  device: auto
  dtype: auto
  thinking: false
  temperature: 0.7
  max_new_tokens: 1024
```

Environment overrides at runtime:

```bash
DIRECTOR_PROVIDER=qwen
DIRECTOR_MODEL=Qwen/Qwen3-7B-A0.5B
DIRECTOR_DEVICE=cuda
DIRECTOR_TEMPERATURE=0.7
SCRIPT_PROVIDER=qwen
SCRIPT_MODEL=Qwen/Qwen3-7B-A0.5B
```

### Fallback Behavior

Fallback depends on `REQUIRE_REAL_LLM`:

1. **Non-strict mode**: If Qwen is unavailable or fails, the director falls back to
   the deterministic planner and the script falls back to `script/writer.py`.
   The pipeline never breaks due to LLM unavailability.
2. **Strict mode (`REQUIRE_REAL_LLM=true`)**: **no fallback.** Any attempt to use a
   mock/deterministic provider (or a CPU-only box) raises a clear `RuntimeError`, so
   the run cannot silently claim GPU validation without a real model.

### Performance Expectations

On T4 GPU (15.8GB VRAM):
- Qwen3-7B-A0.5B model loading: ~30-60s
- Concept generation (2-3 concepts): ~45-120s
- Production plan generation: ~30-60s
- Script generation (narration sections): ~20-60s
- Full pipeline: ~3-5 minutes

### Troubleshooting: Out-of-Memory

The 7B model is fp16 is ~14GB, so it must be loaded **once**, not once per stage.
These mitigations are built in:

- **Shared model cache**: the director and script stages reuse a single loaded
  model (`QwenProvider` class-level cache) instead of loading two copies.
- **Streamed loading**: `low_cpu_mem_usage=True`, `device_map="auto"` with
  `max_memory` headroom (`QWEN_VRAM_RESERVE_GB`, default 2.5) — weights stream
  from disk and overflow spills to CPU instead of crashing.
- **SDPA attention**: memory-efficient attention, no flash-attn required on T4.
- **`torch.cuda.empty_cache()`** between stages.

If you still hit `CUDA out of memory`:

1. Use **4-bit** loading (drops the model to ~4GB):
   ```bash
   export DIRECTOR_DTYPE=4bit
   export SCRIPT_DTYPE=4bit
   ```
   (requires `pip install bitsandbytes`, included in `scripts/colab_setup.sh`)
2. Or reduce per-call output length: `DIRECTOR_MAX_NEW_TOKENS=1024`.
3. If it is **system RAM** (not VRAM) that runs out on free Colab (12GB), keep
   `device_map="auto"` (already default) and avoid running other cells that
   allocate big tensors while the model is loaded.

On A100 (40GB VRAM):
- Qwen3-30B-A3B model loading: ~60-120s
- Concept generation: ~30-60s
- Production plan: ~20-40s
- Full pipeline: ~2-3 minutes
