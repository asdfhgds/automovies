# GPU Validation (Colab)

## Real Qwen director and script run

Use a GPU runtime. Run `bash scripts/colab_setup.sh`, then set
`STUDIO_PROFILE=colab-gpu`, `DIRECTOR_PROVIDER=qwen`, and
`SCRIPT_PROVIDER=qwen`. `scripts/colab_run.py` defaults to the practical
`Qwen/Qwen3-7B-A0.5B`; override `DIRECTOR_MODEL` for larger GPUs.

```bash
python scripts/colab_run.py
python src/main.py init --title "Legal test" --source tests/fixtures/test_speech.mp4
python src/main.py run --project-id <project-id>
pytest -m llm_integration -q
```

Save the `doctor` JSON plus model, device, load time, and generation timings.
This milestone is not GPU-validated until the commands finish on a real GPU.

1) Open a new Google Colab notebook. Set Runtime > Change runtime type > GPU.

2) Run these setup cells in order:

# System packages
!apt-get update -y && apt-get install -y ffmpeg git

# Upgrade pip
!python -m pip install --upgrade pip

# Install PyTorch with CUDA (choose appropriate CUDA version for Colab runtime).
# Example (CUDA 11.8):
!pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install WhisperX and PySceneDetect
!pip install git+https://github.com/m-bain/whisperX.git
!pip install scenedetect opencv-python-headless

# (Optional) pyttsx3 for fixture synthesis
!pip install pyttsx3

3) Clone your repository (replace REPO_URL):

!git clone <REPO_URL> repo
%cd repo

4) Run doctor to confirm GPU + deps:

!python src/main.py doctor

5) Generate tiny test video and run pipeline:

!python tests/fixtures/generate_test_fixture.py tests/fixtures/test_speech.mp4 "Short GPU test"
!python src/main.py init --title "Colab GPU Test" --source tests/fixtures/test_speech.mp4
# note printed project id, then run the pipeline:
!python src/main.py run --project-id <project-id>

6) Inspect outputs in data/<project-id>/
- transcripts/transcript.json
- scenes/scene_index.json (if PySceneDetect produced it)
- scenes/scene_ranking.json
- scenes/selected_scene.json
- assets/scenes/<scene_id>.mp4

7) Use ffprobe to confirm clip properties:

!ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 data/<project-id>/assets/scenes/<scene_id>.mp4

Notes
- Choose the correct torch CUDA wheel matching the Colab runtime. If unsure, replace the pip install with the recommended command from https://pytorch.org/get-started/locally/ for your CUDA version.
- If whisperx fails due to compute type, inspect its compute_type argument and pass 'float16' for common GPUs.


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
- Qwen3-7B-A0.5B: ✅ Fits easily
- Qwen3-30B-A3B: ⚠️ May OOM without quantization
- Qwen3-235B-A22B: ❌ Requires quantization or larger GPU

**Colab A100 (40GB VRAM):**
- Qwen3-30B-A3B: ✅ Fits comfortably
- Qwen3-235B-A22B: ✅ Fits with careful settings

### Setup for Real LLM

```python
# In Colab, set up environment:
!pip install transformers accelerate

# For smaller model (T4 GPU):
%env DIRECTOR_PROVIDER=qwen
%env DIRECTOR_MODEL=Qwen/Qwen3-7B-A0.5B
%env DIRECTOR_DEVICE=cuda
%env DIRECTOR_TEMPERATURE=0.8
```

### Running Pipeline with Real Qwen

```bash
# Enable creative director with real Qwen
export CREATIVE_DIRECTOR_ENABLED=true
export DIRECTOR_PROVIDER=qwen
export DIRECTOR_MODEL=Qwen/Qwen3-30B-A3B  # or smaller model
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
  model: Qwen/Qwen3-30B-A3B
  device: auto  # or "cuda", "cpu"
  dtype: auto   # or "float16", "float32", "bfloat16"
  thinking: true
  temperature: 0.8
  top_p: 0.9
  max_new_tokens: 2048
  timeout_sec: 180
```

Environment overrides at runtime:

```bash
DIRECTOR_PROVIDER=qwen
DIRECTOR_MODEL=Qwen/Qwen3-7B-A0.5B
DIRECTOR_DEVICE=cuda
DIRECTOR_TEMPERATURE=0.7
```

### Fallback Behavior

If Qwen is unavailable or fails:
1. **During tests**: Falls back to MockLLMProvider
2. **During pipeline**: Falls back to deterministic director

This ensures the pipeline never breaks due to LLM unavailability.

### Performance Expectations

On T4 GPU (15.8GB VRAM):
- Qwen3-7B-A0.5B model loading: ~30-60s
- Concept generation (3 concepts): ~45-120s
- Production plan generation: ~30-60s
- Full pipeline: ~3-5 minutes

On A100 (40GB VRAM):
- Qwen3-30B-A3B model loading: ~60-120s
- Concept generation: ~30-60s
- Production plan: ~20-40s
- Full pipeline: ~2-3 minutes
