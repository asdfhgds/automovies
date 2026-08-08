# GPU Validation — Quick Start for Colab

This notebook automates the full GPU validation pipeline.

## Quick Steps

1. **Open in Colab**: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asdfhgds/automovies/blob/asdfhgds-autonomous-movie-studio-spec/GPU_VALIDATION.ipynb)

   (Or manually upload `GPU_VALIDATION.ipynb` to a Colab notebook.)

2. **Select GPU Runtime**:
   - Runtime → Change runtime type → GPU (T4, L4, or A100)
   - Click Save

3. **Run All Cells**:
   - Execute cells in order (Shift+Enter or Runtime → Run all)
   - The notebook will:
     - Install dependencies (ffmpeg, torch, whisperx, scenedetect)
     - Clone the repository
     - Run `python src/main.py doctor` (verify CUDA, GPU, packages)
     - Generate a tiny test video with speech
     - Run the full pipeline end-to-end
     - Validate all artifacts
     - Run integration tests

4. **Review Output**:
   - The final cell prints a summary of artifacts produced
   - Note the project ID

## Expected Output

On success, all these files should be present:

```
data/<project_id>/
├── transcripts/
│   └── transcript.json ✓
├── scenes/
│   ├── scene_index.json ✓
│   ├── scene_ranking.json ✓
│   └── selected_scene.json ✓
├── director_plan.json ✓
└── assets/
    └── scenes/
        └── <scene_id>.mp4 ✓
```

## If Something Fails

1. Check the error message in the cell output
2. Common fixes:
   - **WhisperX model load error**: torch/CUDA version mismatch; update the PyTorch install in the notebook
   - **scenedetect import error**: opencv missing; rerun the opencv-python-headless cell
   - **ffmpeg error**: ensure ffmpeg cell ran successfully
3. Run individual cells to isolate the issue
4. Report the exact error and cell output

## After Success

Copy the output to `PROJECT_STATUS.md` (format in that file shows where to update).

Include:
- GPU type (T4, L4, A100, etc.)
- VRAM
- PyTorch + CUDA versions
- Project ID and artifacts

## Manual Alternative

If notebook issues occur, run commands directly in Colab shell:

```bash
!apt-get update && apt-get install -y ffmpeg
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install git+https://github.com/m-bain/whisperX.git
!pip install scenedetect opencv-python-headless
!git clone https://github.com/asdfhgds/automovies.git repo && cd repo
!python src/main.py doctor
!python src/main.py init --title "Test" --source tests/fixtures/test_speech.mp4
!python src/main.py run --project-id <id>
```
