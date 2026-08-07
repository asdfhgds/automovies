GPU Validation (Colab)

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
