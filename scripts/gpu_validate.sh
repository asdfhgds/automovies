#!/bin/bash
# gpu_validate.sh
# Helper script to run the full GPU validation in a fresh environment (Colab / VM)
# Usage (Colab): run each cell's commands or execute in a shell on the target machine.

set -euo pipefail

echo "== Install system deps =="
apt-get update -y && apt-get install -y ffmpeg git

echo "== Python deps: pip upgrade =="
python -m pip install --upgrade pip

echo "== Install PyTorch with CUDA (example for CUDA 11.8) =="
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "== Install whisperx and scenedetect =="
python -m pip install git+https://github.com/m-bain/whisperX.git
python -m pip install scenedetect opencv-python-headless

echo "== Clone repo and enter workspace =="
# Replace REPO_URL with your fork/clone URL
if [ -z "${REPO_URL:-}" ]; then
  echo "Please set REPO_URL environment variable to your repository URL (https://github.com/owner/repo.git)"
  exit 1
fi

git clone "$REPO_URL" repo
cd repo

echo "== Run doctor =="
python src/main.py doctor

echo "== Generate tiny test fixture =="
python tests/fixtures/generate_test_fixture.py tests/fixtures/test_speech.mp4 "This is a short GPU test sample."

echo "== Init project =="
PROJECT_ID=$(python - <<PY
from src.main import init_project
class A: pass
args = A()
args.title='GPU Validation'
args.source='$(pwd)/tests/fixtures/test_speech.mp4'
print(init_project(args))
PY
)

echo "Project created: $PROJECT_ID"

echo "== Run pipeline =="
python src/main.py run --project-id "$PROJECT_ID"

echo "== Artifacts =="
ls -lh data/$PROJECT_ID/transcripts
ls -lh data/$PROJECT_ID/scenes
ls -lh data/$PROJECT_ID/assets/scenes

echo "== Probe the extracted clip (if any) =="
if ls data/$PROJECT_ID/assets/scenes/*.mp4 1> /dev/null 2>&1; then
  ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 data/$PROJECT_ID/assets/scenes/*.mp4
else
  echo "No extracted scene clip found"
fi

echo "GPU validation script finished. Inspect data/$PROJECT_ID for outputs."