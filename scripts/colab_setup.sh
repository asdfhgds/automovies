#!/usr/bin/env bash
set -euo pipefail
apt-get update -y
apt-get install -y ffmpeg git
python -m pip install --upgrade pip
python -m pip install transformers accelerate scenedetect opencv-python-headless
python -m pip install git+https://github.com/m-bain/whisperX.git
