import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Test Notebook: Stage-Oriented Pipeline\n",
                "\n",
                "This notebook tests the new artifact-driven pipeline architecture.\n",
                "Run on Google Colab with **GPU: T4** runtime."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# @title Mount Google Drive & Setup\n",
                "from google.colab import drive\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "import subprocess\n",
                "import sys\n",
                "import os\n",
                "\n",
                "# Clone repo\n",
                "if not os.path.exists('/content/automovies'):\n",
                "    subprocess.run(['git', 'clone', 'https://github.com/asdfhgds/automovies.git'], check=True)\n",
                "%cd /content/automovies\n",
                "\n",
                "!pip install -e . -q 2>&1 | tail -3\n",
                "sys.path.insert(0, '/content/automovies/src')\n",
                "\n",
                "print('Setup complete')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# @title Verify GPU & Dependencies\n",
                "import torch\n",
                "import sys\n",
                "\n",
                "print(f'Python: {sys.version}')\n",
                "print(f'PyTorch: {torch.__version__}')\n",
                "print(f'CUDA: {torch.cuda.is_available()}')\n",
                "if torch.cuda.is_available():\n",
                "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
                "    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# @title Configure Project & Storage\n",
                "import os\n",
                "import json\n",
                "\n",
                "PROJECT_ID = 'test_stage_pipeline_001'\n",
                "\n",
                "DRIVE_ROOT = '/content/drive/MyDrive/AutoMovies'\n",
                "PROJECTS_DIR = os.path.join(DRIVE_ROOT, 'projects')\n",
                "MOVIES_DIR = os.path.join(DRIVE_ROOT, 'movies')\n",
                "\n",
                "os.makedirs(PROJECTS_DIR, exist_ok=True)\n",
                "os.makedirs(MOVIES_DIR, exist_ok=True)\n",
                "\n",
                "os.environ['AUTOMOVIES_PROJECT_ROOT'] = PROJECTS_DIR\n",
                "\n",
                "project_dir = os.path.join(PROJECTS_DIR, PROJECT_ID)\n",
                "os.makedirs(project_dir, exist_ok=True)\n",
                "\n",
                "print(f'Project: {PROJECT_ID}')\n",
                "print(f'Storage: {PROJECTS_DIR}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# @title Get Source Video\n",
                "import gdown\n",
                "import os\n",
                "\n",
                "# Replace with your Google Drive file ID\n",
                "SOURCE_VIDEO_ID = ''  # e.g., '1ABC123...'\n",
                "SOURCE_VIDEO_NAME = 'source.mp4'\n",
                "\n",
                "source_path = os.path.join(project_dir, SOURCE_VIDEO_NAME)\n",
                "if os.path.exists(source_path):\n",
                "    print(f'✓ Source exists: {source_path}')\n",
                "elif SOURCE_VIDEO_ID:\n",
                "    url = f'https://drive.google.com/uc?id={SOURCE_VIDEO_ID}'\n",
                "    gdown.download(url, source_path, quiet=False)\n",
                "    print('✓ Downloaded')\n",
                "else:\n",
                "    print('⚠ No video ID - upload video to Drive and set SOURCE_VIDEO_ID')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# @title Create Project Meta\n",
                "import json\n",
                "import os\n",
                "\n",
                "meta = {\n",
                "    'project_id': PROJECT_ID,\n",
                "    'title': 'Stage Pipeline Test',\n",
                "    'source_path': source_path if os.path.exists(source_path) else None\n",
                "}\n",
                "with open(os.path.join(project_dir, 'project_meta.json'), 'w') as f:\n",
                "    json.dump(meta, f, indent=2)\n",
                "print('✓ project_meta.json created')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# @title Test Stage Runner - Dry Run Full Pipeline\n",
                "import subprocess\n",
                "import sys\n",
                "import json\n",
                "\n",
                "def run_stage(stage, config=None, force=False, dry_run=False):\n",
                "    cmd = [\n",
                "        sys.executable,\n",
                "        'scripts/run_stage.py',\n",
                "        '--storage-root', PROJECTS_DIR,\n",
                "        '--project-id', PROJECT_ID,\n",
                "        '--stage', stage,\n",
                "    ]\n",
                "    if force: cmd.append('--force')\n",
                "    if dry_run: cmd.append('--dry-run')\n",
                "    if config:\n",
                "        for k, v in config.items():\n",
                "            cmd.extend(['--config', f'{k}={json.dumps(v)}'])\n",
                "    print(f'\\n[RUN] {stage}')\n",
                "    result = subprocess.run(cmd, cwd='/content/automovies', capture_output=True, text=True, timeout=7200)\n",
                "    print(f'Exit: {result.returncode}')\n",
                "    if result.stdout: print(result.stdout[-2000:])\n",
                "    if result.stderr: print('STDERR:', result.stderr[-1000:])\n",
                "    return result.returncode == 0\n",
                "\n",
                "# Dry run full pipeline\n",
                "print('=== DRY RUN: Full Pipeline ===')\n",
                "run_stage('all', config={'grounded': True, 'num_concepts': 3, 'enricher': 'qwen3vl'}, dry_run=True)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# @title Test Individual Stage - Ingest\n",
                "run_stage('ingest', config={'title': 'Stage Test Movie'})"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# @title Test Individual Stage - Transcription\n",
                "run_stage('transcription', config={'model': 'large-v3', 'word_timestamps': True})"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# @title Test Individual Stage - Scene Indexing\n",
                "run_stage('scene_indexing', config={'threshold': 30.0})"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# @title Test Individual Stage - Movie Intelligence\n",
                "run_stage('movie_intelligence', config={'enricher': 'heuristic', 'attach_keyframes': False})"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# @title Test Individual Stage - Director\n",
                "run_stage('director', config={'grounded': True, 'num_concepts': 3, 'min_coverage': 0.4, 'target_sec': 90})"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# @title Check Project Manifest & Artifacts\n",
                "import json\n",
                "from pathlib import Path\n",
                "\n",
                "project_path = Path(PROJECTS_DIR) / PROJECT_ID\n",
                "\n",
                "# Check manifest\n",
                "manifest_path = project_path / 'project_manifest.json'\n",
                "if manifest_path.exists():\n",
                "    with open(manifest_path) as f:\n",
                "        manifest = json.load(f)\n",
                "    print('=== PROJECT MANIFEST ===')\n",
                "    for stage_name in ['ingest', 'transcription', 'scene_indexing', 'movie_intelligence', 'director']:\n",
                "        s = manifest.get('stages', {}).get(stage_name, {})\n",
                "        status = s.get('status', 'unknown')\n",
                "        icon = {'completed': '[OK]', 'running': '[>]', 'failed': '[FAIL]', 'skipped': '[SKIP]', 'not_started': '[ ]'}.get(status, '[?]')\n",
                "        dur = s.get('duration_seconds', 0)\n",
                "        print(f'  {icon} {stage_name}: {status} ({dur:.1f}s)')\n",
                "    print(f'\\nTotal artifacts: {len(manifest.get(\"artifacts\", {}))}')\n",
                "else:\n",
                "    print('No manifest found')\n",
                "\n",
                "# List all artifacts\n",
                "print('\\n=== ARTIFACTS ===')\n",
                "for root, dirs, files in os.walk(project_path):\n",
                "    for f in files:\n",
                "        p = Path(root) / f\n",
                "        rel = p.relative_to(project_path)\n",
                "        size = p.stat().st_size\n",
                "        print(f'  {rel} ({size/1024:.1f} KB)')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# @title Verify Artifact Persistence (Restart Test)\n",
                "# This simulates a Colab restart - the artifacts should persist in Google Drive\n",
                "\n",
                "import json\n",
                "from pathlib import Path\n",
                "\n",
                "project_path = Path(PROJECTS_DIR) / PROJECT_ID\n",
                "\n",
                "# Verify key artifacts exist\n",
                "key_artifacts = [\n",
                "    'project_meta.json',\n",
                "    'project_manifest.json',\n",
                "    'transcripts/transcript.json',\n",
                "    'scenes/scene_index.json',\n",
                "    'movie_index.json',\n",
                "    'director_plan.json',\n",
                "]\n",
                "\n",
                "print('=== PERSISTENCE CHECK ===')\n",
                "for art in key_artifacts:\n",
                "    p = Path(PROJECTS_DIR) / PROJECT_ID / art\n",
                "    if p.exists():\n",
                "        size = p.stat().st_size\n",
                "        print(f'[OK] {art} ({size/1024:.1f} KB)')\n",
                "    else:\n",
                "        print(f'[MISSING] {art}')\n",
                "\n",
                "# Verify manifest loads correctly\n",
                "manifest_path = Path(PROJECTS_DIR) / PROJECT_ID / 'project_manifest.json'\n",
                "if manifest_path.exists():\n",
                "    with open(manifest_path) as f:\n",
                "        m = json.load(f)\n",
                "    print(f'\\nManifest version: {m.get(\"pipeline_version\")}')\n",
                "    print(f'Project: {m.get(\"title\")}')\n",
                "    print(f'Updated: {m.get(\"updated_at\")}')\n",
                "    print(f'Artifacts registered: {len(m.get(\"artifacts\", {}))}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# @title Test Resume Capability (Force Re-run vs Skip)\n",
                "print('=== RESUME TEST ===')\n",
                "print('Running director again WITHOUT --force (should skip)...')\n",
                "run_stage('director', config={'grounded': True, 'num_concepts': 3})\n",
                "\n",
                "print('\\nRunning director WITH --force (should re-run)...')\n",
                "import subprocess\nnimport sys\nimport json\n\ncmd = [\n    sys.executable,\n    'scripts/run_stage.py',\n    '--storage-root', PROJECTS_DIR,\n    '--project-id', PROJECT_ID,\n    '--stage', 'director',\n    '--force',\n    '--config', 'grounded=true',\n    '--config', 'num_concepts=3',\n]\nresult = subprocess.run(cmd, cwd='/content/automovies', capture_output=True, text=True, timeout=1800)\nprint(f'Exit: {result.returncode}')\nif result.stdout: print(result.stdout[-1500:])\nif result.stderr: print('STDERR:', result.stderr[-1000:])"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(r'C:\Users\hp\Documents\Default Project\automovies\TEST_Stage_Pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print('Test notebook written successfully')