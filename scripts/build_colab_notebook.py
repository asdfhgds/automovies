#!/usr/bin/env python
"""Generate notebooks/colab_real_movie_tts.ipynb.

The notebook drives the full real-movie pipeline on a GPU Colab runtime:

  real movie (Google Drive) -> WhisperX -> PySceneDetect -> Qwen director
  -> evidence selection -> Qwen script -> real Kokoro TTS (CUDA) -> FFmpeg
  render (ducking/normalization/subtitles) -> QC + ffprobe validation,
  plus a cross-provider TTS benchmark.

Build with:  python scripts/build_colab_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = []

cells.append(md("""# Real Movie + Real TTS — GPU Validation
Run this notebook on a **GPU runtime** (T4/A100) to prove the end-to-end pipeline
produces an actual video-essay MP4 from a **real movie you legally own**.

- Uses the **same code the orchestrator uses** (`python src/main.py run`).
- Strict modes are ON: `REQUIRE_REAL_LLM=true` (real Qwen director + script) and
  `REQUIRE_REAL_TTS=true` (real TTS; **mock audio is refused**).
- TTS runs on **CUDA only** — real TTS is never synthesized on CPU.
- The movie file is **not** committed to the repo. It is supplied three ways:
  a **Google Drive share link** (`MOVIE_URL`, downloaded with `gdown`, no Drive
  mount needed), a path on the mounted Google Drive, or an upload.

After it finishes, `reports/tts_benchmark.json`, `provider_manifest.json`, and
`reports/qc_report.json` prove exactly which models/providers ran.
"""))

cells.append(md("""### Cell 1 — Runtime & system packages
Runs `apt-get`, clones the repo, and installs base deps (FFmpeg, PyTorch-CUDA,
Transformers, Whisper, PySceneDetect). Set `REPO_URL` / `BRANCH` to your fork and
branch, and supply the movie via `MOVIE_URL` (Google Drive share link) or
`MOVIE_PATH` (Drive/local path)."""))

cells.append(code("""# @title 1) Setup: system + repo + base deps
import os, sys, subprocess, textwrap

REPO_URL = "https://github.com/asdfhgds/automovies.git"  # @param {type:"string"}
BRANCH = "main"  # @param {type:"string"}
MOVIE_URL = ""  # @param {type:"string"} Google Drive share link (e.g. https://drive.google.com/file/d/xxx/view)
MOVIE_PATH = ""  # @param {type:"string"} Drive/local path (used if MOVIE_URL is empty)

def sh(cmd, **kw):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True, **kw)

# Always work from an absolute, deterministic location so re-running this cell
# never nests extra copies of the repo.
ROOT = "/content"
REPO_DIR = os.path.join(ROOT, "automovies")
os.chdir(ROOT)
if not os.path.isdir(os.path.join(REPO_DIR, "src")):
    if os.path.isdir(REPO_DIR):
        sh(f"rm -rf {REPO_DIR}")
    sh(f"git clone --depth 1 -b {BRANCH} {REPO_URL} {REPO_DIR}")
os.chdir(REPO_DIR)
sh("bash scripts/colab_setup.sh")

# Remember the movie source for the next cells
open("/content/movie_path.txt", "w").write(MOVIE_PATH)
open("/content/movie_url.txt", "w").write(MOVIE_URL)
print("Setup complete. Repo:", os.getcwd())
"""))

cells.append(md("""### Cell 2 — Google Drive + TTS deps
Mount Drive so a movie can be supplied without committing it (optional if
`MOVIE_URL` is set), then install the open-source TTS stack (Kokoro required;
Chatterbox / Qwen3-TTS optional)."""))

cells.append(code("""# @title 2) Mount Drive + install TTS deps
from google.colab import drive
drive.mount("/content/drive")

sh("bash scripts/colab_tts_setup.sh")

MOVIE_PATH = open("/content/movie_path.txt").read().strip()
MOVIE_URL = open("/content/movie_url.txt").read().strip()
print("MOVIE_URL =", repr(MOVIE_URL))
print("MOVIE_PATH =", repr(MOVIE_PATH))
"""))

cells.append(md("""### Cell 3 — Get + validate the movie
Priority: if `MOVIE_URL` is set the movie is downloaded from the Google Drive
share link with `gdown` (to a fixed path `/content/movie_download`), and then
verified to actually be a video with `ffprobe` (a shared link that needs
permissions, or a non-video file, fails loudly here instead of mid-pipeline).
Otherwise a `MOVIE_PATH` on the mounted Drive is used, and if that is also empty
the notebook prompts you to upload a file. The movie is **never copied into the
repo** — only its absolute path is registered in the project."""))

cells.append(code("""# @title 3) Get + validate movie file
import os, subprocess
from IPython.display import display, HTML

MOVIE_PATH = open("/content/movie_path.txt").read().strip()
MOVIE_URL = open("/content/movie_url.txt").read().strip()

# Tolerate a URL accidentally pasted into the MOVIE_PATH field
if MOVIE_PATH.startswith("http"):
    MOVIE_URL = MOVIE_URL or MOVIE_PATH
    MOVIE_PATH = ""

if MOVIE_URL and not MOVIE_PATH:
    try:
        import gdown
    except ImportError:
        sh("pip install -q gdown")
        import gdown
    print("Downloading movie from Google Drive link:", MOVIE_URL)
    # Fixed output path: avoids gdown falling back to a URL-basename filename.
    saved = gdown.download(MOVIE_URL, output="/content/movie_download", quiet=False)
    assert saved and os.path.exists(saved), "gdown failed to download the movie"
    MOVIE_PATH = saved
    open("/content/movie_path.txt", "w").write(MOVIE_PATH)

if not MOVIE_PATH:
    from google.colab import files
    print("Uploading a movie from this machine (or set MOVIE_URL / Drive path above):")
    up = files.upload()
    MOVIE_PATH = list(up.keys())[0]
    open("/content/movie_path.txt", "w").write(MOVIE_PATH)

MOVIE_PATH = os.path.abspath(os.path.expanduser(MOVIE_PATH))
print("Movie:", MOVIE_PATH, "exists:", os.path.exists(MOVIE_PATH))
assert os.path.exists(MOVIE_PATH), "Movie file not found"

probe = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration,size",
     "-show_entries", "stream=codec_type,codec_name,width,height",
     "-of", "default=noprint_wrappers=1", MOVIE_PATH],
    capture_output=True, text=True)
print(probe.stdout or probe.stderr)
# The downloaded file must actually be a video, not an HTML error page.
assert "codec_type=video" in (probe.stdout or ""), (
    "The file at MOVIE_PATH is not a valid video. If using MOVIE_URL, make sure "
    "it is a real video file shared as 'Anyone with the link'."
)
print("OK: valid video file.")
"""))

cells.append(md("""### Cell 4 — Environment + doctor
Enables the `colab-gpu` profile and both strict modes. `TTS_PROVIDER=kokoro` is
the default; switch to `chatterbox` / `qwen3_tts` if you installed them.
`TTS_DEVICE=cuda` guarantees real TTS never falls back to CPU."""))

cells.append(code("""# @title 4) Env vars + doctor
import os
os.environ["STUDIO_PROFILE"] = "colab-gpu"
os.environ["REQUIRE_REAL_LLM"] = "true"
os.environ["REQUIRE_REAL_TTS"] = "true"
os.environ["DIRECTOR_PROVIDER"] = "qwen"
os.environ["DIRECTOR_MODEL"] = "Qwen/Qwen3-4B-Instruct-2507"
os.environ["TTS_DEVICE"] = "cuda"
os.environ["TTS_PROVIDER"] = "kokoro"  # kokoro | chatterbox | qwen3_tts
os.environ["TTS_VOICE"] = "am_adam"
os.environ["RUN_TTS_BENCHMARK"] = "true"
os.environ["BURN_SUBTITLES"] = "true"

!python src/main.py doctor 2>&1 | tail -n 40
"""))

cells.append(md("""### Cell 5 — Initialize the project
Registers the real movie (path only, not the file) under `data/<project-id>`."""))

cells.append(code("""# @title 5) init project
import subprocess
from pathlib import Path

MOVIE_PATH = open("/content/movie_path.txt").read().strip()
MOVIE_PATH = Path(MOVIE_PATH).expanduser().resolve()
out = subprocess.run(
    ["python", "src/main.py", "init",
     "--title", "Real Movie Video Essay",
     "--source", str(MOVIE_PATH)],
    capture_output=True, text=True)
print(out.stdout or out.stderr)
import re
m = re.search(r"project ([0-9a-f-]{36})", out.stdout or "")
PROJECT_ID = m.group(1) if m else None
print("PROJECT_ID =", PROJECT_ID)
assert PROJECT_ID, "failed to init project"
open("/content/project_id.txt", "w").write(PROJECT_ID)
"""))

cells.append(md("""### Cell 6 — Run the full pipeline (the long one)
Real Qwen director + script on CUDA, real Kokoro TTS on CUDA, FFmpeg render with
film/music ducking, loudnorm normalization, a true-peak limiter, and burned
subtitles. This is the same entry point the CLI uses, so nothing here is special
to the notebook."""))

cells.append(code("""# @title 6) run pipeline (real LLM + real TTS + render)
import subprocess, time

PROJECT_ID = open("/content/project_id.txt").read().strip()
t0 = time.time()
out = subprocess.run(
    ["python", "src/main.py", "run", "--project-id", PROJECT_ID],
    capture_output=True, text=True, timeout=3600)
dt = time.time() - t0
print(out.stdout or out.stderr)
print(f"--- pipeline wall time: {dt:.1f}s ---")
print("(exit code)", out.returncode)
if out.returncode != 0:
    raise SystemExit(f"pipeline failed: {out.stderr[-4000:]}")
"""))

cells.append(md("""### Cell 7 — TTS benchmark
Synthesizes the same narration text with every installed TTS provider on CUDA and
records model / device / generation time / duration / sample rate / status in
`reports/tts_benchmark.json`."""))

cells.append(code("""# @title 7) TTS benchmark (CUDA)
from generation.tts_benchmark import benchmark_tts

PROJECT_ID = open("/content/project_id.txt").read().strip()
report = benchmark_tts(
    output_dir=f"data/{PROJECT_ID}/reports",
    narration={"tone": "dramatic", "emotion": "tense", "pace": 0.95,
               "energy": 0.6, "dramatic_intensity": 0.8},
)
for r in report["results"]:
    print(f"  [{r['status']:>10}] {r['provider']:<12} "
          f"model={r['model']} device={r['device']} "
          f"gen={r['generation_time_sec']}s dur={r['duration_sec']}s sr={r['sample_rate']}"
          + (f"  error={r['error']}" if r.get('error') else ""))
"""))

cells.append(md("""### Cell 8 — QC + ffprobe validation
Runs the QC critic (artifact existence, real-TTS flag, no clipping via
`volumedetect`, render probe) and shows the provider manifest."""))

cells.append(code("""# @title 8) QC + validation + manifest
import json, subprocess
from pathlib import Path

PROJECT_ID = open("/content/project_id.txt").read().strip()
proj = Path("data") / PROJECT_ID

from qc.critic import run_qc
report = run_qc(proj)
print(json.dumps(report, indent=2))

manifest = json.loads((proj / "provider_manifest.json").read_text())
print("\\n=== PROVIDER MANIFEST (excerpt) ===")
for k in ("strict_mode", "profile", "director_real_generation", "director_model",
          "script_real_generation", "script_model", "tts_provider", "tts_model",
          "tts_device", "tts_real", "transcription_real", "pipeline_total_seconds"):
    print(f"  {k}: {manifest.get(k)}")

render = proj / "renders" / "final_render.mp4"
probe = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
     "-show_entries", "stream=codec_type,codec_name,width,height",
     "-of", "json", str(render)], capture_output=True, text=True)
print("\\n=== RENDER (ffprobe) ===")
print(probe.stdout)
from IPython.display import Video
display(Video(str(render)))
"""))

cells.append(md("""### Interpreting the results
- `tts_real: true` + `tts_provider: kokoro` (or chatterbox/qwen3_tts) + a
  non-empty `render_duration_sec` with `no_clipping: true` => the real pipeline
  produced a playable MP4 with real narration.
- The benchmark shows which providers installed and their generation time /
  duration / sample rate; `cpu_skipped` entries are expected when a provider is
  installed but the runtime lacks CUDA.

**Watch for:** narrations that don't match the selected scenes (script quality),
scenes that don't illustrate the thesis (retrieval), and pacing/emotion gaps
(TTS control). Log what you see — those observations decide the next milestone.
"""))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "name": "colab_real_movie_tts.ipynb"},
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

out_path = ROOT / "notebooks" / "colab_real_movie_tts.ipynb"
out_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {out_path}")
