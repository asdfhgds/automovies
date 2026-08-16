#!/usr/bin/env python
"""Generate notebooks/colab_grounded_movie_pipeline.ipynb.

One-click (Run All) validation of the GROUNDED movie pipeline on a fresh Colab
GPU runtime:

  real movie -> WhisperX -> PySceneDetect -> Qwen3-VL Movie Intelligence
  -> Grounded Creative Director -> Grounding Contract -> Grounded Script
  -> Grounded Editorial Plan -> Real TTS -> Timeline -> FFmpeg -> QC
  -> playable final MP4

Strict modes are ON so the pipeline can never silently fall back to mocks.

Build with:  python scripts/build_colab_grounded_notebook.py
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

cells.append(md("""# Grounded Movie Pipeline — One-Click Colab Validation

Run this notebook on a **GPU runtime** (`Runtime -> Change runtime type -> T4/A100`)
then **`Runtime -> Run all`**. It validates the whole grounded production chain on
a **real movie you legally own**, using **the same code the orchestrator runs**
(`python src/main.py run`).

```
movie
 ├─ WhisperX (real transcription)
 ├─ PySceneDetect (scene detection)
 ├─ Qwen3-VL scene enrichment (Movie Intelligence -> movie_index.json)
 ├─ Grounded Creative Director  -> grounding_contract.json
 ├─ Grounded Script             -> grounded_script.json + reports/script_grounding_report.md
 ├─ Grounded Editorial Plan     -> editorial_plan.json (excerpt windows)
 ├─ Real TTS (Kokoro on CUDA)   -> narration
 ├─ Timeline -> excerpt clips   -> timeline/editorial_timeline.json
 ├─ FFmpeg render               -> renders/final_render.mp4
 └─ QC                          -> reports/qc_report.json
```

### Strict modes are ON — no silent fallbacks
- `REQUIRE_REAL_LLM=true` — real Qwen director; mock/deterministic director is refused.
- `REQUIRE_REAL_TTS=true` — real Kokoro/Chatterbox/Qwen3-TTS narration; mock audio is refused.
- `REQUIRE_REAL_VISION=true` — **Qwen3-VL** scene enrichment; heuristic enrichment is refused.
- `GROUNDED_DIRECTOR=true` — the movie-grounded Creative Director is the source of truth.
- `EDITORIAL_MODE=true` — the evidence-grounded editorial pipeline builds the cut.

If any required real component fails, the pipeline prints **VALIDATION FAILED** with
the actual error and stops. Nothing is asserted as success that did not happen.

### Movie is never committed
Supply the movie via a `MOVIE_PATH` on your mounted Google Drive (only the path is
registered in the project). Optionally trim the input with `TRIM_TO_SEC` for a cheap
first validation run. The generated video targets 60-120 seconds.
"""))

# --------------------------------------------------------------------------- #
# Cell 1 — GPU check (very first thing; no repo needed)
# --------------------------------------------------------------------------- #
cells.append(md("""### Cell 1 — GPU check
First cell: confirm the runtime actually exposes a CUDA GPU with `nvidia-smi`.
No GPU -> the notebook should never proceed (fallbacks are forbidden)."""))

cells.append(code("""# @title 1) GPU check (nvidia-smi)
import subprocess, sys, shutil

# 0) nvidia-smi first, exactly as specified.
r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(r.stdout or r.stderr)
print("--------------")
ok = r.returncode == 0 and "NVIDIA-SMI" in (r.stdout or "")
if not ok:
    raise RuntimeError(
        "VALIDATION FAILED: this Colab runtime has NO usable GPU. "
        "Go to Runtime -> Change runtime type -> T4 GPU (or better) then re-run."
    )
print("GPU present (nvidia-smi ok).")
"""))

# --------------------------------------------------------------------------- #
# Cell 2 — Repo + base deps
# --------------------------------------------------------------------------- #
cells.append(md("""### Cell 2 — Repo + base dependencies
Pulls the **latest `origin/main`** (never an old commit), then runs the project's
own Colab setup scripts. Only what the pipeline actually needs is installed:

- `scripts/colab_setup.sh`       — FFmpeg, CUDA PyTorch, Transformers (Qwen3), Whisper, PySceneDetect
- `scripts/colab_vision_setup.sh` — Qwen2.5/3-VL (Qwen3-VL scene enrichment)
- `scripts/colab_tts_setup.sh`    — Kokoro (real TTS) + optional Chatterbox / Qwen3-TTS

Note: `REPO_URL` / `BRANCH` are the single source of truth for where the code
comes from. Edit them here if you run a fork."""))

cells.append(code("""# @title 2) Repo + base deps
import os, sys, subprocess

REPO_URL = "https://github.com/asdfhgds/automovies.git"  # @param {type:"string"}
BRANCH = "main"  # @param {type:"string"}

def sh(cmd, **kw):
    print("$", cmd)
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)
    sys.stdout.write(p.stdout or "")
    if p.returncode != 0:
        sys.stdout.write(p.stderr or "")
        raise SystemExit(f"command failed ({p.returncode}): {cmd}\\n--- tail ---\\n{(p.stderr or '')[-4000:]}")

# Deterministic location so re-runs never nest extra copies of the repo.
ROOT = "/content"
REPO_DIR = os.path.join(ROOT, "automovies")
os.chdir(ROOT)
if not os.path.isdir(os.path.join(REPO_DIR, "src")):
    if os.path.isdir(REPO_DIR):
        sh(f"rm -rf {REPO_DIR}")
    sh(f"git clone --depth 1 -b {BRANCH} {REPO_URL} {REPO_DIR}")
else:
    # Already cloned: fetch the latest fixes/scripts without wiping anything.
    sh(f"git -C {REPO_DIR} fetch origin {BRANCH}")
    sh(f"git -C {REPO_DIR} reset --hard origin/{BRANCH}")
os.chdir(REPO_DIR)
sh("bash scripts/colab_setup.sh")
print("Setup complete. Repo:", os.getcwd())
"""))

# --------------------------------------------------------------------------- #
# Cell 3 — Drive mount + vision + TTS deps
# --------------------------------------------------------------------------- #
cells.append(md("""### Cell 3 — Drive mount + vision + TTS deps
Mount Google Drive, then install the Qwen3-VL scene-enrichment deps and the
open-source TTS stack. The TTS setup restores a Transformers that still exposes
`Qwen3ForCausalLM` after any package forced a too-new build."""))

cells.append(code("""# @title 3) Mount Drive + install vision/TTS deps
from google.colab import drive
drive.mount("/content/drive")
sh("bash scripts/colab_vision_setup.sh")
sh("bash scripts/colab_tts_setup.sh")
print("Vision + TTS deps installed.")
"""))

# --------------------------------------------------------------------------- #
# Cell 4 — Env + doctor
# --------------------------------------------------------------------------- #
cells.append(md("""### Cell 4 — Production env + doctor
Configures the `colab-gpu` profile and strict real-LLM / real-vision / real-TTS
modes, then runs `python src/main.py doctor`. The doctor prints GPU name, VRAM,
CUDA, PyTorch, FFmpeg, WhisperX, PySceneDetect, Transformers, Accelerate, Qwen3
and Qwen3-VL, and this cell **fails loudly** if any required piece is missing."""))

cells.append(code("""# @title 4) Production env + doctor
import os, json, subprocess, sys

os.environ["STUDIO_PROFILE"] = "colab-gpu"
os.environ["CREATIVE_DIRECTOR_ENABLED"] = "true"
os.environ["GROUNDED_DIRECTOR"] = "true"           # grounded director is the source of truth
os.environ["EDITORIAL_MODE"] = "true"              # evidence-grounded editorial pipeline
os.environ["EDITORIAL_TARGET_SEC"] = "90"          # 60-120s validation target
os.environ["REQUIRE_REAL_LLM"] = "true"
os.environ["REQUIRE_REAL_VISION"] = "true"
os.environ["REQUIRE_REAL_TTS"] = "true"
os.environ["DIRECTOR_PROVIDER"] = "qwen"
os.environ["DIRECTOR_MODEL"] = "Qwen/Qwen3-4B-Instruct-2507"
os.environ["DIRECTOR_DEVICE"] = "cuda"
os.environ["VISION_ENRICHER"] = "qwen3vl"
os.environ["VISION_MODEL"] = "Qwen/Qwen2.5-VL-7B-Instruct"  # Qwen3-VL / Qwen2.5-VL real vision
os.environ["VISION_DEVICE"] = "cuda"
os.environ["TTS_PROVIDER"] = "kokoro"   # kokoro | chatterbox | qwen3_tts
os.environ["TTS_DEVICE"] = "cuda"
os.environ["TTS_VOICE"] = "am_adam"
os.environ["BURN_SUBTITLES"] = "true"

# 1) doctor — human-readable report
out = subprocess.run(["python", "src/main.py", "doctor"],
                     capture_output=True, text=True)
text = (out.stdout or "") + (out.stderr or "")
print(text[-6000:] if len(text) > 6000 else text)

# 2) parse the doctor JSON summary and FAIL LOUDLY on anything required.
summary = None
for line in reversed(((out.stdout or "") + (out.stderr or "")).splitlines()):
    if line.lstrip().startswith("{"):
        try:
            summary = json.loads(line)
            break
        except Exception:
            summary = None
if summary is None:
    raise RuntimeError("VALIDATION FAILED: could not parse doctor JSON summary. "
                       "Check the doctor output above.")
errors = []
if not summary.get("nvidia_smi"):
    errors.append("nvidia-smi missing")
if not (summary.get("torch") or {}).get("cuda_available"):
    errors.append("CUDA unavailable")
if not summary.get("ffmpeg"):
    errors.append("ffmpeg missing")
if not summary.get("whisperx"):
    errors.append("whisperx missing")
if not summary.get("pyscenedetect"):
    errors.append("pyscenedetect missing")
if not summary.get("transformers"):
    errors.append("transformers missing")
if not summary.get("accelerate"):
    errors.append("accelerate missing")
props = (summary.get("torch") or {})
print("\\nGPU:", props.get("gpu_name"), "| VRAM MB:", props.get("total_memory_mb"))
q = summary.get("qwen") or {}
if not q.get("qwen3_available"):
    errors.append("Qwen3ForCausalLM not importable")
if not q.get("qwen_vl_available"):
    errors.append("Qwen2.5/3-VL not importable")
tts = (summary.get("providers") or {}).get("tts") or {}
if not tts.get(summary.get("tts_provider", "kokoro")):
    errors.append("real TTS provider not available: " + summary.get("tts_provider", ""))
if errors:
    raise RuntimeError("VALIDATION FAILED: " + "; ".join(errors))
print("Doctor OK: GPU, CUDA, FFmpeg, WhisperX, PySceneDetect, Transformers, "
      "Accelerate, Qwen3, Qwen3-VL, real TTS all present.")
"""))

# --------------------------------------------------------------------------- #
# Cell 5 — Movie config (the ONLY thing a user must change per movie)
# --------------------------------------------------------------------------- #
cells.append(md("""### Cell 5 — Point at your movie (single path)
Set `MOVIE_PATH` to the **same real movie** you own and used for prior pipeline
validation (so results are directly comparable: grounded concept + grounded script
+ short evidence excerpts instead of a generic chema). Only this path changes
when you use a different movie. A `MOVIE_URL` (Google Drive share link) is optional
and downloads the file to `/content/movie_download`. `TRIM_TO_SEC` keeps the input
cheap for a first validation run (0 = full movie; the output always targets 60-120s)."""))

cells.append(code("""# @title 5) Movie path config
MOVIE_PATH = "/content/drive/MyDrive/movie_tests/my_movie.mp4"  # @param {type:"string"} Drive path to your real movie
MOVIE_URL = ""   # @param {type:"string"} Optional Google Drive share link (https://drive.google.com/file/d/xxx/view)
TRIM_TO_SEC = 0  # @param {type:"number"} Trim input to first N seconds (0 = full movie)
print("MOVIE_PATH =", MOVIE_PATH)
print("MOVIE_URL  =", MOVIE_URL)
print("TRIM_TO_SEC=", TRIM_TO_SEC)
"""))

# --------------------------------------------------------------------------- #
# Cell 6 — Get + validate the movie file
# --------------------------------------------------------------------------- #
cells.append(md("""### Cell 6 — Get + validate the movie
Resolution order: `MOVIE_URL` (Drive share link, via `gdown`) → `MOVIE_PATH`
(Drive path). The file id is normalized to the `uc?id=` form gdown handles
reliably, and the result is verified to actually be a video before the pipeline
starts. A non-video file or a permission-blocked link fails loudly here instead of
mid-pipeline."""))

cells.append(code("""# @title 6) Get + validate movie file
import os, re, subprocess

if MOVIE_URL and not MOVIE_PATH.startswith("/content/drive"):
    try:
        import gdown
    except ImportError:
        subprocess.run(["pip", "install", "-q", "gdown"], check=False)
        import gdown
    m = re.search(r"(?:file/d/|id=)([a-zA-Z0-9_-]{10,})", MOVIE_URL)
    if m:
        MOVIE_URL = f"https://drive.google.com/uc?id={m.group(1)}&export=download"
        print("Using Drive file id:", m.group(1))
    print("Downloading movie from Google Drive link:", MOVIE_URL)
    saved = gdown.download(MOVIE_URL, output="/content/movie_download", quiet=False)
    assert saved and os.path.exists(saved), "gdown failed to download the movie"
    MOVIE_PATH = saved
    size = os.path.getsize(MOVIE_PATH)
    print(f"Downloaded {size/1e6:.1f} MB to {MOVIE_PATH}")
    assert size > 1024 * 1024, (
        f"Downloaded file is only {size} bytes - this is the error page, not the "
        "movie. Make sure the Drive link points to a real video file shared as "
        "'Anyone with the link'."
    )

MOVIE_PATH = os.path.abspath(os.path.expanduser(MOVIE_PATH))
print("Movie:", MOVIE_PATH, "exists:", os.path.exists(MOVIE_PATH))
assert os.path.exists(MOVIE_PATH), (
    "VALIDATION FAILED: movie not found at MOVIE_PATH. Update Cell 5."
)

probe = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration,size",
     "-show_entries", "stream=codec_type,codec_name,width,height",
     "-of", "default=noprint_wrappers=1", MOVIE_PATH],
    capture_output=True, text=True)
print(probe.stdout or probe.stderr)
assert "codec_type=video" in (probe.stdout or ""), (
    "VALIDATION FAILED: the file at MOVIE_PATH is not a valid video."
)

if int(TRIM_TO_SEC or 0) > 0:
    trimmed = "/content/movie_trimmed.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", MOVIE_PATH,
         "-t", str(int(TRIM_TO_SEC)), "-c", "copy", trimmed], check=True)
    MOVIE_PATH = trimmed
    print(f"Trimmed to first {int(TRIM_TO_SEC)}s -> {MOVIE_PATH}")

open("/content/movie_path.txt", "w").write(MOVIE_PATH)
print("OK: valid video file registered.")
"""))

# --------------------------------------------------------------------------- #
# Cell 7 — Init project
# --------------------------------------------------------------------------- #
cells.append(md("""### Cell 7 — Initialize the project
Registers the real movie (path only, never the file) and captures the returned
`PROJECT_ID` programmatically — no copy/paste needed."""))

cells.append(code("""# @title 7) init project
import subprocess, re
from pathlib import Path

MOVIE_PATH = Path(open("/content/movie_path.txt").read().strip()).expanduser().resolve()
assert MOVIE_PATH.exists(), "no movie was registered - run Cell 6 first"
out = subprocess.run(
    ["python", "src/main.py", "init",
     "--title", "Grounded Movie Test",
     "--source", str(MOVIE_PATH)],
    capture_output=True, text=True)
print(out.stdout or out.stderr)
m = re.search(r"project ([0-9a-f-]{36})", out.stdout or "")
PROJECT_ID = m.group(1) if m else None
print("PROJECT_ID =", PROJECT_ID)
assert PROJECT_ID, "VALIDATION FAILED: failed to init project"
open("/content/project_id.txt", "w").write(PROJECT_ID)
"""))

# --------------------------------------------------------------------------- #
# Cell 8 — Run the full grounded pipeline
# --------------------------------------------------------------------------- #
cells.append(md("""### Cell 8 — Run the full grounded pipeline (the long one)
This is the same `python src/main.py run` entry point the CLI uses. It executes
the entire chain with real providers on CUDA, so nothing here is special to the
notebook. Allow up to 2h for a full-length movie (transcription + full-frame scene
detection + Qwen3-VL enrichment + grounded director + script + TTS + render)."""))

cells.append(code("""# @title 8) run grounded pipeline (real LLM + real vision + real TTS + render)
import subprocess, time

PROJECT_ID = open("/content/project_id.txt").read().strip()
t0 = time.time()
out = subprocess.run(
    ["python", "src/main.py", "run", "--project-id", PROJECT_ID, "--profile", "colab-gpu"],
    timeout=10800)
dt = time.time() - t0
print(f"--- pipeline wall time: {dt:.1f}s ---")
print("(exit code)", out.returncode)
if out.returncode != 0:
    raise RuntimeError("VALIDATION FAILED: pipeline failed (see logs above).")
open("/content/pipeline_wall_sec.txt", "w").write(f"{dt:.1f}")
"""))

# --------------------------------------------------------------------------- #
# Cell 9 — Verify the grounded chain artifacts + script grounding
# --------------------------------------------------------------------------- #
cells.append(md("""### Cell 9 — Verify the grounded chain + script grounding
Inspects the actual artifact **contents** (not just existence):

- `director_plan.json`       — grounded plan (must carry `grounded: true`)
- `grounding_contract.json`  — thesis / evidence requirements / supporting scenes / motifs / intent
- `grounded_script.json` + `script.json` — every section with real scene refs
- `reports/script_grounding_report.md`   — human-inspectable grounding report
- `editorial_plan.json`      — segments + excerpt windows
- `timeline/editorial_timeline.json`     — excerpt clips
- `renders/final_render.mp4`, `reports/qc_report.json`, `provider_manifest.json`

Every script section must reference real scene ids; excerpt windows must be valid
(0 <= start < end <= scene end, short caps) and must never exceed their source
scene. If grounding validation fails the notebook stops with the exact issue."""))

cells.append(code("""# @title 9) Verify grounded chain + script grounding
import json, sys
from pathlib import Path

sys.path.insert(0, "src")
PROJECT_ID = open("/content/project_id.txt").read().strip()
proj = Path("data") / PROJECT_ID

def load(rel, default=None):
    p = proj / rel
    if not p.exists():
        p2 = proj / "reports" / rel
        if p2.exists():
            p = p2
        else:
            return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default

errors = []

# --- artifact existence ---
required = [
    "director_plan.json", "grounding_contract.json", "grounded_script.json",
    "script.json", "editorial_plan.json", "timeline/editorial_timeline.json",
    "renders/final_render.mp4", "reports/qc_report.json", "provider_manifest.json",
    "reports/script_grounding_report.md",
]
for rel in required:
    if not (proj / rel).exists():
        errors.append(f"missing artifact: {rel}")
print("Artifacts present:", sum((proj / rel).exists() for rel in required), "/", len(required))

# --- director plan is grounded ---
plan = load("director_plan.json", {})
if not plan.get("grounded"):
    errors.append("director_plan.json is NOT grounded (grounded != true)")
if not plan.get("thesis"):
    errors.append("director_plan.json has no thesis")
if not plan.get("supporting_scenes"):
    errors.append("director_plan.json has no supporting_scenes")

# --- grounding contract ---
contract = load("grounding_contract.json", {})
concept = contract.get("concept") or {}
if not concept.get("thesis"):
    errors.append("grounding_contract.json has no concept.thesis")
if not contract.get("evidence_requirements"):
    errors.append("grounding_contract.json has no evidence_requirements")
if not contract.get("supporting_scenes"):
    errors.append("grounding_contract.json has no supporting_scenes")

# --- movie intelligence scene bounds ---
movie_index = load("movie_index.json", {})
scenes = movie_index.get("scenes") or []
by_id = {s.get("scene_id"): s for s in scenes}
if not scenes:
    errors.append("movie_index.json has no scenes (Qwen3-VL enrichment required)")
else:
    enr = (movie_index.get("provenance") or {}).get("scene_enricher")
    if enr != "qwen3vl":
        errors.append(f"VALIDATION FAILED: movie intelligence was NOT enriched by "
                      f"Qwen3-VL (provenance.scene_enricher={enr!r}, expected 'qwen3vl').")
    print(f"movie_index: {len(scenes)} scenes, enricher={enr}")

# --- grounded script sections are grounded ---
gscript = load("grounded_script.json", {})
sections = gscript.get("sections") or []
print(f"grounded_script: {len(sections)} sections")
if len(sections) < 5:
    errors.append(f"grounded_script has too few sections ({len(sections)})")
for sec in sections:
    sid = sec.get("id")
    if not sec.get("scene_ids"):
        errors.append(f"section {sid} has no scene references")
    for scid in sec.get("scene_ids") or []:
        if scid not in by_id:
            errors.append(f"section {sid} references unknown scene {scid}")
    for ev in sec.get("narrative_evidence") or []:
        esc = ev.get("scene_id")
        if esc not in by_id:
            errors.append(f"section {sid} evidence references unknown scene {esc}")
            continue
        src = by_id[esc]
        w0, w1 = ev.get("start_sec"), ev.get("end_sec")
        if w0 is None or w1 is None:
            errors.append(f"section {sid} evidence {esc} missing excerpt timestamps")
            continue
        if not (0 <= w0 < w1):
            errors.append(f"section {sid} evidence {esc} invalid window {w0}-{w1}")
        if float(w1) > float(src.get("end_sec", 1e18)) + 1e-6:
            errors.append(f"section {sid} evidence {esc} window {w0}-{w1} exceeds "
                          f"scene end {src.get('end_sec')}")
        if float(w1) - float(w0) > 6.0 + 1e-6:
            errors.append(f"section {sid} evidence {esc} excerpt too long "
                          f"({w1 - w0:.1f}s > 6s)")

# --- script metadata contains source movie / project id ---
m = load("script.json", {})
meta = m.get("metadata") or {}
if gscript.get("movie_id") or gscript.get("project_id"):
    if gscript.get("project_id") and gscript.get("project_id") != PROJECT_ID:
        errors.append("grounded_script project_id mismatch")

# --- report md exists + non-empty ---
rep = proj / "reports" / "script_grounding_report.md"
if not rep.exists() or len((rep.read_text(encoding="utf-8") or "").strip()) < 100:
    errors.append("reports/script_grounding_report.md missing or empty")

if errors:
    print("\\nGROUNDING ERRORS:")
    for e in errors:
        print("  -", e)
    raise RuntimeError("VALIDATION FAILED: grounding checks failed (see above).")
print("Grounding chain OK: all script sections cite real scenes with valid "
      "short excerpt windows; contract + report present.")
"""))

# --------------------------------------------------------------------------- #
# Cell 10 — Verify real providers
# --------------------------------------------------------------------------- #
cells.append(md("""### Cell 10 — Verify real providers from the manifest
Confirms via `provider_manifest.json` that:

- Director provider = real Qwen, on CUDA
- Vision provider = Qwen3-VL
- Script provider = real (editorial/grounded path)
- TTS provider = real (Kokoro), on CUDA
- Transcription = real (WhisperX)

If any required component silently used a fallback → **VALIDATION FAILED**."""))

cells.append(code("""# @title 10) Verify real providers
import json
from pathlib import Path

PROJECT_ID = open("/content/project_id.txt").read().strip()
proj = Path("data") / PROJECT_ID
manifest = json.loads((proj / "provider_manifest.json").read_text(encoding="utf-8"))

for k in ("profile", "strict_mode", "transcription_real",
          "director_provider", "director_model", "director_device",
          "director_real_generation", "vision_enricher",
          "script_provider", "script_real_generation", "script_device",
          "tts_provider", "tts_model", "tts_device", "tts_real",
          "editorial_mode", "editorial_plan_built", "editorial_timeline_built",
          "pipeline_total_seconds"):
    print(f"  {k}: {manifest.get(k)}")

errors = []
if manifest.get("director_real_generation") is not True:
    errors.append("Director did NOT run real generation")
if str(manifest.get("director_provider", "")).lower() != "qwen":
    errors.append("Director provider is not real Qwen: " + str(manifest.get("director_provider")))
if manifest.get("vision_enricher") not in ("qwen3vl", None):
    errors.append("Vision provider is not Qwen3-VL: " + str(manifest.get("vision_enricher")))
if manifest.get("script_real_generation") is not True:
    errors.append("Script was not produced by the real (grounded/editorial) path")
if manifest.get("tts_real") is not True:
    errors.append("TTS was NOT real (mock fallback was used)")
if str(manifest.get("tts_device", "")).lower() != "cuda":
    errors.append("TTS did not run on CUDA: " + str(manifest.get("tts_device")))

# Validate the vision provenance from the movie index itself.
mi = json.loads((proj / "movie_index.json").read_text(encoding="utf-8"))
if (mi.get("provenance") or {}).get("scene_enricher") != "qwen3vl":
    errors.append("movie_index provenance.scene_enricher != qwen3vl")

if errors:
    print("\\nPROVIDER ERRORS:")
    for e in errors:
        print("  -", e)
    raise RuntimeError("VALIDATION FAILED: a required real provider silently "
                       "used a fallback (see above). Not claiming success.")
print("Providers OK: real Qwen director + Qwen3-VL vision + real script path + "
      "real CUDA TTS. No mock fallbacks.")
"""))

# --------------------------------------------------------------------------- #
# Cell 11 — Verify video + QC
# --------------------------------------------------------------------------- #
cells.append(md("""### Cell 11 — Verify the video
`ffprobe` inspection of the final render (duration / video codec / audio codec /
resolution / fps / sample rate / size), confirm **H.264 + AAC + duration > 0**,
then run the project's own QC critic which checks playability, real-TTS narration
flag and no clipping."""))

cells.append(code("""# @title 11) Verify video (ffprobe) + QC
import json, subprocess, sys
from pathlib import Path

sys.path.insert(0, "src")
PROJECT_ID = open("/content/project_id.txt").read().strip()
proj = Path("data") / PROJECT_ID
render = proj / "renders" / "final_render.mp4"
assert render.exists() and render.stat().st_size > 0, "VALIDATION FAILED: final_render.mp4 missing/empty"

probe = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries",
     "format=duration,size,bit_rate",
     "-show_entries", "stream=index,codec_type,codec_name,width,height,r_frame_rate,"
     "sample_rate,channels",
     "-of", "json", str(render)], capture_output=True, text=True, check=True)
info = json.loads(probe.stdout)
fmt = info.get("format", {})
streams = {s.get("codec_type"): s for s in info.get("streams", [])}
print("duration(s):", fmt.get("duration"))
print("size(bytes):", fmt.get("size"))
print("video:", {k: streams.get("video", {}).get(k) for k in ("codec_name","width","height","r_frame_rate")})
print("audio:", {k: streams.get("audio", {}).get(k) for k in ("codec_name","sample_rate","channels")})

errors = []
try:
    dur = float(fmt.get("duration") or 0)
except (TypeError, ValueError):
    dur = 0
if dur <= 0:
    errors.append("render duration <= 0")
v = streams.get("video") or {}
a = streams.get("audio") or {}
if str(v.get("codec_name", "")).lower() not in ("h264", "libx264"):
    errors.append("video codec is not H.264: " + str(v.get("codec_name")))
if str(a.get("codec_name", "")).lower() != "aac":
    errors.append("audio codec is not AAC: " + str(a.get("codec_name")))
print("\\nRender size:", round(render.stat().st_size / 1e6, 1), "MB")

# Existing QC critic.
from qc.critic import run_qc
qc = run_qc(proj)
print("\\nQC checks passed subset:")
for k in ("render", "render_non_empty", "render_has_video", "render_has_audio",
          "narration_real_tts", "no_clipping", "editorial_plan", "editorial_timeline"):
    print(f"  {k}: {qc['checks'].get(k)}")
qe = qc.get("checks") or {}
if qe.get("render") is not True:
    errors.append("QC render check failed")
if qe.get("no_clipping") is not True:
    errors.append("QC no_clipping check failed (audio clips)")
if qe.get("narration_real_tts") is not True:
    errors.append("QC narration_real_tts check failed")
qc_out = proj / "reports" / "qc_report.json"
print("QC report:", qc_out)

if errors:
    print("\\nVIDEO ERRORS:"); [print("  -", e) for e in errors]
    raise RuntimeError("VALIDATION FAILED: video/QC checks failed (see above).")
print("Video OK: H.264 + AAC + duration>0; QC passed.")
"""))

# --------------------------------------------------------------------------- #
# Cell 12 — Human-readable report
# --------------------------------------------------------------------------- #
cells.append(md("""### Cell 12 — Human-readable validation report
Writes `GPU_GROUNDED_VALIDATION_REPORT.md` (project id, movie, GPU, VRAM, CUDA,
Python, models, provider runtimes, QC result, final video path). Only real timings
read from the manifest/wall-clock are included — nothing is fabricated."""))

cells.append(code("""# @title 12) Generate GPU_GROUNDED_VALIDATION_REPORT.md
import json, os, platform, time
from pathlib import Path

PROJECT_ID = open("/content/project_id.txt").read().strip()
proj = Path("data") / PROJECT_ID
manifest = json.loads((proj / "provider_manifest.json").read_text(encoding="utf-8"))

import torch
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"
vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) \
    if torch.cuda.is_available() else None
cuda_ver = torch.version.cuda if torch.cuda.is_available() else "n/a"
wall = float(open("/content/pipeline_wall_sec.txt").read().strip() or 0)

movie = manifest.get("movie_title") or ""
mi = json.loads((proj / "movie_index.json").read_text(encoding="utf-8"))
if not movie:
    movie = (mi.get("movie") or {}).get("title") or "unknown"

d = {
    "Project ID": PROJECT_ID,
    "Movie": movie,
    "GPU": gpu_name,
    "VRAM": f"{vram_gb} GB" if vram_gb else "n/a",
    "CUDA": cuda_ver,
    "Python": platform.python_version(),
    "Qwen model (director)": manifest.get("director_model"),
    "Qwen3-VL model (vision)": os.environ.get("VISION_MODEL"),
    "TTS provider": manifest.get("tts_provider"),
    "TTS model": manifest.get("tts_model"),
    "Director runtime (s)": manifest.get("director_seconds"),
    "Vision runtime (s)": manifest.get("vision_seconds"),
    "TTS runtime (s)": manifest.get("tts_seconds"),
    "Script runtime (s)": manifest.get("script_seconds"),
    "Pipeline total (s)": manifest.get("pipeline_total_seconds"),
    "Wall clock run (s)": round(wall, 1),
    "QC result": "PASS" if (json.loads((proj / "reports" / "qc_report.json").read_text()).get("passed")) else "FAIL",
    "Final video path": str(proj / "renders" / "final_render.mp4"),
}
lines = ["# GPU Grounded Pipeline Validation Report", ""]
for k, v in d.items():
    lines.append(f"- **{k}**: {v}")
lines += ["", f"Runtime breakdown (from provider_manifest.json): "
             f"director={d['Director runtime (s)']}s, "
             f"vision={d['Vision runtime (s)']}s, "
             f"tts={d['TTS runtime (s)']}s, "
             f"script={d['Script runtime (s)']}s, "
             f"total={d['Pipeline total (s)']}s."]
lines.append("")
report_path = proj / "GPU_GROUNDED_VALIDATION_REPORT.md"
report_path.write_text("\\n".join(lines), encoding="utf-8")
print("\\n".join(lines))
print("\\nReport written ->", report_path)
"""))

# --------------------------------------------------------------------------- #
# Cell 13 — Show the important outputs
# --------------------------------------------------------------------------- #
cells.append(md("""### Cell 13 — Show the important outputs
Selected concept + thesis, grounding contract summary, script sections with their
scene references, editorial plan segments + excerpt windows, and the playable
final video in-line."""))

cells.append(code("""# @title 13) Show concept / contract / script / plan / video
import json
from pathlib import Path
from IPython.display import display, Video

PROJECT_ID = open("/content/project_id.txt").read().strip()
proj = Path("data") / PROJECT_ID

def load(rel):
    return json.loads((proj / rel).read_text(encoding="utf-8"))

plan = load("director_plan.json")
contract = load("grounding_contract.json")
gscript = load("grounded_script.json")
eplan = load("editorial_plan.json")

print("=== SELECTED CONCEPT ===")
print("Title:", (contract.get("concept") or {}).get("title"))
print("Thesis:", (contract.get("concept") or {}).get("thesis"))
print("Hook:", (contract.get("concept") or {}).get("hook"))
print("Why interesting:", (contract.get("concept") or {}).get("why_interesting"))

print("\\n=== GROUNDING CONTRACT (summary) ===")
print("Evidence requirements:", contract.get("evidence_requirements"))
print("Supporting scenes:", [(s.get("scene_id"), s.get("start_sec"), s.get("end_sec"))
                             for s in contract.get("supporting_scenes", [])])
print("Visual motifs:", contract.get("visual_motifs"))
print("Character focus:", contract.get("character_focus"))
print("Editorial intent:", contract.get("editorial_intent"))

print("\\n=== GROUNDED SCRIPT (sections) ===")
for sec in gscript.get("sections", []):
    print(f"- [{sec.get('id')}] scenes={sec.get('scene_ids')} "
          f"est={sec.get('estimated_seconds')}s")
    print("    narration:", (sec.get('narration') or '')[:110])

print("\\n=== EDITORIAL PLAN (segments + excerpt windows) ===")
for seg in eplan.get("segments", []):
    print(f"- [{seg.get('id')}] {seg.get('purpose')}")
    for ev in seg.get("evidence", []):
        print(f"    evidence {ev.get('scene_id')} {ev.get('start_sec')}-{ev.get('end_sec')}s")
    print("    narration:", (seg.get("narration") or {}).get("text", "")[:100])

render = proj / "renders" / "final_render.mp4"
print("\\n=== FINAL VIDEO ===")
display(Video(str(render), width=480))
"""))

# --------------------------------------------------------------------------- #
# Cell 14 — Download links
# --------------------------------------------------------------------------- #
cells.append(md("""### Cell 14 — Download links
Click-to-download the final video, the validation report, and every decision
artifact produced by the grounded chain."""))

cells.append(code("""# @title 14) Download links
from pathlib import Path
from google.colab import files

PROJECT_ID = open("/content/project_id.txt").read().strip()
proj = Path("data") / PROJECT_ID

to_download = [
    "renders/final_render.mp4",
    "GPU_GROUNDED_VALIDATION_REPORT.md",
    "director_plan.json",
    "grounding_contract.json",
    "grounded_script.json",
    "script.json",
    "reports/script_grounding_report.md",
    "editorial_plan.json",
    "timeline/editorial_timeline.json",
    "reports/qc_report.json",
    "provider_manifest.json",
]
missing = [rel for rel in to_download if not (proj / rel).exists()]
if missing:
    print("Missing:", missing)
else:
    for rel in to_download:
        print("Downloading:", rel)
        files.download(str(proj / rel))
"""))

cells.append(md("""# Interpretation cheatsheet

**A playable MP4 is *not* success** — watch the final video and score:

1. **Idea** — Is the grounded thesis original and specific to this movie?
2. **Movie understanding** — Did Qwen3-VL actually understand the scenes?
3. **Script** — Analysis grounded in real moments, not a generic summary?
4. **Evidence** — Does narration match the exact excerpts shown?
5. **Editing** — Do the short excerpt windows create meaning?
6. **TTS** — Natural real voice, not a robotic read?
7. **Audio** — Narration clear, no clipping (`QC: no_clipping`)?
8. **Overall** — Would you publish this?

Compare against the previous run: *generic concept, generic script, long scene
montage* should now be *grounded concept, grounded script, short evidence
excerpts, grounded editorial plan*.
"""))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "name": "colab_grounded_movie_pipeline.ipynb"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

out_path = ROOT / "notebooks" / "colab_grounded_movie_pipeline.ipynb"
out_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {out_path}")