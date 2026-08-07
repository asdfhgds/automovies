"""FFmpeg editor stub: assemble timeline and produce a placeholder render file."""
import json
from pathlib import Path

def assemble(project_dir: Path):
    renders_dir = Path(project_dir) / 'renders'
    renders_dir.mkdir(parents=True, exist_ok=True)
    out_file = renders_dir / 'final_render.mp4'
    # create a tiny placeholder MP4-like file header (not valid video, but placeholder)
    with out_file.open('wb') as f:
        f.write(b"\x00\x00\x00\x18ftypmp42")
    job = {"output_path": str(out_file), "status": "done"}
    job_path = renders_dir / 'render_job.json'
    with job_path.open('w', encoding='utf-8') as f:
        json.dump(job, f, ensure_ascii=False, indent=2)
    print(f"Assembled render -> {out_file}")
    return out_file
