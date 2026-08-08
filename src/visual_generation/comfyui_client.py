"""ComfyUI client stub: generate placeholder image assets based on asset plan."""
import json
from pathlib import Path


def generate_assets(project_dir: Path):
    out_dir = Path(project_dir) / 'assets'
    out_dir.mkdir(parents=True, exist_ok=True)
    # Simple asset plan
    assets = [
        {"asset_id": "asset-1", "type": "image", "prompt": "symbolic illustration of contrast", "linked_section_id": "scene_discussion", "duration_sec": 6, "engine": "comfyui-stub"}
    ]
    plan_path = Path(project_dir) / 'asset_plan.json'
    with plan_path.open('w', encoding='utf-8') as f:
        json.dump({"project_id": str(project_dir.name), "assets": assets}, f, ensure_ascii=False, indent=2)

    # Create placeholder files
    for a in assets:
        p = out_dir / (a['asset_id'] + '.png')
        with p.open('wb') as img:
            img.write(b"\x89PNG\r\n\x1a\n")
        print(f"Created placeholder asset -> {p}")
    return plan_path
