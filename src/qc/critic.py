"""Quality control stub: perform basic checks on produced artifacts."""
import json
from pathlib import Path


def run_qc(project_dir: Path):
    project_dir = Path(project_dir)
    checks = {}
    # check director plan
    checks['director_plan'] = (project_dir / 'director_plan.json').exists()
    checks['script'] = (project_dir / 'script.json').exists()
    checks['scene_cards'] = (project_dir / 'scenes' / 'scene_cards.json').exists()
    checks['assets'] = any((project_dir / 'assets').glob('*')) if (project_dir / 'assets').exists() else False
    checks['render'] = (project_dir / 'renders' / 'final_render.mp4').exists()

    # multi-scene cut: require a selection file and a clip per selected scene
    selected_scenes_path = project_dir / 'scenes' / 'selected_scenes.json'
    selected_scene_path = project_dir / 'scenes' / 'selected_scene.json'
    if selected_scenes_path.exists():
        selections = json.loads(selected_scenes_path.read_text(encoding='utf-8'))
        checks['selected_scenes'] = isinstance(selections, list) and len(selections) > 0
        checks['scene_clips'] = all(
            (project_dir / 'assets' / 'scenes' / f"{s.get('scene_id')}.mp4").exists()
            for s in selections
        ) if isinstance(selections, list) else False
    elif selected_scene_path.exists():
        checks['selected_scenes'] = True
        sel = json.loads(selected_scene_path.read_text(encoding='utf-8'))
        checks['scene_clips'] = (project_dir / 'assets' / 'scenes' / f"{sel.get('scene_id')}.mp4").exists()
    else:
        checks['selected_scenes'] = False
        checks['scene_clips'] = False

    report = {'project': str(project_dir.name), 'checks': checks}
    out = project_dir / 'reports'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'qc_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"QC report written -> {out / 'qc_report.json'}")
    return report
