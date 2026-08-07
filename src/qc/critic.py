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
    report = {'project': str(project_dir.name), 'checks': checks}
    out = project_dir / 'reports'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'qc_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"QC report written -> {out / 'qc_report.json'}")
    return report
