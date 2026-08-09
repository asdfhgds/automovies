"""Select ranked scenes for a production plan.

``selected_scenes.json`` is the canonical multi-scene selection artifact.  The
legacy ``selected_scene.json`` is retained as the first selection so existing
consumers can continue to run unchanged.
"""
import json
from pathlib import Path
from typing import Optional


def _load_scenes(*paths: Path):
    scenes = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data = data.get("scenes", [])
            for scene in data if isinstance(data, list) else []:
                scene_id = scene.get("scene_id")
                if scene_id:
                    scenes[scene_id] = scene
        except (OSError, ValueError, TypeError):
            continue
    return scenes


def _valid_selection(ranking, scene, evidence_requirement=None):
    start = scene.get("start_sec")
    end = scene.get("end_sec")
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return {
        "scene_id": ranking["scene_id"],
        "start_sec": start,
        "end_sec": end,
        "score": ranking.get("score"),
        "reason": ranking.get("reason"),
        "evidence_requirement": evidence_requirement,
    }


def select_scenes(project_dir: Path, max_scenes: int = 3, scene_requirements=None) -> Path:
    """Select distinct, valid scenes in rank order.

    When a director supplies ``scene_requirements``, each requirement gets a
    best available scene first.  Requirements are advisory: a project with no
    scene-type metadata still receives the highest-ranked distinct scenes.
    """
    project_dir = Path(project_dir)
    ranking_path = project_dir / 'scenes' / 'scene_ranking.json'
    index_path = project_dir / 'scenes' / 'scene_index.json'
    cards_path = project_dir / 'scenes' / 'scene_cards.json'

    if not ranking_path.exists():
        raise FileNotFoundError('scene_ranking.json not found')

    with ranking_path.open('r', encoding='utf-8') as f:
        rankings = json.load(f)

    scenes = _load_scenes(index_path, cards_path)
    selected, used_ids = [], set()
    requirements = scene_requirements or []

    def add_first(requirement=None, preferred_types=None):
        for ranking in rankings:
            scene_id = ranking.get("scene_id")
            if scene_id in used_ids or scene_id not in scenes:
                continue
            scene = scenes[scene_id]
            scene_types = set(scene.get("scene_types", []))
            scene_types.add(scene.get("scene_type", ""))
            if preferred_types and scene_types.isdisjoint(preferred_types):
                continue
            selection = _valid_selection(ranking, scene, requirement)
            if selection:
                selected.append(selection)
                used_ids.add(scene_id)
                return True
        return False

    for requirement in requirements:
        if len(selected) >= max_scenes:
            break
        preferred = set(requirement.get("preferred_scene_types", []))
        # If types cannot be matched, retain ranking-based fallback for this
        # evidence purpose rather than failing the whole production plan.
        if not add_first(requirement, preferred):
            add_first(requirement)
    while len(selected) < max_scenes and add_first():
        pass

    scenes_dir = project_dir / "scenes"
    multi_path = scenes_dir / "selected_scenes.json"
    legacy_path = scenes_dir / "selected_scene.json"
    if not selected:
        for path in (multi_path, legacy_path):
            if path.exists():
                path.unlink()
        raise RuntimeError("No valid scenes selected")
    multi_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    legacy_path.write_text(json.dumps(selected[0], ensure_ascii=False, indent=2), encoding="utf-8")
    return multi_path


def select_best_scene(project_dir: Path, top_n: int = 1) -> Optional[Path]:
    """Backward-compatible single-scene selection API."""
    select_scenes(project_dir, max_scenes=max(1, top_n))
    return Path(project_dir) / "scenes" / "selected_scene.json"
