"""Select the best valid scene from scene_ranking.json.

Writes selected_scene.json with scene_id, start_sec, end_sec, score, reason.
"""
import json
from pathlib import Path
from typing import Optional


def select_best_scene(project_dir: Path, top_n: int = 1) -> Optional[Path]:
    project_dir = Path(project_dir)
    ranking_path = project_dir / 'scenes' / 'scene_ranking.json'
    index_path = project_dir / 'scenes' / 'scene_index.json'
    cards_path = project_dir / 'scenes' / 'scene_cards.json'

    if not ranking_path.exists():
        raise FileNotFoundError('scene_ranking.json not found')

    with ranking_path.open('r', encoding='utf-8') as f:
        rankings = json.load(f)

    # Load scene index to map start/end
    scenes = {}
    for p in (index_path, cards_path):
        if p.exists():
            try:
                with p.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                    for s in data:
                        scenes[s.get('scene_id')] = s
            except Exception:
                pass

    # iterate rankings and pick first valid
    selected = None
    for r in rankings[:top_n]:
        sid = r.get('scene_id')
        scene = scenes.get(sid)
        if not scene:
            continue
        start = scene.get('start_sec')
        end = scene.get('end_sec')
        if start is None or end is None:
            continue
        try:
            start = float(start)
            end = float(end)
        except Exception:
            continue
        if end <= start:
            continue
        selected = {
            'scene_id': sid,
            'start_sec': start,
            'end_sec': end,
            'score': r.get('score'),
            'reason': r.get('reason')
        }
        break

    out_path = project_dir / 'scenes' / 'selected_scene.json'
    if selected:
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(selected, f, ensure_ascii=False, indent=2)
        return out_path
    else:
        # ensure no stale file
        if out_path.exists():
            out_path.unlink()
        raise RuntimeError('No valid scene selected')
