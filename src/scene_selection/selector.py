"""Select the best scenes from scene_ranking.json.

Primary API:
- select_scenes(project_dir, top_n=3) -> list of selected scene entries,
  writing selected_scenes.json (the evidence-driven, multi-scene format).

Backward-compatible API:
- select_best_scene(project_dir, top_n=1) -> Path to selected_scene.json,
  still writing the single-scene file consumed by older tooling.
"""
import json
from pathlib import Path
from typing import List, Optional


def _load_scene_map(project_dir: Path) -> dict:
    """Load scene_id -> scene from the scene index (or legacy cards)."""
    scenes = {}
    for p in (project_dir / 'scenes' / 'scene_index.json', project_dir / 'scenes' / 'scene_cards.json'):
        if not p.exists():
            continue
        try:
            with p.open('r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get("scenes", [])
            for s in data or []:
                scenes[s.get('scene_id')] = s
        except Exception:
            pass
    return scenes


def _valid_times(scene: dict):
    start = scene.get('start_sec')
    end = scene.get('end_sec')
    if start is None or end is None:
        return None, None
    try:
        start = float(start)
        end = float(end)
    except Exception:
        return None, None
    if end <= start:
        return None, None
    return start, end


def select_scenes(
    project_dir: Path,
    top_n: int = 3,
    min_gap_sec: float = 0.0,
    min_duration_sec: float = 0.5,
) -> List[dict]:
    """Select up to top_n highest-scoring, non-overlapping valid scenes.

    Iterates scene_ranking.json (already sorted by descending score) and keeps
    scenes that have valid timestamps and a minimum duration. Scenes that
    overlap an already-chosen scene are skipped so the final cut is not
    redundant. Writes selected_scenes.json.

    Returns the list of selected entries:
    [{"scene_id", "start_sec", "end_sec", "score", "reason", "order"}]
    """
    project_dir = Path(project_dir)
    ranking_path = project_dir / 'scenes' / 'scene_ranking.json'

    if not ranking_path.exists():
        raise FileNotFoundError('scene_ranking.json not found')

    with ranking_path.open('r', encoding='utf-8') as f:
        rankings = json.load(f)

    scenes = _load_scene_map(project_dir)
    count = max(1, int(top_n))

    selected = []
    for r in rankings:
        if len(selected) >= count:
            break
        sid = r.get('scene_id')
        scene = scenes.get(sid)
        if not scene:
            continue
        start, end = _valid_times(scene)
        if start is None:
            continue
        if (end - start) < min_duration_sec:
            continue
        # Skip scenes that genuinely overlap an already-chosen scene
        # (adjacent scenes sharing only a boundary are allowed).
        overlaps = any(
            (start < chosen_end - min_gap_sec) and (end > chosen_start + min_gap_sec)
            for _, chosen_start, chosen_end, _, _ in selected
        )
        if overlaps:
            continue
        selected.append((sid, start, end, r.get('score'), r.get('reason')))

    entries = [
        {
            'scene_id': sid,
            'start_sec': start,
            'end_sec': end,
            'score': score,
            'reason': reason,
            'order': idx,
        }
        for idx, (sid, start, end, score, reason) in enumerate(selected)
    ]

    scenes_dir = project_dir / 'scenes'
    scenes_dir.mkdir(parents=True, exist_ok=True)

    with (scenes_dir / 'selected_scenes.json').open('w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    # Backward compatibility: keep selected_scene.json pointing at the top pick.
    if entries:
        first = dict(entries[0])
        with (scenes_dir / 'selected_scene.json').open('w', encoding='utf-8') as f:
            json.dump(first, f, ensure_ascii=False, indent=2)
    else:
        stale = scenes_dir / 'selected_scene.json'
        if stale.exists():
            stale.unlink()

    return entries


def select_best_scene(project_dir: Path, top_n: int = 1) -> Optional[Path]:
    """Select the single best scene (legacy API). Returns Path to selected_scene.json.

    Derives the single-scene file from an existing multi-scene selection so it
    never clobbers selected_scenes.json. If no selection exists yet, it runs a
    fresh single-scene selection.
    """
    project_dir = Path(project_dir)
    scenes_path = project_dir / 'scenes' / 'selected_scenes.json'
    if not scenes_path.exists():
        select_scenes(project_dir, top_n=top_n)
    entries = json.loads(scenes_path.read_text(encoding='utf-8'))
    single_path = project_dir / 'scenes' / 'selected_scene.json'
    if not entries:
        if single_path.exists():
            single_path.unlink()
        raise RuntimeError('No valid scene selected')
    single_path.write_text(
        json.dumps(dict(entries[0]), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return single_path
