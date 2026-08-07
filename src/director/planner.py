"""Director planner: deterministic planner that generates a director_plan.json.

If a scene index is present in project_dir/scenes/scene_index.json the planner will
choose the scene with the largest transcript (most words) and produce a simple
thesis derived from the scene's transcript. This is deterministic and testable.
"""
import json
import uuid
from pathlib import Path
from collections import Counter
import re


STOPWORDS = {
    'the','a','an','and','or','of','in','on','for','to','is','are','was','were','it','this','that','with','as','by','from','at','be','has','have','had','but','not'
}


def _tokenize(text: str):
    if not text:
        return []
    t = text.lower()
    toks = re.findall(r"[a-z0-9]+", t)
    return [x for x in toks if x not in STOPWORDS]


def _most_common_keywords(text: str, n: int = 3):
    toks = _tokenize(text)
    if not toks:
        return []
    c = Counter(toks)
    return [w for w, _ in c.most_common(n)]


def plan_director(project_dir: Path, title: str = None):
    project_dir = Path(project_dir)
    scenes_file = project_dir / 'scenes' / 'scene_index.json'
    chosen_scene = None

    if scenes_file.exists():
        try:
            with scenes_file.open('r', encoding='utf-8') as f:
                scenes = json.load(f)
            # pick scene with most words in transcript
            best = None
            best_count = 0
            for s in scenes:
                text = s.get('transcript') or s.get('summary') or ''
                wc = len(text.split())
                if wc > best_count:
                    best_count = wc
                    best = s
            if best is not None:
                chosen_scene = best
        except Exception:
            # malformed scene index: fall back to None
            chosen_scene = None

    # build a deterministic thesis
    if chosen_scene:
        scene_id = chosen_scene.get('scene_id', 'scene_unknown')
        transcript = chosen_scene.get('transcript', '')
        keywords = _most_common_keywords(transcript, n=3)
        if keywords:
            thesis_text = f"Explore how {' / '.join(keywords)} shape the emotional arc of {scene_id}."
        else:
            # fallback to short excerpt
            excerpt = ' '.join(transcript.split()[:8])
            thesis_text = f"Analyze the moment in {scene_id}: '{excerpt}'."
    else:
        thesis_text = title or 'A focused analysis of a key scene.'

    plan = {
        "project_id": str(uuid.uuid4()),
        "content_type": "scene_analysis",
        "topic": title or "Generated Director Plan",
        "thesis": thesis_text,
        "hook": "An engaging opening that motivates the thesis",
        "tone": "analytical",
        "structure": [
            {"id": "intro", "goal": "Hook and thesis", "target_seconds": 20},
            {"id": "scene_discussion", "goal": "Explain the scene", "target_seconds": 60},
            {"id": "closing", "goal": "Wrap and CTA", "target_seconds": 10}
        ],
        "visual_strategy": ["use_scene_clip"],
        "music_mood": "subtle_tension",
        "length_target_sec": 90
    }

    out_path = project_dir / 'director_plan.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    print(f"Wrote director plan -> {out_path}")
    return out_path
