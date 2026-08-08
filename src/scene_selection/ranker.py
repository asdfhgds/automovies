"""Scene ranking: deterministic lexical/keyword-based scorer.

API:
- rank_scenes(project_dir: Path, thesis: str, top_k: int = 5) -> list of ranking entries

This module prefers scene_index.json (produced by PySceneDetect), falls back to scene_cards.json.
"""
from pathlib import Path
import json
import re
from collections import Counter
from typing import List, Dict

STOPWORDS = {
    'the','a','an','and','or','of','in','on','for','to','is','are','was','were','it','this','that','with','as','by','from','at','be','has','have','had','but','not'
}


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    # simple tokenization: words of letters and numbers
    tokens = re.findall(r"[a-z0-9]+", text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def score_scene(thesis_tokens: List[str], scene_text: str) -> float:
    scene_tokens = tokenize(scene_text)
    if not thesis_tokens:
        return 0.0
    thesis_set = set(thesis_tokens)
    scene_set = set(scene_tokens)
    overlap = thesis_set & scene_set
    overlap_score = len(overlap) / max(1, len(thesis_set))
    # length score: encourage scenes with some transcript
    length_score = min(1.0, len(scene_tokens) / 80.0)
    # frequency alignment: count how many thesis tokens appear multiple times
    freq = Counter(scene_tokens)
    freq_score = sum(min(3, freq[t]) for t in thesis_set if t in freq) / (3 * max(1, len(thesis_set)))
    # weighted combination
    score = 0.6 * overlap_score + 0.25 * freq_score + 0.15 * length_score
    return round(score, 4)


def rank_scenes(project_dir: Path, thesis: str, top_k: int = 5) -> List[Dict]:
    project_dir = Path(project_dir)
    scenes_file = None
    candidates = [project_dir / 'scenes' / 'scene_index.json', project_dir / 'scenes' / 'scene_cards.json', project_dir / 'scenes' / 'scene_cards.json']
    for c in candidates:
        if c.exists():
            scenes_file = c
            break
    if scenes_file is None:
        raise FileNotFoundError('No scene index found in project')

    with scenes_file.open('r', encoding='utf-8') as f:
        scenes = json.load(f)

    thesis_tokens = tokenize(thesis)

    rankings = []
    for s in scenes:
        text = s.get('transcript') or s.get('summary') or ''
        sc = score_scene(thesis_tokens, text)
        reason_parts = []
        if sc > 0:
            reason_parts.append(f"{len(set(tokenize(text)) & set(thesis_tokens))} keyword overlap")
        if len(text.split()) > 0:
            reason_parts.append(f"{len(text.split())} words in transcript")
        reason = '; '.join(reason_parts) if reason_parts else 'no match'
        rankings.append({
            'scene_id': s.get('scene_id') or s.get('scene_id', 'unknown'),
            'score': sc,
            'reason': reason
        })

    rankings.sort(key=lambda r: r['score'], reverse=True)
    out = rankings[:top_k]

    out_path = project_dir / 'scenes' / 'scene_ranking.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out
