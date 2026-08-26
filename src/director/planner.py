"""Director planner: generates director_plan.json using creative director (with deterministic fallback).

Can use either:
1. CreativeDirector (LLM-backed, generates multiple concepts and selects best)
2. Deterministic planner (simple, reproducible, for testing)

Use env var CREATIVE_DIRECTOR_ENABLED=true to enable LLM-backed director.
"""
import json
import uuid
import os
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


def plan_director(project_dir: Path, title: str = None, use_creative: bool = None):
    """
    Generate director plan.
    
    Args:
        project_dir: Project directory
        title: Optional title for the content
        use_creative: If True, use creative director (LLM). If None, auto-detect from env.
    
    Returns:
        Path to director_plan.json
    """
    # Auto-detect if not specified
    if use_creative is None:
        use_creative = os.getenv('CREATIVE_DIRECTOR_ENABLED', 'false').lower() == 'true'
    
    # Try creative director first if enabled
    if use_creative:
        try:
            return _plan_director_creative(project_dir, title)
        except Exception as e:
            print(f"Creative director failed ({e}), falling back to deterministic planner")
    
    # Fallback to deterministic planner
    return _plan_director_deterministic(project_dir, title)


def _plan_director_creative(project_dir: Path, title: str = None) -> Path:
    """Use LLM-backed creative director."""
    from .creative_director import CreativeDirector
    
    project_dir = Path(project_dir)
    
    # Load scene index and transcript
    scenes_file = project_dir / 'scenes' / 'scene_index.json'
    transcript_file = project_dir / 'transcripts' / 'transcript.json'
    
    scene_index = []
    if scenes_file.exists():
        with scenes_file.open('r', encoding='utf-8') as f:
            scene_index = json.load(f)
    
    transcript = {}
    if transcript_file.exists():
        with transcript_file.open('r', encoding='utf-8') as f:
            transcript = json.load(f)
    
    # Movie metadata (basic)
    movie_metadata = {
        "title": title or "Unknown Movie",
        "source": project_dir.name,
    }
    
    # Run creative director
    director = CreativeDirector()
    result = director.develop_production_plan(
        movie_metadata=movie_metadata,
        scene_index=scene_index,
        transcript=transcript,
        user_topic=None,
        num_concepts=3,
    )
    
    # Save production plan as director plan
    out_path = project_dir / 'director_plan.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(result['production_plan'], f, ensure_ascii=False, indent=2)
    
    # Also save concepts for reference
    concepts_path = project_dir / 'creative_concepts.json'
    with concepts_path.open('w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Wrote creative director plan -> {out_path}")
    print(f"Wrote concepts -> {concepts_path}")
    return out_path

def _plan_director_deterministic(project_dir: Path, title: str = None) -> Path:
    """Deterministic director (original implementation)."""
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
    # NOTE: the thesis is later (re)written into spoken narration, so it must
    # never embed internal identifiers (scene_id). Use neutral prose instead.
    if chosen_scene:
        transcript = chosen_scene.get('transcript', '')
        keywords = _most_common_keywords(transcript, n=3)
        if keywords:
            thesis_text = "Explore how " + " / ".join(keywords) + " shape the emotional arc of the story."
        else:
            # fallback to short excerpt
            excerpt = ' '.join(transcript.split()[:8])
            thesis_text = f"Analyze the moment this story lingers on: '{excerpt}'."
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
