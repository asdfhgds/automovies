"""PySceneDetect adapter: detect scenes and write scene_index.json

This module defers importing the scenedetect package until called. If scenedetect
is not installed, callers should fall back to the stub adapter.
"""
import json
from pathlib import Path
from typing import Optional


def detect_and_build_scene_index(project_dir: Path, source_path: Optional[str] = None):
    try:
        # Import here so the package isn't required at module import time
        from scenedetect import VideoManager, SceneManager
        from scenedetect.detectors import ContentDetector
    except Exception as e:
        raise RuntimeError("PySceneDetect not installed or failed to import")

    if source_path is None:
        raise RuntimeError("Source path required for PySceneDetect scene detection")

    source_path = str(source_path)
    vm = VideoManager([source_path])
    sm = SceneManager()
    sm.add_detector(ContentDetector())

    vm.set_downscale_factor()  # use default downscale
    vm.start()
    sm.detect_scenes(frame_source=vm)
    scene_list = sm.get_scene_list()  # list of (start, end) FrameTimecodes

    scenes = []
    for i, (start_tc, end_tc) in enumerate(scene_list, start=1):
        start_sec = float(start_tc.get_seconds())
        end_sec = float(end_tc.get_seconds())
        scenes.append({
            "scene_id": f"scene-{i}",
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration": end_sec - start_sec,
            "transcript": "",  # filled later by matching transcript
            "key_frames": [],
            "keywords": []
        })

    # Try to attach transcript text overlapping each scene
    transcripts_path = Path(project_dir) / 'transcripts' / 'transcript.json'
    if transcripts_path.exists():
        try:
            with transcripts_path.open('r', encoding='utf-8') as f:
                tr = json.load(f)
            words = tr.get('words', [])
            # words are expected to be objects with start/end/text
            for s in scenes:
                start = s['start_sec']
                end = s['end_sec']
                overlap_texts = [w.get('text','') for w in words if w.get('start') is not None and w.get('end') is not None and not (w['end'] < start or w['start'] > end)]
                s['transcript'] = ' '.join(overlap_texts)
        except Exception:
            pass

    out_dir = Path(project_dir) / 'scenes'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'scene_index.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)

    print(f"Wrote PySceneDetect scene index -> {out_path}")
    vm.release()
    return out_path
