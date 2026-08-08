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
        from scenedetect import detect, ContentDetector
    except Exception as e:
        raise RuntimeError("PySceneDetect not installed or failed to import")

    if source_path is None:
        raise RuntimeError("Source path required for PySceneDetect scene detection")

    source_path = str(source_path)
    # Use ContentDetector for v0.7.1 API (sensitive to content changes)
    scene_list = detect(source_path, ContentDetector(threshold=27.0))

    # If no scenes detected, create one scene spanning the entire video
    if not scene_list:
        try:
            from scenedetect import VideoManager
            vm = VideoManager([source_path])
            vm.start()
            # Get total duration
            for frame in vm.get_frames():
                pass
            duration = vm.get_duration().get_seconds()
            vm.release()
            
            # Create a simple timecode wrapper
            class SimpleTC:
                def __init__(self, seconds):
                    self.seconds = seconds
                def get_seconds(self):
                    return self.seconds
            
            # Create one scene for the entire video
            scene_list = [(SimpleTC(0.0), SimpleTC(duration))]
        except Exception:
            # Fallback: assume 30fps and calculate from ffprobe
            import subprocess
            try:
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                     '-of', 'default=noprint_wrappers=1:nokey=1', source_path],
                    capture_output=True, text=True, timeout=10
                )
                if result.stdout.strip():
                    duration = float(result.stdout.strip())
                    
                    class SimpleTC:
                        def __init__(self, seconds):
                            self.seconds = seconds
                        def get_seconds(self):
                            return self.seconds
                    
                    scene_list = [(SimpleTC(0.0), SimpleTC(duration))]
            except Exception:
                pass

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
            segments = tr.get('segments', [])
            # segments are expected to have start_sec, end_sec, text
            for s in scenes:
                start = s['start_sec']
                end = s['end_sec']
                overlap_texts = [seg.get('text','') for seg in segments if seg.get('start_sec') is not None and seg.get('end_sec') is not None and not (seg['end_sec'] < start or seg['start_sec'] > end)]
                s['transcript'] = ' '.join(overlap_texts)
        except Exception:
            pass

    out_dir = Path(project_dir) / 'scenes'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'scene_index.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)

    print(f"Wrote PySceneDetect scene index -> {out_path}")
    return out_path
