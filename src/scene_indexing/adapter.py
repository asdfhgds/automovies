"""Scene indexing adapter: prefer PySceneDetect, fall back to the stub scene_detector.

Provides build_scene_cards(project_dir: Path, source_path: Optional[str]=None) -> Path
"""
from pathlib import Path
from typing import Optional


def build_scene_cards(project_dir: Path, source_path: Optional[str] = None):
    # Lazy-load PySceneDetect implementation
    try:
        from .pyscenedetect_adapter import detect_and_build_scene_index
        print("Using PySceneDetect adapter for scene detection")
        return detect_and_build_scene_index(project_dir, source_path)
    except Exception as e:
        # Fallback to existing simple stub
        print(f"PySceneDetect failed ({e}) — falling back to scene_detector stub")
        try:
            from .scene_detector import build_scene_cards as stub_build
            return stub_build(project_dir)
        except Exception as fallback_e:
            raise RuntimeError(f"No scene detection backend available: {fallback_e}")
