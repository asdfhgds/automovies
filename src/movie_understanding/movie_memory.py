"""Movie memory: persistence for the movie index artifacts."""
import json
from pathlib import Path
from typing import Any, Dict


def save_json(project_dir: Path, filename: str, data: Any) -> Path:
    path = Path(project_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json(project_dir: Path, filename: str, default: Any = None) -> Any:
    path = Path(project_dir) / filename
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_movie_index(project_dir: Path, movie_index: Dict) -> Path:
    return save_json(project_dir, "movie_index.json", movie_index)


def save_semantic_index(project_dir: Path, semantic_index: Dict) -> Path:
    return save_json(project_dir, "semantic_index.json", semantic_index)


def load_movie_index(project_dir: Path) -> Dict:
    index = load_json(project_dir, "movie_index.json", None)
    return index or {}


def load_semantic_index(project_dir: Path) -> Dict:
    index = load_json(project_dir, "semantic_index.json", None)
    return index or {}
