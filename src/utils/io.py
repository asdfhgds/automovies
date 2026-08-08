"""Simple IO helpers used by the MVP CLI."""
import json
from pathlib import Path


def ensure_dirs(paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path):
    p = Path(path)
    if not p.exists():
        return None
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)
