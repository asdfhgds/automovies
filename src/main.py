"""Minimal CLI for the Autonomous Movie Studio MVP scaffold."""
import argparse
import json
import uuid
from pathlib import Path
from app.orchestrator import start_pipeline
from utils.io import ensure_dirs, write_json

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data"


def init_project(args):
    project_id = str(uuid.uuid4())
    project_dir = DEFAULT_DATA_DIR / project_id
    ensure_dirs([project_dir, project_dir / 'transcripts', project_dir / 'scenes', project_dir / 'assets', project_dir / 'renders', project_dir / 'audio', project_dir / 'reports'])
    meta = {
        "project_id": project_id,
        "title": args.title or "untitled",
        "source_path": None
    }
    if getattr(args, 'source', None):
        # store absolute source path (do not copy by default)
        meta['source_path'] = str(Path(args.source).expanduser().resolve())
    write_json(project_dir / 'project_meta.json', meta)
    print(f"Initialized project {project_id} at {project_dir}")
    if meta['source_path']:
        print(f"Registered source: {meta['source_path']}")
    return project_id


def run(args):
    project_id = args.project_id
    if not project_id:
        print("Error: --project-id required for run")
        return 1
    # Validate project and source
    project_dir = DEFAULT_DATA_DIR / project_id
    meta = None
    try:
        from utils.io import read_json
        meta = read_json(project_dir / 'project_meta.json')
    except Exception:
        pass
    if not meta:
        print(f"Error: project metadata not found at {project_dir / 'project_meta.json'}")
        return 1
    if not meta.get('source_path'):
        print("Warning: No source video registered for this project. Some stages will be skipped until a source is provided.")
    start_pipeline(project_id)
    return 0


def main():
    parser = argparse.ArgumentParser(description='Autonomous Movie Studio CLI')
    sub = parser.add_subparsers(dest='cmd')

    p_init = sub.add_parser('init', help='Create a new project')
    p_init.add_argument('--title', help='Human-friendly title')
    p_init.add_argument('--source', help='Local source video file path (optional)')

    p_run = sub.add_parser('run', help='Run pipeline for a project')
    p_run.add_argument('--project-id', help='Existing project id')

    args = parser.parse_args()
    if args.cmd == 'init':
        init_project(args)
    elif args.cmd == 'run':
        run(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
