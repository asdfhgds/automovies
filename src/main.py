"""Minimal CLI for the Autonomous Movie Studio MVP scaffold."""
import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from app.orchestrator import start_pipeline
from utils.io import ensure_dirs, write_json

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data"

# Shipping-verdict exit codes (enforced by the CLI after the pipeline runs).
EXIT_PASS = 0
EXIT_UNKNOWN = 1   # status missing/unparseable -> fail closed
EXIT_REVISE = 2
EXIT_FAIL = 3


def exit_code_for(status) -> int:
    """Map a pipeline status to the CLI exit code.

    PASS ships (0); REVISE and FAIL do NOT ship (2 / 3); anything unreadable
    fails closed (1). A non-zero exit is what makes the verdict enforceable in
    scripts/CI: the process refuses to claim a completed run is shippable.
    """
    if status == "PASS":
        return EXIT_PASS
    if status == "REVISE":
        return EXIT_REVISE
    if status == "FAIL":
        return EXIT_FAIL
    return EXIT_UNKNOWN


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


def benchmark_tts(args):
    """Benchmark every available TTS provider on a shared narration text."""
    import json as _json

    from generation.tts_benchmark import benchmark_tts

    text = args.text or (
        "Welcome to this deep dive. Every frame we are about to examine was "
        "chosen for a reason. Let us look closer at what the director is "
        "actually doing, and why it matters."
    )
    narration = {}
    if getattr(args, 'emotion', None):
        narration['emotion'] = args.emotion
    if getattr(args, 'pace', None):
        narration['pace'] = float(args.pace)
    if getattr(args, 'tone', None):
        narration['tone'] = args.tone
    report = benchmark_tts(
        text=text,
        output_dir=Path(args.output_dir),
        include_mock=True,
        narration=narration or None,
    )
    for entry in report['results']:
        status = entry['status']
        line = (
            f"  [{status:>10}] {entry['provider']:<12} "
            f"model={entry['model']} device={entry['device']} "
            f"gen={entry['generation_time_sec']}s dur={entry['duration_sec']}s "
            f"sr={entry['sample_rate']}"
        )
        if entry.get('error'):
            line += f" error={entry['error']}"
        print(line)
    return 0


def run(args):
    project_id = args.project_id
    if not project_id:
        print("Error: --project-id required for run")
        return 1
    profile = getattr(args, 'profile', None)
    if profile:
        os.environ['STUDIO_PROFILE'] = profile
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
    manifest = start_pipeline(project_id) or {}
    status = manifest.get('pipeline_status')
    print(f"pipeline_status = {status}")
    for reason in manifest.get('pipeline_status_reasons', []):
        print(f"  - {reason}")
    return exit_code_for(status)


def main():
    parser = argparse.ArgumentParser(description='Autonomous Movie Studio CLI')
    sub = parser.add_subparsers(dest='cmd')

    p_init = sub.add_parser('init', help='Create a new project')
    p_init.add_argument('--title', help='Human-friendly title')
    p_init.add_argument('--source', help='Local source video file path (optional)')

    p_run = sub.add_parser('run', help='Run pipeline for a project')
    p_run.add_argument('--project-id', help='Existing project id')
    p_run.add_argument('--profile', default=None,
                       help='Profile override (local|colab-gpu). Sets STUDIO_PROFILE.')

    p_bench = sub.add_parser('benchmark-tts', help='Benchmark available TTS providers on a shared narration')
    p_bench.add_argument('--text', help='Narration text to synthesize')
    p_bench.add_argument('--tone', help='Tone (analytical, dramatic, ...)')
    p_bench.add_argument('--emotion', help='Emotion override')
    p_bench.add_argument('--pace', help='Pace multiplier override')
    p_bench.add_argument('--output-dir', default='reports', help='Output directory for wavs + report')

    p_doctor = sub.add_parser('doctor', help='Run environment health checks')
    p_doctor.add_argument('--json', action='store_true',
                          help='Print the health-check report as a single-line JSON object '
                               '(for machine parsing; no human-readable sections)')

    args = parser.parse_args()
    if args.cmd == 'init':
        init_project(args)
    elif args.cmd == 'run':
        # The pipeline verdict is the exit status: PASS ships (0), anything
        # else fails the CLI so scripts/CI cannot treat a bad cut as complete.
        sys.exit(run(args))
    elif args.cmd == 'benchmark-tts':
        benchmark_tts(args)
    elif args.cmd == 'doctor':
        # lightweight import of doctor checks
        import json as _json
        from utils.doctor import print_report, run_checks
        if getattr(args, 'json', False):
            print(_json.dumps(run_checks(), ensure_ascii=False))
        else:
            print_report()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
