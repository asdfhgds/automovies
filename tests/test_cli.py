"""CLI shipping-verdict enforcement.

The CLI must turn the pipeline status into an exit code so that scripts/CI
cannot treat a FAIL/REVISE run as a completed, shippable pipeline. PASS ships
(0); REVISE and FAIL do not (2 / 3); a missing status fails closed (1).
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import src.main as cli
from src.main import (
    EXIT_FAIL,
    EXIT_PASS,
    EXIT_REVISE,
    EXIT_UNKNOWN,
    exit_code_for,
    main,
    run,
)


def _seed_project(data_dir: Path, project_id: str = "p"):
    project_dir = data_dir / project_id
    (project_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    (project_dir / "project_meta.json").write_text(
        json.dumps({"project_id": project_id, "title": "t",
                    "source_path": "movie.mp4"}), encoding="utf-8")
    return project_dir


@pytest.mark.parametrize("status,expected", [
    ("PASS", EXIT_PASS),
    ("REVISE", EXIT_REVISE),
    ("FAIL", EXIT_FAIL),
    ("unknown", EXIT_UNKNOWN),
    (None, EXIT_UNKNOWN),
])
def test_exit_code_for(status, expected):
    assert exit_code_for(status) == expected


def test_run_returns_verdict_and_prints_status(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DEFAULT_DATA_DIR", tmp_path)
    _seed_project(tmp_path)
    manifest = {"pipeline_status": "PASS",
                "pipeline_status_reasons": []}
    monkeypatch.setattr(cli, "start_pipeline", lambda pid: manifest)

    code = run(SimpleNamespace(project_id="p", profile=None))
    assert code == EXIT_PASS
    out = capsys.readouterr().out
    assert "pipeline_status = PASS" in out


def test_run_revise_fails_with_reasons(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DEFAULT_DATA_DIR", tmp_path)
    _seed_project(tmp_path)
    manifest = {"pipeline_status": "REVISE",
                "pipeline_status_reasons": ["creative REVISE: only 1 distinct scene(s)"]}
    monkeypatch.setattr(cli, "start_pipeline", lambda pid: manifest)

    code = run(SimpleNamespace(project_id="p", profile=None))
    assert code == EXIT_REVISE
    out = capsys.readouterr().out
    assert "pipeline_status = REVISE" in out
    assert "creative REVISE" in out


def test_run_fail_does_not_ship(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "DEFAULT_DATA_DIR", tmp_path)
    _seed_project(tmp_path)
    monkeypatch.setattr(cli, "start_pipeline", lambda pid: {"pipeline_status": "FAIL"})
    assert run(SimpleNamespace(project_id="p", profile=None)) == EXIT_FAIL


def test_run_fails_closed_when_status_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "DEFAULT_DATA_DIR", tmp_path)
    _seed_project(tmp_path)
    monkeypatch.setattr(cli, "start_pipeline", lambda pid: {})
    assert run(SimpleNamespace(project_id="p", profile=None)) == EXIT_UNKNOWN


def test_run_missing_project_metadata_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "DEFAULT_DATA_DIR", tmp_path)
    assert run(SimpleNamespace(project_id="nope", profile=None)) == 1


def test_main_exits_with_verdict(monkeypatch):
    """``main()`` must turn the run's return value into the process exit code."""
    monkeypatch.setattr(cli, "run", lambda args: EXIT_FAIL)
    monkeypatch.setattr(sys, "argv", ["cli", "run"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == EXIT_FAIL


def test_main_exits_zero_on_pass(monkeypatch):
    monkeypatch.setattr(cli, "run", lambda args: EXIT_PASS)
    monkeypatch.setattr(sys, "argv", ["cli", "run"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == EXIT_PASS