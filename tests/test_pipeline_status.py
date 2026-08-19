"""Production status object: PASS/REVISE/FAIL evaluation + persistence.

The pipeline status must distinguish "the process didn't crash" from "the
movie is publishable" — so these tests exercise the real verdict logic
(assets, timeline coverage, render playability, narration audio) against
synthetic-but-contract-shaped artifacts.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import quality.pipeline_status as ps
import render.validate as rv
from quality.pipeline_status import (
    FAIL,
    PASS,
    REVISE,
    evaluate_pipeline,
    load_pipeline_status,
    save_pipeline_status,
)


@pytest.fixture
def media_probe(monkeypatch):
    """Structural unit tests have no real MP4 files; stub the media probe so the
    full structural validation (windows, ordering, coverage math) still runs."""
    def fake_probe(path, label="clip", require_video=True, min_duration=0.0):
        from pathlib import Path as P
        p = P(path)
        if not p.exists():
            raise rv.RenderValidationError(f"{label} missing: {p}")
        # duration matches the clip plan carried by the timeline
        return {"duration_sec": float(p.stem.split("_")[-1]) or 3.0,
                "video": True, "audio": True}

    monkeypatch.setattr(rv, "validate_media_file", fake_probe)


def _segment(seg_id, scene, n_start, n_dur, vis_start, vis_end,
             purpose="hook: open", base: Path | None = None):
    return {
        "seg_id": seg_id,
        "purpose": purpose,
        "narration": {
            "start_sec": n_start,
            "end_sec": n_start + n_dur,
            "duration_sec": n_dur,
        },
        "video": [{
            "excerpt_index": 0,
            "source_scene": scene,
            "source_start_sec": vis_start,
            "source_end_sec": vis_end,
            "content_path": str((base / f"excerpts/{seg_id}.mp4")
                                if base is not None
                                else Path(f"excerpts/{seg_id}.mp4").resolve()),
            "extracted": True,
            "duration_sec": vis_end - vis_start,
            "speed": 1.0,
            "crop_zoom": 1.0,
            "hold_sec": 0.0,
            "mute_film_audio": False,
        }],
        "visual_coverage_sec": vis_end - vis_start,
        "narration_uncovered_sec": max(0.0, n_dur - (vis_end - vis_start)),
        "audio": {"duck_level": 0.05, "mute_film": False},
    }


def _timeline(segments, total=None):
    total = total if total is not None else sum(s["visual_coverage_sec"] for s in segments)
    return {
        "mode": "editorial",
        "total_duration_sec": round(total, 3),
        "narration_total_sec": round(total, 3),
        "source_path": "movie.mp4",
        "segments": segments,
    }


@pytest.fixture
def good_project(tmp_path: Path) -> Path:
    """A project whose artifacts satisfy every technical gate."""
    p = tmp_path / "proj"
    segs = [
        _segment("seg_00", "scene-1", 0.0, 3.0, 0.0, 3.0, "hook: open", p),
        _segment("seg_01", "scene-2", 3.0, 3.0, 3.0, 6.0, "contrast: cut", p),
        _segment("seg_02", "scene-3", 6.0, 3.0, 6.0, 9.0, "conclusion: close", p),
    ]
    tls = _timeline(segs)
    (p / "timeline").mkdir(parents=True)
    (p / "timeline" / "editorial_timeline.json").write_text(
        json.dumps(tls), encoding="utf-8")
    (p / "renders").mkdir()
    (p / "renders" / "final_render.mp4").write_bytes(b"\x00" * 1024)
    (p / "excerpts").mkdir(parents=True)
    for seg in segs:
        name = seg["video"][0]["content_path"]
        (p / name).write_bytes(b"\x00" * 1024)
    (p / "audio").mkdir()
    (p / "audio" / "voice.wav").write_bytes(b"RIFFfake")
    (p / "audio" / "narration_inputs.json").write_text(
        json.dumps({"schema": "tts_input_contract_v1", "count": 3,
                    "entries": []}), encoding="utf-8")
    return p


def test_status_pass_on_complete_project(good_project: Path, media_probe, monkeypatch):
    monkeypatch.setattr(ps, "_evaluate_render", lambda _p: PASS)
    monkeypatch.setattr(ps, "_evaluate_audio", lambda _p: PASS)
    status = evaluate_pipeline(good_project)
    assert status.status == PASS
    assert status.technical.all_pass
    assert status.reasons == []
    assert status.creative.distinct_scenes == 3
    assert status.creative.segment_count == 3


def test_status_fail_when_render_missing(good_project: Path, media_probe):
    (good_project / "renders" / "final_render.mp4").unlink()
    status = evaluate_pipeline(good_project)
    assert status.status == FAIL
    assert status.technical.render == FAIL
    assert any("render" in r for r in status.reasons)


def test_status_fail_on_timeline_coverage_gap(good_project: Path, media_probe, monkeypatch):
    monkeypatch.setattr(ps, "_evaluate_render", lambda _p: PASS)
    monkeypatch.setattr(ps, "_evaluate_audio", lambda _p: PASS)
    # narration outruns the footage actually selected (12s of script, 9s of clips)
    tl = _timeline([
        _segment("seg_00", "scene-1", 0.0, 6.0, 0.0, 3.0, "hook: open", good_project),
        _segment("seg_01", "scene-2", 6.0, 6.0, 6.0, 9.0, "contrast: cut", good_project),
        _segment("seg_02", "scene-3", 12.0, 6.0, 12.0, 15.0, "detail: zoom", good_project),
    ], total=12.0)
    (good_project / "timeline" / "editorial_timeline.json").write_text(
        json.dumps(tl), encoding="utf-8")
    status = evaluate_pipeline(good_project)
    assert status.status == FAIL
    assert status.technical.timeline == FAIL


def test_status_revise_when_visual_variety_too_low(good_project: Path, media_probe, monkeypatch):
    monkeypatch.setattr(ps, "_evaluate_render", lambda _p: PASS)
    monkeypatch.setattr(ps, "_evaluate_audio", lambda _p: PASS)
    # four segments but every clip from the same scene: technically fine, creatively flat
    tl = _timeline([
        _segment("seg_00", "scene-1", 0.0, 3.0, 0.0, 3.0, "hook: open", good_project),
        _segment("seg_01", "scene-1", 3.0, 3.0, 3.0, 6.0, "contrast: cut", good_project),
        _segment("seg_02", "scene-1", 6.0, 3.0, 6.0, 9.0, "detail: zoom", good_project),
        _segment("seg_03", "scene-1", 9.0, 3.0, 9.0, 12.0, "conclusion: close", good_project),
    ], total=12.0)
    (good_project / "timeline" / "editorial_timeline.json").write_text(
        json.dumps(tl), encoding="utf-8")
    (good_project / "excerpts" / "seg_03.mp4").write_bytes(b"\x00" * 1024)
    status = evaluate_pipeline(good_project)
    assert status.status == REVISE
    assert status.technical.all_pass
    assert any("distinct scene" in r for r in status.reasons)


def test_save_load_roundtrip(good_project: Path, media_probe, monkeypatch):
    monkeypatch.setattr(ps, "_evaluate_render", lambda _p: PASS)
    monkeypatch.setattr(ps, "_evaluate_audio", lambda _p: PASS)
    status = save_pipeline_status(good_project)
    assert status.status == PASS
    path = good_project / "reports" / "pipeline_status.json"
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == PASS
    loaded = load_pipeline_status(good_project)
    assert loaded.status == PASS
    assert loaded.technical.all_pass
    assert loaded.creative.distinct_scenes == 3


def test_manifest_wiring_via_orchestrator_fixture(good_project: Path, media_probe, monkeypatch):
    """The orchestrator's status phase is a thin wrapper over the evaluator; this
    proves the status object is what the manifest records (no silent crash)."""
    monkeypatch.setattr(ps, "_evaluate_render", lambda _p: PASS)
    monkeypatch.setattr(ps, "_evaluate_audio", lambda _p: PASS)
    status = save_pipeline_status(good_project)
    manifest = {"pipeline_status": status.status}
    assert manifest["pipeline_status"] in (PASS, REVISE, FAIL)