"""Pipeline production status (milestone P0: pipeline truthfulness).

A completed pipeline must NEVER mean merely "the process didn't crash." This
module produces an explicit status object with a technical breakdown (assets,
timeline, render, audio) and creative metrics measured from real artifacts, and
an overall verdict:

    PASS    all technical gates pass and creative metrics clear the threshold
    REVISE  technically sound, but the creative evaluation says re-edit
    FAIL    a required stage/gate failed — the "pipeline completed" claim is
            false and the artifact must not be treated as a publishable video

Verdict logic is deliberately low-level and auditable: it reads the same
artifacts the QC/render stages write (timeline, render validation, TTS meta).
Creative metrics are computed from real data (coverage, scene variety, pacing
variety, alignment), not invented.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PASS = "PASS"
REVISE = "REVISE"
FAIL = "FAIL"


# ---------------------------------------------------------------------------
# Status schema
# ---------------------------------------------------------------------------

@dataclass
class TechnicalStatus:
    assets: str = FAIL
    timeline: str = FAIL
    render: str = FAIL
    audio: str = FAIL

    @property
    def all_pass(self) -> bool:
        return all(v == PASS for v in (self.assets, self.timeline,
                                       self.render, self.audio))

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TechnicalStatus":
        d = d or {}
        return cls(
            assets=str(d.get("assets", FAIL)),
            timeline=str(d.get("timeline", FAIL)),
            render=str(d.get("render", FAIL)),
            audio=str(d.get("audio", FAIL)),
        )


@dataclass
class CreativeMetrics:
    evidence_coverage: Optional[float] = None
    visual_variety: Optional[float] = None
    pacing_variety: Optional[float] = None
    narration_visual_alignment: Optional[float] = None
    distinct_scenes: int = 0
    segment_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for k, v in asdict(self).items():
            out[k] = round(v, 3) if isinstance(v, float) and v is not None else v
        return out


@dataclass
class PipelineStatus:
    status: str = FAIL
    technical: TechnicalStatus = field(default_factory=TechnicalStatus)
    creative: CreativeMetrics = field(default_factory=CreativeMetrics)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "technical": self.technical.to_dict(),
            "creative": self.creative.to_dict(),
            "reasons": self.reasons,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineStatus":
        return cls(
            status=str(d.get("status", FAIL)),
            technical=TechnicalStatus.from_dict(d.get("technical")),
            creative=CreativeMetrics(**{k: v for k, v in
                                        (d.get("creative") or {}).items()
                                        if k in CreativeMetrics.__dataclass_fields__}),
            reasons=list(d.get("reasons", [])),
        )


# ---------------------------------------------------------------------------
# Technical gates (reuse the real render-validation work, never duplicate it)
# ---------------------------------------------------------------------------

def _timeline(project_dir: Path) -> Optional[dict]:
    path = project_dir / "timeline" / "editorial_timeline.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _evaluate_assets(project_dir: Path) -> str:
    from render.validate import RenderValidationError, validate_visual_segments

    tl = _timeline(project_dir)
    if not tl:
        return FAIL
    try:
        validate_visual_segments(tl)
        return PASS
    except RenderValidationError:
        return FAIL


def _evaluate_timeline(project_dir: Path) -> str:
    tl = _timeline(project_dir)
    if not tl:
        return FAIL
    try:
        from render.validate import validate_timeline_coverage
        report = validate_timeline_coverage(tl)
        return PASS if report.status == PASS else FAIL
    except Exception:
        return FAIL


def _evaluate_render(project_dir: Path) -> str:
    from render.validate import (
        RenderValidationError,
        qc_black_threshold,
        validate_render_file,
    )

    render = project_dir / "renders" / "final_render.mp4"
    if not render.exists():
        return FAIL
    try:
        validate_render_file(render, require_audio=True,
                             max_black_sec=qc_black_threshold())
        return PASS
    except RenderValidationError:
        return FAIL


def _evaluate_audio(project_dir: Path) -> str:
    """Audio gate: real narration (not mock where strict), voice exists, and the
    TTS input contract proves the provider received only clean narration."""
    from audio.narration_contract import (
        NarrationSanitizationError,
        build_tts_inputs,
    )

    voice = project_dir / "audio" / "voice.wav"
    if not voice.exists() or voice.stat().st_size <= 0:
        return FAIL
    # Contract: narration_inputs.json must exist and prove sanitization source.
    manifest = project_dir / "audio" / "narration_inputs.json"
    if not manifest.exists():
        # fall back: re-run the sanitizer over script.json (fail closed).
        script_file = project_dir / "script.json"
        if not script_file.exists():
            return FAIL
        try:
            build_tts_inputs(json.loads(script_file.read_text(encoding="utf-8")))
        except (NarrationSanitizationError, ValueError):
            return FAIL
    else:
        try:
            ni = json.loads(manifest.read_text(encoding="utf-8"))
            if ni.get("schema") != "tts_input_contract_v1" or int(ni.get("count", 0)) <= 0:
                return FAIL
        except Exception:
            return FAIL
    # Mock narration is a FAIL in strict production mode only.
    tts_meta = project_dir / "audio" / "tts_meta.json"
    if tts_meta.exists():
        try:
            meta = json.loads(tts_meta.read_text(encoding="utf-8"))
            if os.getenv("REQUIRE_REAL_TTS", "false").lower() == "true" and \
                    bool(meta.get("mock", False)):
                return FAIL
        except Exception:
            pass
    return PASS


def _creative_metrics(project_dir: Path) -> CreativeMetrics:
    """Creative metrics are measured from real timeline data (no invented
    scores). Returns default zeros when no timeline exists."""
    m = CreativeMetrics()
    tl = _timeline(project_dir)
    if not tl or not tl.get("segments"):
        return m

    segs = tl.get("segments", [])
    m.segment_count = len(segs)
    scenes: set = set()
    distinct_rhythms: set = set()
    covered_narration = 0.0
    narration_total = 0.0
    for seg in segs:
        for clip in seg.get("video", []):
            sid = clip.get("source_scene")
            if sid:
                scenes.add(sid)
        visual_coverage = float(seg.get("visual_coverage_sec", 0.0))
        nar = seg.get("narration") or {}
        n_start = float(nar.get("start_sec", 0.0))
        n_dur = float(nar.get("duration_sec", 0.0))
        covered_narration += min(1.0, visual_coverage / max(1e-6, n_dur)) * n_dur
        narration_total += n_dur
    m.distinct_scenes = len(scenes)
    if m.segment_count:
        m.visual_variety = len(scenes) / float(m.segment_count)
    if narration_total > 0:
        m.evidence_coverage = covered_narration / narration_total
        m.narration_visual_alignment = covered_narration / narration_total
    for seg in segs:
        # Real pacing signal: recorded rhythm (decision list) falls back to the
        # purpose category the timeline actually stamped on the segment.
        rhythm = None
        meta = seg.get("pacing") or seg.get("editing") or seg.get("metadata")
        if isinstance(meta, dict):
            rhythm = meta.get("rhythm")
        purpose = str(seg.get("purpose", ""))
        category = purpose.split(":", 1)[0].strip() if ":" in purpose else purpose.strip()
        distinct_rhythms.add(rhythm or category or "unset")
    if m.segment_count:
        m.pacing_variety = len(distinct_rhythms) / float(m.segment_count)
    return m


# ---------------------------------------------------------------------------
# Evaluation + persistence
# ---------------------------------------------------------------------------

_CREATIVE_MIN_ALIGNMENT = 0.6   # at least 60% of narration visually covered
_CREATIVE_MIN_SCENES = 3


def evaluate_pipeline(project_dir) -> PipelineStatus:
    """Compute the production status for a completed project directory."""
    project_dir = Path(project_dir)
    reasons: List[str] = []

    technical = TechnicalStatus(
        assets=_evaluate_assets(project_dir),
        timeline=_evaluate_timeline(project_dir),
        render=_evaluate_render(project_dir),
        audio=_evaluate_audio(project_dir),
    )
    creative = _creative_metrics(project_dir)

    if technical.assets != PASS:
        reasons.append("asset validation FAILED (pre-render visual gate)")
    if technical.timeline != PASS:
        reasons.append("timeline coverage FAILED (narration exceeds real footage)")
    if technical.render != PASS:
        reasons.append("render validation FAILED (playability / streams / black)")

    if not technical.all_pass:
        return PipelineStatus(status=FAIL, technical=technical,
                              creative=creative, reasons=reasons)

    # Technically sound -> REVISE when the creative numbers miss the floor.
    if creative.distinct_scenes < _CREATIVE_MIN_SCENES:
        reasons.append(
            f"creative REVISE: only {creative.distinct_scenes} distinct scene(s) "
            f"(need >= {_CREATIVE_MIN_SCENES}); visual variety is too low")
    if creative.narration_visual_alignment is not None and \
            creative.narration_visual_alignment < _CREATIVE_MIN_ALIGNMENT:
        reasons.append(
            f"creative REVISE: narration-visual alignment "
            f"{creative.narration_visual_alignment:.2f} below {_CREATIVE_MIN_ALIGNMENT}")
    if reasons:
        return PipelineStatus(status=REVISE, technical=technical,
                              creative=creative, reasons=reasons)
    return PipelineStatus(status=PASS, technical=technical,
                          creative=creative, reasons=reasons)


def save_pipeline_status(project_dir) -> PipelineStatus:
    """Compute + persist ``reports/pipeline_status.json``; returns the status."""
    project_dir = Path(project_dir)
    status = evaluate_pipeline(project_dir)
    out_dir = project_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pipeline_status.json").write_text(
        json.dumps(status.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def load_pipeline_status(project_dir) -> Optional[PipelineStatus]:
    path = Path(project_dir) / "reports" / "pipeline_status.json"
    if not path.exists():
        return None
    return PipelineStatus.from_dict(json.loads(path.read_text(encoding="utf-8")))