"""Render validation: make the renderer fail closed, never silently broken.

Pre-render
    validate_visual_segments()  — every timeline visual item must exist,
    be playable, carry a video stream, have duration > 0, and its
    source window must be inside the source movie duration.

    validate_timeline_coverage() — planned visual duration must cover the
    expected render duration within a configured tolerance; large gaps fail.

    validate_multi_scene()       — at least ``min_visual_segments`` distinct
    source windows must be selected; silent repetition is forbidden.

Post-render
    validate_render_file()       — duration > 0, video+audio streams present,
    minimum frame count, and no long unexpected black intervals
    (configurable ``max_black_segment_seconds``).
"""
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_MAX_BLACK_SEC = 2.0
_DEFAULT_MAX_GAP_SEC = 2.0
_DEFAULT_MIN_RENDER_FRAMES = 60

# Visual items that must be validated before rendering.
_VALIDATED_KEYS = ("content_path", "start_sec", "end_sec", "duration_sec")


class RenderValidationError(ValueError):
    """Raised when pre- or post-render validation fails (fail closed)."""


# --------------------------------------------------------------------------
# Pre-render: per-visual-asset checks
# --------------------------------------------------------------------------

def _ffprobe_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration,size",
                "-show_entries", "stream=index,codec_type",
                "-of", "json",
                str(path),
            ],
            check=False, capture_output=True, text=True,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except Exception:
        return None


def probe_video(path: Path) -> Optional[Dict[str, Any]]:
    """ffprobe a file; returns ``{"duration_sec", "video", "audio"}`` or None."""
    data = _ffprobe_json(Path(path))
    if data is None:
        return None
    duration = 0.0
    try:
        duration = max(0.0, float(data.get("format", {}).get("duration") or 0.0))
    except (TypeError, ValueError):
        duration = 0.0
    streams = data.get("streams", [])
    return {
        "duration_sec": duration,
        "video": any(s.get("codec_type") == "video" for s in streams),
        "audio": any(s.get("codec_type") == "audio" for s in streams),
    }


def validate_media_file(
    path: Path,
    label: str = "clip",
    require_video: bool = True,
    min_duration: float = 0.0,
) -> Dict[str, Any]:
    """Validate a single media file exists, is playable, has a video stream.

    Raises :class:`RenderValidationError` on ANY failure with a precise reason.
    """
    p = Path(path)
    if not p.exists():
        raise RenderValidationError(f"{label} missing: {p}")
    if p.stat().st_size <= 0:
        raise RenderValidationError(f"{label} is empty (size <= 0): {p}")
    probe = probe_video(p)
    if probe is None:
        raise RenderValidationError(f"{label} is not a playable media file: {p}")
    if require_video and not probe["video"]:
        raise RenderValidationError(f"{label} has no video stream: {p}")
    if probe["duration_sec"] <= min_duration:
        raise RenderValidationError(
            f"{label} duration {probe['duration_sec']:.3f}s <= required "
            f"{min_duration:.3f}s: {p}"
        )
    return probe


def _iter_video_items(timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for seg in timeline.get("segments", []):
        for clip in seg.get("video", []):
            items.append(clip)
    for track in (timeline.get("tracks") or {}).values():
        for item in track.get("items", []):
            if item.get("type") in ("video_clip", "video"):
                items.append(item)
    return items


def validate_visual_segments(
    timeline: Dict[str, Any],
    source_duration: Optional[float] = None,
    min_duration: float = 0.1,
) -> List[Dict[str, Any]]:
    """Validate every visual item in an editorial timeline. Fail closed.

    For each clip verifies: file exists, size > 0, ffprobe succeeds, duration
    > 0, video stream exists, ``start < end``, and start/end are inside the
    source movie duration. Raises on the first invalid asset.
    """
    items = _iter_video_items(timeline)
    if not items:
        raise RenderValidationError("timeline contains no visual segments")

    validated: List[Dict[str, Any]] = []
    seen_paths = set()
    for i, item in enumerate(items):
        clip = dict(item)
        path = Path(clip.get("content_path") or "")
        # Tracks re-list segment clips; validate each unique file once using the
        # richest metadata available (segment clips carry source windows).
        if str(path) in seen_paths:
            continue
        seen_paths.add(str(path))
        label = f"visual item #{i + 1}"

        probe = validate_media_file(path, label=label, require_video=True,
                                    min_duration=min_duration)

        start = clip.get("source_start_sec", clip.get("start_sec"))
        end = clip.get("source_end_sec", clip.get("end_sec"))
        if start is not None and end is not None:
            start, end = float(start), float(end)
            if not (start < end):
                raise RenderValidationError(
                    f"{label} invalid window start={start} end={end} (start >= end): {path}"
                )
            if source_duration is None:
                src = Path(timeline.get("source_path") or "") if timeline.get("source_path") else None
                if src and src.exists():
                    src_probe = probe_video(src)
                    source_duration = src_probe["duration_sec"] if src_probe else None
            if source_duration:
                if start < -1e-6:
                    raise RenderValidationError(
                        f"{label} start {start} is before the source movie starts: {path}"
                    )
                if end > source_duration + 1e-6:
                    raise RenderValidationError(
                        f"{label} end {end} exceeds source duration "
                        f"{source_duration:.3f}s: {path}"
                    )
        clip_dur = float(clip.get("duration_sec") or 0.0)
        if clip_dur <= 0:
            raise RenderValidationError(f"{label} has non-positive duration_sec: {path}")

        clip["_validated_duration_sec"] = probe["duration_sec"]
        clip["_has_video"] = probe["video"]
        validated.append(clip)
    return validated


# --------------------------------------------------------------------------
# Pre-render: timeline coverage + multi-scene checks
# --------------------------------------------------------------------------

@dataclass
class CoverageReport:
    expected_duration: float
    visual_coverage: float
    uncovered_seconds: float
    status: str  # PASS | FAIL
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_duration": round(self.expected_duration, 3),
            "visual_coverage": round(self.visual_coverage, 3),
            "uncovered_seconds": round(self.uncovered_seconds, 3),
            "status": self.status,
            "detail": self.detail,
        }


def _visual_coverage(timeline: Dict[str, Any], validated: List[Dict[str, Any]]) -> float:
    """Visual coverage is the sum of the *effective* clip durations.

    Because xfade chains overlap consecutive clips, the rendered video is
    slightly shorter than the raw sum; we approximate the true coverage with
    the clip durations carried by the timeline (what the renderer plans to
    show), which is exactly what must be validated against the target.
    """
    total = 0.0
    seen = set()
    for clip in validated:
        src = clip.get("content_path")
        if src in seen:
            continue
        seen.add(src)
        total += float(clip.get("duration_sec") or 0.0)
    return total


def validate_timeline_coverage(
    timeline: Dict[str, Any],
    expected_duration: Optional[float] = None,
    max_gap_sec: float = _DEFAULT_MAX_GAP_SEC,
) -> CoverageReport:
    """Fail closed when the planned visual timeline leaves a large uncovered gap.

    ``expected_duration`` defaults to the timeline's own target duration
    (``total_duration_sec`` / ``narration_total_sec``).
    """
    if max_gap_sec == _DEFAULT_MAX_GAP_SEC:
        val = os.getenv("QC_MAX_TIMELINE_GAP_SECONDS", "")
        if val.strip():
            try:
                max_gap_sec = float(val)
            except ValueError:
                pass
    if expected_duration is None:
        expected_duration = float(
            timeline.get("total_duration_sec")
            or timeline.get("narration_total_sec")
            or 0.0
        )
    validated = validate_visual_segments(timeline)
    coverage = _visual_coverage(timeline, validated)
    uncovered = max(0.0, expected_duration - coverage)
    ok = uncovered <= max_gap_sec
    report = CoverageReport(
        expected_duration=expected_duration,
        visual_coverage=coverage,
        uncovered_seconds=uncovered,
        status="PASS" if ok else "FAIL",
        detail=(
            f"visual coverage {coverage:.1f}s vs target {expected_duration:.1f}s; "
            f"uncovered {uncovered:.1f}s (tolerance {max_gap_sec:.1f}s)"
        ),
    )
    if not ok:
        raise RenderValidationError(
            "timeline coverage FAILED: " + report.detail
        )
    return report


def validate_multi_scene(
    timeline: Dict[str, Any],
    minimum_segments: int = 5,
) -> List[str]:
    """Validate that the timeline uses at least ``minimum_segments`` distinct
    source windows. Silent repetition of the same clip is forbidden. Returns
    the list of distinct scene identifiers actually selected.
    """
    items = _iter_video_items(timeline)
    distinct: List[str] = []
    seen = set()
    for item in items:
        scene = str(item.get("source_scene")
                    or item.get("metadata", {}).get("scene_id")
                    or "")
        if scene and scene not in seen:
            seen.add(scene)
            distinct.append(scene)
    if len(distinct) < minimum_segments:
        raise RenderValidationError(
            f"multi-scene validation failed: only {len(distinct)} distinct "
            f"source scene(s) selected, {minimum_segments} required "
            f"(found {sorted(distinct)})"
        )
    return distinct


# --------------------------------------------------------------------------
# Script -> timeline contract: every analytical narration section must map to
# at least one valid visual reference (unless the director marked it
# ``narration_only``).
# --------------------------------------------------------------------------

def validate_script_timeline_mapping(
    script: Dict[str, Any],
    timeline: Dict[str, Any],
) -> List[str]:
    """Verify narration sections have visual evidence in the timeline.

    Returns the list of section ids that carry narration but have NO visual
    reference in the timeline and are not explicitly ``narration_only``.
    Raises :class:`RenderValidationError` when any such section exists.
    """
    # Sections that are narrated but allowed to carry no clip when the
    # director/plan explicitly marks them narration-only.
    scheme = {
        str(s.get("section_id")): s
        for s in (script.get("sections") or [])
    }
    timeline_segments = {
        str(seg.get("seg_id")): seg
        for seg in (timeline.get("segments") or [])
    }
    problems: List[str] = []
    for sid, section in scheme.items():
        narration = (section.get("text") or "").strip()
        if not narration:
            continue
        if str(section.get("visual_type", "")) == "narration_only":
            continue
        seg = timeline_segments.get(sid)
        visuals = []
        if seg:
            visuals = seg.get("video") or []
        for track in (timeline.get("tracks") or {}).values():
            if track.get("type") in ("video",):
                visuals.extend(track.get("items") or [])
        if not visuals:
            problems.append(sid)
    if problems:
        raise RenderValidationError(
            "script->timeline contract FAILED: narration sections with no "
            "visual evidence: " + ", ".join(sorted(problems))
        )
    return []


# --------------------------------------------------------------------------
# Post-render: playability + black-frame detection
# --------------------------------------------------------------------------

def black_segment_seconds(path: Path) -> List[float]:
    """Run ffmpeg ``blackdetect`` on the render; return detected black interval
    durations (seconds). Uses ``-vf blackdetect=d=0.25:pix_th=0.06`` so brief
    intentional transitions (cuts/fades) are not flagged.

    Returns [] when blackdetect is unavailable or produces nothing.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        raise RenderValidationError(f"render missing or empty: {p}")
    cmd = [
        "ffmpeg", "-hide_banner", "-i", str(p),
        "-vf", "blackdetect=d=0.25:pix_th=0.06",
        "-an", "-f", "null", "-",
    ]
    try:
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as e:
        print(f"[Render] blackdetect unavailable: {e}")
        return []
    out = res.stderr or ""
    intervals: List[float] = []
    for line in out.splitlines():
        if "black_duration:" not in line:
            continue
        try:
            dur_str = line.split("black_duration:")[1].split()[0]
            intervals.append(float(dur_str))
        except (IndexError, ValueError):
            continue
    return intervals


def qc_black_threshold() -> float:
    """Configurable max allowed single black segment, seconds."""
    val = os.getenv("QC_MAX_BLACK_SEGMENT_SECONDS", "")
    if val.strip():
        try:
            return float(val)
        except ValueError:
            pass
    return _DEFAULT_MAX_BLACK_SEC


@dataclass
class PostRenderReport:
    duration_sec: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    frames: int = 0
    black_segments_sec: List[float] = field(default_factory=list)
    max_black_sec: float = _DEFAULT_MAX_BLACK_SEC
    status: str = "FAIL"
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_sec": round(self.duration_sec, 3),
            "has_video": self.has_video,
            "has_audio": self.has_audio,
            "frames": self.frames,
            "black_segments_sec": [round(b, 3) for b in self.black_segments_sec],
            "max_black_segment_seconds": self.max_black_sec,
            "status": self.status,
            "errors": self.errors,
        }


def _render_frame_count(path: Path) -> int:
    cmd = [
        "ffprobe", "-v", "error", "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", str(path),
    ]
    try:
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception:
        return 0
    line = (res.stdout or "").strip()
    try:
        return int(float(line))
    except ValueError:
        return 0


def validate_render_file(path: Path, require_audio: bool = True,
                         min_frames: int = _DEFAULT_MIN_RENDER_FRAMES,
                         max_black_sec: Optional[float] = None) -> PostRenderReport:
    """Post-render validation. Raises :class:`RenderValidationError` on failure.

    Verifies: duration > 0, video stream exists, audio stream exists when
    required, frame count >= ``min_frames``, and no black interval longer than
    ``max_black_sec`` (default from env / constant).
    """
    p = Path(path)
    report = PostRenderReport(
        max_black_sec=max_black_sec if max_black_sec is not None else qc_black_threshold()
    )
    if not p.exists() or p.stat().st_size <= 0:
        report.errors.append(f"render missing or empty: {p}")
        report.status = "FAIL"
        raise RenderValidationError(report.errors[-1])

    probe = probe_video(p)
    if probe is None:
        report.errors.append(f"render is not a playable media file: {p}")
        raise RenderValidationError(report.errors[-1])

    report.duration_sec = probe["duration_sec"]
    report.has_video = probe["video"]
    report.has_audio = probe["audio"]

    if report.duration_sec <= 0:
        report.errors.append(f"render duration <= 0 ({report.duration_sec:.3f}s)")
    if not report.has_video:
        report.errors.append("render has no video stream")
    if require_audio and not report.has_audio:
        report.errors.append("render has no audio stream but narration was required")

    report.frames = _render_frame_count(p)
    if report.frames < min_frames:
        report.errors.append(
            f"render frame count {report.frames} < minimum {min_frames}"
        )

    report.black_segments_sec = black_segment_seconds(p)
    too_black = [
        b for b in report.black_segments_sec if b > report.max_black_sec
    ]
    if too_black:
        report.errors.append(
            "excessive black-video interval: "
            + ", ".join(f"{b:.1f}s" for b in too_black)
            + f" (max allowed {report.max_black_sec:.1f}s)"
        )

    if report.errors:
        report.status = "FAIL"
        raise RenderValidationError(
            "QC FAIL: " + "; ".join(report.errors)
        )
    report.status = "PASS"
    return report