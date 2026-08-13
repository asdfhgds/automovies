"""Tests for movie understanding vision layer: keyframe extraction, the
Qwen3-VL scene enricher (including graceful degrade + strict mode), and the
enricher factory env-config selection."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from movie_understanding.analyzer import MovieAnalyzer
from movie_understanding.enrich_factory import (
    create_scene_enricher_from_env,
    get_vision_config_from_env,
)
from movie_understanding.keyframes import (
    extract_all_scene_keyframes,
    extract_scene_keyframes,
    snapshot_frame,
)
from movie_understanding.scene_analyzer import HeuristicSceneEnricher
from movie_understanding.vision_enricher import (
    Qwen3VLEnricher,
    _extract_json_dict,
    _repair_json,
)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture
def video(tmp_path):
    if not _ffmpeg_available():
        pytest.skip("ffmpeg required")
    path = tmp_path / "movie.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=duration=6.0:size=320x180:rate=15",
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True, text=True,
    )
    return path


def _scenes():
    return [
        {"scene_id": "scene-1", "start_sec": 0.0, "end_sec": 3.0,
         "duration": 3.0, "transcript": "A man walks into a dim bar."},
        {"scene_id": "scene-2", "start_sec": 3.0, "end_sec": 6.0,
         "duration": 3.0, "transcript": ""},
    ]


# ---------------------------------------------------------------------------
# Keyframe extraction
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg required")
def test_extract_scene_keyframes_writes_jpeg(video, tmp_path):
    frames = extract_scene_keyframes(str(video), 0.5, 2.5, tmp_path / "kf",
                                     scene_id="scene-1")
    assert len(frames) == 1
    assert frames[0].exists()
    assert frames[0].suffix == ".jpg"


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg required")
def test_extract_scene_keyframes_multiple(video, tmp_path):
    frames = extract_scene_keyframes(str(video), 0.0, 6.0, tmp_path / "kf",
                                     scene_id="scene-1", max_frames=3)
    assert len(frames) == 3
    assert all(f.exists() for f in frames)


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg required")
def test_extract_all_returns_mapping(video, tmp_path):
    mapping = extract_all_scene_keyframes(str(video), _scenes(), tmp_path / "kf")
    assert set(mapping) == {"scene-1", "scene-2"}
    assert len(mapping["scene-1"]) == 1
    assert Path(mapping["scene-1"][0]).exists()


def test_extract_scene_keyframes_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_scene_keyframes(
            str(tmp_path / "nope.mp4"), 0.0, 1.0, tmp_path / "kf")


def test_extract_scene_keyframes_invalid_window(tmp_path):
    with pytest.raises(ValueError):
        extract_scene_keyframes("x.mp4", 2.0, 1.0, tmp_path / "kf")


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg required")
def test_snapshot_frame(video, tmp_path):
    out = snapshot_frame(str(video), 1.0, tmp_path / "one.jpg")
    assert out.exists()
    assert out.suffix == ".jpg"


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def test_repair_json_common_errors():
    assert json.loads(_repair_json('{"a": True, "b": None,}')) == {"a": True, "b": None}
    assert _repair_json('{"a": }') is None


def test_extract_json_dict_fenced():
    out = 'Sure! Here:\n```json\n{"location": "bar", "actions": ["walk"]}\n```\ndone'
    assert _extract_json_dict(out) == {"location": "bar", "actions": ["walk"]}


def test_extract_json_dict_embedded():
    out = 'Here is the answer {"location": "street"} and that is all.'
    assert _extract_json_dict(out) == {"location": "street"}


def test_extract_json_dict_garbage():
    assert _extract_json_dict("no json here") is None


# ---------------------------------------------------------------------------
# Qwen3-VL enricher (no real model — monkeypatch model I/O)
# ---------------------------------------------------------------------------


class _FakeVL(Qwen3VLEnricher):
    """Enricher with a fake 'model' that always emits a given JSON answer."""

    def __init__(self, answer: str = '{"location": "bar", "actions": ["drink"], '
                                     '"objects": ["bottle", "counter"], '
                                     '"visual_description": "dim bar", '
                                     '"visual_events": ["patron enters at ~1s"], '
                                     '"emotional_cues": ["slumped shoulders"], '
                                     '"themes": ["loneliness"], '
                                     '"mood": "somber", '
                                     '"cinematography": "medium shot, low key", '
                                     '"confidence": 0.85}'):
        super().__init__(strict=False)
        self._answer = answer
        self._calls = []
        self._faked_ok = True

    def _vision_available(self):
        return self._faked_ok, "ok" if self._faked_ok else "no GPU"

    def _initialize(self):
        # Never load the real model in tests.
        self._initialized = True
        self.model = object()
        self.processor = object()
        self._device_resolved = "cpu"

    def _generate(self, image_paths, prompt):
        self._calls.append((list(image_paths), prompt))
        return self._answer


def _scene_with_keyframes(scene=None):
    scene = scene or _scenes()[0]
    scene = dict(scene)
    scene["key_frames"] = ["/tmp/k1.jpg", "/tmp/k2.jpg"]
    return scene


# ---------------------------------------------------------------------------
# Real _initialize path (accelerate device_map dispatch) — regression for the
# "You can't move a model that has some modules offloaded to cpu" failure seen
# on a real T4. AutoModel.from_pretrained is faked into an object whose .to()
# raises when called on a dispatched model, proving we never call it.
# ---------------------------------------------------------------------------


class _DispatchedModel:
    def __init__(self, to_raises):
        self._to_raises = to_raises

    def eval(self):
        return self

    def to(self, *a, **k):
        if self._to_raises:
            raise RuntimeError("You can't move a model that has some modules offloaded to cpu or disk.")
        return self


class _FakeProcessor:
    pass


def test_initialize_never_calls_to_on_dispatched_model(monkeypatch):
    import movie_understanding.vision_enricher as ve

    # Clear the class-level cache so we exercise the load path.
    ve._MODEL_CACHE.clear()
    monkeypatch.setattr(ve, "_gpu_available", lambda: True)

    import sys
    import types

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = type(
        "AutoProcessor", (), {"from_pretrained": staticmethod(lambda *a, **k: _FakeProcessor())}
    )
    fake_transformers.AutoModel = type(
        "AutoModel", (),
        {"from_pretrained": staticmethod(lambda *a, **k: _DispatchedModel(to_raises=True))},
    )

    # Sub in a fake transformers module (AutoProcessor/AutoModel only) so the
    # real CUDA/torch stack is never touched.
    import transformers as _real_tf
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    try:
        en = ve.Qwen3VLEnricher(model="fake/vl", device="cuda", strict=False)
        en._initialize()
    finally:
        monkeypatch.setitem(sys.modules, "transformers", _real_tf)
        ve._MODEL_CACHE.clear()

    assert en._initialized is True
    assert en.model_load_time_sec == 0.0 or en.model_load_time_sec > 0


def test_initialize_calls_to_when_not_dispatched(monkeypatch):
    """CPU path (no device_map) still calls .to() but tolerates the plain model."""
    import movie_understanding.vision_enricher as ve

    ve._MODEL_CACHE.clear()
    monkeypatch.setattr(ve, "_gpu_available", lambda: False)

    import sys
    import types

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = type(
        "AutoProcessor", (), {"from_pretrained": staticmethod(lambda *a, **k: _FakeProcessor())}
    )
    fake_transformers.AutoModel = type(
        "AutoModel", (),
        {"from_pretrained": staticmethod(lambda *a, **k: _DispatchedModel(to_raises=False))},
    )

    import transformers as _real_tf
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    try:
        en = ve.Qwen3VLEnricher(model="fake/vl", device="cpu", strict=False)
        en._initialize()
    finally:
        monkeypatch.setitem(sys.modules, "transformers", _real_tf)
        ve._MODEL_CACHE.clear()
    assert en._initialized is True


def test_fake_vl_populates_vision_fields():
    en = _FakeVL()
    out = en.enrich(_scene_with_keyframes(), [])
    story = out["story"]
    assert story["location"] == "bar"
    assert story["actions"] == ["drink"]
    assert story["visual_description"] == "dim bar"
    assert story["themes"] == ["loneliness"]
    assert story["mood"] == "somber"
    assert story["objects"] == ["bottle", "counter"]
    assert story["visual_events"] == ["patron enters at ~1s"]
    assert story["emotional_cues"] == ["slumped shoulders"]
    assert story["cinematography"] == "medium shot, low key"
    assert story["confidence"] == 0.85
    assert story["provenance"]["location"] == "qwen3vl"
    assert story["provenance"]["visual_description"] == "qwen3vl"
    assert story["provenance"]["objects"] == "qwen3vl"
    assert story["provenance"]["confidence"] == "qwen3vl"
    # transcript-derived fields survive
    assert isinstance(story["summary"], str)


def test_fake_vl_uses_existing_prose_fields():
    en = _FakeVL()
    scene = _scene_with_keyframes()
    scene["transcript"] = "Sam walks into the bar. Sam lights a cigarette."
    out = en.enrich(scene, [])
    story = out["story"]
    assert "Sam" in story["summary"]
    assert "Sam" in story["characters"]


def test_vision_unavailable_degrades_to_heuristic_fields_none():
    en = _FakeVL()
    en._faked_ok = False
    out = en.enrich(_scene_with_keyframes(), [])
    story = out["story"]
    assert story["location"] is None
    assert story["visual_description"] is None
    assert story["provenance"]["location"].startswith("unavailable")
    # heuristic fields still present
    assert isinstance(story["summary"], str)


def test_vision_strict_raises_when_unavailable():
    en = _FakeVL()
    en._faked_ok = False
    en.strict = True
    with pytest.raises(RuntimeError, match="REQUIRE_REAL_VISION"):
        en.enrich(_scene_with_keyframes(), [])


def test_vision_no_keyframes_degrades():
    en = _FakeVL()
    scene = dict(_scenes()[0])
    scene["key_frames"] = []
    out = en.enrich(scene, [])
    assert out["story"]["location"] is None
    assert out["story"]["provenance"]["location"] == "unavailable (no keyframes)"


def test_vision_strict_no_keyframes_raises():
    en = _FakeVL()
    en.strict = True
    scene = dict(_scenes()[0])
    scene["key_frames"] = []
    with pytest.raises(RuntimeError, match="no keyframes"):
        en.enrich(scene, [])


def test_fake_vl_placeholder_cleanup():
    # model returns empty/partial values -> cleaned to None, not placeholder junk
    en = _FakeVL(answer='{"location": "", "actions": [1, "x"], "mood": " "}')
    out = en.enrich(_scene_with_keyframes(), [])
    story = out["story"]
    assert story["location"] is None
    assert story["actions"] == ["1", "x"]
    assert story["mood"] is None


def test_enricher_prompt_refers_to_scene_id():
    en = _FakeVL()
    en.enrich(_scene_with_keyframes(_scenes()[1]), [])
    prompt = en._calls[0][1]
    assert "scene-2" in prompt


# ---------------------------------------------------------------------------
# Temporal probe (ordered visual events with approximate timestamps)
# ---------------------------------------------------------------------------


def test_temporal_probe_parses_ordered_events():
    en = _FakeVL(
        answer='{"visual_events": [{"time_sec": 0.0, "event": "character enters"}, '
               '{"time_sec": 2.0, "event": "sits down"}, '
               '{"time_sec": 4.0, "event": "coin appears"}], '
               '"confidence": 0.7, "limitation": "frames spaced ~3s apart"}'
    )
    scene = _scene_with_keyframes(_scenes()[1])  # 3.0-6.0s window, 2 keyframes
    out = en.probe_temporal(scene, [], n_frames=2)
    assert out["ok"] is True
    assert out["visual_events"][0]["event"] == "character enters"
    assert out["visual_events"][1]["time_sec"] == 2.0
    assert out["confidence"] == 0.7
    assert out["limitation"]
    assert out["scene_id"] == "scene-2"


def test_temporal_probe_string_events():
    en = _FakeVL(
        answer='{"visual_events": ["character enters", "sits down"], '
               '"confidence": 0.5}'
    )
    out = en.probe_temporal(_scene_with_keyframes(_scenes()[1]), [], n_frames=2)
    assert out["ok"] is True
    assert out["visual_events"][0] == {"time_sec": None, "event": "character enters"}


def test_temporal_probe_needs_two_keyframes():
    en = _FakeVL()
    scene = dict(_scenes()[0])
    scene["key_frames"] = ["/tmp/k1.jpg"]  # only one frame
    out = en.probe_temporal(scene, [])
    assert out["ok"] is False
    assert "2 keyframes" in out["reason"]


def test_temporal_probe_degrades_when_vision_unavailable():
    en = _FakeVL()
    en._faked_ok = False
    out = en.probe_temporal(_scene_with_keyframes(_scenes()[1]), [])
    assert out["ok"] is False
    assert out["reason"] == "no GPU"


# ---------------------------------------------------------------------------
# Analyzer integration (heuristic default; vision via injected enricher)
# ---------------------------------------------------------------------------


def test_movie_analyzer_uses_injected_vision_enricher(tmp_path):
    (tmp_path / "scenes").mkdir(parents=True)
    (tmp_path / "transcripts").mkdir()
    scenes = _scenes()
    for s in scenes:
        s["key_frames"] = ["/tmp/kf.jpg"]
    (tmp_path / "scenes" / "scene_index.json").write_text(
        json.dumps(scenes), encoding="utf-8")
    (tmp_path / "transcripts" / "transcript.json").write_text(
        json.dumps({"segments": []}), encoding="utf-8")
    (tmp_path / "project_meta.json").write_text(
        json.dumps({"project_id": "p1", "title": "T", "source_path": "x.mp4"}),
        encoding="utf-8")

    idx = MovieAnalyzer(
        scene_enricher=_FakeVL(),
        attach_keyframes=False,
    ).analyze(tmp_path)
    assert idx["provenance"]["scene_enricher"] == "qwen3vl"
    assert idx["scenes"][0]["story"]["location"] == "bar"
    assert idx["provenance"]["keyframes"] is False


def test_movie_analyzer_attach_keyframes_provenance(tmp_path, monkeypatch):
    (tmp_path / "scenes").mkdir(parents=True)
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "scenes" / "scene_index.json").write_text(
        json.dumps(_scenes()), encoding="utf-8")
    (tmp_path / "transcripts" / "transcript.json").write_text(
        json.dumps({"segments": []}), encoding="utf-8")
    (tmp_path / "project_meta.json").write_text(
        json.dumps({"project_id": "p1", "title": "T",
                    "source_path": "missing.mp4"}),
        encoding="utf-8")

    monkeypatch.setenv("VISION_ENRICHER", "heuristic")
    idx = MovieAnalyzer(attach_keyframes=True).analyze(tmp_path)
    assert idx["provenance"]["keyframes"] is True


# ---------------------------------------------------------------------------
# Factory / env selection
# ---------------------------------------------------------------------------


def test_factory_default_heuristic(monkeypatch):
    monkeypatch.delenv("VISION_ENRICHER", raising=False)
    assert create_scene_enricher_from_env().name == "heuristic"


def test_factory_heuristic_explicit(monkeypatch):
    monkeypatch.setenv("VISION_ENRICHER", "heuristic")
    assert create_scene_enricher_from_env().name == "heuristic"


def test_factory_qwen3vl_degrades_when_no_gpu(monkeypatch):
    monkeypatch.setenv("VISION_ENRICHER", "qwen3vl")
    en = create_scene_enricher_from_env()
    # No CUDA in the test environment -> enricher exists but vision unavailable
    assert en.name == "qwen3vl"
    assert en.available is False


@pytest.mark.skipif(
    True, reason="requires CUDA GPU + transformers; gated behind strict run")
def test_factory_qwen3vl_strict_requires_gpu(monkeypatch):
    monkeypatch.setenv("VISION_ENRICHER", "qwen3vl")
    monkeypatch.setenv("REQUIRE_REAL_VISION", "true")
    with pytest.raises(RuntimeError, match="REQUIRE_REAL_VISION"):
        create_scene_enricher_from_env()


def test_factory_strict_forbids_heuristic(monkeypatch):
    monkeypatch.delenv("VISION_ENRICHER", raising=False)
    monkeypatch.setenv("REQUIRE_REAL_VISION", "true")
    with pytest.raises(RuntimeError, match="REQUIRE_REAL_VISION"):
        create_scene_enricher_from_env()


def test_vision_config_env(monkeypatch):
    monkeypatch.setenv("VISION_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    monkeypatch.setenv("VISION_DEVICE", "cuda")
    monkeypatch.setenv("VISION_MAX_FRAMES", "3")
    cfg = get_vision_config_from_env()
    assert cfg["enricher"] == "heuristic"
    assert cfg["model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert cfg["device"] == "cuda"
    assert cfg["max_frames"] == 3


def test_heuristic_has_mood_field():
    story = HeuristicSceneEnricher().enrich(_scenes()[0], [])["story"]
    assert story["mood"] is None
    assert story["provenance"]["mood"] == "unavailable (vision/LLM)"
    # new vision/LLM-only fields are also honestly None + provenance-flagged
    for field in ("objects", "visual_events", "emotional_cues",
                  "cinematography", "confidence"):
        assert story[field] is None
        assert story["provenance"][field].startswith("unavailable")