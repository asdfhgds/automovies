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
    extract_all_scene_keyframes_with_times,
    extract_scene_keyframes,
    extract_scene_keyframes_with_times,
    frame_times_for_window,
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


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg required")
def test_extract_with_times_matches_shared_coordinates(video, tmp_path):
    pairs = extract_scene_keyframes_with_times(
        str(video), 3.0, 6.0, tmp_path / "kf", scene_id="scene-2", max_frames=3)
    assert len(pairs) == 3
    assert all(Path(p).exists() for p, _ in pairs)
    expected = frame_times_for_window(3.0, 6.0, 3)
    assert [t for _, t in pairs] == expected


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg required")
def test_extract_all_with_times(video, tmp_path):
    mapping = extract_all_scene_keyframes_with_times(
        str(video), _scenes(), tmp_path / "kf", max_frames_per_scene=2)
    assert set(mapping) == {"scene-1", "scene-2"}
    assert len(mapping["scene-1"]["frames"]) == len(mapping["scene-1"]["times_sec"]) == 2
    assert mapping["scene-1"]["times_sec"] == frame_times_for_window(0.0, 3.0, 2)


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


def test_frame_times_exact_and_deterministic():
    # repair #3: sample coordinates are exact floats, never rounded, and the
    # shared function is the single source of truth for extraction + probe.
    assert frame_times_for_window(0.0, 6.0, 2) == [1.5, 4.5]
    assert frame_times_for_window(3.0, 6.0, 2) == [3.75, 5.25]
    # a 2-decimal dt can't survive round-trips through 2-decimal rounding
    assert frame_times_for_window(0.1234567, 6.0, 3) == [
        0.1234567 + 5.8765433 * (0.5 / 3),
        0.1234567 + 5.8765433 * (1.5 / 3),
        0.1234567 + 5.8765433 * (2.5 / 3),
    ]
    assert frame_times_for_window(0.0, 6.0, 1) == [6.0 * 0.35]


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


def test_extract_json_dict_salvages_truncated_array():
    out = ('```json\n{"location": "small shop or garage", '
           '"actions": ["standing still", "looking left"], '
           '"objects": ["man in plaid", "tools')
    parsed = _extract_json_dict(out)
    assert parsed is not None
    assert parsed["location"] == "small shop or garage"
    assert parsed["actions"] == ["standing still", "looking left"]
    assert parsed["objects"] == ["man in plaid", "tools"]


def test_extract_json_dict_salvages_truncated_string_value():
    parsed = _extract_json_dict('{"location": "a test plac')
    assert parsed == {"location": "a test plac"}


def test_extract_json_dict_truncated_prose_still_none():
    assert _extract_json_dict("the man walks over to the counter and picks") is None


def test_enricher_default_max_new_tokens_is_1024():
    import inspect
    from movie_understanding.vision_enricher import Qwen3VLEnricher
    sig = inspect.signature(Qwen3VLEnricher.__init__)
    assert sig.parameters["max_new_tokens"].default == 1024


def test_enricher_retries_truncated_response_before_failing():
    en = _FakeVL(answer="ignored")
    answers = iter(["not json at all", '{"location": "ok", "actions": ["go"]}'])

    def fake_generate(image_paths, prompt):
        return next(answers)

    en._generate = fake_generate
    result = en.enrich(_scene_with_keyframes(), [])
    assert result["analysis"]["visual"]["location"] == "ok"
    assert result["analysis"]["visual"]["actions"] == ["go"]


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

    def generate(self, **k):
        return None

    def to(self, *a, **k):
        if self._to_raises:
            raise RuntimeError("You can't move a model that has some modules offloaded to cpu or disk.")
        return self


class _FakeProcessor:
    pass


def _make_fake_transformers(model_cls, from_pretrained=None):
    """transformers stub with AutoProcessor + AutoModelForCausalLM (+ AutoModel
    fallback), so the real CUDA/torch stack is never touched."""
    import types

    from_pretrained = from_pretrained or (lambda *a, **k: model_cls())
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = type(
        "AutoProcessor", (), {"from_pretrained": staticmethod(lambda *a, **k: _FakeProcessor())}
    )
    for name in ("AutoModelForCausalLM", "AutoModel"):
        setattr(
            fake_transformers, name, type(
                name, (),
                {"from_pretrained": staticmethod(from_pretrained)},
            )
        )
    return fake_transformers


def test_initialize_never_calls_to_on_dispatched_model(monkeypatch):
    import movie_understanding.vision_enricher as ve

    # Clear the class-level cache so we exercise the load path.
    ve._MODEL_CACHE.clear()
    monkeypatch.setattr(ve, "_gpu_available", lambda: True)

    import sys

    fake_transformers = _make_fake_transformers(
        _DispatchedModel,
        from_pretrained=lambda *a, **k: _DispatchedModel(to_raises=True),
    )

    # Sub in a fake transformers module (AutoProcessor/AutoModel* only) so the
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

    fake_transformers = _make_fake_transformers(
        _DispatchedModel,
        from_pretrained=lambda *a, **k: _DispatchedModel(to_raises=False),
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


class _GenModel:
    def generate(self, **k):
        return None


class _NoGenModel:
    pass


def _fake_transformers_cls(cls_name, returns):
    return type(cls_name, (), {"from_pretrained": staticmethod(lambda *a, **k: returns())})


def test_load_conditional_vl_resolves_to_vision2seq_when_causal_returns_base(monkeypatch):
    """Regression: when AutoModelForCausalLM/AutoModel return a base model
    without .generate() (as seen on real Colab with Qwen2.5-VL-7B), the
    resolver must find the *VLForConditionalGeneration wrapper."""
    import sys
    import types
    import movie_understanding.vision_enricher as ve

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoModel = _fake_transformers_cls("AutoModel", _NoGenModel)
    fake_tf.AutoModelForCausalLM = _fake_transformers_cls("AutoModelForCausalLM", _NoGenModel)
    fake_tf.AutoModelForVision2Seq = _fake_transformers_cls("AutoModelForVision2Seq", _GenModel)

    import transformers as _real_tf
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    try:
        model = ve._load_conditional_vl("fake/vl", {})
    finally:
        monkeypatch.setitem(sys.modules, "transformers", _real_tf)
    assert isinstance(model, _GenModel)


def test_load_conditional_vl_falls_back_to_concrete_qwen_module(monkeypatch):
    """Regression: when every auto entry point returns base models, the
    concrete transformers.models.qwen*_vl ForConditionalGeneration class
    still yields a generate()-able model (cross-version robustness)."""
    import sys
    import types
    import movie_understanding.vision_enricher as ve

    fake_tf = types.ModuleType("transformers")
    for name in ("AutoModel", "AutoModelForCausalLM",
                 "AutoModelForVision2Seq", "AutoModelForImageTextToText"):
        setattr(fake_tf, name, _fake_transformers_cls(name, _NoGenModel))

    models_pkg = types.ModuleType("transformers.models")
    fake_tf.models = models_pkg
    qwen_mod = types.ModuleType("transformers.models.qwen2_5_vl")
    qwen_mod.Qwen2_5_VLForConditionalGeneration = type(
        "Qwen2_5_VLForConditionalGeneration", (), {
            "from_pretrained": staticmethod(lambda *a, **k: _GenModel()),
        }
    )

    sentinel_tf = sys.modules["transformers"]
    sentinel_models = sys.modules.get("transformers.models")
    sentinel_qwen = sys.modules.get("transformers.models.qwen2_5_vl")
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "transformers.models", models_pkg)
    monkeypatch.setitem(sys.modules, "transformers.models.qwen2_5_vl", qwen_mod)
    try:
        model = ve._load_conditional_vl("fake/vl", {})
    finally:
        monkeypatch.setitem(sys.modules, "transformers", sentinel_tf)
        monkeypatch.setitem(sys.modules, "transformers.models", sentinel_models)
        monkeypatch.setitem(sys.modules, "transformers.models.qwen2_5_vl", sentinel_qwen)
    assert isinstance(model, _GenModel)


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
    # repair #4: the visual observations also live in analysis.visual,
    # separate from the transcript half
    analysis = out["analysis"]
    assert analysis["visual"]["location"] == "bar"
    assert analysis["visual"]["provenance"]["mood"] == "qwen3vl"
    assert analysis["visual"]["objects"] == ["bottle", "counter"]
    assert isinstance(analysis["transcript"]["summary"], str)
    assert analysis["transcript"]["provenance"]["characters"] == "diarization_speaker_labels"


def test_fake_vl_uses_existing_prose_fields():
    en = _FakeVL()
    scene = _scene_with_keyframes()
    scene["transcript"] = "Sam walks into the bar. Sam lights a cigarette."
    out = en.enrich(scene, [])
    story = out["story"]
    assert "Sam" in story["summary"]
    # repair #5: capitalized transcript words are NOT characters — and there
    # are no diarization speaker labels here, so characters stay empty.
    assert story["characters"] == []
    assert out["analysis"]["transcript"]["characters"] == []


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
    # String events carry no explicit time -> anchored to the sampled frame time,
    # expressed in scene-relative seconds (offsets 0.75 and 2.25 for 3.0-6.0s).
    # (The fake returns the same 2 events for both frames → 4 anchored events.)
    assert out["visual_events"][0]["event"] == "character enters"
    assert out["visual_events"][0]["time_sec"] == 0.75
    assert out["visual_events"][1] == {"time_sec": 0.75, "event": "sits down"}
    assert out["visual_events"][2]["time_sec"] == 2.25


def test_temporal_probe_single_image_per_frame():
    """Regression: the probe must prompt keyframes one at a time (single image
    per forward pass) so peak VRAM equals scene enrichment - the multi-image
    batch path OOM'd on a 16GB T4."""
    en = _FakeVL(
        answer='{"visual_events": [{"time_sec": 0.5, "event": "enter"}], '
               '"confidence": 0.8}'
    )
    scene = _scene_with_keyframes(_scenes()[1])  # 3.0-6.0s, 2 keyframes
    out = en.probe_temporal(scene, [], n_frames=2)
    assert out["ok"] is True
    # One _generate call per keyframe, each with exactly one image path.
    assert len(en._calls) == 2
    assert all(len(paths) == 1 for paths, _ in en._calls)
    assert out["method"] == "per-frame single-image sampling"
    # Exact, unrounded capture coordinates (absolute movie seconds + offsets).
    assert out["sampled_times_sec"] == [3.75, 5.25]
    assert out["sampled_times_rel_sec"] == [0.75, 2.25]
    # Duplicate event from the two frames is deduped after ordering.
    assert out["visual_events"] == [
        {"time_sec": 0.5, "event": "enter"},
    ]


def test_temporal_probe_uses_stored_exact_keyframe_times():
    """Repair #2: the probe must reuse the exact times the frames were
    extracted at, never recompute its own spacing."""
    en = _FakeVL(
        answer='{"visual_events": [{"time_sec": 1.0, "event": "gesture"}], '
               '"confidence": 0.6}'
    )
    scene = _scene_with_keyframes(_scenes()[1])  # 3.0-6.0s
    scene["key_frame_times_sec"] = [3.124518, 5.987123]  # exact extraction coords
    out = en.probe_temporal(scene, [], n_frames=2)
    assert out["ok"] is True
    # The reported sample coordinates match the stored ones verbatim
    # (float-exact; the relative offsets are the raw difference, unrounded).
    assert out["sampled_times_sec"] == [3.124518, 5.987123]
    assert out["sampled_times_rel_sec"] == pytest.approx([0.124518, 2.987123])
    # The prompt observes the frames at those exact offsets.
    p0 = en._calls[0][1]
    assert "0.124518" in p0 or "0.125" in p0


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
        # Mark these as already-grouped narrative scenes so the analyzer keeps
        # the pre-attached keyframes instead of re-extracting (no video here).
        s["shot_ids"] = [s["scene_id"]]
        s["shots"] = [dict(start_sec=s["start_sec"], end_sec=s["end_sec"],
                           transcript=s.get("transcript", ""), **{"shot_id": s["scene_id"]})]
        s.pop("scene_id", None)
        s.pop("duration", None)
        s["scene_id"] = f"narrative-{s['shot_ids'][0]}"
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


def test_movie_analyzer_releases_vision_model_after_analyze(tmp_path, monkeypatch):
    """The VL model must hand VRAM back after analyze (OOM guard)."""
    monkeypatch.setenv("VISION_ENRICHER", "heuristic")
    (tmp_path / "scenes").mkdir(parents=True)
    (tmp_path / "transcripts").mkdir()
    scenes = _scenes()
    for s in scenes:
        s["shot_ids"] = [s["scene_id"]]
        s["shots"] = [dict(start_sec=s["start_sec"], end_sec=s["end_sec"],
                           shot_id=s["scene_id"])]
    (tmp_path / "scenes" / "scene_index.json").write_text(
        json.dumps(scenes), encoding="utf-8")
    (tmp_path / "transcripts" / "transcript.json").write_text(
        json.dumps({"segments": []}), encoding="utf-8")
    (tmp_path / "project_meta.json").write_text(
        json.dumps({"project_id": "p1", "title": "T", "source_path": "x.mp4"}),
        encoding="utf-8")

    en = _FakeVL()
    MovieAnalyzer(scene_enricher=en, attach_keyframes=False).analyze(tmp_path)
    # enrich() initialized the (fake) model; analyze must release it after.
    assert en.model is None
    assert en.processor is None
    assert en._initialized is False


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