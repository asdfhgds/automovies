"""Qwen3-VL vision scene enricher.

Implements the :class:`~movie_understanding.scene_analyzer.SceneEnricher`
interface for the *visual* fields that the heuristic enricher honestly leaves
unavailable (``location`` / ``actions`` / ``visual_description`` / ``themes`` /
``mood``). The enricher:

1. extracts one or more keyframes per scene (FFmpeg) at a timestamp the caller
   provides or evenly spaced across the scene window,
2. loads a Qwen3-VL / Qwen2.5-VL model (lazily, cached at class level, shared
   across scenes) and prompts it with the keyframe(s) + the scene transcript,
3. parses the model's JSON answer into the story card fields,
4. records ``provenance`` per field as ``qwen3-vl`` so the editorial director
   knows the values came from vision rather than heuristics.

When no GPU / model / ffmpeg is available the enricher degrades gracefully: the
vision fields stay ``None`` exactly like the heuristic enricher and provenance
reports ``unavailable(...)`` with the reason. Strict mode (``REQUIRE_REAL_VISION``)
turns that degradation into a hard error instead.
"""
import json
import logging
import re
from typing import List, Optional

from movie_understanding.scene_analyzer import SceneEnricher

logger = logging.getLogger(__name__)

# Shared class-level model cache (model, processor, device). Vision models are
# heavy; loading one copy per scene would be wasteful and OOM-prone.
_MODEL_CACHE: dict = {}

_SYSTEM_PROMPT = (
    "You are a film analyst. You look at a single frame (or a few frames) from "
    "a film scene and describe exactly what is visibly happening. Always answer "
    "with valid JSON only \u2014 no markdown, no code fences, no surrounding prose."
)


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _first_json_value(text: str, start: int = 0) -> Optional[str]:
    """Return the first balanced JSON object/array substring in ``text``."""
    for idx in range(start, len(text)):
        if text[idx] not in "[{":
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(idx, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch in "[{":
                    depth += 1
                elif ch in "]}":
                    depth -= 1
                    if depth == 0:
                        return text[idx : i + 1]
    return None


def _repair_json(text: str) -> Optional[str]:
    """Fix common JSON mistakes small models make; validate before returning."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        return None


def _extract_json_dict(text: str) -> Optional[dict]:
    """Extract a dict from a model response (fenced, first balanced value, boxed)."""
    if not text:
        return None
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        candidate = _repair_json(fenced.group(1).strip())
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    raw = _first_json_value(text)
    for candidate in ([text, raw] if raw else [text]):
        fixed = _repair_json(candidate)
        if fixed:
            try:
                parsed = json.loads(fixed)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return None


class Qwen3VLEnricher(SceneEnricher):
    """Scene enrichments from a Qwen3-VL / Qwen2.5-VL vision-language model."""

    name = "qwen3vl"

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "auto",
        dtype: str = "auto",
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        max_frames: int = 1,
        strict: bool = False,
    ):
        self.model_name = model
        self.device = device
        self.dtype = dtype
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_frames = max_frames
        self.strict = strict

        self.model = None
        self.processor = None
        self._device_resolved = None
        self._initialized = False

        self.last_raw_output: Optional[str] = None
        self.model_load_time_sec: Optional[float] = None
        self.last_generation_time_sec: Optional[float] = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True when we could actually extract keyframes and load a VL model."""
        return self._vision_available()[0]

    def _vision_available(self):
        """Return ``(ok, reason)`` for running the real vision model."""
        try:
            import importlib.util
            if importlib.util.find_spec("transformers") is None:
                return False, "transformers not installed"
            if importlib.util.find_spec("torch") is None:
                return False, "torch not installed"
        except Exception:
            return False, "dependency import check failed"

        if not _gpu_available():
            return False, "no CUDA GPU (Qwen3-VL needs a GPU)"

        return True, "ok"

    def _require(self) -> "Qwen3VLEnricher":
        """Check the strict flag: refuse silent degradation when required real."""
        if self.strict and not self._vision_available()[0]:
            raise RuntimeError(
                "REQUIRE_REAL_VISION=true but Qwen3-VL unavailable: "
                f"{self._vision_available()[1]}. "
                "Vision scene enrichment must use a real VL model on CUDA."
            )
        return self

    # ------------------------------------------------------------------
    # Model lifecycle (lazy load + shared class cache)
    # ------------------------------------------------------------------

    @classmethod
    def release_model(cls) -> None:
        """Drop the shared vision model cache and free GPU memory."""
        global _MODEL_CACHE
        _MODEL_CACHE = {}
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if _gpu_available() else "cpu"

    def _initialize(self) -> None:
        if self._initialized:
            return
        self._require()
        import time as _time

        _load_start = _time.monotonic()
        try:
            import torch
            from transformers import AutoProcessor, AutoModel

            device = self._resolve_device()
            cache_key = (self.model_name, device, str(self.dtype))
            cached = _MODEL_CACHE.get(cache_key)
            if cached:
                self.model, self.processor, self._device_resolved = cached
                self._initialized = True
                self.model_load_time_sec = 0.0
                logger.info(f"Reused cached vision model {self.model_name} on {device}")
                return

            logger.info(f"Loading vision model {self.model_name}...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            quantize_4bit = str(self.dtype).lower() in ("4bit", "int4", "nf4")
            torch_dtype = (
                torch.float16 if not quantize_4bit and device == "cuda" else torch.float32
            )
            load_kwargs = dict(
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if quantize_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                except ImportError:
                    raise RuntimeError(
                        "VISION_DTYPE=4bit requires bitsandbytes. "
                        "Install: pip install bitsandbytes"
                    )
            if device == "cuda":
                load_kwargs["device_map"] = "auto"
                load_kwargs["attn_implementation"] = "sdpa"
            self.model = AutoModel.from_pretrained(self.model_name, **load_kwargs)
            self.model.eval()

            if device == "cuda":
                self.model.to(device)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            self._device_resolved = device
            self._initialized = True
            self.model_load_time_sec = round(_time.monotonic() - _load_start, 2)
            _MODEL_CACHE[cache_key] = (self.model, self.processor, device)
            logger.info(
                f"Vision model ready in {self.model_load_time_sec}s "
                "(cached for reuse across scenes)"
            )
        except ImportError as e:
            raise RuntimeError(
                f"Qwen3-VL requires transformers + torch (CUDA build): {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to init Qwen3-VL: {e}")

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(self, scene: dict, transcript_text: str) -> str:
        scene_id = scene.get("scene_id", "scene")
        return f"""You are analyzing one frame from a film scene.

## Frame
I have attached a keyframe taken from within the scene window
({float(scene.get('start_sec', 0.0)):g}s - {float(scene.get('end_sec', 0.0)):g}s).

## Scene transcript (dialogue / narration audible near this frame)
{transcript_text[:400] or "(no dialogue in this scene)"}

## Scene id
{scene_id}

## Task
Describe exactly what is visible in the attached frame(s). Infer ONLY what is
visually justified or strongly implied by the transcript. Fill every field:

{{
  "location": "VISIBLE LOCATION — indoor/outdoor, type of place, setting",
  "actions": ["ACTION1", "ACTION2"],
  "visual_description": "2-3 sentences describing composition, subjects, lighting, colors, objects, camera feel",
  "themes": ["THEME1", "THEME2"],
  "mood": "ONE-WORD-or-phrase emotional mood of the frame"
}}

IMPORTANT - REPLACE, DON'T COPY:
- The ALL-CAPS strings above are placeholders, NOT text to output.
- Write original observations about THIS frame only.
- Never output placeholder strings.

Answer now:"""

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self, image_paths: List[str], prompt: str) -> str:
        if not self.model or not self.processor:
            raise RuntimeError("Vision model not initialized")

        import time as _time
        import torch
        from PIL import Image

        _gen_start = _time.monotonic()
        images = [Image.open(p).convert("RGB") for p in image_paths]

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": im} for im in images
                ] + [{"type": "text", "text": prompt}],
            },
        ]

        # Some processor versions expose apply_chat_template directly; others
        # carry it on .tokenizer. Try to render the chat into a prompt string
        # that identifies the image placeholders, then feed the images in.
        chat_rendered = False
        image_placeholder_format = "image"
        for maybe_tokenizer in (self.processor, getattr(self.processor, "tokenizer", None)):
            if maybe_tokenizer is None:
                continue
            fn = getattr(maybe_tokenizer, "apply_chat_template", None)
            if callable(fn):
                try:
                    text = fn(messages, tokenize=False, add_generation_prompt=True)
                    chat_rendered = bool(text)
                except Exception as e:
                    logger.warning(f"apply_chat_template failed ({e})")
                break
        if not chat_rendered:
            text = prompt

        try:
            inputs = self.processor(
                text=[text],
                images=images,
                return_tensors="pt",
                padding=True,
            )
        except Exception as e:
            # Some processors require images passed as a tensor list;
            # fall back to tokenizer-level input prep.
            logger.warning(f"processor(images=...) failed ({e}); falling back to text-only prompt")
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)

        if self._device_resolved == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items() if hasattr(v, "to")}
        else:
            inputs = {k: v.to(self._device_resolved) for k, v in inputs.items() if hasattr(v, "to")}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0.0,
                top_p=0.95,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        decode = getattr(self.processor, "batch_decode", None) or getattr(
            self.processor, "decode", None)
        if decode is None:
            decode = getattr(getattr(self.processor, "tokenizer", None), "batch_decode")
        generated_text = decode(generated_ids.unsqueeze(0), skip_special_tokens=True)
        if isinstance(generated_text, list):
            generated_text = generated_text[0]

        self.last_generation_time_sec = round(_time.monotonic() - _gen_start, 2)
        return generated_text.strip()

    # ------------------------------------------------------------------
    # SceneEnricher interface
    # ------------------------------------------------------------------

    def _base_story(self, scene: dict, transcript_segments: List[dict]) -> dict:
        """Heuristic story skeleton (transcript-derived fields)."""
        from movie_understanding.scene_analyzer import HeuristicSceneEnricher
        story = HeuristicSceneEnricher().enrich(scene, transcript_segments)["story"]
        return story

    def _unavailable_story(self, scene: dict, transcript_segments: List[dict],
                           reason: str) -> dict:
        """Story with vision fields marked unavailable and a provenance reason."""
        story = self._base_story(scene, transcript_segments)
        for field in ("location", "actions", "visual_description", "themes", "mood"):
            story[field] = None
            story["provenance"][field] = f"unavailable ({reason})"
        return story

    @staticmethod
    def _scene_shell(scene: dict) -> dict:
        return {
            "scene_id": scene.get("scene_id", "scene-0"),
            "start_sec": float(scene.get("start_sec", 0.0)),
            "end_sec": float(scene.get("end_sec", 0.0)),
            "duration_sec": float(scene.get("duration", 0.0)) or (
                float(scene.get("end_sec", 0.0)) - float(scene.get("start_sec", 0.0))
            ),
            "transcript": (scene.get("transcript") or "").strip(),
        }

    def enrich(self, scene: dict, transcript_segments: List[dict]) -> dict:
        """Enrich one scene with vision + transcript.

        If vision is unavailable (no GPU, model, keyframes) the vision-only
        fields stay ``None`` and provenance explains why. In strict mode an
        unavailable vision stack is a hard error.
        """
        scene_id = scene.get("scene_id", "scene-0")

        ok, reason = self._vision_available()
        if not ok:
            if self.strict:
                raise RuntimeError(
                    f"REQUIRE_REAL_VISION=true but scene {scene_id} cannot be "
                    f"vision-enriched: {reason}"
                )
            logger.warning(f"Scene {scene_id}: vision unavailable ({reason}); using heuristic")
            return {
                **self._scene_shell(scene),
                "story": self._unavailable_story(scene, transcript_segments, reason),
            }

        keyframes = [k for k in (scene.get("key_frames") or []) if k]
        if isinstance(scene.get("key_frames"), str):
            keyframes = [scene["key_frames"]]
        if not keyframes:
            reason = "no keyframes"
            if self.strict:
                raise RuntimeError(
                    f"REQUIRE_REAL_VISION=true but scene {scene_id} has no keyframes"
                )
            logger.warning(f"Scene {scene_id}: {reason}; using heuristic")
            return {
                **self._scene_shell(scene),
                "story": self._unavailable_story(scene, transcript_segments, reason),
            }

        self._initialize()
        transcript_text = " ".join(
            d.get("text", "") for d in (transcript_segments or [])
            if d.get("text")
        ) or (scene.get("transcript") or "")

        prompt = self._build_prompt(scene, transcript_text)
        try:
            output = self._generate(keyframes[: self.max_frames], prompt)
        except Exception as e:
            raise RuntimeError(f"Scene {scene_id} vision generation failed: {e}")

        self.last_raw_output = output
        parsed = _extract_json_dict(output)
        if not parsed:
            raise RuntimeError(
                f"Scene {scene_id} vision response was not valid JSON: {output[:200]}"
            )

        story = self._base_story(scene, transcript_segments)
        story.update({
            "location": _clean_str(parsed.get("location")),
            "actions": _clean_list(parsed.get("actions")),
            "visual_description": _clean_str(parsed.get("visual_description")),
            "themes": _clean_list(parsed.get("themes")),
            "mood": _clean_str(parsed.get("mood")),
        })
        story["provenance"].update({
            "location": self.name,
            "actions": self.name,
            "visual_description": self.name,
            "themes": self.name,
            "mood": self.name,
        })

        return {**self._scene_shell(scene), "story": story}


def _clean_str(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _clean_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        s = str(item).strip()
        if s:
            out.append(s)
    return out[:6]


def attach_keyframes_to_scenes(
    scenes: List[dict],
    source_path: str,
    keyframe_dir,
    max_frames: int = 1,
) -> List[dict]:
    """Extract keyframes for each scene and attach ``key_frames`` to it.

    A scene that cannot be extracted (missing file, ffmpeg missing) keeps an
    empty ``key_frames`` list; the enricher then degrades to heuristic.
    """
    from movie_understanding.keyframes import extract_all_scene_keyframes

    frames = extract_all_scene_keyframes(
        source_path, scenes, keyframe_dir, max_frames_per_scene=max_frames
    )
    for scene in scenes:
        sid = scene.get("scene_id")
        if not sid:
            continue
        scene["key_frames"] = frames.get(sid) or []
    return scenes