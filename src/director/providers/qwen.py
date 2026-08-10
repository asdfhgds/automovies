"""Qwen LLM provider for real creative direction."""
import json
import logging
import os
import re
from typing import Dict, Any, List, Optional

from .base import LLMProvider
from utils.json_guard import contains_placeholder

logger = logging.getLogger(__name__)

# Shared model cache keyed by (model_name, device, dtype, quantized).
# The director and script stages each create a QwenProvider; without sharing,
# both would load a full (~14GB fp16) copy of the model and OOM a 16GB T4.
_MODEL_CACHE: Dict[Any, Any] = {}

# System prompt used when the tokenizer exposes a chat template. Instruct models
# (e.g. Qwen3-4B-Instruct-2507) respond far more reliably (clean JSON, no chatty
# prose) when we wrap the raw task prompt in a chat turn.
_SYSTEM_PROMPT = (
    "You are a film critic and creative director. You analyze films to develop "
    "engaging video-essay concepts and production plans. Always answer with "
    "valid JSON only \u2014 no markdown, no code fences, no surrounding prose."
)


def _gpu_memory_gb() -> Optional[float]:
    """Total VRAM in GB of the first CUDA device, or None if unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return None


def _first_json_value(text: str, start: int = 0) -> Optional[str]:
    """Return the first balanced JSON object/array substring in ``text``.

    String-aware (handles quotes/escapes) so braces inside strings don't count.
    Returns None if no balanced value is found.
    """
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
    """Fix the common JSON mistakes small LLMs make; validate before returning."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(
        r"(?<![\w])'([^'\\]*(?:\\.[^'\\]*)*)'(?![\w])",
        lambda m: '"' + m.group(1).replace('"', '\\"') + '"',
        text,
    )
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        return None


class QwenProvider(LLMProvider):
    """Real LLM provider using Qwen models."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: str = "auto",
        dtype: str = "auto",
        thinking: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.9,
        max_new_tokens: int = 2048,
        timeout_sec: int = 180,
    ):
        """
        Initialize Qwen provider with lazy model loading.

        Args:
            model: Model identifier from HuggingFace (e.g., "Qwen/Qwen3-4B-Instruct-2507")
            device: "auto", "cuda", or "cpu"
            dtype: "auto", "float16", "float32"
            thinking: Enable extended thinking if supported
            temperature: Generation temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate
            timeout_sec: Generation timeout
        """
        self.model_name = model
        self.device = device
        self.dtype = dtype
        self.thinking = thinking
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.timeout_sec = timeout_sec

        # Lazy-loaded
        self.model = None
        self.tokenizer = None
        self._device_resolved = None
        self._initialized = False

        # Last raw generation output (for debugging failed JSON parsing)
        self.last_raw_output: Optional[str] = None

        # Real execution timing (recorded for validation reports)
        self.model_load_time_sec = None
        self.last_generation_time_sec = None
        self.generation_times = []

    @property
    def device_resolved(self) -> Optional[str]:
        """Device actually used after initialization (None before load)."""
        return self._device_resolved

    @classmethod
    def release_model(cls) -> None:
        """Drop the shared model cache and free GPU memory.

        Call between stages when you know the next stage uses a *different*
        model than the cached one (otherwise the cache is exactly what we want).
        """
        global _MODEL_CACHE
        _MODEL_CACHE = {}
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _initialize(self) -> None:
        """Lazy initialization of model and tokenizer."""
        if self._initialized:
            return

        logger.info(f"Initializing Qwen provider with model: {self.model_name}")
        import time as _time

        _load_start = _time.monotonic()

        try:
            import torch

            # Resolve device
            self._device_resolved = self._resolve_device()
            logger.info(f"Using device: {self._device_resolved}")

            # Resolve dtype
            dtype_map = {
                "auto": torch.float16 if self._device_resolved == "cuda" else torch.float32,
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            quantize_4bit = str(self.dtype).lower() in ("4bit", "int4", "nf4")
            torch_dtype = dtype_map.get(self.dtype, torch.float32)
            if quantize_4bit:
                torch_dtype = torch.float16
            logger.info(f"Using dtype: {torch_dtype} (4bit={quantize_4bit})")

            # Cache check happens BEFORE importing transformers so a hit never
            # pays for (or twice-loads) the model stack.
            cache_key = (self.model_name, self._device_resolved, str(torch_dtype), quantize_4bit)
            cached = _MODEL_CACHE.get(cache_key)
            if cached:
                self.model, self.tokenizer, cached_device = cached
                self._device_resolved = cached_device
                self._initialized = True
                self.model_load_time_sec = 0.0
                logger.info(
                    f"Reused cached model {self.model_name} on {cached_device} "
                    "(avoided a second full model load)"
                )
                return

            from transformers import AutoTokenizer, AutoModelForCausalLM

            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # Load model (memory-efficient: stream shards, keep headroom on GPU)
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
                        "Qwen dtype='4bit' requires bitsandbytes. "
                        "Install: pip install bitsandbytes"
                    )

            if self._device_resolved == "cuda":
                # device_map="auto" lets accelerate decide placement and spool
                # overflow to CPU instead of crashing with CUDA OOM.
                load_kwargs["device_map"] = "auto"
                mem_gb = _gpu_memory_gb()
                if mem_gb:
                    reserve = float(os.getenv("QWEN_VRAM_RESERVE_GB", "2.5"))
                    gpu_gb = max(2.0, mem_gb - reserve)
                    load_kwargs["max_memory"] = {0: f"{int(gpu_gb)}GiB"}
                    logger.info(f"CUDA max_memory set to {load_kwargs['max_memory']}")
                # Prefer memory-efficient SDPA attention (no flash-attn needed on T4).
                load_kwargs["attn_implementation"] = "sdpa"
            else:
                load_kwargs["device_map"] = None

            logger.info(f"Loading model {self.model_name}...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
            except (ValueError, NotImplementedError, RuntimeError) as e:
                # Some model configs reject explicit attn_implementation; retry with default.
                logger.warning(f"sdpa attention rejected ({e}); retrying with default attention")
                load_kwargs.pop("attn_implementation", None)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

            if self._device_resolved == "cuda":
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                    logger.info(f"GPU: {gpu_name} ({gpu_vram:.1f} GB VRAM)")
                except Exception as e:
                    logger.warning(f"Could not get GPU info: {e}")

            self._initialized = True
            self.model_load_time_sec = round(_time.monotonic() - _load_start, 2)
            _MODEL_CACHE[cache_key] = (self.model, self.tokenizer, self._device_resolved)
            logger.info(
                f"Qwen provider initialized successfully in {self.model_load_time_sec}s "
                f"(model cached for reuse)"
            )

        except ImportError as e:
            raise RuntimeError(
                f"Failed to import required packages for Qwen: {e}. "
                "Install: pip install torch transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qwen provider: {e}")

    def _resolve_device(self) -> str:
        """Resolve device: auto -> cuda/cpu, or return specified device."""
        if self.device != "auto":
            return self.device

        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            logger.info(f"Device auto-detected: {device}")
            return device

        except ImportError:
            logger.warning("PyTorch not available, using CPU")
            return "cpu"

    def _extract_json(self, text: str, expected_key: str = None) -> Optional[Dict]:
        """
        Extract JSON from LLM response, handling fenced JSON and wrapping text.

        Args:
            text: Raw LLM output
            expected_key: Expected top-level key (e.g., "concepts", "scores")

        Returns:
            Parsed JSON dict or None
        """
        text = (text or "").strip()
        if not text:
            return None

        candidates = []

        # Try 1: Direct JSON parsing
        candidates.append(text)

        # Try 2: Fenced JSON (```json ... ```)
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fenced:
            candidates.append(fenced.group(1).strip())

        # Try 3: First balanced JSON value inside prose ("Sure! Here it is: {...}")
        first_val = _first_json_value(text)
        if first_val and first_val != text:
            candidates.append(first_val)

        # Try 4: Targeted lookup of a specific key's value
        if expected_key:
            key_match = re.search(r'"' + re.escape(expected_key) + r'"\s*:', text)
            if key_match:
                val = _first_json_value(text, key_match.end())
                if val:
                    candidates.append(val)

        for candidate in candidates:
            for variant in (candidate, _repair_json(candidate)):
                if not variant:
                    continue
                try:
                    return json.loads(variant)
                except json.JSONDecodeError:
                    continue

        logger.warning(f"Failed to extract JSON from response: {text[:100]}")
        return None

    def _validate_concepts(self, concepts: List[Dict]) -> bool:
        """Validate concept structure and diversity."""
        required_fields = ["title", "hook", "thesis", "why_interesting"]

        for i, concept in enumerate(concepts):
            for field in required_fields:
                if field not in concept or not concept[field]:
                    logger.warning(
                        f"Concept {i} missing or empty field: {field}"
                    )
                    return False

        # Check for diversity (concepts shouldn't be nearly identical)
        if len(concepts) >= 2:
            # Simple heuristic: theses should be somewhat different
            theses = [c.get("thesis", "") for c in concepts]
            if len(theses) == len(set(theses)):
                # All unique, good
                logger.info("Concepts are diverse (unique theses)")
                return True
            else:
                logger.warning("Some concepts have identical theses")
                return False

        return True

    def _build_generation_prompt(
        self,
        movie_metadata: Dict[str, Any],
        scene_index: List[Dict[str, Any]],
        transcript: Dict[str, Any],
        creative_memory: str,
        user_topic: str = None,
        num_concepts: int = 3,
    ) -> str:
        """Build concept generation prompt."""
        movie_title = movie_metadata.get("title", "Unknown")
        movie_duration = movie_metadata.get("duration_sec", 0)

        # Build scene summary (limit to avoid token explosion)
        scene_summary = ""
        if scene_index:
            # Include first 5 scenes with timestamps
            for scene in scene_index[:5]:
                scene_id = scene.get("scene_id", "unknown")
                start = scene.get("start_sec", 0)
                end = scene.get("end_sec", 0)
                transcript_text = scene.get("transcript", "")[:200]
                scene_summary += f"\n- {scene_id}: {start:.1f}s-{end:.1f}s | {transcript_text}"

        # Build memory context
        memory_context = ""
        if creative_memory and creative_memory.strip():
            memory_context = f"\n\n## Previous Concepts (for avoiding repetition):\n{creative_memory}"

        # Build transcript summary (limit)
        transcript_text = ""
        if transcript and "segments" in transcript:
            segments = transcript["segments"][:10]  # First 10 segments
            for seg in segments:
                text = seg.get("text", "")[:100]
                transcript_text += f"\n- {text}"

        prompt = f"""You are a creative director analyzing a film to generate engaging video essay concepts.

## Movie Details
- Title: {movie_title}
- Duration: {movie_duration:.1f} seconds

## Scene Breakdown{scene_summary}

## Transcript (first segments){transcript_text}{memory_context}

## Task
Generate {num_concepts} distinct creative concepts for this film. Each concept must:
1. Have a SPECIFIC claim (not generic platitudes)
2. Reference actual scenes or dialogue from the film
3. Be visually compelling and suitable for a video essay
4. Have a clear hook and thesis
5. Be diverse in approach (psychology, philosophy, symbolism, narrative structure, etc.)

Do NOT generate generic concepts like:
- "This movie teaches us about life"
- "Characters show emotions"
- "Good versus evil"

Instead, generate specific interpretations with evidence.

{f"User Focus: {user_topic}" if user_topic else ""}

Return ONLY valid JSON (no markdown, no code fences) with this structure:

{{
  "concepts": [
    {{
      "title": "YOUR_ORIGINAL_TITLE_HERE",
      "hook": "YOUR_ORIGINAL_OPENING_HOOK_HERE",
      "thesis": "YOUR_SPECIFIC_EVIDENCE_BASED_THESIS_HERE",
      "why_interesting": "WHY_THIS_ANGLE_MATTERS_HERE",
      "supporting_scene_types": ["A_SCENE_TYPE_FROM_THE_MATERIAL_ABOVE"],
      "tone": "A_SINGLE_TONE_FROM: analytical_philosophical|psychological_intimate|metanarrative_interrogative|other",
      "visual_strategy": "YOUR_VISUAL_APPROACH_HERE",
      "estimated_duration_sec": 85
    }}
  ]
}}

IMPORTANT - REPLACE, DON'T COPY:
- Every ALL-CAPS string above is a placeholder, NOT text to output.
- Write original, specific content based ONLY on THIS film's transcript and scenes above.
- Your title, hook, and thesis must reference concrete details from the material above.
- Never output placeholder strings like "YOUR_ORIGINAL_TITLE_HERE".

Generate now:"""

        return prompt

    def _build_critique_prompt(self, concept: Dict[str, Any]) -> str:
        """Build concept critique prompt."""
        prompt = f"""You are a film critic evaluating a creative concept.

## Concept to Critique
Title: {concept.get('title', 'Untitled')}
Hook: {concept.get('hook', '')}
Thesis: {concept.get('thesis', '')}
Why Interesting: {concept.get('why_interesting', '')}

## Evaluation Criteria
Score each 0.0-1.0:
1. originality: How unique is this interpretation?
2. thesis_strength: Is the thesis clear, specific, and compelling?
3. evidence_strength: Could this concept be supported by scenes?
4. visual_potential: Can this be made visually compelling?
5. audience_curiosity: Would audiences find this interesting?
6. feasibility: Can this realistically be produced?

Return ONLY valid JSON (no markdown, no fencing):
{{
  "scores": {{
    "originality": 0.7,
    "thesis_strength": 0.8,
    "evidence_strength": 0.6,
    "visual_potential": 0.8,
    "audience_curiosity": 0.7,
    "feasibility": 0.8
  }},
  "overall": 0.75,
  "critique": "Brief summary of strengths and areas for improvement"
}}

Evaluate now:"""

        return prompt

    def _build_production_plan_prompt(
        self,
        concept: Dict[str, Any],
        scene_index: List[Dict[str, Any]],
    ) -> str:
        """Build production plan prompt."""
        duration = concept.get("estimated_duration_sec", 90)

        # Scene types available
        scene_types = set()
        for scene in scene_index:
            types = scene.get("supporting_scene_types", [])
            if isinstance(types, list):
                scene_types.update(types)

        scene_types_str = ", ".join(sorted(scene_types)[:10]) if scene_types else "unknown"

        prompt = f"""You are a film director planning production for a video essay.

## Concept
Title: {concept.get('title', 'Untitled')}
Thesis: {concept.get('thesis', '')}
Tone: {concept.get('tone', 'analytical')}

## Constraints
- Target duration: {duration} seconds
- Available scenes: {scene_types_str}

## Task
Create a detailed production plan with:
1. Structure (distinct sections with durations)
2. Scene requirements for evidence
3. Visual strategy (cinematography, editing, effects)
4. Music strategy (mood, instrumentation)
5. Editing strategy (pacing, cuts, transitions)

Return ONLY valid JSON (no markdown, no code fences) with this structure:

{{
  "structure": [
    {{"section": "YOUR_SECTION_NAME_HERE", "duration_sec": 15}}
  ],
  "scene_requirements": [
    {{"purpose": "YOUR_PURPOSE_HERE", "preferred_scene_types": ["YOUR_SCENE_TYPE_HERE"]}}
  ],
  "visual_strategy": "YOUR_VISUAL_STRATEGY_HERE",
  "music_strategy": "YOUR_MUSIC_STRATEGY_HERE",
  "editing_strategy": "YOUR_EDITING_STRATEGY_HERE"
}}

IMPORTANT - REPLACE, DON'T COPY:
- The ALL-CAPS strings above are placeholders, NOT text to output.
- Give real section names and strategies grounded in the concept above.
- Never output placeholder strings like "YOUR_SECTION_NAME_HERE".

Plan now:"""

        return prompt

    def generate_concepts(
        self,
        movie_metadata: Dict[str, Any],
        scene_index: List[Dict[str, Any]],
        transcript: Dict[str, Any],
        creative_memory: str,
        user_topic: str = None,
        num_concepts: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate creative concepts using Qwen."""
        self._initialize()

        prompt = self._build_generation_prompt(
            movie_metadata,
            scene_index,
            transcript,
            creative_memory,
            user_topic,
            num_concepts,
        )

        logger.info(f"Generating {num_concepts} concepts with Qwen...")
        last_error: Optional[str] = None
        # One retry with a different sampling seed often fixes flaky JSON.
        for attempt in range(2):
            output = self._generate_text(prompt, seed_override=attempt)
            self.last_raw_output = output

            result = self._extract_json(output, "concepts")
            if not result:
                last_error = f"Failed to extract JSON from generation output: {output[:400]}"
                logger.warning(f"Concept parse failed (attempt {attempt + 1}): {last_error}")
                continue

            concepts = self._coerce_concepts(result)
            if not concepts:
                last_error = (
                    f"Generation returned empty concepts list. "
                    f"Raw output: {output[:400]}"
                )
                logger.warning(f"Concept coercion empty (attempt {attempt + 1}): {last_error}")
                continue

            # Small models sometimes echo the prompt's example JSON instead of
            # writing original concepts; treat that as a failed generation.
            if self._has_placeholder_concepts(concepts):
                last_error = (
                    "Generation echoed prompt placeholder text. "
                    f"Raw output: {output[:400]}"
                )
                logger.warning(f"Concept placeholders (attempt {attempt + 1}): {last_error}")
                continue

            # Validate
            if not self._validate_concepts(concepts):
                logger.warning("Concepts failed validation, but proceeding")

            logger.info(f"Generated {len(concepts)} concepts")
            return concepts[:num_concepts]

        raise ValueError(last_error or "Qwen concept generation failed")

    @staticmethod
    def _has_placeholder_concepts(concepts: List[Dict[str, Any]]) -> bool:
        """True if any concept looks like it copied the prompt's example text."""
        for concept in concepts:
            for field in ("title", "hook", "thesis", "why_interesting", "visual_strategy"):
                if contains_placeholder(concept.get(field)):
                    return True
        return False

    @staticmethod
    def _coerce_concepts(result: Any) -> List[Dict[str, Any]]:
        """Coerce whatever the model returned into a list of concept dicts.

        Tolerant of the shapes small instruct models actually emit:
          {"concepts": [ ... ]}
          [ ... ]                              (bare array)
          {"concept": {...}} / {"selected_concept": {...}}
          {...}                                (single concept at top level)
        """
        if isinstance(result, list):
            return [c for c in result if isinstance(c, dict)]
        if not isinstance(result, dict):
            return []

        concepts = result.get("concepts")
        if isinstance(concepts, list):
            return [c for c in concepts if isinstance(c, dict)]

        for key in ("concept", "selected_concept"):
            value = result.get(key)
            if isinstance(value, dict):
                return [value]

        if all(k in result for k in ("title", "hook", "thesis")):
            return [result]

        return []

    def refine_concept(
        self, concept: Dict[str, Any], feedback: str
    ) -> Dict[str, Any]:
        """Refine concept based on feedback."""
        self._initialize()

        prompt = f"""Refine this concept based on feedback:

## Original Concept
Title: {concept.get('title', '')}
Thesis: {concept.get('thesis', '')}

## Feedback
{feedback}

Provide refined concept as JSON:
{{
  "title": "...",
  "hook": "...",
  "thesis": "...",
  "why_interesting": "...",
  "tone": "...",
  "visual_strategy": "..."
}}"""

        try:
            output = self._generate_text(prompt)
            refined = self._extract_json(output)
            if refined:
                # Merge with original
                concept.update(refined)
            return concept
        except Exception as e:
            logger.error(f"Qwen refinement failed: {e}")
            return concept

    def generate_production_plan(
        self,
        concept: Dict[str, Any],
        scene_index: List[Dict[str, Any]],
        transcript: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate production plan for concept."""
        self._initialize()

        prompt = self._build_production_plan_prompt(concept, scene_index)

        logger.info("Generating production plan with Qwen...")
        last_error: Optional[str] = None
        for attempt in range(2):
            output = self._generate_text(prompt, seed_override=attempt)
            self.last_raw_output = output

            plan = self._extract_json(output)
            if not plan:
                last_error = f"Failed to extract JSON from production plan: {output[:400]}"
                logger.warning(f"Plan parse failed (attempt {attempt + 1}): {last_error}")
                continue

            # Reject a plan that just copied the prompt's example placeholders.
            text_fields = [
                plan.get("visual_strategy"),
                plan.get("music_strategy"),
                plan.get("editing_strategy"),
            ]
            for req in plan.get("scene_requirements") or []:
                if isinstance(req, dict):
                    text_fields.append(req.get("purpose"))
                    types = req.get("preferred_scene_types")
                    if isinstance(types, list):
                        text_fields.extend(types)
            if any(contains_placeholder(f) for f in text_fields):
                last_error = (
                    "Production plan echoed prompt placeholder text. "
                    f"Raw output: {output[:400]}"
                )
                logger.warning(f"Plan placeholders (attempt {attempt + 1}): {last_error}")
                continue

            logger.info("Production plan generated")
            return plan

        raise ValueError(last_error or "Qwen production plan generation failed")

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Public convenience wrapper for raw text generation.

        Used by the script stage and any caller that needs ad-hoc model output.
        Ensures the model is initialized (loaded) before generation.
        """
        self._initialize()
        if max_new_tokens is not None:
            self.max_new_tokens = int(max_new_tokens)
        return self._generate_text(prompt)

    def _generate_text(self, prompt: str, seed_override: int = None) -> str:
        """
        Generate text using the model.

        Args:
            prompt: Input prompt
            seed_override: Optional int to add to the generation seed so a
                retry draws a different sample without reloading the model

        Returns:
            Generated text (only the newly generated tokens, decoded)
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized")

        import time as _time
        _gen_start = _time.monotonic()

        try:
            import torch

            # Honor the thinking flag when supported by the model.
            try:
                if hasattr(self.model, "generation_config"):
                    self.model.generation_config.enable_thinking = bool(self.thinking)
            except Exception:
                pass

            # Wrap the task in a chat turn when the tokenizer supports it.
            # Instruct models follow chat formatting much better than raw
            # completion prompts, which yields cleaner JSON output.
            try:
                if hasattr(self.tokenizer, "apply_chat_template"):
                    chat = self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if chat:
                        prompt = chat
            except Exception as e:
                logger.warning(f"apply_chat_template failed ({e}); using raw prompt")

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            )
            input_len = inputs["input_ids"].shape[1]

            # Move to device
            if self._device_resolved == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                # Vary the RNG so a retry draws a different sample (older
                # transformers versions reject generate(seed=...)).
                if seed_override is not None:
                    torch.manual_seed(42 + seed_override)
                    if self._device_resolved == "cuda":
                        torch.cuda.manual_seed_all(42 + seed_override)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                )

            # Decode ONLY the new tokens (drop prompt + chat wrappers).
            generated_ids = outputs[0][input_len:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            self.last_generation_time_sec = round(_time.monotonic() - _gen_start, 2)
            self.generation_times.append(self.last_generation_time_sec)

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
