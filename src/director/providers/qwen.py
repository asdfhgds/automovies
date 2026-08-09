"""Qwen LLM provider for real creative direction."""
import json
import logging
import os
import re
from typing import Dict, Any, List, Optional

from .base import LLMProvider

logger = logging.getLogger(__name__)


class QwenProvider(LLMProvider):
    """Real LLM provider using Qwen models."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-7B-A0.5B",
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
            model: Model identifier from HuggingFace (e.g., "Qwen/Qwen3-7B-A0.5B")
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

        # Real execution timing (recorded for validation reports)
        self.model_load_time_sec = None
        self.last_generation_time_sec = None
        self.generation_times = []

    @property
    def device_resolved(self) -> Optional[str]:
        """Device actually used after initialization (None before load)."""
        return self._device_resolved

    def _initialize(self) -> None:
        """Lazy initialization of model and tokenizer."""
        if self._initialized:
            return

        logger.info(f"Initializing Qwen provider with model: {self.model_name}")
        import time as _time

        _load_start = _time.monotonic()

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

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
            torch_dtype = dtype_map.get(self.dtype, torch.float32)
            logger.info(f"Using dtype: {torch_dtype}")

            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # Load model
            logger.info(f"Loading model {self.model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=self._device_resolved,
                trust_remote_code=True,
            )

            if self._device_resolved == "cuda":
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                    logger.info(f"GPU: {gpu_name} ({gpu_vram:.1f} GB VRAM)")
                except Exception as e:
                    logger.warning(f"Could not get GPU info: {e}")

            self._initialized = True
            self.model_load_time_sec = round(_time.monotonic() - _load_start, 2)
            logger.info(
                f"Qwen provider initialized successfully in {self.model_load_time_sec}s"
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
        text = text.strip()

        # Try 1: Direct JSON parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try 2: Fenced JSON (```json ... ```)
        json_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try 3: Look for JSON object within text
        # Find first { and last }
        start = text.find("{")
        if start != -1:
            # Find matching closing brace
            brace_count = 0
            end = -1
            for i in range(start, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break

            if end != -1:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

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

Return ONLY valid JSON (no markdown, no fencing) with this structure:
{{
  "concepts": [
    {{
      "title": "Specific concept title",
      "hook": "Engaging question or statement that draws viewers in",
      "thesis": "The core argument about the film (specific, evidence-based)",
      "why_interesting": "Why this interpretation reveals something important",
      "supporting_scene_types": ["type1", "type2"],
      "tone": "analytical_philosophical|psychological_intimate|metanarrative_interrogative|other",
      "visual_strategy": "How to visualize this concept in editing, effects, etc.",
      "estimated_duration_sec": 85
    }}
  ]
}}

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

Return ONLY valid JSON (no markdown, no fencing):
{{
  "structure": [
    {{"section": "hook", "duration_sec": 10}},
    {{"section": "setup", "duration_sec": 20}},
    {{"section": "analysis", "duration_sec": 50}},
    {{"section": "conclusion", "duration_sec": 10}}
  ],
  "scene_requirements": [
    {{"purpose": "demonstrate_thesis", "preferred_scene_types": ["type1", "type2"]}}
  ],
  "visual_strategy": "Description of visual approach",
  "music_strategy": "Description of music approach",
  "editing_strategy": "Description of editing approach"
}}

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

        try:
            prompt = self._build_generation_prompt(
                movie_metadata,
                scene_index,
                transcript,
                creative_memory,
                user_topic,
                num_concepts,
            )

            logger.info(f"Generating {num_concepts} concepts with Qwen...")
            output = self._generate_text(prompt)

            # Extract and parse JSON
            result = self._extract_json(output, "concepts")
            if not result:
                raise ValueError("Failed to extract JSON from generation output")

            concepts = result.get("concepts", [])
            if not concepts:
                raise ValueError("Generation returned empty concepts list")

            # Validate
            if not self._validate_concepts(concepts):
                logger.warning("Concepts failed validation, but proceeding")

            logger.info(f"Generated {len(concepts)} concepts")
            return concepts[:num_concepts]

        except Exception as e:
            logger.error(f"Qwen concept generation failed: {e}")
            raise

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

        try:
            prompt = self._build_production_plan_prompt(concept, scene_index)

            logger.info("Generating production plan with Qwen...")
            output = self._generate_text(prompt)

            # Extract and parse JSON
            plan = self._extract_json(output)
            if not plan:
                raise ValueError("Failed to extract JSON from production plan")

            logger.info("Production plan generated")
            return plan

        except Exception as e:
            logger.error(f"Qwen production plan generation failed: {e}")
            raise

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

    def _generate_text(self, prompt: str) -> str:
        """
        Generate text using the model.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
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

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            )

            # Move to device
            if self._device_resolved == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                )

            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt from output
            if prompt in generated_text:
                generated_text = generated_text[generated_text.index(prompt) + len(prompt) :]

            self.last_generation_time_sec = round(_time.monotonic() - _gen_start, 2)
            self.generation_times.append(self.last_generation_time_sec)

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
