"""Qwen-backed narration script generation (real LLM).

Produces script.json using the same schema as the deterministic writer
(src/script/writer.py) so all downstream stages (TTS, timeline, render) work
unchanged. Records provider metadata so strict GPU validation can confirm a real
Qwen model actually generated the script.

No silent fallback: on any failure this module raises.
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_selected_scenes(project_dir: Path) -> List[Dict[str, Any]]:
    data = _load_json(project_dir / "scenes" / "selected_scenes.json")
    if isinstance(data, dict):
        data = [data]
    if isinstance(data, list) and data:
        return data
    data = _load_json(project_dir / "scenes" / "selected_scene.json")
    if isinstance(data, dict):
        data = [data]
    return data if isinstance(data, list) else []


def _load_scene_index(project_dir: Path) -> Dict[str, Any]:
    data = _load_json(project_dir / "scenes" / "scene_index.json")
    if isinstance(data, dict):
        data = data.get("scenes", [])
    return {s.get("scene_id"): s for s in (data or []) if s.get("scene_id")}


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start != -1:
        cnt = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                cnt += 1
            elif text[i] == "}":
                cnt -= 1
                if cnt == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def _build_prompt(
    thesis: str,
    tone: str,
    structure: List[Dict[str, Any]],
    selected_scenes: List[Dict[str, Any]],
    scene_map: Dict[str, Any],
) -> str:
    sections_desc = "\n".join(
        f"- {s.get('id', 'section')}: {s.get('goal', 'develop the analysis')}"
        f" (~{int(s.get('target_seconds', 15))}s)"
        for s in structure
    )
    scene_lines = []
    for sc in selected_scenes:
        sid = sc.get("scene_id")
        info = scene_map.get(sid, {})
        text = (info.get("transcript") or info.get("summary") or "").strip()[:220]
        scene_lines.append(f"- {sid} [{sc.get('start_sec', 0):.1f}s-{sc.get('end_sec', 0):.1f}s]: {text}")

    return f"""You are a screenwriter for a short video-essay. Write the narration script.

## Thesis
{thesis}

## Tone
{tone}

## Required sections
{sections_desc}

## Selected scenes (evidence for the analysis)
{''.join(scene_lines) if scene_lines else '- (no scenes listed)'}

## Rules
- The hook/intro section must open with the thesis.
- Analysis sections must reference the listed scenes as evidence.
- The closing section must conclude the thesis.
- estimated_seconds for all sections should total roughly the sum of the target seconds.
- scene_ids must be chosen ONLY from the scene ids listed above.

Return ONLY valid JSON (no markdown, no code fences) with this exact structure:
{{
  "sections": [
    {{"section_id": "intro", "text": "Narration...", "estimated_seconds": 15, "scene_ids": ["scene-1"]}}
  ]
}}"""


def _normalize_sections(raw: Any, structure: List[Dict[str, Any]], selected_scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Coerce model output into the deterministic script schema."""
    ids = [s.get("section_id") for s in structure]
    known_scene_ids = [s.get("scene_id") for s in selected_scenes]

    if not isinstance(raw, list) or not raw:
        raise ValueError("Qwen returned no usable sections")

    sections = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        section_id = item.get("section_id")
        if section_id in (None, ""):
            section_id = f"section_{len(sections) + 1}"
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        try:
            est = int(float(item.get("estimated_seconds", 15)))
        except (TypeError, ValueError):
            est = 15
        est = max(1, min(est, 300))
        scene_ids = item.get("scene_ids")
        if not isinstance(scene_ids, list) or not scene_ids:
            scene_ids = known_scene_ids[:1]
        scene_ids = [s for s in scene_ids if s in known_scene_ids][:3] or known_scene_ids[:1]
        sections.append({
            "section_id": section_id,
            "text": text,
            "estimated_seconds": est,
            "scene_ids": scene_ids,
        })
        if len(sections) >= 12:
            break

    if not sections:
        raise ValueError("Qwen returned no valid narration sections")
    return sections


def generate_script_qwen(
    project_dir: Path,
    model: str = "Qwen/Qwen3-7B-A0.5B",
    device: str = "cuda",
    max_new_tokens: int = 1024,
    thinking: bool = False,
    temperature: float = 0.7,
    dtype: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate script.json with a real Qwen model.

    Loads the director plan + selected scenes, calls Qwen, parses the narration
    sections, and writes script.json in the canonical schema.

    `dtype` defaults to the SCRIPT_DTYPE env var (or "auto"). Use "4bit" to load
    the model in NF4 (~4GB) when VRAM is tight. The model is shared with the
    director stage through QwenProvider's class-level cache, so the 7B weights
    are only resident once.

    Returns the script dict (also persisted). Raises on any failure.
    """
    if dtype is None:
        dtype = os.getenv("SCRIPT_DTYPE", "auto")

    project_dir = Path(project_dir)
    plan = _load_json(project_dir / "director_plan.json")
    if not plan:
        raise FileNotFoundError("director_plan.json missing")

    thesis = plan.get("thesis") or "the meaning hidden inside a pivotal scene"
    tone = plan.get("tone") or "analytical"
    structure = plan.get("structure") or [
        {"id": "intro", "goal": "Hook and thesis", "target_seconds": 20},
        {"id": "scene_discussion", "goal": "Explain the scene", "target_seconds": 60},
        {"id": "closing", "goal": "Wrap and CTA", "target_seconds": 10},
    ]

    selected = _load_selected_scenes(project_dir)
    scene_map = _load_scene_index(project_dir)
    if not selected:
        raise FileNotFoundError("selected_scenes.json / selected_scene.json missing")

    from director.providers.qwen import QwenProvider

    provider = QwenProvider(
        model=model,
        device=device,
        dtype=dtype,
        thinking=thinking,
        temperature=temperature,
        top_p=0.9,
        max_new_tokens=max_new_tokens,
    )

    prompt = _build_prompt(thesis, tone, structure, selected, scene_map)
    output = provider.generate_text(prompt, max_new_tokens=max_new_tokens)

    parsed = _extract_json(output)
    if not parsed:
        raise RuntimeError(
            f"Qwen script response was not valid JSON: {output[:300]}"
        )

    sections = _normalize_sections(parsed.get("sections"), structure, selected)
    voiceover = " ".join(s["text"] for s in sections)

    meta_path = project_dir / "project_meta.json"
    project_id = project_dir.name
    if meta_path.exists():
        meta = _load_json(meta_path) or {}
        project_id = meta.get("project_id", project_id)

    script = {
        "project_id": project_id or project_dir.name,
        "voiceover_text": voiceover,
        "sections": sections,
        "cta": "Subscribe for more",
        "style_notes": f"{tone} commentary with a concise, cinematic pace",
        "scene_ids": [s.get("scene_id") for s in selected],
        "script_provider": "qwen",
        "script_model": model,
        "script_device": provider.device_resolved or device,
        "script_dtype": dtype,
        "qwen_load_time_sec": provider.model_load_time_sec,
        "qwen_generation_time_sec": provider.last_generation_time_sec,
    }

    out_path = project_dir / "script.json"
    out_path.write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote Qwen script -> {out_path}")
    return script