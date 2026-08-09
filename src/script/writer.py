"""Generate the first usable narration script for a project.

The MVP intentionally keeps script generation deterministic, but it now consumes
the same scene-index formats used by the rest of the pipeline.  This makes the
output useful to the TTS and editing stages instead of being a disconnected
placeholder.
"""
import json
import os
from pathlib import Path


def _load_scenes(project_dir: Path):
    """Load the preferred scene index, falling back to legacy scene cards."""
    for path in (
        project_dir / "scenes" / "scene_index.json",
        project_dir / "scenes" / "scene_cards.json",
    ):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("scenes", [])
        if isinstance(data, list):
            return data
    return []


def generate_script(project_dir: Path):
    project_dir = Path(project_dir)
    plan_p = project_dir / 'director_plan.json'
    out_dir = project_dir
    if not plan_p.exists():
        raise FileNotFoundError('director_plan.json missing')
    with plan_p.open('r', encoding='utf-8') as f:
        plan = json.load(f)
    scenes = _load_scenes(project_dir)
    if not scenes:
        raise FileNotFoundError('scene_index.json or scene_cards.json missing')

    selected_path = project_dir / "scenes" / "selected_scenes.json"
    selected_ids = []
    if selected_path.exists():
        selected_ids = [item.get("scene_id") for item in json.loads(selected_path.read_text(encoding="utf-8"))]
    scenes_by_id = {scene.get("scene_id"): scene for scene in scenes}
    selected_scenes = [scenes_by_id[scene_id] for scene_id in selected_ids if scene_id in scenes_by_id] or [scenes[0]]
    require_real_llm = os.getenv("REQUIRE_REAL_LLM", "false").lower() == "true"
    if os.getenv("SCRIPT_PROVIDER", "mock").lower() == "qwen":
        try:
            from director.provider_factory import get_director_config_from_env, get_llm_provider_from_config
            from script.providers import QwenScriptProvider
            config = get_director_config_from_env()
            config["provider"] = "qwen"
            config["model"] = os.getenv("SCRIPT_MODEL", config.get("model"))
            config["device"] = os.getenv("SCRIPT_DEVICE", config.get("device", "auto"))
            provider = get_llm_provider_from_config(config)
            if provider is None:
                raise RuntimeError("No Qwen provider available")
            target_duration = int(plan.get("length_target_sec", 60))
            generated = QwenScriptProvider(provider).generate_script(
                thesis=plan.get("thesis", ""), selected_scenes=selected_scenes,
                movie_context={"title": plan.get("topic", "")}, target_duration=target_duration,
                tone=plan.get("tone", "analytical"),
            )
            generated["project_id"] = plan.get("project_id", project_dir.name)
            generated["cta"] = "Subscribe for more"
            generated["provider_metadata"] = provider.model_info() if hasattr(provider, "model_info") else {"provider": "qwen"}
            out_path = project_dir / "script.json"
            out_path.write_text(json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote Qwen script -> {out_path}")
            return out_path
        except Exception as exc:
            if require_real_llm:
                raise RuntimeError(f"Real Qwen script generation failed: {exc}") from exc
            print(f"Qwen script generation failed ({exc}); using deterministic fallback")
    elif require_real_llm:
        raise RuntimeError("Strict validation requires SCRIPT_PROVIDER=qwen")
    first_scene = selected_scenes[0]
    thesis = plan.get("thesis") or "the meaning hidden inside a pivotal scene"
    summary = first_scene.get("summary") or first_scene.get("transcript") or "a pivotal moment"
    structure = plan.get("structure") or [
        {"id": "intro", "goal": "Hook and thesis", "target_seconds": 20},
        {"id": "scene_discussion", "goal": "Explain the scene", "target_seconds": 60},
        {"id": "closing", "goal": "Wrap and CTA", "target_seconds": 10},
    ]

    sections = []
    for section in structure:
        section_id = section.get("id", f"section_{len(sections) + 1}")
        goal = section.get("goal", "Develop the analysis")
        duration = max(1, int(section.get("target_seconds", 15)))
        if section_id in ("intro", "hook"):
            text = f"At first glance, this moment seems simple. But {thesis}"
        elif section_id in ("closing", "conclusion"):
            text = f"That is why this scene matters: {thesis}."
        else:
            text = f"{goal}. In this scene, {summary}. This gives us a way to see how {thesis}"
        sections.append({
            "section_id": section_id,
            "text": text,
            "estimated_seconds": duration,
            "scene_ids": [scene.get("scene_id", "scene-1") for scene in selected_scenes],
        })

    voiceover = " ".join(section["text"] for section in sections)
    project_id = plan.get("project_id")
    meta_path = project_dir / "project_meta.json"
    if meta_path.exists():
        project_id = json.loads(meta_path.read_text(encoding="utf-8")).get("project_id", project_id)
    script = {
        "project_id": project_id or project_dir.name,
        "voiceover_text": voiceover,
        "sections": sections,
        "cta": "Subscribe for more",
        "style_notes": f"{plan.get('tone', 'analytical')} commentary with a concise, cinematic pace",
    }
    out_path = out_dir / 'script.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(script, f, ensure_ascii=False, indent=2)
    print(f"Wrote script -> {out_path}")
    return out_path
