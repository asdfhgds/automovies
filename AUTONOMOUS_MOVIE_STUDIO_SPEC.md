Autonomous Movie & TV Video Studio — Full Specification

Project Vision

This project is an autonomous movie-and-TV video studio: a commentary-first creative system that uses a director-style planning layer, scene understanding, AI narration, generative visuals, and automated editing to produce varied, polished commentary videos from film and television content.

Goals

- Produce diverse video styles: scene analysis, philosophical essays, theory videos, character breakdowns, recaps, montages, and experimental visual essays.
- Deliver finished videos (voiceover, music, visuals, subtitles, cuts) rather than raw clips.
- Avoid repetitive outputs by enforcing novelty constraints and creative memory.

Core Idea

Design the app as a pipeline of specialized modules rather than a monolithic generator. Key components:

- Director (planner) — chooses concept, structure, tone, and novelty constraints.
- Research/metadata — fetches TMDb and source info.
- Transcription & Scene Indexing — transcripts with timestamps, shot/scene detection, scene cards.
- Scene Selection — semantic matching between thesis and scene cards.
- Script/Writer — generates narration, hooks, and section text.
- Visual Planner & Generation — decides assets and renders images/video inserts via ComfyUI or other engines.
- Audio — TTS narration and music generation/selection and mixing.
- Editor/Assembler — builds timeline with FFmpeg or an editor layer, adds subtitles and exports.
- Quality Control/Critic — evaluates novelty, coherence, pacing, and visual match; triggers partial regenerations when needed.

Design Principles

- Director-first: creative decisions originate from a planning layer.
- Specialized modules: use tailored tools for transcription, TTS, generation, and editing.
- Pipeline over monolith: break the workflow into stages that can be checkpointed and resumed.
- Variation by design: use memory and scoring to avoid repetition.
- Legal-first testing: start with licensed or public-domain content for experiments.

User-Facing Flow (MVP)

1. User selects a movie or episode (or a seed topic).
2. App fetches metadata and registers the source.
3. Transcribe the source, detect shots/scenes, build scene cards.
4. Director produces candidate plans and chooses the best novel plan.
5. Scene selection ranks scene cards and picks best-supporting clip(s).
6. Script module writes narration and sections to a target duration.
7. Visual planner maps sections to assets (clip, generated image, title card, overlay).
8. Generate assets: TTS, images, short motion inserts, music bed.
9. Editor assembles timeline, mixes audio, adds subtitles.
10. QC runs checks and optionally regenerates weak parts.
11. Export final video and save metadata to memory for novelty tracking.

Scene-Finding & Understanding

Approach:
- Transcribe with word-level timestamps.
- Detect shot/scene boundaries via scene detection.
- Build scene cards containing start/end times, transcript, visual notes, keyframes, shot count, summary, and keywords.
- Semantic ranking: compare director thesis (as search terms or embeddings) to scene cards; score by dialogue match, emotional tone, visual tension, and keywords.
- Fallbacks: widen search windows, search adjacent scenes, rephrase thesis, or replan.

Asset Generation Strategy

- TTS: use high-quality TTS stack for narration; allow multiple voices and prosody tuning.
- Image generation: use ComfyUI or an open-source image model for illustrations, thumbnails, and stylized assets.
- Video inserts: small motion clips for transitions or emphasis; do not attempt to generate long continuous scenes.
- Music: generate or select a music bed; set metadata for mood and avoid copyright issues in testing.
- Each asset type uses a dedicated engine with caching and reproducible prompts.

ComfyUI Role

ComfyUI functions as a production visual engine: given structured asset plans, it produces images, refines styles, and builds reusable workflows. The app’s director should call ComfyUI with an asset prompt and parameters; ComfyUI is not the decision-maker.

Novelty & Anti-Repetition

- Maintain a memory of recent outputs (N last projects) recording topic, hook style, visual style, pacing, structure, music mood, and ending type.
- Produce 3–5 candidate plans; score for novelty vs recent outputs and usability; reject near-duplicates.
- Enforce minimal style distance parameter and explicit novelty constraints in director plan schema.

Quality Control

QC must inspect:
- Structural coherence and hook strength.
- Alignment between script and selected scene(s).
- Repetition vs recent outputs.
- Audio levels and render integrity.
QC should aim to regenerate only failing components rather than rebuilding the whole video.

Legal & Test Library

- Begin with licensed content, public-domain films, or internal test set.
- Tests cover scene detection, transcript timing, selection quality, script quality, asset generation, and render success.

Recommended Architecture

Layers:
- Input Layer: title, file/source, style, length, thesis.
- Planning Layer: director plan JSON.
- Scene Intelligence: transcript, scene detection, scene cards, ranking.
- Generation Layer: voice, images, motion, music.
- Assembly Layer: editing timeline, audio mix, subtitles, export.
- Evaluation Layer: novelty check, coherence check, critic validations.

Repository Layout (recommended)

project-root/
- README.md, pyproject.toml, .env.example
- configs/
  - app.yaml, models.yaml, styles.yaml
  - prompts/*.md (director, writer, critic, scene_selector)
  - schemas/*.schema.json
- data/{library,transcripts,scenes,assets,renders,reports}
- assets/{music,fonts,overlays,templates}
- src/ with modules: app, director, research, transcription, scene_indexing, scene_selection, script, visual_planner, visual_generation, audio, editor, qc, export, utils
- scripts/{run_pipeline,ingest_movie,build_scene_index,render_video,benchmark}
- tests/
- notebooks/

Schemas (examples)

- director_plan.schema.json: structured plan including project_id, content_type, topic, thesis, hook, tone, structure (sections), visual_strategy, music_mood, length_target_sec, novelty_constraints.
- scene_card.schema.json: scene_id, title_id, start_sec, end_sec, transcript, summary, shot_count, key_frames, speaker_labels, keywords.
- script.schema.json: project_id, voiceover_text, sections (section_id + text + estimated_seconds), cta, style_notes.
- asset_plan.schema.json: assets list with asset_id, type (image/video/motion/title/overlay), prompt, linked_section_id, duration_sec, engine.
- render_job.schema.json: timeline entries with start_sec/end_sec, source_type, source_path, audio_mix and export settings.

Exact Pipeline Order (phases)

Phase 0 — Project bootstrap: create project ID, load config, model registry, initialize memory, create working dirs.

Phase 1 — Input & Research: fetch metadata, register source, check licensing/test library.

Phase 2 — Source Indexing: transcribe, detect speakers, detect scenes/shots, sample keyframes, build scene cards, store index.

Phase 3 — Director Planning: generate candidate plans, pick plan, define sections and novelty constraints.

Phase 4 — Scene Matching: convert thesis to search, compare to scene cards, rank and choose scenes.

Phase 5 — Script Generation: write narration, fit text to target length, add hooks/cta, run coherence check.

Phase 6 — Visual Planning: map sections to asset types, produce asset plan.

Phase 7 — Asset Generation: generate/extract images, motion clips, TTS, music; store outputs.

Phase 8 — Assembly: build timeline, insert clips and assets, add narration and music, subtitles, transitions, normalize audio.

Phase 9 — Quality Control: repetition check, hook strength, alignment, pacing, render integrity; regenerate if needed.

Phase 10 — Export: render final video, save report, metadata, and update memory for novelty tracking.

MVP Scope

- Support one title, one thesis, one commentary style, one scene or small group of scenes.
- Provide TTS narration, basic generated visuals, music bed, subtitles, FFmpeg assembly, and a single exported video.
- Do not start with automatic topic discovery, many styles, long multi-episode videos, or complex animations.

Milestones (recommended)

1. Metadata ingestion, transcription, scene cards.
2. Director planner and script generator.
3. Scene ranking and selection.
4. TTS, image generation, and basic rendering.
5. Assembly, subtitles, audio mixing.
6. Novelty memory and critic checks.
7. Expand to multiple formats and advanced video generation.

Testing Plan

- Use a small legal/public-domain test set.
- For each title measure: scene detection quality, transcript accuracy, scene ranking quality, script quality, asset generation success, render success, output variety.
- Key test cases: dialogue-heavy, emotional, philosophical, short action, recap-style.

GPU & Execution Considerations

- Design for Colab/Kaggle-like GPU usage: model caching, checkpointed pipeline steps, resumable jobs, modular execution, small batch ops, short rerunnable steps.
- Avoid long uninterrupted sessions; prefer restartable stages and artifact caching.

Success Criteria

The app is successful when it can take a movie/episode, choose a strong angle, find supporting scene(s), produce coherent narration, generate matching visuals, assemble a polished video, and make the next output demonstrably different from the last.

Developer Notes & Next Steps

- Start with the MVP pipeline and a small, legally safe test library.
- Implement director as a JSON-plan-producing model with novelty constraints built-in.
- Build robust scene indexing (transcripts + scene detection + scene cards).
- Integrate ComfyUI as a visual engine and FFmpeg for assembly.
- Add QC checks that trigger partial regeneration rather than full rebuilds.

One-sentence Summary

An autonomous movie-and-TV video studio that thinks like a director, understands scenes, generates narration and visuals, and assembles varied commentary videos automatically.


(End of spec)
