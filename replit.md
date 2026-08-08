# Autonomous Movie & TV Video Studio

## Project overview

This is a Python MVP pipeline for producing commentary videos from a source
movie or TV clip. The pipeline indexes a source, creates a director plan,
selects a scene, generates deterministic narration and local mock-TTS audio,
builds a timeline, renders an MP4 with FFmpeg, and writes a QC report.

The repository keeps heavyweight providers optional. WhisperX, PySceneDetect,
Qwen, ComfyUI, and production TTS providers can be enabled later; local
fallbacks are intentionally available for development and tests.

## User preferences

- Preserve the existing modular Python architecture and provider interfaces.
- Prefer small, testable pipeline milestones over broad restructures.
- Keep local development runnable without GPU-only dependencies.
- Use the project status and roadmap documents to choose the next milestone.