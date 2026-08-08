---
name: Local media pipeline
description: Durable constraint for the local profile's mock audio/video artifacts.
---

The local development profile must produce media that real downstream tools can
consume. Mock TTS output needs a valid WAV header and sample data, and rendered
video output should be created by FFmpeg rather than a placeholder file.

**Why:** Signature-only placeholder files allow early stages to appear complete
but fail when timeline assembly, ffprobe, or QC validates the final artifacts.

**How to apply:** When adding or replacing a local mock provider, validate its
output with the same tool used by the next pipeline stage before considering
the stage complete.