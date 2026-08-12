# NEXT MILESTONE

**Last Updated**: editorial pipeline GPU run ready — `notebooks/colab_editorial_gpu.ipynb`.

## Status Tags

- **PROVEN**
  - Editorial planning subsystem (`src/editorial/`) — plan → evidence-aligned script → timeline → FFmpeg render → QC.
  - Movie intelligence layer (`src/movie_understanding/`) — analyzer, scene/character/event/semantic indexing (heuristic-first).
  - Local E2E: 138 tests pass; editorial orchestrator E2E produces all expected artifacts
    (`director_plan.json`, `editorial_plan.json`, `script.json`, `movie_index.json`,
    `timeline/editorial_timeline.json`, excerpt clips, `renders/final_render.mp4`,
    `provider_manifest.json`, `reports/qc_report.json`).
  - Renderer fix: per-clip `fps=` normalization so `xfade` sees uniform timebases
    (fixes "First input link timebase … do not match … xfade timebase" on mixed-rate sources).

- **EXPERIMENTAL**
  - Evidence retrieval (semantic TF-IDF + lexical + dialogue overlap) — heuristics only, no embedder/LLM.
  - Editorial narration is deterministic around the real (Qwen) thesis — not yet an LLM editorial writer.
  - Real TTS (Kokoro priority) — quality/performance needs a real-GPU eval.

- **FAILED (baseline to beat)**
  - The first real-movie output (pre-editorial) was: clips in sequence + robotic TTS + paragraph subtitles.

- **KNOWN LIMITATIONS**
  - Scene enrichment fields that need vision/LLM (location, actions, visual_description, themes) are `None` + provenance-flagged.
  - Captions are chunked short (≤3 words) but timed by even distribution unless real word timestamps exist.
  - TTS emotion/pace control is approximate per provider (Kokoro voice+speed; not true expressive control).

- **CURRENT BOTTLECK**: movie understanding / evidence quality (semantic retrieval needs real embeddings and/or LLM/vision scene enrichment). Verified rendering is no longer the blocker.

- **NEXT ACTION**
  1. Run `notebooks/colab_editorial_gpu.ipynb` on a GPU runtime with a user-supplied movie (T4+).
  2. Watch `renders/final_render.mp4` and score idea/understanding/script/evidence/editing/TTS/subtitles/audio.
  3. Choose the next milestone from the **largest visible weakness** (Movie Understanding,
     Semantic Evidence Retrieval, Editorial Director, TTS Performance, Subtitle System,
     Audio Design, Visual Generation) — do NOT pick from a predetermined feature list.