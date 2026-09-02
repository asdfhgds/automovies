# Future Distributed Architecture

> This document describes how the current artifact-driven pipeline architecture can evolve into a fully distributed system with independent services.

---

## Vision

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Intelligence    │     │   Director       │     │   Creative       │
│  Service         │────▶│   Service        │────▶│   Service        │
│                  │     │                  │     │                  │
│ - Transcription  │     │ - Grounded Dir   │     │ - Script Gen     │
│ - Scene Detect   │     │ - Creative Dir   │     │ - Editorial      │
│ - Vision Enrich  │     │ - Evidence Gate  │     │ - Timeline       │
│ - Semantic Index │     │ - Plan Contract  │     │                  │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ARTIFACT STORE (S3 / GCS / POSIX)                │
│  project_manifest.json  |  movie_index.json  |  director_plan.json  │
│  grounded_script.json   |  editorial_plan.json | script.json        │
│  editorial_timeline.json|  tts_audio.wav       | final_render.mp4   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Service Boundaries

### 1. Intelligence Service (`automovies-intelligence`)

**Responsibility**: Transform raw video into structured movie intelligence.

**Inputs**:
- Source video file (path or stream)

**Outputs** (Artifacts):
- `transcript.json` — WhisperX word-level transcription
- `scene_index.json` — PySceneDetect shot boundaries
- `movie_index.json` — Enriched narrative scenes
- `semantic_index.json` — TF-IDF/embedding search index
- `characters.json` — Character index
- `events.json` — Event index
- `manifest.json` — Provenance metadata

**GPU Requirements**: T4 (16GB) for WhisperX + optional Qwen3-VL

**Scaling**: Stateless workers; scale horizontally per video

---

### 2. Director Service (`automovies-director`)

**Responsibility**: Generate grounded creative concepts and production plans.

**Inputs** (Artifacts):
- `movie_index.json`
- `semantic_index.json`
- `project_meta.json` (for title, duration)

**Outputs** (Artifacts):
- `director_plan.json` — Thesis, structure, scenes, editorial intent
- `grounded_script.json` — Evidence-aligned script (if grounded)
- `director_reasoning.md` — Human-readable reasoning

**GPU Requirements**: T4/A100 for Qwen 4B/7B

**Key Contract**: `director_plan.json` must satisfy:
- Structured `editorial_plan` with controlled vocabularies
- `grounding_audit` with `overall_valid: true`
- Evidence strategy mapping to real scenes

---

### 3. Creative Service (`automovies-creative`)

**Responsibility**: Transform director plan into renderable assets.

**Sub-stages**:
1. **Editorial** — `editorial_plan.json`, `editorial_timeline.json`, `script.json`
2. **Script** — `script.json` (if not from editorial)
3. **Visual Generation** — `asset_plan.json` → `generated_visuals/`
4. **TTS** — `script.json` → `tts_audio.wav`
5. **Clip Extraction** — `selected_scenes.json` → `assets/scenes/*.mp4`

**Outputs** (Artifacts):
- `editorial_plan.json`
- `editorial_timeline.json`
- `editorial_decisions.json`
- `script.json`
- `asset_plan.json`
- `generated_visuals/`
- `tts_audio.wav`, `tts_meta.json`
- `assets/scenes/*.mp4`

**GPU Requirements**: T4 for TTS (Kokoro/Chatterbox), optional for ComfyUI

---

### 4. Render Service (`automovies-renderer`)

**Responsibility**: Assemble final video from all assets.

**Inputs** (Artifacts):
- `editorial_timeline.json` + `script.json` (editorial mode)
- `assets/scenes/*.mp4` + `tts_audio.wav` (standard mode)
- `generated_visuals/` (optional)

**Outputs** (Artifacts):
- `final_render.mp4`
- `render_job.json`

**GPU Requirements**: CPU (FFmpeg); optional GPU for filters

---

### 5. QC Service (`automovies-qc`)

**Responsibility**: Validate final output and record pipeline verdict.

**Inputs** (Artifacts):
- `final_render.mp4`
- All stage manifests

**Outputs** (Artifacts):
- `pipeline_status.json` — PASS/REVISE/FAIL
- `qc_report.json`
- `provider_manifest.json`

---

## Artifact Store Interface

All services communicate **exclusively** through the artifact store.

```python
class ArtifactStore:
    def put(self, project_id: str, artifact_type: str, 
            local_path: Path, metadata: dict) -> str:
        """Upload artifact, return artifact_id"""
    
    def get(self, project_id: str, artifact_id: str) -> Path:
        """Download artifact to local cache, return local path"""
    
    def exists(self, project_id: str, artifact_id: str) -> bool:
        """Check if artifact exists"""
    
    def list(self, project_id: str, artifact_type: str = None) -> list:
        """List artifacts for project"""
    
    def get_manifest(self, project_id: str) -> ProjectManifest:
        """Load project manifest"""
    
    def update_manifest(self, project_id: str, manifest: ProjectManifest) -> None:
        """Update project manifest atomically"""
```

**Implementations**:
- `LocalArtifactStore` — POSIX filesystem (dev, single-machine)
- `S3ArtifactStore` — AWS S3 / MinIO (production)
- `GCSArtifactStore` — Google Cloud Storage (Colab/production)
- `DriveArtifactStore` — Google Drive API (Colab persistence)

---

## Service Communication Protocol

### Stage Execution Request
```json
{
  "project_id": "uuid",
  "stage": "director",
  "config": {
    "grounded": true,
    "num_concepts": 6,
    "provider": "qwen"
  },
  "input_artifacts": [
    {"artifact_id": "movie_index_abc123", "local_path": "/cache/movie_index.json"}
  ],
  "output_artifacts": [
    {"artifact_type": "director_plan", "relative_path": "director_plan.json"}
  ]
}
```

### Stage Execution Response
```json
{
  "success": true,
  "stage": "director",
  "duration_seconds": 45.2,
  "output_artifact_ids": ["director_plan_xyz789"],
  "metrics": {"provider": "qwen", "grounded": true}
}
```

---

## Orchestration Layer

A lightweight orchestrator coordinates stages:

```python
class PipelineOrchestrator:
    def __init__(self, artifact_store: ArtifactStore, 
                 service_endpoints: Dict[str, str]):
        self.store = artifact_store
        self.endpoints = service_endpoints
    
    def run_stage(self, project_id: str, stage: str, 
                  config: dict, force: bool = False) -> StageResult:
        # 1. Load manifest
        manifest = self.store.get_manifest(project_id)
        
        # 2. Check dependencies
        if not self._check_deps(manifest, stage):
            raise DependencyError(f"Missing deps for {stage}")
        
        # 3. Check skip
        if not force and self._can_skip(manifest, stage):
            return StageResult(skipped=True)
        
        # 4. Download input artifacts to local cache
        inputs = self._download_inputs(manifest, stage)
        
        # 4. Call stage service (HTTP/gRPC)
        result = self._call_service(stage, config, inputs)
        
        # 5. Upload outputs
        self._upload_outputs(project_id, result.output_artifacts)
        
        # 6. Update manifest
        self.store.update_manifest(project_id, manifest)
        
        return result
```

---

## Configuration Management

Each service reads configuration from:

1. **Project manifest** (`config` field) — project-specific
2. **Service config file** — service defaults
3. **Environment variables** — runtime overrides
4. **Stage request** — per-execution params

Priority: 4 > 3 > 1 > 2

---

## Deployment Models

### Development (Local)
```
Local filesystem artifact store
All services as local Python modules
Single GPU (T4)
```

### Colab (Interactive)
```
Google Drive artifact store
Services as local modules + GPU
Manual stage triggering via notebook
```

### Production (Kubernetes)
```
S3/MinIO artifact store
Each service as separate Deployment
HorizontalPodAutoscaler per service
GPU node pools per service type
Argo Workflows / Temporal for orchestration
```

---

## Migration Path from Current Architecture

### Phase 1: Internal Refactor (Current Milestone)
- [x] Project manifest with stage/artifact tracking
- [x] Artifact registry with validation
- [x] Stage contracts with explicit I/O
- [x] Stage runner CLI with resume/skip
- [x] Configurable artifact root (local, Drive, S3)
- [x] JSON schemas for all artifacts
- [x] All existing tests pass

### Phase 2: Service Extraction (Next Milestone)
- [ ] Extract `movie_intelligence` as standalone module with CLI
- [ ] Extract `director` as standalone module with CLI
- [ ] Extract `creative` (editorial/script/tts/visual) as module
- [ ] Extract `renderer` as module
- [ ] Extract `qc` as module
- [ ] Each module has `run(project_id, config)` entry point

### Phase 3: Service Deployment (Future)
- [ ] Containerize each module
- [ ] Deploy to Kubernetes with GPU node pools
- [ ] Set up MinIO/S3 artifact store
- [ ] Implement orchestrator (Argo/Temporal)
- [ ] Add observability (Prometheus, Grafana, Jaeger)

### Phase 4: Multi-User / SaaS (Future)
- [ ] Multi-tenant project isolation
- [ ] Auth / RBAC
- [ ] Web UI for project management
- [ ] Webhooks for stage completion
- [ ] Billing / quota management

---

## API Contracts Between Services

The JSON schemas in `schemas/` become the **immutable API contracts** between services.

| Schema | Producer | Consumers |
|--------|----------|-----------|
| `movie_index.schema.json` | Intelligence | Director, Creative |
| `director_output.schema.json` | Director | Creative, Render |
| `script.schema.json` | Creative | TTS, Render |
| `tts_manifest.schema.json` | TTS | Render |
| `render_manifest.schema.json` | Render | QC, Delivery |

**Contract Rule**: Schema changes require semantic versioning. Breaking changes = new artifact type.

---

## Security Considerations

- Artifact store: Signed URLs for upload/download
- Service-to-service: mTLS + JWT
- Project isolation: Prefix all keys with `projects/{project_id}/`
- Audit logging: All artifact access logged
- Encryption: At-rest (SSE-S3) + in-transit (TLS)

---

## Observability

Each service emits:
- Structured logs (JSON) to stdout
- Metrics: duration, GPU utilization, artifact sizes
- Traces: OpenTelemetry spans for each stage

---

## Summary

The current monolithic orchestrator is refactored into **independently deployable services** that communicate **only through versioned artifact contracts** in a shared object store. This enables:

1. **GPU specialization** — each service runs on optimal hardware
2. **Independent scaling** — scale intelligence workers separately from renderers
3. **Fault isolation** — director failure doesn't block intelligence
4. **Polyglot future** — services can be rewritten in Go/Rust
5. **Colab-to-Production parity** — same artifacts, different storage backend