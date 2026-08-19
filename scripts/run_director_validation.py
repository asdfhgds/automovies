"""Real Qwen Creative Director validation harness (gated).

Runs the Movie-grounded Creative Director against the persisted Movie
Intelligence for a project and writes an inspectable reasoning report.

This intentionally uses the REAL Qwen provider (it must — the milestone is to
validate the director grounds its concepts in the real movie). If the Qwen
provider cannot be created it exits non-zero rather than silently using mocks,
matching the strict-mode philosophy of ``evaluate_retrieval.py --method
embedding``.

Usage::

    python scripts/run_director_validation.py --project data/<project-id>

Writes:
- ``<project>/reports/director_reasoning.md`` — human-inspectable candidate /
  rejected / selected reasoning
- ``<project>/reports/director_validation.json`` — machine-readable result

Env (mirrors the provider factory): DIRECTOR_MODEL, DIRECTOR_DEVICE,
DIRECTOR_DTYPE, DIRECTOR_TEMPERATURE, REQUIRE_REAL_LLM.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from director.scene_facts import SceneFacts  # noqa: E402
from director.grounded import MovieGroundedDirector  # noqa: E402
from director.report import build_report, write_report  # noqa: E402
from director.evidence import EvidenceAnalyzer  # noqa: E402

NUM_CONCEPTS = 5
MIN_COVERAGE = 0.4
DURATION_SEC = 90

GATED_PROMPT = (
    "Create an original 60-120 second movie-analysis concept based only on what "
    "is actually present in this movie."
)


def load_movie_index(project_dir: Path):
    def read(name):
        p = Path(project_dir) / name
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    idx = read("movie_index.json") or read("movie_memory/movie_index.json")
    if not idx:
        raise FileNotFoundError(f"no movie_index.json in {project_dir}")
    return idx


def build_qwen_provider(config: dict = None):
    """Create a real QwenProvider with env-driven settings."""
    import os

    from director.providers.qwen import QwenProvider

    cfg = config or {}
    model = cfg.get("model") or os.getenv("DIRECTOR_MODEL") or "Qwen/Qwen3-4B-Instruct-2507"
    device = cfg.get("device") or os.getenv("DIRECTOR_DEVICE") or "auto"
    dtype = cfg.get("dtype") or os.getenv("DIRECTOR_DTYPE") or "auto"
    try:
        temperature = float(os.getenv("DIRECTOR_TEMPERATURE", "0.8"))
    except ValueError:
        temperature = 0.8

    strict = os.getenv("REQUIRE_REAL_LLM", "false").lower() == "true"
    if strict and device in ("auto", "cuda"):
        from utils.strict import require_cuda
        require_cuda()
        device = "cuda"

    return QwenProvider(model=model, device=device, dtype=dtype,
                        temperature=temperature)


def _gpu_info() -> Dict[str, Any]:
    """Honest GPU measurements. Returns {"name", "vram_total_gb"} when CUDA is
    visible, otherwise marks them unmeasured — never fabricated."""
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "cuda unavailable"}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "vram_total_gb": round(props.total_memory / 1e9, 2),
        "vram_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
    }


def _timed_sec(started: float) -> float:
    return round(__import__("time").monotonic() - started, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True,
                        help="project directory (data/<project-id>)")
    parser.add_argument("--num-concepts", type=int, default=NUM_CONCEPTS)
    parser.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    parser.add_argument("--duration-sec", type=int, default=DURATION_SEC)
    args = parser.parse_args()

    project_dir = Path(args.project)
    idx = load_movie_index(project_dir)
    metadata = {
        "title": idx.get("movie", {}).get("title", "Unknown"),
        "duration_sec": idx.get("movie", {}).get("duration_sec", 0),
        "source": idx.get("source_path"),
    }
    facts = SceneFacts.from_movie_intelligence(movie_index=idx)
    if not len(facts):
        print(f"No scenes loaded from {project_dir}", file=sys.stderr)
        sys.exit(2)

    provider = build_qwen_provider()
    director = MovieGroundedDirector(llm=provider.generate_text)
    _run_started = __import__("time").monotonic()
    result = director.develop(
        movie_metadata=metadata,
        scale_facts=facts,
        num_concepts=args.num_concepts,
        min_coverage=args.min_coverage,
        user_topic=GATED_PROMPT,
        duration_sec=args.duration_sec,
    )
    wall_clock_sec = _timed_sec(_run_started)
    # Real (measured) provider + GPU facts. Only what is actually observed.
    gpu = _gpu_info()
    gen_times = list(getattr(provider, "generation_times", []) or [])
    runtime = {
        "gpu": gpu,
        "model": getattr(provider, "model_name", None) or provider.__dict__.get("model_name"),
        "device": getattr(provider, "device", None),
        "dtype": getattr(provider, "dtype", None),
        "model_load_time_sec": getattr(provider, "model_load_time_sec", None),
        "generation_times_sec": gen_times,
        "total_generation_time_sec": round(sum(gen_times), 2) if gen_times else None,
        "llm_calls": (result.get("llm_stats") or {}).get("llm_calls"),
        "regeneration_rounds": (result.get("llm_stats") or {}).get("regeneration_rounds"),
        "substitutes_generated": (result.get("llm_stats") or {}).get("substitutes_generated"),
        "wall_clock_sec": wall_clock_sec,
    }

    # Machine-readable output.
    out_json = {
        "project_id": idx.get("project_id"),
        "movie": result["movie"],
        "gated_prompt": GATED_PROMPT,
        "context_meta": result["context_meta"],
        "runtime": runtime,
        "generated_concepts": [
            {k: v for k, v in c.items() if not k.startswith("_") and k != "critique"}
            for c in result["generated_concepts"]
        ],
        "rejected_concepts": result["rejected_concepts"],
        "selected_concept_index": result["selected_concept_index"],
        "selected_concept": {
            k: v for k, v in (result["selected_concept"] or {}).items()
            if not k.startswith("_") and k != "critique"
        },
        "plan": result["plan"],
        "diversity_metric": result["diversity_metric"],
    }
    report_dir = project_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "director_validation.json"
    json_path.write_text(json.dumps(out_json, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    md = build_report(
        movie_title=result["movie"],
        concepts=result["generated_concepts"],
        rejected=result["rejected_concepts"],
        selected=result["selected_concept"],
        selected_index=result["selected_concept_index"],
        analyzer=EvidenceAnalyzer(facts),
        plan=result["plan"],
        diversity_metric=result["diversity_metric"],
    )
    md_path = write_report(project_dir, md)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"generated={len(result['generated_concepts'])} "
          f"rejected={len(result['rejected_concepts'])} "
          f"selected={result['selected_concept']['title'] if result['selected_concept'] else 'NONE'}"
          f" diversity={result['diversity_metric']:.3f}")
    print("runtime: "
          f"gpu={gpu.get('name') or gpu.get('reason') or 'not measured'} "
          f"model_load={runtime['model_load_time_sec']}s "
          f"total_generation={runtime['total_generation_time_sec']}s "
          f"llm_calls={runtime['llm_calls']} "
          f"regenerated={runtime['substitutes_generated']} "
          f"wall_clock={wall_clock_sec}s")


if __name__ == "__main__":
    main()
