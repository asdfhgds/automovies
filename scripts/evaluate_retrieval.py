"""Retrieval evaluation harness for the Movie Intelligence Layer.

Evaluation-only tooling (NOT production code — do not import from the
pipeline). Runs a fixed set of natural-language queries against the persisted
``movie_index.json`` / ``semantic_index.json`` for a project and writes:

- ``reports/retrieval_evaluation.json`` — machine-readable records
- ``reports/retrieval_evaluation.md`` — human-readable report

Each record carries ``human_assessment`` (``GOOD`` / ``PARTIAL`` / ``WRONG``)
left blank for a human reviewer to fill in after inspecting the top results,
per the validation milestone. Nothing here fakes accuracy: the automated
fields are the scene ids / timestamps / scores the index actually returned.

Two retrieval methods are supported:

- ``--method tfidf`` (default) — TF-IDF cosine + transcript/dialogue overlap.
- ``--method embedding`` — dense-embedding cosine (sentence-transformers, or a
  ``module:attr`` factory via ``--embedder`` / ``RETRIEVAL_EMBEDDER``). If the
  requested embedder cannot be created, the harness exits **non-zero** with an
  actionable message instead of silently substituting TF-IDF scores.

Usage::

    python scripts/evaluate_retrieval.py --project data/<project-id>
    python scripts/evaluate_retrieval.py --project data/<project-id> --method embedding

Optional: ``--k 5`` (top results per query), ``--queries path/to/queries.json``
to supply your own query list (a JSON array of strings), ``--embedder
module:attr`` to force a specific embedder factory.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_understanding.embedding_retriever import create_embedder_from_env  # noqa: E402
from movie_understanding.semantic_index import SemanticIndex  # noqa: E402

# Evaluation queries for this validation milestone. These deliberately require
# real understanding (who / what / where / why / when), not keyword lookup.
EVAL_QUERIES = [
    "Find a scene where one character appears to have a choice but another character controls the situation.",
    "Find the strongest moment of tension between two characters.",
    "Find a scene where the meaning comes mostly from what characters do rather than what they say.",
    "Find a moment where an important object is visually emphasized.",
    "Find a scene that demonstrates the protagonist's relationship with fate.",
    "Find a scene where a character's behavior contradicts what they say.",
    "Who is present in this scene and what are they doing?",
    "When does the most important visual event occur?",
]

METHOD_LABELS = {
    "tfidf": "TF-IDF semantic + transcript + dialogue overlap.",
    "embedding": "Dense embeddings (cosine) + transcript + dialogue overlap.",
}


def load_project(project_dir: Path):
    def read(name):
        p = Path(project_dir) / name
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    movie_index = read("movie_index.json")
    semantic = read("semantic_index.json")
    if not movie_index:
        raise FileNotFoundError(f"no movie_index.json in {project_dir}")
    return movie_index, semantic


def get_embedder(embedder_spec: str):
    """Create an embedder from ``module:attr`` or the environment.

    Raises ImportError/ValueError with an actionable message when unavailable.
    """
    if embedder_spec:
        module_name, attr_name = embedder_spec.split(":", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, attr_name)
        embedder = factory()
        if not callable(embedder):
            raise ValueError(
                f"--embedder {embedder_spec!r}: {attr_name}() did not return a "
                "callable embedder")
        return embedder
    return create_embedder_from_env()


def build_index(movie_index: dict, method: str = "tfidf", embedder=None) -> SemanticIndex:
    index = SemanticIndex()
    if method == "embedding":
        index.build(movie_index.get("scenes", []), embedder=embedder)
    else:
        index.build(movie_index.get("scenes", []))
    return index


def run_queries(movie_index: dict, queries, k: int,
                method: str = "tfidf", embedder=None) -> list:
    index = build_index(movie_index, method=method, embedder=embedder)
    scenes_by_id = {s.get("scene_id"): s for s in movie_index.get("scenes", [])}
    records = []
    for query in queries:
        hits = index.search(query, k=k)
        top = []
        for h in hits:
            sid = h["scene_id"]
            scene = scenes_by_id.get(sid) or {}
            top.append({
                "scene_id": sid,
                "score": h["score"],
                "rationale": h.get("rationale", ""),
                "start_sec": scene.get("start_sec"),
                "end_sec": scene.get("end_sec"),
            })
        records.append({
            "query": query,
            "top_scene_ids": [h["scene_id"] for h in hits],
            "timestamps": [
                {"start_sec": t["start_sec"], "end_sec": t["end_sec"]}
                for t in top
            ],
            "scores": [t["score"] for t in top],
            "model_reason": [t["rationale"] for t in top],
            "human_assessment": None,  # GOOD / PARTIAL / WRONG — filled by human
            "human_notes": "",
        })
    return records


def write_report(project_dir: Path, records: list, method: str) -> Path:
    report_dir = Path(project_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "retrieval_evaluation.json"
    json_path.write_text(
        json.dumps({"method": method, "queries": records},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Retrieval Evaluation",
        "",
        f"Method: {METHOD_LABELS.get(method, method)}",
        "",
        "Assessment scale: **GOOD** / **PARTIAL** / **WRONG** "
        "(fill in the table by hand).",
        "",
        "| # | Query | Top scenes (timestamps) | Scores | Human assessment |",
        "|---|-------|-------------------------|--------|------------------|",
    ]
    for i, r in enumerate(records, 1):
        top = ", ".join(
            f"{sid} ({t.get('start_sec')}-{t.get('end_sec')}s)"
            for sid, t in zip(r["top_scene_ids"], r["timestamps"])
        ) or "(none)"
        scores = ", ".join(str(s) for s in r["scores"]) or "(none)"
        lines.append(
            f"| {i} | {r['query'][:80]} | {top[:120]} | {scores[:40]} | _blank_ |"
        )
    md_path = report_dir / "retrieval_evaluation.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="project directory (data/<id>)")
    parser.add_argument("--k", type=int, default=5, help="top results per query")
    parser.add_argument("--queries", default=None, help="optional JSON array of queries")
    parser.add_argument("--method", choices=("tfidf", "embedding"), default="tfidf",
                        help="retrieval method (default: tfidf)")
    parser.add_argument("--embedder", default=None,
                        help="module:attr embedder factory (overrides RETRIEVAL_EMBEDDER)")
    args = parser.parse_args()

    queries = EVAL_QUERIES
    if args.queries:
        queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))

    embedder = None
    if args.method == "embedding":
        try:
            embedder = get_embedder(args.embedder)
        except (ImportError, ValueError) as exc:
            print(f"embedding method unavailable: {exc}", file=sys.stderr)
            print("Install sentence-transformers or set RETRIEVAL_EMBEDDER / "
                  "--embedder to a working module:attr factory; refusing to "
                  "silently fall back to TF-IDF.", file=sys.stderr)
            sys.exit(2)

    movie_index, _ = load_project(Path(args.project))
    records = run_queries(movie_index, queries, args.k,
                          method=args.method, embedder=embedder)
    out = write_report(Path(args.project), records, args.method)
    print(f"Wrote {out}  (method={args.method})")
    for r in records:
        print(f"  top={r['top_scene_ids']}  {r['query'][:70]}")


if __name__ == "__main__":
    main()