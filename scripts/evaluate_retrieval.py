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

Usage::

    python scripts/evaluate_retrieval.py --project data/<project-id>

Optional: ``--k 5`` (top results per query), ``--queries path/to/queries.json``
to supply your own query list (a JSON array of strings).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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


def build_index(movie_index: dict) -> SemanticIndex:
    index = SemanticIndex()
    index.build(movie_index.get("scenes", []))
    return index


def run_queries(movie_index: dict, queries, k: int) -> list:
    index = build_index(movie_index)
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


def write_report(project_dir: Path, records: list) -> Path:
    report_dir = Path(project_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "retrieval_evaluation.json"
    json_path.write_text(
        json.dumps({"method": "tfidf", "queries": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Retrieval Evaluation",
        "",
        "Method: TF-IDF semantic + transcript + dialogue overlap.",
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
    args = parser.parse_args()

    queries = EVAL_QUERIES
    if args.queries:
        queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))

    movie_index, _ = load_project(Path(args.project))
    records = run_queries(movie_index, queries, args.k)
    out = write_report(Path(args.project), records)
    print(f"Wrote {out}")
    for r in records:
        print(f"  top={r['top_scene_ids']}  {r['query'][:70]}")


if __name__ == "__main__":
    main()
