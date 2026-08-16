"""Fill human assessments into the embedding retrieval eval and write a
combined TF-IDF vs embedding comparison report for project bc6384be.

Evaluation-only tooling; leaves the raw records in place and writes two
artifacts under reports/:
  - retrieval_evaluation.json  (embedding records with human_assessment filled)
  - retrieval_comparison_tfidf_vs_embedding.md
"""
import json
from pathlib import Path

PROJ = Path("data/bc6384be-47a5-4ee8-8674-7ff861472026")
REPORTS = PROJ / "reports"

# Verdicts are keyed by the eval query index (1-based) so they stay explicit.
# label: GOOD / PARTIAL / WRONG   note: concise human rationale.
VERDICTS = [
    {"label": "PARTIAL",
     "note": ("dense cosine surfaces the decision/confrontation scenes "
              "(23 sheriffs at burning car, 22 contemplating next move) instead "
              "of TF-IDF's random garage chats; but no scene truly dramatizes one "
              "character holding a choice while another controls it.")},
    {"label": "WRONG",
     "note": ("embedding ranks calm bar/garage conversation (5,18,7) first and "
              "misses the genuinely tense scenes (22/23 sheriffs at the burning "
              "car, 17 night confrontation). TF-IDF hit those two; embedding "
              "regressed here.")},
    {"label": "WRONG",
     "note": ("neither method finds the real action-over-dialogue beats (19 horse "
              "handling, 20 driving trailer, 24 riding); returns standing/sheriff "
              "scenes instead. No semantic reasoning.")},
    {"label": "WRONG",
     "note": ("no scene in this clip is a clear visually-emphasized-object beat; "
              "both methods recycle mid-list scenes. The burning car (23) is the "
              "closest object emphasis but ranks 3rd.")},
    {"label": "PARTIAL",
     "note": ("embedding surfaces the contemplation/reflection scenes (2 man with "
              "dog, 22 decision-making, 10 contemplating) -- a better thematic "
              "match than TF-IDF's garage chats, though nothing literal about "
              "fate exists.")},
    {"label": "WRONG",
     "note": ("no scene contains a behavior-contradicts-speech beat; both methods "
              "return token/similarity noise with no intent understanding.")},
    {"label": "GOOD",
     "note": ("embedding's best query: rich who-is-present-and-doing content "
              "(23 sheriffs, 22 sheriffs, 12 two people) with the single highest "
              "score in the whole eval (0.2024). TF-IDF's variant is PARTIAL at "
              "best.")},
    {"label": "PARTIAL",
     "note": ("embedding surfaces the burning-car reveal (23) at 3rd and the "
              "night confrontation (17) 2nd -- a temporal-adjacent answer; still "
              "not a true time-anchored lock, so not GOOD. TF-IDF misses the "
              "key event entirely.")},
]

# For the markdown table we also record each method's top-5 scene ids.
def read_method(name):
    j = json.loads((REPORTS / name).read_text(encoding="utf-8"))
    return j["queries"]

embed = read_method("retrieval_evaluation_embedding.json")
tfidf = read_method("retrieval_evaluation_tfidf.json")

# 1) Fill assessments into the embedding records (the canonical eval file).
out_records = []
for i, rec in enumerate(embed):
    v = VERDICTS[i]
    rec = dict(rec)
    rec["human_assessment"] = v["label"]
    rec["human_notes"] = v["note"]
    out_records.append(rec)
(REPORTS / "retrieval_evaluation.json").write_text(
    json.dumps({"method": "embedding", "queries": out_records},
               ensure_ascii=False, indent=2), encoding="utf-8")

# 2) Combined comparison markdown.
lines = [
    "# Retrieval Comparison -- TF-IDF vs Embedding (10-min western clip)",
    "",
    "Project: `bc6384be-47a5-4ee8-8674-7ff861472026` (600s, 24 scenes / 102 shots).",
    "Same enriched corpus, same 8 evaluation queries. `human_assessment` filled from",
    "inspection of the actual scene content.",
    "",
    "| # | Query | TF-IDF top-5 | Embed top-5 | Assessment (embed) |",
    "|---|-------|--------------|-------------|--------------------|",
]
for i, (e, t) in enumerate(zip(embed, tfidf)):
    v = VERDICTS[i]
    lines.append(
        f"| {i+1} | {e['query'][:70]} | "
        f"{', '.join(t['top_scene_ids'])} | "
        f"{', '.join(e['top_scene_ids'])} | "
        f"{v['label']} |"
    )
lines += [
    "",
    "## Per-query human notes",
    "",
]
for i, v in enumerate(VERDICTS, 1):
    lines.append(f"{i}. **{v['label']}** -- {v['note']}")
lines += [
    "",
    "## Score summary",
    "",
    f"- TF-IDF top-1 scores: {', '.join(format(t['scores'][0],'.4f') for t in tfidf)}",
    f"- Embed top-1 scores:  {', '.join(format(e['scores'][0],'.4f') for e in embed)}",
    "",
    "Embedding scores are 5-10x higher and it fixes the floor-scoring failure on the",
    "thematic/temporal queries (Q1, Q5, Q8) and gives the single best result (Q7, 0.20).",
    "But it is NOT uniformly better: Q2 (strongest tension) is a clean TF-IDF win --",
    "literal 'tense/serious' overlap surfaced the real confrontation scenes that",
    "embedding skipped. Q3/Q4/Q6 stay WRONG for both: no vector index alone provides",
    "the narrative reasoning those identification queries need.",
]
(REPORTS / "retrieval_comparison_tfidf_vs_embedding.md").write_text(
    "\n".join(lines), encoding="utf-8")

print("Wrote reports/retrieval_evaluation.json (embedding, assessments filled)")
print("Wrote reports/retrieval_comparison_tfidf_vs_embedding.md")
