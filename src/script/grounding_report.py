"""Human-inspectable grounding report for the generated script.

Produces ``reports/script_grounding_report.md`` showing, for every script
section, the selected concept, the thesis, the evidence requirements, and the
exact scene/evidence references + narration —— so a human can verify the script
is *using evidence* rather than repeating the director or inventing material.
"""
from pathlib import Path
from typing import Any, Dict, List


def build_grounding_report(
    contract: Dict[str, Any],
    script: Dict[str, Any],
) -> str:
    concept = contract.get("concept") or {}
    intents = contract.get("editorial_intent") or {}
    sections = script.get("sections") or []
    evidence_by_id = {e.get("id"): e.get("claim", "")
                      for e in (script.get("evidence") or [])}

    lines = [
        "# Script Grounding Report",
        "",
        f"- **Project**: {script.get('project_id', '')}",
        f"- **Generator**: {script.get('grounded', '')} "
        f"({script.get('provenance', {}).get('generator', 'grounded')})",
        "",
        "## Selected Concept",
        "",
        f"- **Title**: {concept.get('title', '')}",
        f"- **Thesis**: {concept.get('thesis', '')}",
        f"- **Hook**: {concept.get('hook', '')}",
        "",
        "## Evidence Requirements",
        "",
    ]
    for e in script.get("evidence") or []:
        lines.append(f"- `{e.get('id')}` — {e.get('claim', '')}")
    if not (script.get("evidence") or []):
        lines.append("- (none)")

    lines += [
        "",
        "## Editorial Intent",
        "",
        f"- **Pacing**: {intents.get('pacing', '')}",
        f"- **Tone**: {intents.get('tone', '')}",
        f"- **Visual style**: {intents.get('visual_style', '')}",
        f"- **Audio style**: {intents.get('audio_style', '')}",
        "",
        "## Script Sections",
        "",
    ]

    for section in sections:
        sid = section.get("id", "")
        s_type = section.get("type", "")
        narration = section.get("narration", "")
        scene_ids = section.get("scene_ids", [])
        evidence_ids = section.get("evidence_ids", [])
        narrative_evidence = section.get("narrative_evidence", [])

        ev_labels = []
        for eid in evidence_ids:
            claim = evidence_by_id.get(eid, "")
            ev_labels.append(f"{eid} ({claim})" if claim else eid)
        lines += [
            f"### {sid} ({s_type})",
            "",
            f"- **Scenes**: {', '.join(scene_ids) or '—'}",
            f"- **Evidence**: {', '.join(ev_labels) or '—'}",
            f"- **Narration**: {narration}",
            "",
        ]
        if narrative_evidence:
            lines.append("  On-screen moments:")
            for ev in narrative_evidence:
                lines.append(
                    f"  - `{ev.get('scene_id')}` {ev.get('start_sec')}–{ev.get('end_sec')}s "
                    f"— {ev.get('reason', '')}"
                )
            lines.append("")

    return "\n".join(lines)


def write_script_grounding_report(
    project_dir: Path,
    contract: Dict[str, Any],
    script: Dict[str, Any],
) -> Path:
    project_dir = Path(project_dir)
    report_dir = project_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "script_grounding_report.md"
    path.write_text(build_grounding_report(contract, script), encoding="utf-8")
    return path