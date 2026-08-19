"""Inspectable director reasoning report.

Writes ``reports/director_reasoning.md`` so a human can see, for every candidate
concept, its thesis, the evidence the movie actually contains, strengths and
weaknesses, its feasibility score, and — critically — WHY the selected concept
won. Nothing is hidden behind the model's answer.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional

from director.evidence import EvidenceAnalyzer
from director.concepts import render_ref


def _line(label, value, indent="  "):
    return f"{indent}{label}: {value}"


def _format_score(score) -> str:
    if score is None:
        return "—"
    return f"{float(score):.2f}"


def build_report(
    movie_title: str,
    concepts: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    selected: Optional[Dict[str, Any]],
    selected_index: Optional[int],
    analyzer: EvidenceAnalyzer,
    plan: Optional[Dict[str, Any]] = None,
    diversity_metric: float = 0.0,
) -> str:
    """Render the full director reasoning report."""
    lines = [
        "# Director Reasoning Report",
        "",
        f"- **Movie**: {movie_title}",
        f"- **Concepts generated**: {len(concepts)}",
        f"- **Concepts rejected (no evidence)**: {len(rejected)}",
        f"- **Diversity metric (0..1)**: {diversity_metric:.3f}",
        "",
    ]

    # Candidate sections with index labels A/B/C...
    labels = "ABCDEFGH"
    for i, concept in enumerate(concepts):
        tag = labels[i] if i < len(labels) else str(i + 1)
        lines += _candidate_section(tag, concept, analyzer)

    if rejected:
        lines += [
            "## Rejected Concepts (insufficient evidence)",
            "",
        ]
        for j, concept in enumerate(rejected, 1):
            ev = analyzer.concept_evidence(concept)
            lines += [
                f"### Rejected {j}. {concept.get('title', '?')}",
                _line("Thesis", concept.get("thesis", "")),
                _line("Claim coverage", f"{ev['claim_coverage']} "
                      f"({ev['claim_matched']}/{max(1, len(ev['claim_refs']))} claim refs matched)"),
                "",
            ]
            if ev["claim_missing_refs"]:
                lines.append("  Ungrounded claims (NOT in the movie):")
                for ref in ev["claim_missing_refs"]:
                    lines.append(f"    - {render_ref(ref)}")
                lines.append("")

    if selected is not None:
        lines += [
            "## SELECTED CONCEPT",
            "",
            _line("Title", selected.get("title", "")),
            _line("Thesis", selected.get("thesis", "")),
            _line("Why selected", _selection_reason(selected, concepts, selected_index)),
            "",
        ]
        if plan:
            ev = selected.get("_evidence")
            if ev and ev.get("supporting_scene_ids"):
                lines.append(_line("Supporting scenes", ", ".join(ev["supporting_scene_ids"])))
            motifs = (ev or {}).get("visual_motifs") or []
            if motifs:
                lines.append(_line("Visual opportunities", ", ".join(motifs)))
            ed = plan.get("editorial_direction") or {}
            lines.append("")
            for k in ("pacing", "visual_style", "audio_style", "editing_style"):
                if ed.get(k):
                    lines.append(_line(k.replace("_", " ").title(), ed.get(k)))

    return "\n".join(lines)


def _candidate_section(tag: str, concept: Dict[str, Any], analyzer: EvidenceAnalyzer):
    preview = analyzer.evidence_preview_md(concept)
    critique = concept.get("critique") or {}
    score = critique.get("overall")
    strengths = [
        d for d, v in critique.items()
        if isinstance(v, (int, float)) and v >= 0.7 and d not in ("overall",)
    ]
    weaknesses = [
        d for d, v in critique.items()
        if isinstance(v, (int, float)) and v <= 0.4 and d not in ("overall",)
    ]
    section = [
        f"## Candidate {tag}: {concept.get('title', '?')}",
        "",
    ]
    section += [f"  {ln}" if ln else "" for ln in preview.splitlines()]
    section += [
        "",
        f"  Feasibility score: {_format_score(score)}",
        f"  Strengths: {', '.join(strengths) or '—'}",
        f"  Weaknesses: {', '.join(weaknesses) or '—'}",
        "",
    ]
    return section


def _selection_reason(selected, concepts, selected_index):
    ev = selected.get("_evidence") or {}
    parts = []
    if ev.get("claim_coverage"):
        parts.append(f"claim evidence coverage {ev['claim_coverage']}")
    score = (selected.get("critique") or {}).get("overall")
    if score is not None:
        parts.append(f"overall feasibility {float(score):.2f}")
    return ("Selected as the strongest grounded concept (" + ", ".join(parts) + ")"
            if parts else "Selected as the strongest grounded concept.")


def write_report(project_dir: Path, text: str, filename: str = "director_reasoning.md") -> Path:
    project_dir = Path(project_dir)
    report_dir = project_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / filename
    path.write_text(text, encoding="utf-8")
    return path
