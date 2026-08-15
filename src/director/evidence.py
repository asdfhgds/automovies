"""Evidence analysis: grounds a concept's claims to the actual scene index.

The Director must not trust its own inventions. For every concept we ask "what
scenes / characters / visual patterns / dialogue would I need?" and then check,
lexically, whether the movie intelligence actually contains that material. This
module is entirely deterministic (no LLM, no new retrieval layer): it matches
evidence phrases against the on-screen facts already in ``SceneFacts``.

It is the source of truth for:

- ``evidence_preview`` per concept (which scenes support it, visual
  opportunities, an overall coverage level),
- the final plan's ``evidence_strategy`` (scene_ids, required_moments,
  character_focus, visual_motifs),
- deciding whether a concept should be *rejected* for lack of evidence.
"""
import re
from typing import Dict, Any, List, Optional

from director.scene_facts import SceneFacts

COVERAGE_HIGH = "HIGH"
COVERAGE_MEDIUM = "MED"
COVERAGE_LOW = "LOW"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


class EvidenceAnalyzer:
    """Matches concept evidence asks against actual scene facts."""

    def __init__(self, scene_facts: SceneFacts):
        self.facts = scene_facts
        # Precompute per-scene fact text + token sets once.
        self._scene_text = {sf.scene_id: sf.fact_text().lower() for sf in scene_facts}
        self._scene_tokens = {
            sid: set(_tokenize(txt)) for sid, txt in self._scene_text.items()
        }

    # -- Matching primitives -----------------------------------------------

    def _scene_matches_phrase(self, scene_id: str, phrase: str) -> bool:
        """Does a scene's facts contain ``phrase`` (as a whole phrase or tokens)?"""
        phrase = (phrase or "").strip()
        if not phrase:
            return False
        text = self._scene_text.get(scene_id, "")
        if phrase.lower() in text:
            return True
        # All significant tokens present?
        tokens = [t for t in _tokenize(phrase) if len(t) > 1]
        if not tokens:
            return False
        scene_tokens = self._scene_tokens.get(scene_id, set())
        return all(t in scene_tokens for t in tokens)

    def find_scenes(self, phrase: str, k: Optional[int] = None) -> List[str]:
        """Scene ids whose facts contain ``phrase``, preserving index order."""
        hits = [sid for sid in self.facts.used_scene_ids()
                if self._scene_matches_phrase(sid, phrase)]
        return hits[:k] if k is not None else hits

    # -- Required-evidence coverage for a concept ---------------------------

    def concept_evidence(
        self,
        concept: Dict[str, Any],
        required_evidence: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute evidence availability for a concept.

        Returns a dict with ``required_evidence``, per-claim matched scenes,
        coverage ratio (0..1) and a HIGH / MED / LOW label.
        """
        required = required_evidence
        if required is None:
            required = concept.get("required_evidence") or []
        if isinstance(required, str):
            required = [required]
        required = [str(r).strip() for r in required if str(r).strip()]

        claim_scenes = []
        unmatched = []
        ratio_num = 0
        for claim in required:
            scenes = self.find_scenes(claim)
            if scenes:
                ratio_num += 1
                claim_scenes.append({"claim": claim, "scenes": scenes})
            else:
                unmatched.append(claim)

        total = max(1, len(required))
        ratio = ratio_num / total if required else 0.0
        coverage = self._coverage_label(ratio, has_claims=bool(required))

        # Also mine character/object focus from the claim text.
        character_focus = [
            c for c in self.facts.known_characters()
            if any(self._scene_matches_phrase(sid, c) for sid in self._all_sids())
        ]
        visual_motifs = self._visual_motifs(concept)

        return {
            "required_evidence": required,
            "claim_scenes": claim_scenes,
            "unmatched_claims": unmatched,
            "matched_claims": ratio_num,
            "coverage_ratio": round(ratio, 2),
            "coverage": coverage,
            "character_focus": character_focus,
            "visual_motifs": visual_motifs,
            "supporting_scene_ids": self._union(claim_scenes),
        }

    def _all_sids(self) -> List[str]:
        return self.facts.used_scene_ids()

    @staticmethod
    def _union(claim_scenes: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for item in claim_scenes:
            for sid in item["scenes"]:
                if sid not in out:
                    out.append(sid)
        return out

    @staticmethod
    def _coverage_label(ratio: float, has_claims: bool) -> str:
        if not has_claims:
            return COVERAGE_LOW
        if ratio >= 0.7:
            return COVERAGE_HIGH
        if ratio >= 0.4:
            return COVERAGE_MEDIUM
        return COVERAGE_LOW

    def _visual_motifs(self, concept: Dict[str, Any]) -> List[str]:
        """Derive visual motifs from grounded objects/locations + visual text."""
        stop = {
            "the", "a", "an", "of", "to", "in", "and", "or", "for", "on", "is",
            "use", "using", "used", "with", "from", "as", "by", "that", "this",
            "when", "how", "what", "are", "be", "it", "you", "your",
        }
        seen, out = set(), []

        def _add(token: str):
            tok = str(token).strip().lower()
            if not tok or tok in stop or tok in seen:
                return
            seen.add(tok)
            out.append(tok)

        # Grounded objects/locations that the concept actually references.
        blob = " ".join([
            str(concept.get("thesis") or ""),
            str(concept.get("visual_opportunity") or ""),
            " ".join(str(e) for e in (concept.get("required_evidence") or [])),
        ]).lower()
        for obj in self.facts.known_objects():
            if obj.lower() in blob:
                for token in _tokenize(obj):
                    _add(token)
        for loc in self.facts.known_locations():
            if loc.lower() in blob:
                for token in _tokenize(loc):
                    _add(token)

        # Significant tokens from the visual-opportunity text.
        text = str(concept.get("visual_opportunity") or "").lower()
        for tok in _tokenize(text):
            if len(tok) >= 3:
                _add(tok)
        return out[:8]

    # -- Rejection gate -------------------------------------------------------

    def is_sufficient(
        self,
        concept: Dict[str, Any],
        min_coverage: float = 0.4,
        required_evidence: Optional[List[str]] = None,
    ) -> bool:
        """A concept is admissible only if enough of its evidence is real."""
        ev = self.concept_evidence(concept, required_evidence=required_evidence)
        if not ev["required_evidence"]:
            return False
        return ev["coverage_ratio"] >= min_coverage

    # -- Plan evidence strategy ----------------------------------------------

    def build_evidence_strategy(
        self, concept: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the final plan's grounded ``evidence_strategy``."""
        ev = self.concept_evidence(concept)
        supporting = ev["supporting_scene_ids"]
        scenes = [
            self.facts.by_id(sid) for sid in supporting if self.facts.by_id(sid)
        ]
        required_moments = []
        for s in scenes:
            moments = (s.visual_events or [])[:2]
            for m in moments:
                required_moments.append(f"{s.scene_id}: {m}")
        return {
            "scene_ids": supporting,
            "required_moments": required_moments,
            "character_focus": ev["character_focus"],
            "visual_motifs": ev["visual_motifs"],
            "evidence_coverage": ev["coverage"],
        }

    def evidence_preview_md(self, concept: Dict[str, Any]) -> str:
        """A compact evidence preview (markdown) for the reasoning report."""
        ev = self.concept_evidence(concept)
        lines = [
            f"Thesis: {concept.get('thesis', '')}",
            "",
            "Potential supporting scenes:",
        ]
        if ev["supporting_scene_ids"]:
            for sid in ev["supporting_scene_ids"]:
                lines.append(f"- {sid}")
        else:
            lines.append("- (none — no grounded evidence found)")
        lines.append("")
        lines.append("Visual opportunities:")
        motifs = ev["visual_motifs"] or [
            m.strip() for m in str(concept.get("visual_opportunity") or "").split(",")
            if m.strip()
        ]
        if motifs:
            for m in motifs[:5]:
                lines.append(f"- {m}")
        else:
            lines.append("- (not specified)")
        lines.append("")
        lines.append(f"Evidence coverage: {ev['coverage']} "
                     f"({ev['matched_claims']}/{max(1, len(ev['required_evidence']))} claims matched)")
        if ev["unmatched_claims"]:
            lines.append("Unmatched claims (NOT in the movie):")
            for claim in ev["unmatched_claims"]:
                lines.append(f"- {claim}")
        return "\n".join(lines)
