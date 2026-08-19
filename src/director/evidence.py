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

Matching contract (deterministic, no LLM):

- **exact scene id first** — a ``scene`` ref must name a real scene id;
- then **canonical identifier / alias** matching against the known character,
  object and location vocabulary (see ``SceneFacts``);
- then **exact token containment** — every significant token of the ref value
  must literally appear in a scene's facts. Arbitrary substring containment is
  never used as a primary rule (so ``son`` never matches ``person``).
"""
import re
from typing import Dict, Any, List, Optional

from director.concepts import concept_refs, render_ref
from director.scene_facts import (
    SceneFacts,
    normalize_entity,
    significant_tokens,
    strip_articles,
)

COVERAGE_HIGH = "HIGH"
COVERAGE_MEDIUM = "MED"
COVERAGE_LOW = "LOW"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


#: Conservative, controllable synonym map: canonical value -> additional
#: aliases the matcher may accept for the same on-screen entity. Kept tiny and
#: explicit — never a free-text semantic matcher.
SYNONYM_ALIASES: Dict[str, List[str]] = {}


def _scene_id_norm(value: str) -> str:
    return re.sub(r"[\s\-_]+", "", (value or "").lower().strip())


class EvidenceAnalyzer:
    """Matches concept evidence asks against actual scene facts."""

    def __init__(
        self,
        scene_facts: SceneFacts,
        synonym_map: Optional[Dict[str, List[str]]] = None,
    ):
        self.facts = scene_facts
        # Precompute per-scene fact text + token sets once.
        self._scene_text = {sf.scene_id: sf.fact_text().lower() for sf in scene_facts}
        self._scene_tokens = {
            sid: set(_tokenize(txt)) for sid, txt in self._scene_text.items()
        }
        self._synonym_map = dict(SYNONYM_ALIASES)
        if synonym_map:
            self._synonym_map.update(synonym_map)
        # Alias -> scenes lookup for the canonical vocabulary (entity kinds).
        self._entity_scenes: Dict[str, Dict[str, List[str]]] = self._build_entity_index()

    # -- Canonical vocabulary index -----------------------------------------

    def _build_entity_index(self) -> Dict[str, Dict[str, List[str]]]:
        index = {}
        for kind, vocab in (
            ("character", self.facts.character_vocabulary()),
            ("object", self.facts.object_vocabulary()),
            ("location", self.facts.location_vocabulary()),
        ):
            table: Dict[str, List[str]] = {}
            for entry in vocab:
                canonical = entry["canonical"]
                aliases = list(entry["aliases"])
                for extra in self._synonym_map.get(canonical, []):
                    norm_extra = normalize_entity(strip_articles(extra))
                    if norm_extra:
                        aliases.append(norm_extra)
                for alias in set(aliases):
                    table.setdefault(alias, []).append(entry["canonical"])
            index[kind] = table
        return index

    # -- Matching primitives -------------------------------------------------

    def _scene_ids_for_entity(self, kind: str, alias: str) -> List[str]:
        """Scene ids for an exact canonical/alias hit in the vocabulary."""
        canonical = normalize_entity(strip_articles(alias))
        hits = self._entity_scenes.get(kind, {}).get(canonical, [])
        scenes = set()
        for canon in hits:
            for entry in self._scenes_by_canonical(kind, canon):
                scenes.update(entry["scenes"])
        return [sid for sid in self.facts.used_scene_ids() if sid in scenes]

    def _scenes_by_canonical(self, kind: str, canonical: str):
        vocab = {
            "character": self.facts.character_vocabulary(),
            "object": self.facts.object_vocabulary(),
            "location": self.facts.location_vocabulary(),
        }[kind]
        return [e for e in vocab if e["canonical"] == canonical]

    def _exact_scene_id(self, value: str) -> Optional[str]:
        norm = _scene_id_norm(value)
        if not norm:
            return None
        for sid in self.facts.used_scene_ids():
            if _scene_id_norm(sid) == norm:
                return sid
        return None

    def _token_scenes(self, value: str) -> List[str]:
        """Scenes whose facts contain every significant token of ``value``
        (exact token containment — no arbitrary substring)."""
        tokens = significant_tokens(value)
        if not tokens:
            return []
        return [
            sid for sid in self.facts.used_scene_ids()
            if all(t in self._scene_tokens[sid] for t in tokens)
        ]

    def _match_ref(self, ref: Dict[str, Any]) -> List[str]:
        """Resolve one evidence ref to the scene ids that support it."""
        kind = ref.get("kind", "text")
        if kind == "scene":
            sid = self._exact_scene_id(ref.get("scene_id") or ref.get("value") or "")
            return [sid] if sid else []
        value = ref.get("value") or ""
        if not value.strip():
            return []
        if kind in ("character", "object", "location"):
            scenes = self._scene_ids_for_entity(kind, value)
            if scenes:
                return scenes
        return self._token_scenes(value)

    def find_scenes(self, phrase: str, k: Optional[int] = None) -> List[str]:
        """Scene ids whose facts contain ``phrase`` (exact token containment)."""
        hits = self._token_scenes(phrase)
        return hits[:k] if k is not None else hits

    # -- Required-evidence coverage for a concept ---------------------------

    def concept_evidence(
        self,
        concept: Dict[str, Any],
        required_evidence: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute evidence availability for a concept.

        The authoritative input is ``concept["evidence_refs"]``; the legacy
        ``required_evidence`` strings are accepted as ``text`` refs when the
        structured refs are absent (or via the ``required_evidence`` argument).

        Returns both the structured ref analysis and the legacy summary keys:

        - ``requested_refs`` / ``matched_refs`` / ``missing_refs`` — per ref
        - ``matched_scenes`` (== ``supporting_scene_ids``) — union of matched scenes
        - ``required_evidence`` — one rendered line per ref
        - ``matched_claims`` / ``unmatched_claims`` / ``coverage_ratio`` /
          ``coverage`` — the HIGH / MED / LOW coverage summary
        """
        if required_evidence is None:
            refs = concept_refs(concept)
        else:
            legacy = required_evidence
            if isinstance(legacy, str):
                legacy = [legacy]
            refs = [
                {"kind": "text", "value": str(item).strip()}
                for item in (legacy or []) if str(item).strip()
            ]

        matched_refs: List[Dict[str, Any]] = []
        missing_refs: List[Dict[str, Any]] = []
        claim_scenes: List[Dict[str, Any]] = []
        scenes_seen: List[str] = []
        for ref in refs:
            scenes = self._match_ref(ref)
            record = dict(ref)
            if scenes:
                record["matched_scenes"] = scenes
                matched_refs.append(record)
                claim_scenes.append({"claim": render_ref(ref), "scenes": scenes})
                for s in scenes:
                    if s not in scenes_seen:
                        scenes_seen.append(s)
            else:
                missing_refs.append(record)

        required = [r for r in (render_ref(r) for r in refs) if r]
        total = max(1, len(refs))
        ratio = len(matched_refs) / total if refs else 0.0
        coverage = self._coverage_label(ratio, has_claims=bool(refs))

        character_focus = [
            str(r["value"]) for r in matched_refs
            if r.get("kind") == "character" and r.get("value")
        ]
        if not character_focus:
            matched_text_tokens = {
                t for sid in scenes_seen for t in self._scene_tokens.get(sid, set())
            }
            character_focus = [
                c for c in self.facts.known_characters()
                if all(t in matched_text_tokens for t in significant_tokens(c))
            ]
        visual_motifs = self._visual_motifs(concept)

        return {
            "required_evidence": required,
            "requested_refs": refs,
            "matched_refs": matched_refs,
            "missing_refs": missing_refs,
            "matched_scenes": scenes_seen,
            "claim_scenes": claim_scenes,
            "unmatched_claims": [render_ref(r) for r in missing_refs],
            "matched_claims": len(matched_refs),
            "coverage_ratio": round(ratio, 2),
            "coverage": coverage,
            "character_focus": character_focus,
            "visual_motifs": visual_motifs,
            "supporting_scene_ids": scenes_seen,
        }

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

        # Only objects/locations the concept actually references (via its refs).
        ref_values = [r for r in (render_ref(r) for r in concept_refs(concept)) if r]
        blob = " ".join([
            str(concept.get("thesis") or ""),
            str(concept.get("visual_opportunity") or ""),
            " ".join(ref_values),
        ]).lower()
        blob_tokens = set(_tokenize(blob))
        for obj in self.facts.known_objects():
            toks = significant_tokens(obj)
            if toks and set(toks).issubset(blob_tokens):
                for token in _tokenize(obj):
                    _add(token)
        for loc in self.facts.known_locations():
            toks = significant_tokens(loc)
            if toks and set(toks).issubset(blob_tokens):
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
        """A concept is admissible only if it carries evidence refs and enough
        of them resolve to real scenes."""
        ev = self.concept_evidence(concept, required_evidence=required_evidence)
        if not ev["requested_refs"]:
            return False
        if not ev["matched_refs"]:
            return False
        return ev["coverage_ratio"] >= min_coverage

    # -- Plan evidence strategy ----------------------------------------------

    def build_evidence_strategy(
        self, concept: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the final plan's grounded ``evidence_strategy``.

        Only scenes that actually matched an evidence ref are used — the
        strategy is deterministic, never model-invented.
        """
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
            "Evidence references:",
        ]

        def _ref_key(r):
            return (r.get("kind"), r.get("scene_id") or r.get("value"))

        if ev["requested_refs"]:
            for ref in ev["requested_refs"]:
                label = render_ref(ref)
                matched = next(
                    (r for r in ev["matched_refs"] if _ref_key(r) == _ref_key(ref)),
                    None,
                )
                status = f"-> {' '.join(matched['matched_scenes'])}" if matched else "NOT FOUND"
                lines.append(f"- {label} [{status}]")
        else:
            lines.append("- (none — concept carries no evidence refs)")
        lines.append("")
        lines.append("Scene focus:")
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
                     f"({ev['matched_claims']}/{max(1, len(ev['requested_refs']))} refs matched)")
        if ev["missing_refs"]:
            lines.append("Missing evidence (NOT in the movie):")
            for ref in ev["missing_refs"]:
                lines.append(f"- {render_ref(ref)}")
        return "\n".join(lines)
