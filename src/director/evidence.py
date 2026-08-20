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

#: Claim ref kinds that ground on CONCRETE on-screen content (objects,
#: characters, locations, actions, events, dialogue). Moods and themes are
#: interpretation, not content — a concept grounded ONLY on those (e.g.
#: ``"calm"`` / ``"dark"`` matching a dozen scenes) must not carry a run.
CONCRETE_CLAIM_KINDS = frozenset({
    "character", "object", "location", "action", "event", "dialogue",
})


#: Editorial/craft vocabulary allowed in plan ``editorial_direction`` prose.
#: These describe HOW to cut / score / frame the essay, never claims about
#: on-screen content, so the plan auditor must not flag them as invented.
#: Also includes common neutral process/generic verbs and abstract staging
#: nouns that appear in ANY editorial prose (focusing, shifts, moments, ...)
#: regardless of the movie — only concrete content nouns get audited.
PLAN_EDITORIAL_TERMS = frozenset({
    "abstraction", "absence", "ambient", "angle", "angles", "artificial",
    "atmosphere", "audio", "beat", "build", "builds", "camera", "capture",
    "captures", "capturing", "cinematography", "close", "closeup",
    "closeups", "color", "colour", "colours", "composition", "continuity",
    "contrast", "contrasts", "create", "creates", "creating", "crossfade",
    "cut", "cuts", "dark", "depth", "dialogue", "dim", "distant", "draw",
    "draws", "drawing", "dynamic", "echo", "echoes", "edit", "editing",
    "edits", "edges", "emphasis", "emphasize", "emphasizes", "emphasizing",
    "enable", "enables", "enabling", "evoke", "evokes", "evoking", "extreme",
    "fade", "fades", "focus", "focused", "focuses", "focusing", "frame",
    "frames", "framing", "gain", "gesture", "gestures", "giving", "ground",
    "grounded", "hard", "heighten", "heightens", "hint", "hints", "hold",
    "holds", "holding", "imagery", "imply", "implies", "interplay",
    "internal", "intimate", "jump", "keep", "keeps", "keeping", "layers",
    "light", "lighting", "lights", "long", "lot", "make", "makes", "making",
    "mark", "marks", "measured", "minimal", "moment", "moments", "mood",
    "motion", "movement", "murmur", "music", "narration", "natural",
    "offscreen", "off-screen", "pace", "pacing", "palette", "panel",
    "panels", "parallel", "parallels", "pauses", "point", "points",
    "positioning", "punctuated", "quiet", "reflect", "reflects",
    "reflecting", "resonance", "resonates", "resonate", "reveal", "reveals",
    "revealing", "rhythm", "root", "roots", "score", "shadow", "shadowing",
    "shadows", "sharp", "shift", "shifts", "shifting", "shot", "shots",
    "show", "shows", "showing", "signal", "signals", "silence", "slow",
    "slower", "slowly", "soft", "sound", "sparse", "static", "steady",
    "still", "stillness", "subdued", "subtle", "suggest", "suggests",
    "suggesting", "takes", "tap", "tempo", "texture", "timing", "tone",
    "tones", "transition", "transitions", "turn", "turns", "turning",
    "underscore", "underscores", "underscoring", "unfolds", "unfolding",
    "use", "uses", "using", "vast", "voice", "weave", "weaves", "wide",
    "widescreen", "zoom",
})

#: Plural / verb-form suffixes stripped when classifying plan prose tokens so
#: "chairs" is checked as "chair", "throwing" as "throw", etc. Ordered so the
#: LEAST destructive removal (just "s") is tried first and kept when it yields
#: a known form; longer suffixes are fallbacks.
_PLAN_INFLECTION = ("s", "es", "ing", "ed")


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
        # Concrete-only facts per scene (objects / characters / locations /
        # actions / visual_events / dialogue / visual_description / transcript
        # — NOT mood / theme / emotional_cues / cinematography). Used so a
        # concept grounded ONLY on abstract moods/themes cannot carry a run.
        self._scene_concrete_text = {
            sf.scene_id: self._concrete_fact_text(sf) for sf in scene_facts
        }
        self._scene_concrete_tokens = {
            sid: set(_tokenize(txt))
            for sid, txt in self._scene_concrete_text.items()
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

    @staticmethod
    def _concrete_fact_text(sf: "SceneFact") -> str:
        """Facts that name on-screen CONTENT (not abstract moods/themes)."""
        parts = [
            sf.transcript,
            sf.dialogue_text,
            sf.location or "",
            " ".join(sf.characters),
            " ".join(sf.actions),
            " ".join(sf.objects),
            " ".join(sf.visual_events),
            sf.visual_description or "",
        ]
        return " ".join(parts)

    def _claim_is_concrete(self, ref: Dict[str, Any], scenes: List[str]) -> bool:
        """True when a matched claim grounds on concrete on-screen content.

        A ref of a concrete kind (object/character/location/action/event/
        dialogue) is concrete by definition. A free-form text ref is concrete
        only if every significant token of its value literally appears in the
        CONCRETE facts of one of its matched scenes (so ``"calm"`` tag as a
        mood does NOT count even if a mood field contains it).
        """
        kind = ref.get("kind", "text")
        if kind in CONCRETE_CLAIM_KINDS:
            return True
        if kind != "text":
            return False
        tokens = significant_tokens(ref.get("value") or "")
        if not tokens:
            return False
        for sid in scenes:
            conc = self._scene_concrete_tokens.get(sid, set())
            if all(t in conc for t in tokens):
                return True
        return False

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

        # Claim refs are everything EXCEPT scene refs. A concept is only
        # genuinely grounded if its claims (characters, objects, actions,
        # dialogue, moods, themes...) resolve to real scenes — matching a bare
        # scene id proves nothing about the claims themselves.
        scene_refs = [r for r in refs if r.get("kind") == "scene"]
        claim_refs = [r for r in refs if r.get("kind") != "scene"]
        claim_matched = [
            r for r in matched_refs if r.get("kind") != "scene"
        ]
        claim_missing = [
            r for r in missing_refs if r.get("kind") != "scene"
        ]
        claim_total = max(1, len(claim_refs))
        claim_ratio = len(claim_matched) / claim_total if claim_refs else 0.0
        claim_coverage = self._coverage_label(claim_ratio, has_claims=bool(claim_refs))

        # Concrete grounded claims — on-screen content, not mere moods/themes.
        concrete_matched = [
            r for r in claim_matched if self._claim_is_concrete(r, r.get("matched_scenes", []))
        ]

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
            # Claim (non-scene) grounding — the strict gate input.
            "scene_refs": scene_refs,
            "claim_refs": claim_refs,
            "claim_matched_refs": claim_matched,
            "claim_missing_refs": claim_missing,
            "claim_matched": len(claim_matched),
            "claim_ratio": round(claim_ratio, 2),
            "claim_coverage": claim_coverage,
            "concrete_matched_refs": concrete_matched,
            "concrete_matched": len(concrete_matched),
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
        require_concrete: bool = True,
    ) -> bool:
        """A concept is admissible only if its CLAIM refs (everything except
        scene ids) resolve to real scenes at or above ``min_coverage``.

        Bare scene-id refs prove nothing about the concept's claims, so they do
        not count toward admissibility. A concept also needs at least one
        matched scene to build a plan against. When ``require_concrete`` is
        true (the milestone default) at least ONE matched claim must ground on
        concrete on-screen content (object / character / location / action /
        event / dialogue) — a concept backed only by moods/themes cannot carry
        a run.
        """
        ev = self.concept_evidence(concept, required_evidence=required_evidence)
        if not ev["claim_refs"]:
            return False
        if ev["claim_ratio"] < min_coverage:
            return False
        if not ev["supporting_scene_ids"]:
            return False
        if require_concrete and ev["concrete_matched"] < 1:
            return False
        return True

    # -- Plan editorial-direction grounding audit -----------------------------

    def plan_grounding(
        self,
        editorial_direction: Optional[Dict[str, Any]],
        evidence_scene_ids: Optional[List[str]] = None,
        min_coverage: float = 0.55,
    ) -> Dict[str, Any]:
        """Audit a plan's ``editorial_direction`` prose for invented content.

        The plan stage already imposes a prompt rule ("describe only the
        moments/objects/characters that appear in the evidence scenes"), but a
        weak model still writes concrete claims like "empty chairs", "open
        windows" or "the rhythm of grief". This deterministic audit token-checks
        the prose against the evidence scenes' ACTUAL fact tokens and reports:

        - ``grounded_terms``  — tokens that appear in the evidence scenes,
        - ``elsewhere_terms`` — tokens that exist in the movie but NOT in the
          evidence scenes (scope leak),
        - ``invented_terms``  — tokens found in no scene at all (hallucination),
        - ``coverage``        — grounded / (grounded + invented + elsewhere),
        - ``sufficient``      — coverage >= ``min_coverage`` AND no invented
          terms that are plainly unsupported.

        ``min_coverage`` is a soft advisory threshold: the gate that decides
        whether a plan is acceptable lives in the caller (bounded regeneration
        with per-term feedback), so a single prose word never silently fails a
        whole plan — it is surfaced and corrected.
        """
        ed = editorial_direction or {}
        blob = " ".join(
            str(v) for v in ed.values() if isinstance(v, str) and v.strip()
        )
        sig = significant_tokens(blob)
        seen: List[str] = []
        grounded: List[str] = []
        elsewhere: List[str] = []
        invented: List[str] = []

        def _stem(token: str) -> str:
            for suffix in _PLAN_INFLECTION:
                if (
                    token.endswith(suffix)
                    and len(token) > len(suffix) + 2
                ):
                    return token[: -len(suffix)]
            return token

        evidence_ids = [
            sid for sid in self.facts.used_scene_ids()
            if not evidence_scene_ids or sid in set(evidence_scene_ids or [])
        ]
        evidence_tokens = set()
        for sid in evidence_ids:
            evidence_tokens |= self._scene_tokens.get(sid, set())
        movie_tokens = set()
        for sid in self.facts.used_scene_ids():
            movie_tokens |= self._scene_tokens.get(sid, set())
        evidence_stems = {_stem(t) for t in evidence_tokens}
        movie_stems = {_stem(t) for t in movie_tokens}

        for tok in sig:
            if tok in seen:
                continue
            seen.append(tok)
            stem = _stem(tok)
            if stem in PLAN_EDITORIAL_TERMS or tok in PLAN_EDITORIAL_TERMS:
                continue
            if (
                tok in evidence_tokens
                or stem in evidence_tokens
                or stem in evidence_stems
            ):
                grounded.append(tok)
            elif (
                tok in movie_tokens
                or stem in movie_tokens
                or stem in movie_stems
            ):
                elsewhere.append(tok)
            else:
                invented.append(tok)

        denominator = max(1, len(grounded) + len(invented) + len(elsewhere))
        coverage = round(len(grounded) / denominator, 3)
        sufficient = (
            coverage >= min_coverage
            and len(invented) == 0
        )
        return {
            "grounded_terms": grounded,
            "elsewhere_terms": elsewhere,
            "invented_terms": invented,
            "coverage": round(coverage, 3),
            "min_coverage": min_coverage,
            "sufficient": bool(sufficient),
        }

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
                     f"({ev['matched_claims']}/{max(1, len(ev['requested_refs']))} refs matched); "
                     f"claim coverage: {ev['claim_coverage']} "
                     f"({ev['claim_matched']}/{max(1, len(ev['claim_refs']))})")
        if ev["missing_refs"]:
            lines.append("Missing evidence (NOT in the movie):")
            for ref in ev["missing_refs"]:
                lines.append(f"- {render_ref(ref)}")
        return "\n".join(lines)

    # -- Structured failed-reference feedback ------------------------------

    def ref_feedback(
        self, concept: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Structured per-ref feedback so corrections are deterministic.

        For EVERY requested ref returns a record: ``{"kind", "value", "found",
        "scenes", "suggestions"}`` where ``suggestions`` for a NOT-FOUND ref
        lists verbatim identifiers of the same ``kind`` from the movie's actual
        vocabulary (a scene id, an on-screen object, an action, a location, a
        character, a theme, a mood, or dialogue lines). The corrective prompt
        renders this so the model can swap a hallucinated value for a real one
        instead of guessing again.
        """
        if not concept:
            return []
        ev = self.concept_evidence(concept)
        out: List[Dict[str, Any]] = []
        for ref in ev["requested_refs"]:
            kind = str(ref.get("kind") or "text")
            value = ref.get("value") or ref.get("scene_id") or ""
            matches = ev.get("matched_refs") or []
            matches = [
                m for m in matches
                if m.get("kind") == ref.get("kind")
                and m.get("matched_scenes")
            ]
            if ref.get("kind") == "scene":
                matched = next(
                    (
                        m for m in matches
                        if (m.get("scene_id") or m.get("value"))
                        == (ref.get("scene_id") or ref.get("value"))
                    ),
                    None,
                )
            else:
                matched = next(
                    (
                        m for m in matches
                        if m.get("value") == ref.get("value")
                    ),
                    None,
                )
            record: Dict[str, Any] = {
                "kind": kind,
                "value": value,
                "found": matched is not None,
                "scenes": list(matched["matched_scenes"]) if matched else [],
                "suggestions": [],
            }
            if not matched:
                record["suggestions"] = self._verbatim_suggestions(kind)
            out.append(record)
        return out

    def _verbatim_suggestions(self, kind: str, limit: int = 8) -> List[str]:
        """Verbatim identifiers of the requested kind from the movie's facts."""
        kind = str(kind).lower()
        pools = {
            "scene": self.facts.used_scene_ids(),
            "object": self.facts.known_objects(),
            "character": self.facts.known_characters(),
            "location": self.facts.known_locations(),
            "action": self.facts.known_actions(),
            "theme": self.facts.known_themes(),
            "mood": self.facts.known_moods(),
            "dialogue": self.facts.known_dialogue(),
        }
        pool = pools.get(kind)
        # free-form "text" claims carry no kind of their own; offer the movie's
        # concrete on-screen vocabulary (objects/characters/locations/actions)
        # so the model can cite something real instead of guessing again.
        if pool is None and kind in ("text", "claim", ""):
            pool = (
                self.facts.known_objects()
                + self.facts.known_characters()
                + self.facts.known_locations()
                + self.facts.known_actions()
            )
        pool = pool or []
        # Prefer short, concrete items so the model picks a citable identifier.
        pool = [str(p).strip() for p in pool if str(p).strip()]
        pool.sort(key=lambda p: (len(p.split()), p))
        return pool[:limit]

    # -- Deterministic ref derivation (the model is not trusted here) -------

    def derive_refs(
        self,
        concept: Dict[str, Any],
        max_refs: int = 6,
        prefer_concrete: bool = True,
    ) -> List[Dict[str, Any]]:
        """Synthesize ``evidence_refs`` deterministically from the concept's own
        prose (title / hook / thesis / why_interesting / visual_opportunity) —
        never from the model's declared refs.

        This is the core anti-hallucination fix: a weak generator writes prose
        like "the child walks into the forest" and *claims* refs that don't
        exist. Derivation instead scans the prose for the movie's ACTUAL known
        vocabulary (objects, locations, characters, actions, themes, moods,
        dialogue, scene ids) and emits only verbatim, groundable refs. If the
        prose mentions an invented noun (kitchen, notebook, father...), no ref
        is emitted for it because it is not in the vocabulary.

        If the prose does not name a scene id but matched some concrete
        vocabulary, a scene ref for the strongest supporting scene is appended
        so the concept always has an anchor for the plan.
        """
        if not concept:
            return []
        prose_raw = " ".join(
            str(concept.get(k) or "")
            for k in ("title", "hook", "thesis", "why_interesting",
                      "visual_opportunity")
        )
        prose_lower = prose_raw.lower()
        prose = _tokenize(prose_lower)
        # Significant (content) tokens only — stopwords like "in"/"with" inside
        # a vocabulary phrase ("man IN plaid shirt", "counter WITH various
        # items") must NEVER match generic prose that merely contains the same
        # function word. Without this, every concept derives the same broad
        # refs and the gate passes theses about objects absent from the movie.
        prose_set = set(prose)
        prose_sig_set = set(significant_tokens(prose_raw))

        refs: List[Dict[str, Any]] = []
        seen = set()

        def _add_ref(kind: str, value: str, scenes_ok: bool = True) -> None:
            if scenes_ok and not self._match_ref({
                    "kind": kind, "value": value}):
                return  # never emit a claim the movie cannot actually support
            value = str(value).strip()
            if not value:
                return
            key = (kind, value.lower())
            if key in seen:
                return
            seen.add(key)
            refs.append({"kind": kind, "value": value})

        # 1. Scene ids named in the prose (e.g. "scene-1" / "scene 3").
        #    Must match on the word-boundary numeric form, NOT a bare
        #    substring ("scene3" must not match inside "scene30").
        scene_number_re = re.compile(r"\bscene[\s\-_]*(\d+)\b")
        mentioned_scene_nums = set(scene_number_re.findall(prose_lower))
        for sid in self.facts.used_scene_ids():
            num = _scene_id_norm(sid).lstrip("scene")
            if num in mentioned_scene_nums or sid.lower() in prose_set:
                refs.append({"kind": "scene", "scene_id": sid})
                seen.add(("scene", sid))

        # 2. Concrete vocabulary the prose actually mentions: objects,
        #    locations, characters, actions (checked first when preferred).
        concrete = [
            ("object", self.facts.known_objects()),
            ("location", self.facts.known_locations()),
            ("character", self.facts.known_characters()),
            ("action", self.facts.known_actions()),
        ]
        abstract = [
            ("theme", self.facts.known_themes()),
            ("mood", self.facts.known_moods()),
            ("dialogue", self.facts.known_dialogue()),
        ]
        order = concrete + abstract
        for kind, values in order:
            if len(refs) >= max_refs:
                break
            for value in values:
                if len(refs) >= max_refs:
                    break
                tokens = significant_tokens(str(value))
                if not tokens:
                    continue
                # Match on ANY significant (content) token so prose that says
                # "saloon" derives the canonical location "saloon, dim light".
                # Stopwords are excluded: a vocab phrase like "man IN plaid
                # shirt" must not fire on prose that merely contains "in".
                # The ref is only emitted after verifying the FULL value
                # grounds.
                if any(t in prose_sig_set for t in tokens):
                    _add_ref(kind, value)

        # 3. Ensure at least one scene ref when concrete claims matched.
        if not any(r.get("kind") == "scene" for r in refs):
            supporting: List[str] = []
            for r in refs:
                for sid in self._match_ref(r):
                    if sid not in supporting:
                        supporting.append(sid)
            if supporting:
                # Prefer the scene supporting the most claims.
                counts: Dict[str, int] = {}
                for r in refs:
                    for sid in self._match_ref(r):
                        counts[sid] = counts.get(sid, 0) + 1
                if counts:
                    top = max(counts, key=counts.get)
                    refs.append({"kind": "scene", "scene_id": top})

        return refs[:max_refs]

    def is_sufficient_refs(
        self,
        ev: Dict[str, Any],
        min_coverage: float = 0.4,
        require_concrete: bool = True,
    ) -> bool:
        """Admissibility test on a PRE-computed ``concept_evidence`` dict
        (from derived refs) instead of a raw concept — the gate runs on the
        deterministic refs, never on the model's declared ones."""
        if not ev.get("claim_refs"):
            return False
        if ev["claim_ratio"] < min_coverage:
            return False
        if not ev["supporting_scene_ids"]:
            return False
        if require_concrete and ev["concrete_matched"] < 1:
            return False
        return True
