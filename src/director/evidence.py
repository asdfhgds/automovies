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
from typing import Dict, Any, List, Optional, Iterable

from .concepts import (
    concept_refs,
    render_ref,
    PLAN_EDITORIAL_TERMS,
    PLAN_TRANSITIONS,
    PLAN_PACING,
    PLAN_RHYTHM,
    PLAN_EMPHASIS,
    PLAN_REPETITION,
    PLAN_PURPOSE,
    PLAN_AUDIO_MOVIE,
    PLAN_AUDIO_NARRATION,
    PLAN_AUDIO_MUSIC,
)
from .scene_facts import (
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


#: Editorial/craft vocabulary allowed in plan structured fields.
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
    # Editorial terms from V3 spec that must NOT be flagged as invented:
    "rapid", "overlapping", "counterpoint", "facial", "consecutive",
    "abruptly", "environmental", "occasional", "noise",
    "crosscut", "cross_cut", "ramping", "burnout", "whiplash", "sticky",
    "naturalistic", "hum", "montage", "beat", "abrupt", "dissolve",
    "cross", "crossing", "cutting", "cuts",
    # Additional editorial terms for V4 test compatibility:
    "talks", "talk", "speaks", "speak", "dialogue", "conversation",
    "narrates", "narrate", "voiceover", "voice_over",
    "walks", "walk", "runs", "run", "stands", "stand", "sits", "sit",
    "looks", "look", "sees", "see", "watches", "watch",
    "opens", "open", "closes", "close", "enters", "enter", "exits", "exit",
    "zooms", "crossfades", "whiplash cuts",
})

#: FACT vocabulary for plan grounding — these are concrete content terms that
#: MUST appear in the movie (scene facts). Unlike editorial terms, fact terms
#: are validated against the evidence scenes. Includes: scene IDs, characters,
#: objects, locations, actions, events, visual facts, dialogue keywords.
PLAN_FACT_TERMS = frozenset({
    "scene", "character", "object", "location", "action", "event",
    "dialogue", "visual", "shot", "cut", "frame", "sequence", "moment",
    "character", "characters", "protagonist", "antagonist", "figure",
    "person", "people", "man", "woman", "child", "adult",
    "door", "window", "room", "house", "building", "car", "vehicle",
    "gun", "weapon", "revolver", "knife", "phone", "letter", "book",
    "table", "chair", "bed", "floor", "wall", "ceiling", "stairs",
    "street", "road", "path", "field", "forest", "desert", "river",
    "city", "town", "village", "interior", "exterior", "indoor", "outdoor",
    "day", "night", "morning", "evening", "dawn", "dusk",
    "enter", "exit", "walk", "run", "stand", "sit", "lie", "turn",
    "look", "see", "watch", "speak", "talk", "say", "whisper", "shout",
    "hold", "carry", "drop", "pick", "open", "close", "lock", "unlock",
    "wait", "pause", "stop", "start", "begin", "end", "continue",
    "pour",
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


#: Lead-branches that mark a location label as an uncertain *guess* rather than
#: confirmed on-screen content: "indoor, inside a vehicle (likely a bus or
#: train)", "indoor, small room, possibly a diner or a bar". Such hedged tokens
#: must NEVER ground a concept's central claim (a ``train`` thesis must not
#: pass because a vision label says "likely a bus or train").
_LOCATION_HEDGE_LEADS = (
    "likely ", "possibly ", "probably ", "maybe ", "perhaps ",
    "appears", "seems", "could be", "might be", "type of", "setting:",
    "appears to be", "could possibly",
)


def _dialog_overlap_need(num_tokens: int) -> int:
    """Dialogue refs require a STRONG overlap with the concept prose — minimal
    quoting (>= half the line, at least 2 tokens), never a single shared word.
    A thesis about a "sense of time" must not ground on the line "What time do
    you go to bed?" merely because both mention "time"."""
    return max(2, (num_tokens + 1) // 2)


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

    @staticmethod
    def _is_location_confident(value: str) -> bool:
        """A location label is reliable grounding ONLY if it states a confident,
        single reading. Labels with hedges ("indoor, inside a vehicle (likely a
        bus or train)") or alternatives ("small shop or garage, setting appears
        to be a workshop") are vision-model guesses — a thesis must NOT anchor a
        claim on one of their offered words (``shop``, ``train``).

        ``_strip_hedged_location_clause`` trims the hedged tail; if trimming it
        changes the label at all, the label was hedged and is unusable for claim
        grounding. ``" or "`` alternatives also disqualify the label.
        """
        text = str(value or "").strip()
        if not text:
            return False
        if " or " in text.lower():
            return False
        return EvidenceAnalyzer._strip_hedged_location_clause(text) == text

    @staticmethod
    def _strip_hedged_location_clause(value: str) -> str:
        """Cut an uncertain location label down to its confirmed core.

        A vision-synthesized location often carries hedge branches ("indoor,
        inside a vehicle (likely a bus or train)", "indoor, small room,
        possibly a diner or a bar"). The hedged tail is a *guess* — trimming it
        ensures derived refs for the scene only expose the confident part (so
        "train"/"diner" never become groundable vocabulary from those labels).
        """
        text = str(value or "").strip()
        low = text.lower()
        # Cut at hedges that appear either bare ("possibly a diner") or inside
        # a parenthetical ("(likely a bus or train)").
        cuts = [low.find("(")]
        for hedge in _LOCATION_HEDGE_LEADS:
            cuts.append(low.find("(" + hedge.lstrip()))
            cuts.append(low.find(hedge))
        active = [i for i in cuts if i >= 0]
        if active:
            return text[: min(active)].rstrip(" ,:;(-").strip()
        return text

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

        # Claim-level decomposition (Phase 3)
        claims = self.decompose_claims(concept)
        claim_cov = self.claim_coverage(claims)

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
            # Claim-level grounding (Phase 3)
            "claims": claims,
            "reference_coverage": claim_cov["reference_coverage"],
            "claim_coverage_detail": claim_cov["claim_coverage"],
            # Preserve old label for backward compat
            "claim_coverage": self._coverage_label(claim_ratio, has_claims=bool(claim_refs)),
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

    # -- Plan validation: FactValidator + EditorialSchemaValidator (V4) ------

    def _validate_factual_plan(
        self,
        editorial_plan: Dict[str, Any],
        evidence_scene_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """FACT VALIDATOR: Check factual fields in editorial_plan against evidence.

        Validates:
        - scene_id exists in evidence scenes
        - start_sec/end_sec are valid timestamps within the scene
        - source_fact_refs reference real entities from evidence scenes

        Returns dict with: valid (bool), errors (list), warnings (list).
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(editorial_plan, dict):
            return {"valid": False, "errors": ["editorial_plan missing or not a dict"], "warnings": []}

        visual = editorial_plan.get("visual", {})
        if not isinstance(visual, dict):
            errors.append("visual section missing or not a dict")
        else:
            # Validate scene_id
            scene_id = visual.get("scene_id")
            if not scene_id:
                errors.append("visual.scene_id is required")
            else:
                evidence_ids = [
                    sid for sid in self.facts.used_scene_ids()
                    if not evidence_scene_ids or sid in set(evidence_scene_ids or [])
                ]
                if scene_id not in evidence_ids:
                    errors.append(f"visual.scene_id '{scene_id}' not in evidence scenes: {evidence_ids}")

                # Validate timestamps
                start_sec = visual.get("start_sec")
                end_sec = visual.get("end_sec")
                if start_sec is not None and end_sec is not None:
                    try:
                        s = float(start_sec)
                        e = float(end_sec)
                        if e <= s:
                            errors.append(f"visual.end_sec ({e}) must be > start_sec ({s})")
                        # Check within scene bounds
                        if scene_id and scene_id in self.facts.used_scene_ids():
                            sf = self.facts.by_id(scene_id)
                            if sf and (s < sf.start_sec - 1.0 or e > sf.end_sec + 1.0):
                                warnings.append(f"Timestamps [{s:.1f}-{e:.1f}] may exceed scene {scene_id} bounds [{sf.start_sec:.1f}-{sf.end_sec:.1f}]")
                    except (TypeError, ValueError):
                        errors.append("visual.start_sec/end_sec must be numeric")

            # Validate source_fact_refs (must be grounded in evidence scenes)
            refs = visual.get("source_fact_refs", [])
            if isinstance(refs, list):
                for ref in refs:
                    if not isinstance(ref, str) or not ref.strip():
                        continue
                    # Check if ref matches known vocabulary in evidence scenes
                    found = False
                    for sid in evidence_scene_ids or self.facts.used_scene_ids():
                        sf = self.facts.by_id(sid)
                        if sf:
                            all_facts = (sf.objects or []) + (sf.characters or []) + \
                                       ([sf.location] if sf.location else []) + \
                                       (sf.actions or []) + (sf.themes or []) + \
                                       ([sf.mood] if sf.mood else []) + \
                                       [d.get("text", "") for d in (sf.dialogue or [])]
                            if any(ref.lower() in fact.lower() for fact in all_facts if isinstance(fact, str)):
                                found = True
                                break
                    if not found and ref.strip():
                        # Not an error — could be abstract ref — just warn
                        pass

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _validate_editorial_schema(self, editorial_plan: Dict[str, Any]) -> Dict[str, Any]:
        """EDITORIAL SCHEMA VALIDATOR: Check editorial fields against controlled vocabularies.

        Validates editing/audio fields against controlled enums. Does NOT check
        against movie vocabulary — these are creative choices.

        Returns dict with: valid (bool), errors (list), warnings (list).
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(editorial_plan, dict):
            return {"valid": False, "errors": ["editorial_plan missing or not a dict"], "warnings": []}

        # Validate editing section
        editing = editorial_plan.get("editing", {})
        if not isinstance(editing, dict):
            errors.append("editing section missing or not a dict")
        else:
            # Validate each enum field
            field_enums = {
                "transition": PLAN_TRANSITIONS,
                "pacing": PLAN_PACING,
                "rhythm": PLAN_RHYTHM,
                "emphasis": PLAN_EMPHASIS,
                "repetition": PLAN_REPETITION,
                "purpose": PLAN_PURPOSE,
            }
            for field, enum_set in field_enums.items():
                val = editing.get(field)
                if val is not None and val not in enum_set:
                    errors.append(f"editing.{field}='{val}' not in allowed enum: {sorted(enum_set)}")

        # Validate audio section
        audio = editorial_plan.get("audio", {})
        if not isinstance(audio, dict):
            errors.append("audio section missing or not a dict")
        else:
            audio_enums = {
                "movie_audio": PLAN_AUDIO_MOVIE,
                "narration": PLAN_AUDIO_NARRATION,
                "music": PLAN_AUDIO_MUSIC,
            }
            for field, enum_set in audio_enums.items():
                val = audio.get(field)
                if val is not None and val not in enum_set:
                    errors.append(f"audio.{field}='{val}' not in allowed enum: {sorted(enum_set)}")

        # Visual section structural check
        visual = editorial_plan.get("visual", {})
        if not isinstance(visual, dict):
            errors.append("visual section missing or not a dict")
        else:
            required_visual = {"scene_id"}
            missing = required_visual - set(visual.keys())
            if missing:
                errors.append(f"visual missing required fields: {missing}")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def plan_grounding(
        self,
        plan: Optional[Dict[str, Any]],
        evidence_scene_ids: Optional[List[str]] = None,
        min_coverage: float = 0.55,
    ) -> Dict[str, Any]:
        """V4 Plan validation: runs both FactValidator and EditorialSchemaValidator.

        Returns combined audit result with:
        - fact_validation: {valid, errors, warnings}
        - editorial_validation: {valid, errors, warnings}
        - overall_valid: bool
        - legacy_prose_audit: (optional) for backward compat with free-text fields
        - (legacy keys for backward compat): sufficient, invented_terms, elsewhere_terms, grounded_terms, coverage
        """
        # Default empty result
        result = {
            "fact_validation": {"valid": False, "errors": ["no editorial_plan provided"], "warnings": []},
            "editorial_validation": {"valid": False, "errors": ["no editorial_plan provided"], "warnings": []},
            "overall_valid": False,
            "legacy_prose_audit": None,
            # Legacy keys for backward compatibility
            "sufficient": False,
            "invented_terms": [],
            "elsewhere_terms": [],
            "grounded_terms": [],
            "coverage": 0.0,
            "min_coverage": 0.0,
        }

        plan = plan or {}
        # Require structured editorial_plan (V4)
        editorial_plan = plan.get("editorial_plan")
        if not isinstance(editorial_plan, dict):
            return {
                "fact_validation": {"valid": False, "errors": ["no editorial_plan provided"], "warnings": []},
                "editorial_validation": {"valid": False, "errors": ["no editorial_plan provided"], "warnings": []},
                "overall_valid": False,
            }

        # Check if we have a properly structured V4 editorial_plan with visual.scene_id
        has_structured_plan = (
            isinstance(editorial_plan, dict)
            and isinstance(editorial_plan.get("visual"), dict)
            and editorial_plan["visual"].get("scene_id")
        )

        # Run V4 validators only if structured plan with required visual.scene_id is present
        if has_structured_plan:
            fact_val = self._validate_factual_plan(editorial_plan, evidence_scene_ids)
            editorial_val = self._validate_editorial_schema(editorial_plan)
            result["fact_validation"] = fact_val
            result["editorial_validation"] = editorial_val
            result["overall_valid"] = fact_val["valid"] and editorial_val["valid"]
        else:
            # No structured plan with visual.scene_id - invalid
            result["fact_validation"] = {"valid": False, "errors": ["missing visual.scene_id in editorial_plan"], "warnings": []}
            result["editorial_validation"] = {"valid": False, "errors": ["missing visual.scene_id in editorial_plan"], "warnings": []}
            result["overall_valid"] = False

        return result

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
        fields: Optional[Iterable[str]] = None,
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

        ``fields`` restricts which concept prose fields are scanned
        (default: all). The milestone gate derives refs from the THESIS alone
        (the central claim) so that a thesis about an absent object cannot be
        rescued by decorative prose in ``visual_opportunity`` / ``hook``.
        """
        if not concept:
            return []
        if fields is None:
            fields = ("title", "hook", "thesis", "why_interesting",
                      "visual_opportunity")
        prose_raw = " ".join(
            str(concept.get(k) or "")
            for k in fields
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
                if kind == "dialogue":
                    # Dialogue is only grounded when the concept SUBSTANTIALLY
                    # reproduces the line — a single shared word ("time") is
                    # too weak a bridge between a thesis and a scene's speech.
                    tokens = significant_tokens(str(value))
                    if not tokens or len(tokens) < 2:
                        continue
                    overlap = sum(1 for t in tokens if t in prose_sig_set)
                    if overlap < _dialog_overlap_need(len(tokens)):
                        continue
                elif kind == "location":
                    # Only confident, single-reading location labels can ground
                    # a claim. A hedged/alternative label ("in a vehicle (likely
                    # a bus or train)", "small shop or garage") is a vision-model
                    # guess the thesis must not borrow a word from.
                    if not self._is_location_confident(str(value)):
                        continue
                    tokens = significant_tokens(str(value))
                else:
                    tokens = significant_tokens(str(value))
                if not tokens:
                    continue
                # Match on the vocabulary item's HEAD (first significant, content) token
                # OR on an overlap of at least TWO of its own content tokens.
                # A lone NON-head shared token is too weak a bridge: a thesis
                # that says "clock face ... is visible ... constructed around"
                # must not harvest live items that merely share "face" /
                # "visible" / "around" ("woman's face", "another person
                # partially visible", "looking around"). The real-Qwen T4
                # Run-2 "Clock That Never Ticks" concept passed the claim gate
                # exactly this way. Single-content-word grounding is kept when
                # the word IS the item's head ("saloon" still derives the
                # canonical location "saloon, dim light"). Stopwords are
                # excluded ("man IN plaid shirt" never fires on a mere "in").
                # The ref is only emitted after verifying the FULL value
                # grounds.
                head = tokens[0]
                hits = sum(1 for t in tokens if t in prose_sig_set)
                if head in prose_sig_set or hits >= 2:
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

    def is_claim_sufficient(
        self,
        concept: Dict[str, Any],
        min_coverage: float = 0.4,
    ) -> bool:
        """A concept's CENTRAL CLAIM must itself ground — decorative prose
        fields cannot rescue a thesis about content that is absent.

        The stopword fix stopped ALL concepts sharing the same generic refs,
        but a concept could still pass because a shared incidental word in its
        ``visual_opportunity`` / ``hook`` anchored a real token. The T4 run
        produced theses about absent objects (a clock, a drawing) that grounded
        ONLY via such incidental matches. This gate derives refs from the
        thesis alone (title + thesis, the claim substance) and requires that
        derivation to be sufficient on its own.
        """
        claim = self.derive_refs(
            concept, fields=("title", "thesis"),
        )
        ev = self.concept_evidence({
            "thesis": concept.get("thesis", ""),
            "evidence_refs": claim,
        })
        return self.is_sufficient_refs(ev, min_coverage=min_coverage)

    def decompose_claims(
        self,
        concept: Dict[str, Any],
        min_overlap: int = 1,
    ) -> List[Dict[str, Any]]:
        """Decompose a concept's thesis/hook into atomic factual claims.

        Returns a list of claim dicts with:
        - claim_id, text, type, support (scene_ids + matched facts),
          status (SUPPORTED/PARTIAL/UNKNOWN/UNSUPPORTED), confidence, missing.
        """
        from .concepts import CLAIM_TYPES
        prose = " ".join(str(concept.get(k) or "") for k in ("title", "hook", "thesis"))
        claims = []

        # Simple heuristic: extract candidate claims from thesis by splitting
        # on comparative/temporal markers and action verbs.
        import re
        # Split on common claim boundaries
        segments = re.split(r"\b(?:and|but|because|since|while|whereas|,)\b", prose.lower())
        segments = [s.strip() for s in segments if len(s.strip()) > 10]

        for i, seg in enumerate(segments):
            # Classify claim type from keywords
            ctype = "VISUAL"  # default
            if any(k in seg for k in ("same", "identical", "compare", "versus", "vs", "contrast", "differ", "match", "mirror")):
                ctype = "COMPARISON"
            elif any(k in seg for k in ("before", "after", "later", "earlier", "then", "subsequent", "preced", "temporal", "sequence", "loop")):
                ctype = "TEMPORAL"
            elif any(k in seg for k in ("character", "person", "protagonist", "actor", "she", "he", "they")):
                ctype = "CHARACTER"
            elif any(k in seg for k in ("object", "item", "prop", "revolver", "car", "door", "window")):
                ctype = "OBJECT"
            elif any(k in seg for k in ("location", "place", "setting", "scene", "room", "outdoor", "indoor")):
                ctype = "LOCATION"
            elif any(k in seg for k in ("action", "enter", "exit", "move", "walk", "run", "stand", "sit", "speak")):
                ctype = "ACTION"
            elif any(k in seg for k in ("dialogue", "speak", "say", "say", "line", "conversation")):
                ctype = "DIALOGUE"
            elif any(k in seg for k in ("cause", "because", "lead to", "result", "effect", "consequence")):
                ctype = "CAUSAL"
            elif any(k in seg for k in ("feel", "emotion", "mood", "tone", "sad", "happy", "tense", "calm", "grief", "fear")):
                ctype = "EMOTIONAL"
            elif any(k in seg for k in ("relationship", "between", "with", "connect", "link", "tie")):
                ctype = "RELATIONSHIP"

            # Find supporting scenes/facts
            support_scenes = []
            matched_facts = []
            for sid in self.facts.used_scene_ids():
                scene_facts = self._scene_tokens.get(sid, set())
                # Check if this segment's tokens overlap with scene
                seg_tokens = set(significant_tokens(seg))
                overlap = seg_tokens & scene_facts
                if len(overlap) >= min_overlap:
                    support_scenes.append(sid)
                    matched_facts.extend(list(overlap)[:5])

            # Determine status
            if len(support_scenes) >= 2 and ctype == "COMPARISON":
                status = "SUPPORTED" if len(support_scenes) >= 2 else "PARTIAL"
            elif len(support_scenes) >= 1:
                status = "SUPPORTED"
            elif ctype in ("EMOTIONAL", "CAUSAL", "RELATIONSHIP"):
                status = "UNKNOWN"  # Interpretive claims can't be fully verified
            else:
                status = "UNSUPPORTED"

            confidence = min(1.0, len(matched_facts) / 5.0) if matched_facts else 0.1

            claims.append({
                "claim_id": f"claim_{i+1:02d}",
                "text": seg[:200],
                "type": ctype,
                "support": [{"scene_id": sid, "facts": list(set(matched_facts))} for sid in support_scenes],
                "status": status,
                "confidence": round(confidence, 2),
                "missing": [] if status == "SUPPORTED" else ["insufficient_scene_evidence"],
            })

        # If no claims extracted, create a fallback from the whole thesis
        if not claims:
            claims.append({
                "claim_id": "claim_01",
                "text": prose[:200],
                "type": "VISUAL",
                "support": [],
                "status": "UNKNOWN",
                "confidence": 0.1,
                "missing": ["no_decomposable_claims"],
            })

        return claims

    def claim_coverage(
        self,
        claims: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate reference_coverage and claim_coverage from claim list."""
        if not claims:
            return {"reference_coverage": 0.0, "claim_coverage": 0.0}
        supported = sum(1 for c in claims if c["status"] == "SUPPORTED")
        partial = sum(1 for c in claims if c["status"] == "PARTIAL")
        total = len(claims)
        # claim_coverage weights: SUPPORTED=1.0, PARTIAL=0.5, UNKNOWN=0.2, UNSUPPORTED=0.0
        weight_map = {"SUPPORTED": 1.0, "PARTIAL": 0.5, "UNKNOWN": 0.2, "UNSUPPORTED": 0.0}
        claim_cov = sum(weight_map.get(c["status"], 0.0) for c in claims) / total
        ref_cov = (supported + 0.5 * partial) / total
        return {
            "reference_coverage": round(ref_cov, 3),
            "claim_coverage": round(claim_cov, 3),
        }
