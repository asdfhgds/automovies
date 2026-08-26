"""Scenario-driven deterministic mock LLM for the director validation harness.

The real validation ``scripts/run_director_validation.py`` requires a GPU + real
Qwen. This mock lets the SAME harness run end-to-end locally with no model
download, exercising every stage (project load -> SceneFacts -> context builder
-> concept generation -> grounding -> admission -> plan -> plan grounding ->
verdict -> report writing).

Each ``scenario`` deterministically produces raw LLM text that mimics a real
model behavior that must be proven:

- ``grounded``          — every concept carries verbatim, resolvable vocabulary
                          (object + confident location); the plan's
                          ``editorial_plan`` also stays inside the evidence
                          scene's facts. Verdict must be PASS.
- ``hallucinated``      — concepts name objects/locations/characters that do
                          NOT exist in the movie. The gate must reject them; the
                          bounded regeneration then substitutes grounded
                          concepts so the run still succeeds (proving bounded
                          regeneration + honest rejection counts).
- ``invalid``           — thesis is structurally fine but its evidence refs are
                          unsupported; regeneration keeps failing, so no
                          concept/plan is emitted (verdict FAIL).
- ``hedged``            — concepts lean on a hedged location branch ("small
                          shop or garage", "(likely a bus or train)"). Because
                          ``_is_location_confident`` disqualifies them, the
                          claim must NOT ground on those words -> rejected.
- ``partial``           — concept has some real + some absent vocabulary; the
                          derived refs cover it only partially, so coverage is
                          MED/LOW and admission follows the policy.
- ``none``              — every candidate fails (even after regeneration);
                          selected concept is None, plan is None, verdict FAIL.
- ``plan_rejected``     — a grounded concept is selected but the plan's
                          editorial_plan invents unsupported content; the
                          strict plan gate records the rejection (verdict FAIL,
                          plan_rejection present).

The mock uses the movie's REAL vocabulary (from the scene index) when it must
be grounded, and a fixed set of KNOWN-ABSENT tokens when it must be rejected,
so determinism matches the deterministic matcher.
"""
import json
from typing import Any, Dict, List, Optional

from .evidence import EvidenceAnalyzer
from .scene_facts import SceneFacts

# Tokens that never appear in the real movie facts (verified for bc6384be).
ABSENT = {
    "object": "flying saucer",
    "character": "Sherlock Holmes",
    "location": "enchanted castle",
    "action": "tap dancing",
}
ABSENT_THESIS = (
    "the flying saucer and the enchanted castle oppose the true meaning of "
    "the family dinner"
)


def _rich_scene(movie_index: Dict[str, Any]) -> Dict[str, Any]:
    """The most citable scene: most objects+actions and a confident location."""
    best, best_score = None, -1
    for scene in (movie_index.get("scenes") or []):
        story = scene.get("story") or {}
        loc = story.get("location") or ""
        if not EvidenceAnalyzer._is_location_confident(loc):
            continue
        score = len(story.get("objects") or []) + len(story.get("actions") or [])
        if score > best_score:
            best, best_score = scene, score
    if best is not None:
        return best
    return (movie_index.get("scenes") or [{}])[0]


class MockValidationLLM:
    """``Callable[[str], str]`` that plays one scenario deterministically."""

    def __init__(
        self,
        movie_index: Dict[str, Any],
        scenario: str = "grounded",
        num_concepts: int = 5,
        plan_direction_stays_grounded: bool = True,
    ):
        self.movie_index = movie_index
        self.facts = SceneFacts.from_movie_intelligence(movie_index=movie_index)
        self.scenario = scenario
        self.num_concepts = num_concepts
        # The plan_rejected scenario MUST invent unsupported content so the
        # strict plan gate is exercised; ignore the caller's preference there.
        self.plan_direction_stays_grounded = (
            False if scenario == "plan_rejected" else plan_direction_stays_grounded
        )
        self.calls: List[str] = []
        self._substitutes = 0

        # Honest provider-style attributes the harness runtime block reads.
        self.model_name = "mock-validation"
        self.device = "mock"
        self.dtype = "mock"
        self.model_load_time_sec = 0.0
        self.generation_times: List[float] = []

    # -- Provider-style surface -------------------------------------------

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if "finalizing the plan" in prompt:
            return json.dumps(self._plan(), ensure_ascii=False)
        if "re-running" in prompt:
            self._substitutes += 1
            return json.dumps({"concepts": self._concepts()}, ensure_ascii=False)
        return json.dumps({"concepts": self._concepts()}, ensure_ascii=False)

    # -- Concept inventory per scenario -------------------------------------

    def _concepts(self) -> List[Dict[str, Any]]:
        if self.scenario == "grounded":
            return self._grounded_concepts()
        if self.scenario == "hallucinated":
            # First batch hallucinates; substitutes are grounded (bounded
            # regeneration recovers the run).
            if self._substitutes >= 1:
                return self._grounded_concepts()
            return self._hallucinated_concepts()
        if self.scenario == "invalid":
            # Always invalid: thesis ok, refs not; no grounded substitute.
            return self._invalid_concepts()
        if self.scenario == "hedged":
            if self._substitutes >= 1:
                return self._grounded_concepts()
            return self._hedged_concepts()
        if self.scenario == "partial":
            return self._partial_concepts()
        if self.scenario == "none":
            return self._hallucinated_concepts()
        if self.scenario == "plan_rejected":
            return self._grounded_concepts()
        return self._grounded_concepts()

    def _grounded_concepts(self) -> List[Dict[str, Any]]:
        scene = _rich_scene(self.movie_index)
        story = scene.get("story") or {}
        objs = story.get("objects") or []
        loc = story.get("location") or ""
        acts = story.get("actions") or []
        them = story.get("themes") or []
        obj = objs[0] if objs else "the frame"
        act = (acts[0] if acts else "moving") or "moving"
        theme = (them[0] if them else "reflection") or "reflection"
        sid = scene.get("scene_id") or "scene-1"
        out = []
        for i in range(self.num_concepts):
            out.append({
                "title": f"The {obj} in frame {i + 1}",
                "hook": "The details of this scene carry the whole argument.",
                "thesis": (
                    f"the {obj} in the {loc} shows the film's idea of {theme} "
                    f"through the act of {act}, grounded in {sid}"
                ),
                "why_interesting": "what the camera chooses to show mirrors "
                                   "what the film really believes",
                "visual_opportunity": f"a slow push on the {obj} inside the {loc}",
                "format": "short_video_essay",
                "diversity_angle": "symbolism" if i % 2 == 0 else "cinematography",
            })
        return out

    def _hallucinated_concepts(self) -> List[Dict[str, Any]]:
        out = []
        for i in range(self.num_concepts):
            out.append({
                "title": f"Absent Idea {i + 1}",
                "hook": "We must follow a claim the movie never makes.",
                "thesis": ABSENT_THESIS,
                "why_interesting": "the argument comes from nowhere in the footage",
                "visual_opportunity": f"an extreme close-up of the {ABSENT['object']}",
                "format": "short_video_essay",
                "diversity_angle": "symbolism",
            })
        return out

    def _invalid_concepts(self) -> List[Dict[str, Any]]:
        out = []
        for i in range(self.num_concepts):
            out.append({
                "title": f"Structurally Fine {i + 1}",
                "hook": "A tidy premise.",
                "thesis": (
                    f"the thesis is well formed but its only support is the "
                    f"{ABSENT['object']}, the {ABSENT['character']} and the "
                    f"{ABSENT['location']}"
                ),
                "why_interesting": "never grounded in any real scene",
                "visual_opportunity": f"a crane shot over the {ABSENT['location']}",
                "format": "short_video_essay",
                "diversity_angle": "narrative structure",
            })
        return out

    def _hedged_concepts(self) -> List[Dict[str, Any]]:
        out = []
        for i in range(self.num_concepts):
            out.append({
                "title": f"Hedge Claim {i + 1}",
                "hook": "A plausible word hides an uncertain label.",
                "thesis": (
                    "the train platform becomes the film's image of escape, "
                    "with the shop as its small world"
                ),
                "why_interesting": "these words only appear inside hedged guesses",
                "visual_opportunity": "a wide shot of the platform",
                "format": "short_video_essay",
                "diversity_angle": "symbolism",
            })
        return out

    def _partial_concepts(self) -> List[Dict[str, Any]]:
        scene = _rich_scene(self.movie_index)
        story = scene.get("story") or {}
        objs = story.get("objects") or []
        loc = story.get("location") or ""
        obj = objs[0] if objs else "the frame"
        out = []
        for i in range(self.num_concepts):
            out.append({
                "title": f"Partial Coverage {i + 1}",
                "hook": "Half real, half imagined.",
                "thesis": (
                    f"the {obj} in the {loc} is real, yet the {ABSENT['object']} "
                    f"and the {ABSENT['action']} are not"
                ),
                "why_interesting": "only part of the evidence exists",
                "visual_opportunity": f"a slow push on the {obj}",
                "format": "short_video_essay",
                "diversity_angle": "symbolism",
                # Declared (model-provided) refs mixing real + absent — the
                # deterministic derivation keeps only the grounded half, so the
                # evidence preview shows the missing refs honestly.
                "evidence_refs": [
                    {"kind": "object", "value": obj},
                    {"kind": "object", "value": ABSENT["object"]},
                    {"kind": "character", "value": ABSENT["character"]},
                ],
            })
        return out

    # -- Plan --------------------------------------------------------------

    def _plan(self) -> Dict[str, Any]:
        scene = _rich_scene(self.movie_index)
        story = scene.get("story") or {}
        objs = story.get("objects") or []
        loc = story.get("location") or ""
        obj = objs[0] if objs else "the frame"
        if self.plan_direction_stays_grounded:
            # Only tokens that are either PLAN_EDITORIAL_TERMS or verbatim
            # vocabulary of the evidence scene (so the deterministic audit
            # scores grounded coverage high and no invented terms).
            ed = {
                "pacing": "slow measured pacing, quiet and minimal",
                "visual_style": (
                    f"slow zooms and quiet cuts holding on the {obj} inside "
                    f"the {loc}"
                ),
                "audio_style": "minimal sound, sparse and still",
                "editing_style": "quiet cuts and slow transitions",
            }
        else:
            ed = {
                "pacing": "frenetic",
                "visual_style": (
                    f"empty chairs, an open window and a flying saucer parked "
                    f"outside the {loc}"
                ),
                "audio_style": "noise",
                "editing_style": "crash zooms",
            }
        return {
            "concept": {"title": "t", "hook": "h", "thesis": "s"},
            "format": {"type": "short_video_essay", "duration_sec": 90},
            "editorial_plan": {
                "visual": {
                    "scene_id": "scene-1",
                    "start_sec": 1.0,
                    "end_sec": 3.0,
                    "source_fact_refs": []
                },
                "editing": {
                    "transition": "cut",
                    "pacing": "steady",
                    "rhythm": "steady",
                    "emphasis": "character",
                    "repetition": "none",
                    "purpose": "contrast"
                },
                "audio": {
                    "movie_audio": "retain",
                    "narration": "moderate",
                    "music": "low"
                }
            },
        }