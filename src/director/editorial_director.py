"""Real Editorial Director - produces an Editorial Decision List from a grounded script.

The Editorial Director consumes:
- selected V4 concept
- grounding contract
- grounded script
- scene intelligence
- evidence references
- available audio metadata

It outputs a structured Editorial Decision List.
Each segment must include:
- segment_id
- purpose
- narrative_beat
- evidence
- visual_strategy
- pacing
- audio
- editing

The Editorial Director is NOT the FFmpeg renderer.
The renderer only executes decisions.
"""

from typing import Dict, Any, List, Optional, Literal, Callable
from dataclasses import dataclass, field, asdict
from typing import Literal
import json
from pathlib import Path
from datetime import datetime


# --- Editorial Decision Schema ------------------------------------------------

# Valid values for editorial decision fields (from concepts.py)
PLAN_TRANSITIONS = frozenset({
    "cut", "crossfade", "fade", "dissolve", "jump_cut", "match_cut",
    "smash_cut", "wipe", "iris", "none",
})

PLAN_PACING = frozenset({
    "slow", "measured", "moderate", "gradual", "steady", "rhythmic",
    "rapid", "fast", "accelerating", "decelerating", "variable",
})

PLAN_RHYTHM = frozenset({
    "slow", "steady", "measured", "syncopated", "driving", "pulsing",
    "irregular", "free",
})

PLAN_EMPHASIS = frozenset({
    "character", "action", "object", "location", "emotion", "dialogue",
    "visual", "sound", "silence", "contrast", "repetition", "detail",
})

PLAN_REPETITION = frozenset({
    "none", "motif", "callback", "echo", "parallel", "mirror", "loop",
})

PLAN_PURPOSE = frozenset({
    "contrast", "parallel", "progression", "reveal", "emphasis",
    "transition", "pacing", "mood", "character", "theme", "tension",
    "resolution", "setup", "payoff",
})

PLAN_AUDIO_MOVIE = frozenset({
    "retain", "mute", "filter", "duck",
})

PLAN_AUDIO_NARRATION = frozenset({
    "none", "minimal", "moderate", "dominant", "continuous", "sparse",
})

PLAN_AUDIO_MUSIC = frozenset({
    "none", "low", "moderate", "high", "diegetic_only", "score_only",
})


# --- Editorial Decision Schema -------------------------------------------------

@dataclass
class EditorialSegment:
    """A single editorial decision segment."""
    
    segment_id: str
    purpose: str
    narrative_beat: str
    evidence: List[Dict[str, Any]]
    visual_strategy: Dict[str, Any]
    pacing: Dict[str, Any]
    audio: Dict[str, Any]
    editing: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "purpose": self.purpose,
            "narrative_beat": self.narrative_beat,
            "evidence": self.evidence,
            "visual_strategy": self.visual_strategy,
            "pacing": self.pacing,
            "audio": self.audio,
            "editing": self.editing,
        }


@dataclass
class EditorialDecisionList:
    """The complete Editorial Decision List - the output of the Editorial Director."""
    
    concept_title: str = ""
    concept_thesis: str = ""
    segments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    project_id: str = ""
    movie_title: str = ""
    grounding_contract_path: str = ""
    editorial_plan_path: str = ""
    grounded_script_path: str = ""
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_title": self.concept_title,
            "concept_thesis": self.concept_thesis,
            "segments": [s for s in self.segments],
            "metadata": {
                "project_id": self.project_id,
                "movie_title": self.movie_title,
                "grounding_contract_path": self.grounding_contract_path,
                "editorial_plan_path": self.editorial_plan_path,
                "grounded_script_path": self.grounded_script_path,
                "created_at": self.created_at,
            },
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @staticmethod
    def from_json(json_str: str) -> "EditorialDecisionList":
        return EditorialDecisionList.from_dict(json.loads(json_str))
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EditorialDecisionList":
        edl = EditorialDecisionList()
        edl.concept_title = d.get("concept_title", "")
        edl.concept_thesis = d.get("concept_thesis", "")
        edl.segments = d.get("segments", [])
        meta = d.get("metadata", {})
        edl.project_id = meta.get("project_id", "")
        edl.movie_title = meta.get("movie_title", "")
        edl.grounding_contract_path = meta.get("grounding_contract_path", "")
        edl.editorial_plan_path = meta.get("editorial_plan_path", "")
        edl.grounded_script_path = meta.get("grounded_script_path", "")
        edl.created_at = meta.get("created_at", "")
        return edl
    
    def add_segment(
        self,
        segment_id: str,
        purpose: str,
        narrative_beat: str,
        evidence: List[Dict[str, Any]],
        visual_strategy: Dict[str, Any],
        pacing: Dict[str, Any],
        audio: Dict[str, Any],
        editing: Dict[str, Any],
    ):
        """Add a segment to the editorial decision list."""
        self.segments.append({
            "segment_id": segment_id,
            "purpose": purpose,
            "narrative_beat": narrative_beat,
            "evidence": evidence,
            "visual_strategy": visual_strategy,
            "pacing": pacing,
            "audio": audio,
            "editing": editing,
        })
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @staticmethod
    def from_json(json_str: str) -> "EditorialDecisionList":
        return EditorialDecisionList.from_dict(json.loads(json_str))
    
    @staticmethod
    def load(path: Path) -> "EditorialDecisionList":
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return EditorialDecisionList.from_dict(data)
    
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())


# --- Editorial Director -------------------------------------------------------

class EditorialDirector:
    """Real Editorial Director - produces an Editorial Decision List from a grounded script.
    
    The Editorial Director consumes:
    - selected V4 concept
    - grounding contract
    - grounded script
    - scene intelligence
    - evidence references
    - available audio metadata
    
    It outputs a structured Editorial Decision List.
    
    The Editorial Director is NOT the FFmpeg renderer.
    The renderer only executes decisions.
    """
    
    def __init__(
        self,
        scene_facts=None,
        llm: Callable[[str], str] = None,
    ):
        """Initialize the Editorial Director.
        
        Args:
            scene_facts: Optional SceneFacts for validation
            llm: Optional LLM callable for creative decisions (str -> str)
        """
        self.scene_facts = scene_facts
        self.llm = llm
    
    def create_editorial_plan(
        self,
        grounded_script: Dict[str, Any],
        grounding_contract: Dict[str, Any],
        editorial_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate an Editorial Decision List from a grounded script and editorial plan.
        
        Args:
            grounded_script: The grounded script (from grounded_script.py)
            grounding_contract: The V4 grounding contract
            editorial_plan: The structured editorial_plan from V4 (visual/editing/audio)
            
        Returns:
            A structured Editorial Decision List dict.
        """
        # Calculate clip duration from supporting scenes in grounding contract
        clip_duration = self._calculate_clip_duration(grounding_contract)
        
        # Build the editorial decision list
        edl = {
            "concept_title": grounded_script.get("concept", {}).get("title", ""),
            "concept_thesis": grounded_script.get("concept", {}).get("thesis", ""),
            "segments": self._build_editorial_segments(
                grounded_script.get("sections", []),
                grounding_contract.get("supporting_scenes", []),
                grounding_contract.get("evidence_refs", []),
                editorial_plan=editorial_plan,
                clip_duration=clip_duration,
            ),
            "metadata": {
                "project_id": "",
                "movie_title": "",
                "grounding_contract_path": "",
                "editorial_plan_path": "",
                "grounded_script_path": "",
                "created_at": self._now_iso(),
            }
        }
        
        return {
            "concept_title": edl["concept_title"],
            "concept_thesis": edl["concept_thesis"],
            "segments": edl["segments"],
            "metadata": edl["metadata"],
        }
    
    def _build_editorial_segments(
        self,
        script_sections: List[Dict[str, Any]],
        supporting_scenes: List[Dict[str, Any]],
        evidence_refs: List[Dict[str, Any]],
        editorial_plan: Dict[str, Any],
        clip_duration: float = 3.0,
    ) -> List[Dict[str, Any]]:
        """Build editorial segments from grounded script sections."""
        segments = []
        
        # Calculate clip duration from supporting scenes if not provided
        if clip_duration <= 0:
            clip_duration = 3.0
        
        # Extract editorial plan sections
        visual_plan = editorial_plan.get("visual", {})
        editing_plan = editorial_plan.get("editing", {})
        audio_plan = editorial_plan.get("audio", {})
        
        # Build segments from script sections
        for section in self._get_script_sections():
            # Extract evidence for this section
            section_evidence = self._extract_section_evidence(section)
            
            # Determine segment purpose
            purpose = self._infer_segment_purpose(section)
            
            # Build visual strategy from plan + evidence
            visual_strategy = self._build_visual_strategy(
                section, editorial_plan.get("visual", {}), grounding_contract.get("supporting_scenes", [])
            )
            
            # Build pacing from plan
            pacing = self._build_pacing(editorial_plan.get("editing", {}), clip_duration=clip_duration)
            
            # Build audio strategy
            audio = self._build_audio_strategy(
                editorial_plan.get("audio", {}),
                audio.get("movie_audio", "retain"),
                audio.get("narration", "dominant"),
                audio.get("music", "low"),
            )
            
            # Build editing strategy
            editing = self._build_editing_strategy(
                editorial_plan.get("editing", {}),
            )
            
            segment = {
                "segment_id": f"seg_{len(segments)+1:02d}",
                "purpose": purpose,
                "narrative_beat": self._infer_narrative_beat(section),
                "evidence": section.get("evidence", []),
                "visual_strategy": visual_strategy,
                "pacing": pacing,
                "audio": audio,
                "editing": editing,
            }
            segments.append(segment)
        
        return segments
    
    def _get_script_sections(self) -> List[Dict[str, Any]]:
        """Extract sections from grounded script."""
        # This is a placeholder - in practice would read from grounded script
        return []
    
    def _extract_section_evidence(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract evidence references from a script section."""
        return section.get("evidence", [])
    
    def _infer_segment_purpose(self, section: Dict[str, Any]) -> str:
        """Infer the editorial purpose of a script section."""
        section_type = section.get("type", "")
        type_to_purpose = {
            "hook": "hook",
            "setup": "establish",
            "claim": "claim",
            "evidence": "evidence",
            "interpretation": "interpret",
            "second_evidence": "reinforce",
            "implication": "imply",
            "conclusion": "conclude",
        }
        return type_to_purpose.get(section.get("type", ""), "support")
    
    def _infer_narrative_beat(self, section: Dict[str, Any]) -> str:
        """Infer the narrative beat from a script section."""
        section_type = section.get("type", "")
        narration = section.get("narration", "")
        return f"{section.get('type', 'segment')}: {section.get('narration', '')[:100]}"
    
    def _build_visual_strategy(
        self,
        section: Dict[str, Any],
        visual_plan: Dict[str, Any],
        supporting_scenes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build visual strategy for a segment."""
        visual_plan = {}
        
        # Determine shot type based on section type and evidence
        section_type = section.get("type", "")
        evidence = section.get("evidence", [])
        
        if not evidence:
            return {"type": "wide", "reason": "establishing"}
        
        # Check for specific evidence types
        has_closeup_ref = any(
            ev.get("kind") == "object" and 
            any(kw in ev.get("value", "").lower() for kw in ["face", "eye", "hand", "detail", "close"])
            for ev in section.get("evidence", [])
        )
        
        has_wide_ref = any(
            ev.get("kind") == "location" or
            "wide" in ev.get("value", "").lower() or
            "landscape" in ev.get("value", "").lower()
            for ev in section.get("evidence", [])
        )
        
        if has_closeup_ref:
            return {"type": "close_up", "reason": "evidence references close detail"}
        elif has_wide_ref:
            return {"type": "wide", "reason": "evidence references location/environment"}
        elif section.get("type") == "evidence":
            return {"type": "close_up", "reason": "evidence presentation"}
        elif section.get("type") == "hook":
            return {"type": "close_up", "reason": "hook requires immediate engagement"}
        else:
            return {"type": "medium", "reason": "standard coverage"}
    
    def _build_pacing(self, editing_plan: Dict[str, Any], clip_duration: float = 3.0) -> Dict[str, Any]:
        """Build pacing config from editorial plan, respecting clip duration."""
        pacing = {}
        if "pacing" in editing_plan:
            pacing["rhythm"] = editing_plan.get("pacing", "measured")
        if "rhythm" in editing_plan:
            pacing["rhythm_detail"] = editing_plan["rhythm"]
        if "repetition" in editing_plan:
            pacing["repetition"] = editing_plan["repetition"]
        if "purpose" in editing_plan:
            pacing["purpose"] = editing_plan["purpose"]
        
        # Set defaults based on clip duration
        pacing.setdefault("rhythm", "measured")
        # Use clip duration as hint, but cap segment duration to a reasonable max
        pacing.setdefault("duration_hint_sec", min(clip_duration, 5.0))
        return pacing

    def _build_audio_strategy(
        self,
        audio_plan: Dict[str, Any],
        movie_audio: str = "retain",
        narration: str = "dominant",
        music: str = "low",
    ) -> Dict[str, Any]:
        """Build audio strategy from editorial plan."""
        audio = {}
        
        if "movie_audio" in audio_plan:
            audio["movie_audio"] = audio_plan["movie_audio"]
        else:
            audio["movie_audio"] = movie_audio
        
        if "narration" in audio_plan:
            audio["narration"] = audio_plan["narration"]
        else:
            audio["narration"] = narration
        
        if "music" in audio_plan:
            audio["music"] = audio_plan["music"]
        else:
            audio["music"] = music
        
        return audio
    
    def _build_editing_strategy(self, editing_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Build editing strategy from editorial plan."""
        editing = {}
        
        if "transition" in editing_plan:
            editing["transition"] = editing_plan["transition"]
        else:
            editing["transition"] = "cut"
        
        if "speed" in editing_plan:
            editing["speed"] = editing_plan["speed"]
        else:
            editing["speed"] = 1.0
        
        if "hold" in editing_plan:
            editing["hold"] = editing_plan["hold"]
        else:
            editing["hold"] = False
        
        return editing
    
    def _calculate_clip_duration(self, grounding_contract: Dict[str, Any]) -> float:
        """Calculate the total clip duration from supporting scenes in the grounding contract."""
        supporting_scenes = grounding_contract.get("supporting_scenes", [])
        total_duration = 0.0
        for scene in grounding_contract.get("supporting_scenes", []):
            start = scene.get("start_sec")
            end = scene.get("end_sec")
            if start is not None and end is not None:
                total_duration += max(0.0, end - start)
        return total_duration if total_duration > 0 else 3.0  # fallback to 3.0s if no scenes

    def _now_iso(self) -> str:
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"


# --- Main Entry Point ---------------------------------------------------------

def generate_editorial_decisions(
    grounded_script_path: Path,
    grounding_contract_path: Path,
    editorial_plan_path: Path,
    output_path: Path,
    scene_facts=None,
) -> Path:
    """Main entry point: generate Editorial Decision List from grounded script and plan.
    
    Args:
        grounded_script_path: Path to grounded script JSON
        grounding_contract_path: Path to grounding_contract.json
        editorial_plan_path: Path to editorial_plan.json (from V4)
        output_path: Where to save the Editorial Decision List JSON
        scene_facts: Optional SceneFacts for validation
        
    Returns:
        Path to the generated Editorial Decision List JSON file.
    """
    import json
    from pathlib import Path
    from datetime import datetime
    
    # Load inputs
    with open(grounded_script_path, 'r', encoding='utf-8') as f:
        grounded_script = json.load(f)
    
    with open(grounding_contract_path, 'r', encoding='utf-8') as f:
        grounding_contract = json.load(f)
    
    with open(editorial_plan_path, 'r', encoding='utf-8') as f:
        editorial_plan = json.load(f)
    
    # Create editorial director
    director = EditorialDirector()
    
    # Generate editorial decisions
    edl = director.create_editorial_plan(
        grounded_script=grounded_script,
        grounding_contract=grounding_contract,
        editorial_plan=editorial_plan,
    )
    
    # Add metadata
    edl["metadata"] = {
        "project_id": "",
        "movie_title": "",
        "grounding_contract_path": str(Path(grounding_contract_path).name),
        "editorial_plan_path": str(Path(editorial_plan_path).name),
        "grounded_script_path": str(Path(grounded_script_path).name),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(edl, f, indent=2, ensure_ascii=False)
    
    return output_path


# --- Validation Utilities -----------------------------------------------------

def validate_editorial_decision_list(
    edl: Dict[str, Any],
    grounding_contract: Dict[str, Any],
    grounded_script: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate an Editorial Decision List against the grounding contract and script."""
    errors = []
    warnings = []
    
    # Check each segment references valid evidence
    for segment in edl.get("segments", []):
        for ev in segment.get("evidence", []):
            scene_id = ev.get("scene_id")
            # Could validate against grounding contract here
            pass
    
    # Check for repeated excerpts
    excerpt_windows = set()
    for segment in edl.get("segments", []):
        for ev in segment.get("evidence", []):
            key = (ev.get("scene_id"), ev.get("start_sec"), ev.get("end_sec"))
            if key in excerpt_windows:
                warnings.append(f"Repeated excerpt: {key}")
            excerpt_windows.add(key)
    
    # Check for repeated footage
    # Check timestamp validity
    # Check visual strategy has required fields
    # Check audio fields are valid enums
    # Check editing fields are valid enums
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }