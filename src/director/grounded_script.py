"""Grounded Script Generator - converts V4 grounding contract into a structured script.

The script is the narrative blueprint for the editorial director. It maps every
analytical/interpretive claim to concrete evidence from the grounding contract.
No invented characters, objects, locations, or dialogue allowed.

Script structure (flexible, but must map to evidence):
  HOOK -> SETUP -> CLAIM -> EVIDENCE -> INTERPRETATION -> SECOND_EVIDENCE -> IMPLICATION -> CONCLUSION

Every section must map to concrete evidence with scene IDs and timestamps.
"""

from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field, asdict
from typing import Literal
import json
from pathlib import Path
from datetime import datetime


# --- Script Schema ------------------------------------------------------------

class ScriptEvidenceRef:
    """A single evidence reference in the script."""
    def __init__(
        self,
        scene_id: str,
        start_sec: float,
        end_sec: float,
        purpose: str = "",
    ):
        self.scene_id = scene_id
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.purpose = purpose

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "purpose": self.purpose,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ScriptEvidenceRef":
        return ScriptEvidenceRef(
            scene_id=d["scene_id"],
            start_sec=d["start_sec"],
            end_sec=d["end_sec"],
            purpose=d.get("purpose", ""),
        )


class ScriptSection:
    """A single section of the grounded script."""
    
    # Section types that correspond to the flexible script structure
    SectionType = Literal[
        "hook", "setup", "claim", "evidence", 
        "interpretation", "second_evidence", 
        "implication", "conclusion",
        "custom"  # for custom treatment-specific sections
    ]
    
    def __init__(
        self,
        section_id: str,
        section_type: str,
        narration: str = "",
        evidence: List[Dict[str, Any]] = None,
    ):
        self.section_id = section_id
        self.section_type = section_type
        self.narration = narration
        self.evidence = evidence or []

    def add_evidence(self, scene_id: str, start_sec: float, end_sec: float, purpose: str = ""):
        self.evidence.append({
            "scene_id": scene_id,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "purpose": purpose,
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "type": self.section_type,
            "narration": self.narration,
            "evidence": self.evidence,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ScriptSection":
        return ScriptSection(
            section_id=d["section_id"],
            section_type=d["type"],
            narration=d.get("narration", ""),
            evidence=d.get("evidence", []),
        )


@dataclass
class GroundedScript:
    """The complete grounded script derived from a V4 grounding contract."""
    
    # Core concept info (from grounding contract)
    concept_title: str = ""
    concept_hook: str = ""
    thesis: str = ""
    why_interesting: str = ""
    target_duration_sec: int = 90
    
    # Structured sections - flexible but must map to evidence
    sections: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    project_id: str = ""
    movie_title: str = ""
    grounding_contract_path: str = ""
    
    # Metadata
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept": {
                "title": self.concept_title,
                "hook": self.concept_hook,
                "thesis": self.thesis,
                "why_interesting": self.why_interesting,
            },
            "format": {"type": "short_video_essay", "duration_sec": self.target_duration_sec},
            "sections": [s for s in self.sections],
            "metadata": {
                "project_id": self.project_id,
                "movie_title": self.movie_title,
                "grounding_contract_path": self.grounding_contract_path,
                "created_at": self.created_at,
            },
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GroundedScript":
        script = GroundedScript()
        script.concept_title = d.get("concept", {}).get("title", "")
        script.concept_hook = d.get("concept", {}).get("hook", "")
        script.thesis = d.get("concept", {}).get("thesis", "")
        script.why_interesting = d.get("concept", {}).get("why_interesting", "")
        script.target_duration_sec = d.get("format", {}).get("duration_sec", 90)
        script.sections = d.get("sections", [])
        meta = d.get("metadata", {})
        script.project_id = meta.get("project_id", "")
        script.movie_title = meta.get("movie_title", "")
        script.grounding_contract_path = meta.get("grounding_contract_path", "")
        script.created_at = meta.get("created_at", "")
        return script
    
    def add_section(
        self,
        section_id: str,
        section_type: str,
        narration: str = "",
        evidence: List[Dict[str, Any]] = None,
    ):
        """Add a section to the script."""
        self.sections.append({
            "section_id": section_id,
            "type": section_type,
            "narration": narration,
            "evidence": evidence or [],
        })
    
    def validate_against_evidence(self, contract: Dict[str, Any]) -> Dict[str, Any]:
        """Validate script against the grounding contract."""
        # Check that all scene references exist in supporting_scenes
        supporting_scenes = {s["scene_id"] for s in contract.get("supporting_scenes", [])}
        evidence_refs = {e["value"] for e in contract.get("evidence_refs", [])}
        
        errors = []
        warnings = []
        
        for section in self.sections:
            for ev in section.get("evidence", []):
                scene_id = ev.get("scene_id")
                if scene_id and scene_id not in [s["scene_id"] for s in contract.get("supporting_scenes", [])]:
                    return {
                        "valid": False,
                        "errors": [f"Section {section.get('section_id')}: scene_id '{ev['scene_id']}' not in supporting scenes"],
                        "warnings": []
                    }
            
            # Check timestamps are within scene bounds
            # (would need scene facts for full validation)
            
        return {"valid": True, "errors": [], "warnings": []}
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @staticmethod
    def from_json(json_str: str) -> "GroundedScript":
        return GroundedScript.from_dict(json.loads(json_str))
    
    @staticmethod
    def load(path: Path) -> "GroundedScript":
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return GroundedScript.from_dict(data)
    
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())


# --- Script Generator -------------------------------------------------------

class GroundedScriptGenerator:
    """Generates a grounded script from a V4 grounding contract."""
    
    def __init__(self, scene_facts=None):
        """Initialize with optional SceneFacts for validation."""
        self.scene_facts = scene_facts
    
    def generate(self, contract: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a grounded script from a V4 grounding contract.
        
        Args:
            contract: The V4 grounding contract (from grounding_contract.py)
            
        Returns:
            A structured grounded script dictionary.
        """
        # Extract core concept info
        concept = contract.get("concept", {})
        format_info = contract.get("format", {})
        
        # Calculate the actual available duration from supporting scenes
        available_duration = self._calculate_available_duration(contract)
        
        # Build the script structure
        script = {
            "concept": {
                "title": contract.get("concept", {}).get("title", ""),
                "hook": contract.get("concept", {}).get("hook", ""),
                "thesis": contract.get("concept", {}).get("thesis", ""),
                "why_interesting": contract.get("concept", {}).get("why_interesting", ""),
            },
            "format": {
                "type": contract.get("format", {}).get("type", "short_video_essay"),
                "duration_sec": int(available_duration),  # Use actual clip duration
            },
            "sections": self._build_sections(contract),
            "metadata": {
                "project_id": contract.get("project_id", ""),
                "movie_title": contract.get("concept", {}).get("title", ""),
                "grounding_contract_path": "",
                "created_at": self._now_iso(),
            }
        }
        
        return {
            "concept": script["concept"],
            "format": script["format"],
            "sections": script["sections"],
            "metadata": script["metadata"],
        }
    
    def _build_sections(self, contract: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build script sections from the grounded contract."""
        sections = []
        
        # Extract evidence from contract
        evidence_refs = contract.get("evidence_refs", [])
        supporting_scenes = contract.get("supporting_scenes", [])
        visual_motifs = contract.get("visual_motifs", [])
        character_focus = contract.get("character_focus", [])
        
        sections = []
        
        # HOOK - Opening hook
        hook_section = {
            "section_id": "hook",
            "type": "hook",
            "narration": "",
            "evidence": []
        }
        sections.append(hook_section)
        
        # SETUP - Establish the world/context
        setup_evidence = []
        for ev in contract.get("evidence_refs", []):
            if ev.get("kind") in ("scene", "location", "character"):
                ev_dict = dict(ev)
                ev_dict["purpose"] = "establish_context"
                sections.append({"section_id": f"setup_{len(sections)+1}", "type": "setup", "narration": "", "evidence": [ev_dict]})
        
        # CLAIM - The central thesis
        claim_evidence = []
        for ev in contract.get("evidence_refs", []):
            if ev.get("kind") in ("object", "character", "action", "location"):
                ev_dict = dict(ev)
                ev_dict["purpose"] = "support_claim"
                sections.append({"section_id": f"claim_{len(sections)+1}", "type": "claim", "narration": "", "evidence": [ev_dict]})
        
        return sections
    
    def _calculate_available_duration(self, contract: Dict[str, Any]) -> float:
        """Calculate the total available duration from supporting scenes."""
        supporting_scenes = contract.get("supporting_scenes", [])
        total_duration = 0.0
        for scene in contract.get("supporting_scenes", []):
            start = scene.get("start_sec")
            end = scene.get("end_sec")
            if start is not None and end is not None:
                total_duration += max(0.0, end - start)
        return total_duration if total_duration > 0 else 90.0  # fallback to 90s if no scenes

    def _now_iso(self) -> str:
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"


# --- Validation Utilities -----------------------------------------------------

def validate_script_against_contract(
    script: Dict[str, Any],
    contract: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate a grounded script against its grounding contract."""
    errors = []
    warnings = []
    
    # Check that all scene references exist in supporting_scenes
    contract_scenes = {s["scene_id"] for s in contract.get("supporting_scenes", [])}
    evidence_values = {e["value"] for e in contract.get("evidence_refs", [])}
    
    # Check each section's evidence
    for section in contract.get("sections", []):
        for ev in section.get("evidence", []):
            scene_id = ev.get("scene_id")
            if scene_id and scene_id not in [s.get("scene_id") for s in contract.get("supporting_scenes", [])]:
                return {
                    "valid": False,
                    "errors": [f"Section {section.get('section_id')}: scene_id '{ev.get('scene_id')}' not in supporting scenes"],
                    "warnings": []
                }
    
    return {"valid": True, "errors": [], "warnings": []}


# --- Main Entry Point ---------------------------------------------------------

def generate_grounded_script(
    contract_path: Path,
    output_path: Path,
    scene_facts=None,
) -> Path:
    """Main entry point: generate grounded script from contract file.
    
    Args:
        contract_path: Path to grounding_contract.json
        output_path: Where to save the grounded script JSON
        scene_facts: Optional SceneFacts for validation
        
    Returns:
        Path to the generated script JSON file.
    """
    import json
    from pathlib import Path
    
    # Load contract
    with open(contract_path, 'r', encoding='utf-8') as f:
        contract = json.load(f)
    
    # Generate script
    generator = GroundedScriptGenerator()
    script = generator.generate({"contract": contract})  # adapter for now
    
    # TODO: Pass contract properly to generator
    # For now, adjust to match the actual contract structure
    pass