with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\evidence.py', 'r', encoding='utf-8') as f:
    source = f.read()

# Normalize line endings
source = source.replace('\r\n', '\n').replace('\r', '\n')

# Find the plan_grounding method and replace it entirely
import re

# Pattern to match the entire plan_grounding method
pattern = r'(    def plan_grounding\(.*?)(?=\n    def |\Z)'

def replace_method(match):
    method_text = match.group(1)
    # Rebuild the method with proper structure
    new_method = '''    def plan_grounding(
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
        editorial_plan = plan.get("editorial_plan")

        # Run V4 validators if structured plan present
        if isinstance(editorial_plan, dict):
            fact_val = self._validate_factual_plan(editorial_plan, evidence_scene_ids)
            editorial_val = self._validate_editorial_schema(editorial_plan)
            result["fact_validation"] = fact_val
            result["editorial_validation"] = editorial_val
            result["overall_valid"] = fact_val["valid"] and editorial_val["valid"]

        # Legacy: audit free-text editorial_direction prose (backward compat)
        # Only runs if no structured plan, or as supplementary info
        editorial_direction = plan.get("editorial_direction", {})
        if isinstance(editorial_direction, dict) and ("pacing" in editorial_direction or "visual_style" in editorial_direction):
            legacy_audit = self._legacy_prose_audit(editorial_direction, evidence_scene_ids, min_coverage)
            result["legacy_prose_audit"] = legacy_audit
            # Merge legacy keys for backward compatibility
            result["sufficient"] = legacy_audit["sufficient"]
            result["invented_terms"] = legacy_audit["invented_terms"]
            result["elsewhere_terms"] = legacy_audit["elsewhere_terms"]
            result["grounded_terms"] = legacy_audit["grounded_terms"]
            result["coverage"] = legacy_audit["coverage"]
            result["min_coverage"] = legacy_audit["min_coverage"]
            # If no structured plan, use legacy for overall_valid
            if not isinstance(editorial_plan, dict):
                result["overall_valid"] = legacy_audit["sufficient"]

        return result'''

    return match.group(0)[:match.start(1)] + replacement + source[match.end():]

# Apply the replacement
new_source = re.sub(
    r'(    def plan_grounding\(.*?)(?=\n    def |\Z)',
    '''    def plan_grounding(
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
        editorial_plan = plan.get("editorial_plan")

        # Run V4 validators if structured plan present
        if isinstance(editorial_plan, dict):
            fact_val = self._validate_factual_plan(editorial_plan, evidence_scene_ids)
            editorial_val = self._validate_editorial_schema(editorial_plan)
            result["fact_validation"] = fact_val
            result["editorial_validation"] = editorial_val
            result["overall_valid"] = fact_val["valid"] and editorial_val["valid"]

        # Legacy: audit free-text editorial_direction prose (backward compat)
        # Only runs if no structured plan, or as supplementary info
        editorial_direction = plan.get("editorial_direction", {})
        if isinstance(editorial_direction, dict) and ("pacing" in editorial_direction or "visual_style" in editorial_direction):
            legacy_audit = self._legacy_prose_audit(editorial_direction, evidence_scene_ids, min_coverage)
            result["legacy_prose_audit"] = legacy_audit
            # Merge legacy keys for backward compatibility
            result["sufficient"] = legacy_audit["sufficient"]
            result["invented_terms"] = legacy_audit["invented_terms"]
            result["elsewhere_terms"] = legacy_audit["elsewhere_terms"]
            result["grounded_terms"] = legacy_audit["grounded_terms"]
            result["coverage"] = legacy_audit["coverage"]
            result["min_coverage"] = legacy_audit["min_coverage"]
            # If no structured plan, use legacy for overall_valid
            if not isinstance(editorial_plan, dict):
                result["overall_valid"] = legacy_audit["sufficient"]

        return result''',
    source,
    flags=re.DOTALL
)

# Apply the fix
new_source = re.sub(
    r'(    def plan_grounding\(.*?)(?=\n    def |\Z)',
    '''    def plan_grounding(
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
        editorial_plan = plan.get("editorial_plan")

        # Run V4 validators if structured plan present
        if isinstance(editorial_plan, dict):
            fact_val = self._validate_factual_plan(editorial_plan, evidence_scene_ids)
            editorial_val = self._validate_editorial_schema(editorial_plan)
            result["fact_validation"] = fact_val
            result["editorial_validation"] = editorial_val
            result["overall_valid"] = fact_val["valid"] and editorial_val["valid"]

        # Legacy: audit free-text editorial_direction prose (backward compat)
        # Only runs if no structured plan, or as supplementary info
        editorial_direction = plan.get("editorial_direction", {})
        if isinstance(editorial_direction, dict) and ("pacing" in editorial_direction or "visual_style" in editorial_direction):
            legacy_audit = self._legacy_prose_audit(editorial_direction, evidence_scene_ids, min_coverage)
            result["legacy_prose_audit"] = legacy_audit
            # Merge legacy keys for backward compatibility
            result["sufficient"] = legacy_audit["sufficient"]
            result["invented_terms"] = legacy_audit["invented_terms"]
            result["elsewhere_terms"] = legacy_audit["elsewhere_terms"]
            result["grounded_terms"] = legacy_audit["grounded_terms"]
            result["coverage"] = legacy_audit["coverage"]
            result["min_coverage"] = legacy_audit["min_coverage"]
            # If no structured plan, use legacy for overall_valid
            if not isinstance(editorial_plan, dict):
                result["overall_valid"] = legacy_audit["sufficient"]

        return result''',
    source,
    flags=re.DOTALL
)

with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\evidence.py', 'w', encoding='utf-8') as f:
    f.write(new_source)

print('Fixed plan_grounding method')