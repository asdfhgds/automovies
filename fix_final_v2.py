import re

with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\evidence.py', 'r', encoding='utf-8') as f:
    source = f.read()

# Normalize line endings
source = source.replace('\r\n', '\n').replace('\r', '\n')

# Pattern to match the plan_grounding method up to _legacy_prose_audit
pattern = r'(    def plan_grounding\(.*?)(?=\n    def _legacy_prose_audit\()'

# Replacement method with clean structure
replacement = '''    def plan_grounding(
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
        # Accept both editorial_plan (V4 structured) and editorial_direction (legacy) as structured plan
        editorial_plan = plan.get("editorial_plan") or plan.get("editorial_direction")

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
            # No structured plan - use legacy audit results for validation
            result["fact_validation"] = {"valid": True, "errors": [], "warnings": []}
            result["editorial_validation"] = {"valid": True, "errors": [], "warnings": []}
            result["overall_valid"] = False  # Will be set by legacy audit if applicable

        return result'''

# Apply the replacement using regex
new_source = re.sub(
    r'(    def plan_grounding\(.*?)(?=\n    def _legacy_prose_audit\()',
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
        # Accept both editorial_plan (V4 structured) and editorial_direction (legacy) as structured plan
        editorial_plan = plan.get("editorial_plan") or plan.get("editorial_direction")

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
            # No structured plan - use legacy audit results for validation
            result["fact_validation"] = {"valid": True, "errors": [], "warnings": []}
            result["editorial_validation"] = {"valid": True, "errors": [], "warnings": []}
            result["overall_valid"] = False  # Will be set by legacy audit if applicable

        return result''',
    source,
    flags=re.DOTALL
)

with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\evidence.py', 'w', encoding='utf-8') as f:
    f.write(new_source)

print('Fixed')