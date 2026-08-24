with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\evidence.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Replace lines 706-766 (0-indexed) with clean method
# Line 707 (1-indexed) = "def plan_grounding(" -> index 706
# Line 767 = "def _legacy_prose_audit(" -> index 766

new_method = [
    '    def plan_grounding(\n',
    '        self,\n',
    '        plan: Optional[Dict[str, Any]],\n',
    '        evidence_scene_ids: Optional[List[str]] = None,\n',
    '        min_coverage: float = 0.55,\n',
    '    ) -> Dict[str, Any]:\n',
    '        """V4 Plan validation: runs both FactValidator and EditorialSchemaValidator.\n',
    '\n',
    '        Returns combined audit result with:\n',
    '        - fact_validation: {valid, errors, warnings}\n',
    '        - editorial_validation: {valid, errors, warnings}\n',
    '        - overall_valid: bool\n',
    '        - legacy_prose_audit: (optional) for backward compat with free-text fields\n',
    '        - (legacy keys for backward compat): sufficient, invented_terms, elsewhere_terms, grounded_terms, coverage\n',
    '        """\n',
    '        # Default empty result\n',
    '        result = {\n',
    '            "fact_validation": {"valid": False, "errors": ["no editorial_plan provided"], "warnings": []},\n',
    '            "editorial_validation": {"valid": False, "errors": ["no editorial_plan provided"], "warnings": []},\n',
    '            "overall_valid": False,\n',
    '            "legacy_prose_audit": None,\n',
    '            # Legacy keys for backward compatibility\n',
    '            "sufficient": False,\n',
    '            "invented_terms": [],\n',
    '            "elsewhere_terms": [],\n',
    '            "grounded_terms": [],\n',
    '            "coverage": 0.0,\n',
    '            "min_coverage": 0.0,\n',
    '        }\n',
    '\n',
    '        plan = plan or {}\n',
    '        # Accept both editorial_plan (V4 structured) and editorial_direction (legacy) as structured plan\n',
    '        editorial_plan = plan.get("editorial_plan") or plan.get("editorial_direction")\n',
    '\n',
    '        # Check if we have a properly structured V4 editorial_plan with visual.scene_id\n',
    '        has_structured_plan = (\n',
    '            isinstance(editorial_plan, dict)\n',
    '            and isinstance(editorial_plan.get("visual"), dict)\n',
    '            and editorial_plan["visual"].get("scene_id")\n',
    '        )\n',
    '\n',
    '        # Run V4 validators only if structured plan with required visual.scene_id is present\n',
    '        if has_structured_plan:\n',
    '            fact_val = self._validate_factual_plan(editorial_plan, evidence_scene_ids)\n',
    '            editorial_val = self._validate_editorial_schema(editorial_plan)\n',
    '            result["fact_validation"] = fact_val\n',
    '            result["editorial_validation"] = editorial_val\n',
    '            result["overall_valid"] = fact_val["valid"] and editorial_val["valid"]\n',
    '        else:\n',
    '            # No structured plan - use legacy audit results for validation\n',
    '            result["fact_validation"] = {"valid": True, "errors": [], "warnings": []}\n',
    '            result["editorial_validation"] = {"valid": True, "errors": [], "warnings": []}\n',
    '            result["overall_valid"] = False  # Will be set by legacy audit if applicable\n',
    '\n',
    '        return result\n',
]

# Replace lines 706-766 (0-indexed: 706-766) with new method
# Line 707 (1-indexed) = "def plan_grounding(" -> index 706
# Line 767 = "def _legacy_prose_audit(" -> index 766

with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\evidence.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = lines[:706] + new_method + lines[766:]

with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\evidence.py', 'w', encoding='utf-8') as f:
    f.writelines(lines[:706] + new_method + lines[766:])

print('Rewrote plan_grounding method')