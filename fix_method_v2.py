with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\evidence.py', 'r') as f:
    lines = f.readlines()

# Replace lines 706-766 (0-indexed: 706-766) with clean method
# Line 706 is "def plan_grounding(" (0-indexed: 706)
# Line 767 is "def _legacy_prose_audit("

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
    '        editorial_plan = plan.get("editorial_plan")\n',
    '\n',
    '        # Run V4 validators if structured plan present\n',
    '        if isinstance(editorial_plan, dict):\n',
    '            fact_val = self._validate_factual_plan(editorial_plan, evidence_scene_ids)\n',
    '            editorial_val = self._validate_editorial_schema(editorial_plan)\n',
    '            result["fact_validation"] = fact_val\n',
    '            result["editorial_validation"] = editorial_val\n',
    '            result["overall_valid"] = fact_val["valid"] and editorial_val["valid"]\n',
    '\n',
    '        # Legacy: audit free-text editorial_direction prose (backward compat)\n',
    '        # Only runs if no structured plan, or as supplementary info\n',
    '        editorial_direction = plan.get("editorial_direction", {})\n',
    '        if isinstance(editorial_direction, dict) and ("pacing" in editorial_direction or "visual_style" in editorial_direction):\n',
    '            legacy_audit = self._legacy_prose_audit(editorial_direction, evidence_scene_ids, min_coverage)\n',
    '            result["legacy_prose_audit"] = legacy_audit\n',
    '            # Merge legacy keys for backward compatibility\n',
    '            result["sufficient"] = legacy_audit["sufficient"]\n',
    '            result["invented_terms"] = legacy_audit["invented_terms"]\n',
    '            result["elsewhere_terms"] = legacy_audit["elsewhere_terms"]\n',
    '            result["grounded_terms"] = legacy_audit["grounded_terms"]\n',
    '            result["coverage"] = legacy_audit["coverage"]\n',
    '            result["min_coverage"] = legacy_audit["min_coverage"]\n',
    '            # If no structured plan, use legacy for overall_valid\n',
    '            if not isinstance(editorial_plan, dict):\n',
    '                result["overall_valid"] = legacy_audit["sufficient"]\n',
    '\n',
    '        return result\n',
]

new_lines = lines[:706] + new_method + lines[767:]

with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\evidence.py', 'w') as f:
    f.writelines(lines[:706] + new_method + lines[767:])

print('Rewrote plan_grounding method')