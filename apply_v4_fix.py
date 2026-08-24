#!/usr/bin/env python3
"""Apply V4 changes to grounded.py"""

with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\grounded.py', 'r') as f:
    lines = f.read().split('\n')

v4_build_plan = '''    def _build_plan(
        self,
        movie_metadata: Dict[str, Any],
        selected: Dict[str, Any],
        scene_facts: SceneFacts,
        analyzer: EvidenceAnalyzer,
        duration_sec: int,
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Build the scene-aware plan (deterministic concept + structured editorial plan).

        Returns ``(plan, None)`` on success, or ``(None, rejection)`` when the
        plan's editorial_plan cannot be validated after the bounded corrective
        retry. A rejected plan is NEVER emitted downstream -- the strict plan gate
        mirrors the concept gate's honest FAIL.
        """
        evidence_strategy = analyzer.build_evidence_strategy(selected)
        plan_ctx = self.context_builder.build_plan_context(
            selected, scene_facts, evidence_strategy.get("scene_ids", [])
        )
        scene_ids = evidence_strategy.get("scene_ids", [])

        plan = self._plan_once(plan_ctx, duration_sec)
        # V4: Validate structured editorial_plan + legacy prose
        audit = analyzer.plan_grounding(
            plan.get("editorial_direction"), scene_ids,
        )

        def _extract_corrections(audit_result: Dict[str, Any]) -> List[str]:
            """Extract correction terms from V4 audit for corrective retry."""
            corrections: List[str] = []
            # Fact validation errors
            fact_val = audit.get("fact_validation", {})
            corrections.extend(fact_val.get("errors", []))
            # Editorial validation errors
            editorial_val = audit.get("editorial_validation", {})
            corrections.extend(editorial_val.get("errors", []))
            # Legacy prose invented terms
            legacy = audit.get("legacy_prose_audit")
            if legacy:
                corrections.extend(legacy.get("invented_terms", []))
            return corrections

        # First attempt: if not overall_valid, do ONE corrective retry
        if not audit.get("overall_valid", False):
            corrections = _extract_corrections(audit)
            if corrections:
                plan = self._plan_once(
                    plan_ctx, duration_sec, grounding_warnings=corrections,
                )
                audit = analyzer.plan_grounding(
                    plan.get("editorial_direction"), scene_ids,
                )

        # Final gate
        if not audit.get("overall_valid", False):
            rejection = {
                "reason": "plan editorial_plan not valid after "
                          "bounded corrective retry (strict plan gate)",
                "audit": audit,
                "evidence_scene_ids": scene_ids,
            }
            return None, rejection

        plan["grounding_audit"] = audit
        plan.setdefault("format", {"type": "short_video_essay",
                                    "duration_sec": duration_sec})
        plan.setdefault("editorial_direction", {})
        # The plan is FOR the selected concept: its concept block is fixed and
        # deterministic (never re-imagined by the model). The model only
        # contributes format + editorial_plan + editorial_direction (prose fallback),
        # grounded in the evidence scenes below.
        plan["concept"] = {
            "title": selected.get("title", ""),
            "hook": selected.get("hook", ""),
            "thesis": selected.get("thesis", ""),
        }
        # The evidence_strategy is deterministic, not model-invented.
        plan["evidence_strategy"] = evidence_strategy
        return plan, None'''

# Read the file
with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\grounded.py', 'r') as f:
    lines = f.read().split('\n')

# Replace lines 318-386 (0-indexed) with V4 version
# Line 319 in 1-indexed = index 318
# Line 386 in 1-indexed = index 386
# We need to find the exact boundaries

# Find the start of _build_plan
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if line.strip().startswith('def _build_plan('):
        start_idx = i
    if line.strip().startswith('def _plan_once('):
        end_idx = i
        break

print(f'Found _build_plan at line {start_idx+1}, _plan_once at line {end_idx+1}')

# Replace lines[start_idx:end_idx] with V4 version
v4_lines = v4_build_plan.split('\n')
new_lines = lines[:start_idx] + v4_build_plan.split('\n') + lines[end_idx:]

with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\grounded.py', 'w') as f:
    f.write('\n'.join(new_lines))

print('Replaced _build_plan with V4 version')