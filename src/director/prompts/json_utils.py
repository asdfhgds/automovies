"""Utilities for JSON parsing, validation, and schema handling."""
import json
import logging
import re
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def extract_json(text: str, expected_key: Optional[str] = None) -> Optional[Dict]:
    """
    Extract JSON from LLM response, handling multiple formats.
    
    Handles:
    - Direct JSON
    - Fenced JSON (```json ... ```)
    - JSON within text
    
    Args:
        text: Raw LLM output
        expected_key: Expected top-level key (optional hint)
        
    Returns:
        Parsed JSON dict or None
    """
    text = text.strip()

    # Try 1: Direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try 2: Fenced JSON (```json ... ```)
    json_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try 3: Look for JSON object within text
    # Find first { and last }
    start = text.find("{")
    if start != -1:
        # Find matching closing brace
        brace_count = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

        if end != -1:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

    logger.warning(f"Failed to extract JSON from response (first 100 chars): {text[:100]}")
    return None


def validate_concepts_schema(concepts: List[Dict]) -> tuple[bool, str]:
    """
    Validate concept list against expected schema.
    
    Args:
        concepts: List of concept dicts
        
    Returns:
        (is_valid, error_message)
    """
    required_fields = ["title", "hook", "thesis", "why_interesting"]

    if not isinstance(concepts, list):
        return False, "Concepts must be a list"

    if len(concepts) == 0:
        return False, "No concepts provided"

    for i, concept in enumerate(concepts):
        if not isinstance(concept, dict):
            return False, f"Concept {i} is not a dict"

        for field in required_fields:
            if field not in concept:
                return False, f"Concept {i} missing field: {field}"
            if not concept[field] or (isinstance(concept[field], str) and not concept[field].strip()):
                return False, f"Concept {i} has empty field: {field}"

    return True, ""


def validate_production_plan_schema(plan: Dict) -> tuple[bool, str]:
    """
    Validate production plan against expected schema.
    
    Args:
        plan: Production plan dict
        
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(plan, dict):
        return False, "Plan must be a dict"

    if "structure" not in plan:
        return False, "Plan missing 'structure' field"

    structure = plan["structure"]
    if not isinstance(structure, list) or len(structure) == 0:
        return False, "Structure must be non-empty list"

    for i, section in enumerate(structure):
        if not isinstance(section, dict):
            return False, f"Section {i} is not a dict"
        if "section" not in section or "duration_sec" not in section:
            return False, f"Section {i} missing required fields"

    return True, ""


def check_concept_diversity(concepts: List[Dict]) -> tuple[bool, str]:
    """
    Check if concepts are diverse (not just variations of same idea).
    
    Simple heuristic: theses should not all be identical.
    
    Args:
        concepts: List of concept dicts
        
    Returns:
        (is_diverse, message)
    """
    if len(concepts) < 2:
        return True, "Too few concepts to check diversity"

    theses = [c.get("thesis", "") for c in concepts]

    # Check for exact duplicates
    if len(set(theses)) < len(theses):
        return False, "Some concepts have identical theses"

    # Check for high similarity (very simplistic: just check length variance)
    lengths = [len(t) for t in theses]
    if max(lengths) > 0 and min(lengths) > 0:
        variance = max(lengths) / min(lengths)
        if variance < 1.2:
            logger.warning(f"Concepts may lack diversity (length variance: {variance:.2f})")
            return False, "Concepts appear similar in scope"

    return True, "Concepts are diverse"


def extract_and_validate_concepts(
    response: str,
) -> tuple[Optional[List[Dict]], str]:
    """
    Extract concepts from LLM response and validate.
    
    Args:
        response: Raw LLM output
        
    Returns:
        (concepts_list, error_message) - one will be None/empty if error
    """
    # Extract JSON
    result = extract_json(response, "concepts")
    if not result:
        return None, "Failed to extract JSON"

    concepts = result.get("concepts", [])

    # Validate schema
    is_valid, error = validate_concepts_schema(concepts)
    if not is_valid:
        return None, f"Schema validation failed: {error}"

    # Check diversity
    is_diverse, div_msg = check_concept_diversity(concepts)
    if not is_diverse:
        logger.warning(f"Diversity check: {div_msg}")

    return concepts, ""


def extract_and_validate_production_plan(
    response: str,
) -> tuple[Optional[Dict], str]:
    """
    Extract production plan from LLM response and validate.
    
    Args:
        response: Raw LLM output
        
    Returns:
        (plan_dict, error_message) - one will be None/empty if error
    """
    # Extract JSON
    plan = extract_json(response)
    if not plan:
        return None, "Failed to extract JSON"

    # Validate schema
    is_valid, error = validate_production_plan_schema(plan)
    if not is_valid:
        return None, f"Schema validation failed: {error}"

    return plan, ""
