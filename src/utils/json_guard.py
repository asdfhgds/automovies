"""Detect when a small LLM echoes the prompt's example placeholder text.

Small instruct models often mirror the JSON example that was shown to them
instead of writing original content. These helpers flag the known placeholder
strings (both the old realistic-looking ones and the ALL-CAPS markers used in
current prompts) so a generation can be retried instead of silently accepted.
"""

PLACEHOLDER_MARKERS = (
    # Old realistic-looking example values that models would copy verbatim.
    "specific concept title",
    "core argument about the film",
    "engaging question or statement",
    "why this interpretation",
    "how to visualize this concept",
    "description of visual approach",
    "description of music approach",
    "description of editing approach",
    "narration...",
    "subscribe for more",
    "type1",
    "type2",
    "analytical_philosophical",
    # ALL-CAPS markers used in the current prompts (matched case-insensitively).
    "your_original_title",
    "your_original_opening",
    "your_specific_evidence",
    "why_this_angle",
    "your_visual_approach",
    "your_section_name",
    "your_purpose_here",
    "your_scene_type",
    "your_visual_strategy",
    "your_music_strategy",
    "your_editing_strategy",
    "your_narration_text",
    "your_voiceover",
    "your_section_id",
)

# Values that are placeholders only when a whole short string equals them.
PLACEHOLDER_EXACT = {
    "narration",
    "narration...",
    "...",
    "your narration here",
    "write the narration here",
    "text",
}


def contains_placeholder(value, markers=PLACEHOLDER_MARKERS) -> bool:
    """True if ``value`` contains any known placeholder substring (case-insensitive)."""
    if value is None:
        return False
    low = str(value).lower()
    return any(marker in low for marker in markers)


def is_exact_placeholder(value) -> bool:
    """True if a short string is exactly a placeholder (e.g. 'Narration...')."""
    if value is None:
        return False
    low = str(value).strip().lower()
    return low in PLACEHOLDER_EXACT
