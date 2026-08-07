# Creative Director Integration Guide

## Overview

The creative director framework is a **mock-first, LLM-ready system** for generating creative video concepts. It's currently using a deterministic mock provider for testing. This guide explains how to integrate real LLM providers and use the system.

## Quick Start

### Run Pipeline with Creative Director (Mock)

```bash
export CREATIVE_DIRECTOR_ENABLED=true
python src/main.py init --title "My Movie" --source path/to/video.mp4
python src/main.py run --project-id <project-id>
```

This will:
1. Transcribe the video using Whisper/WhisperX
2. Detect scenes using PySceneDetect
3. **Generate 3-5 creative concepts using MockLLMProvider**
4. Critique and select the strongest concept
5. Rank scenes based on the creative thesis
6. Extract clips and produce final render

### Run Pipeline WITHOUT Creative Director (Deterministic Fallback)

```bash
export CREATIVE_DIRECTOR_ENABLED=false
# or omit the env var (defaults to false)
python src/main.py run --project-id <project-id>
```

This uses the original deterministic director planner.

## Architecture

### Component Map

```
src/director/
├── planner.py              # Entry point; routes to creative or deterministic
├── creative_director.py    # Main orchestrator
├── memory.py              # Concept persistence (JSONL)
├── critic.py              # Concept evaluation (6 dimensions)
└── providers/
    ├── base.py            # Abstract LLMProvider interface
    ├── mock_llm.py        # Current: deterministic mock
    ├── anthropic_provider.py  # TODO: Real Anthropic Claude
    ├── openai_provider.py     # TODO: Real OpenAI GPT-4
    └── replicate_provider.py  # TODO: Replicate (open models)
```

### Data Flow

```
scene_index.json + transcript.json + movie_metadata
    ↓
CreativeDirector.develop_production_plan()
    ├─ CreativeMemory.get_concepts_summary() → retrieve previous concepts
    ├─ LLMProvider.generate_concepts() → 3-5 concepts (mock or real LLM)
    ├─ ConceptCritic.critique() → score each on 6 dimensions
    ├─ Select best concept by overall score
    ├─ LLMProvider.generate_production_plan() → structure & timing
    └─ CreativeMemory.add_concept() → store for future reference
    ↓
director_plan.json
```

## MockLLMProvider (Current)

### What It Does

Generates deterministic, philosophically-grounded concepts:

1. **Thematic Analysis** — explores central themes and motifs
2. **Character Psychology** — examines internal motivations and conflicts
3. **Narrative Structure** — analyzes plot progression and causality
4. **Visual Metaphor** — interprets symbolic visual language
5. **Metanarrative** — examines how the film comments on itself

Each concept includes:
- `title` — compelling concept name
- `thesis` — core analytical statement
- `hook` — audience hook/pitch
- `why_interesting` — what makes it worth exploring
- `tone` — mood/style recommendations
- `structure` — section breakdown with durations
- `visual_strategy` — visual approach/style

### Example Output

```json
{
  "title": "The Architecture of Desire: Psychology of Longing",
  "thesis": "The film explores how characters construct psychological defenses against vulnerability, using repetitive patterns and symbolic objects to impose meaning on chaos.",
  "hook": "A deep dive into the hidden architectures of human longing and self-protection.",
  "tone": "psychological_intimate",
  "why_interesting": "Reveals unconscious patterns that drive character decisions and plot.",
  "estimated_duration_sec": 75,
  "structure": [
    {"section": "hook", "duration_sec": 7},
    {"section": "setup", "duration_sec": 11},
    {"section": "analysis", "duration_sec": 41},
    {"section": "conclusion", "duration_sec": 15}
  ]
}
```

### Testing with Mock

```bash
# Run unit tests (no API calls, fast)
pytest tests/test_creative_director.py -v

# Run E2E tests (full pipeline with mock)
pytest tests/test_creative_director_e2e.py -v

# Run full suite
pytest -q
# Expected: 21 passed, 1 skipped in ~144 seconds
```

## Integrating a Real LLM Provider

### Step 1: Choose Your Provider

| Provider | Cost | Speed | Quality | Setup |
|----------|------|-------|---------|-------|
| Anthropic Claude | $$$ | Medium | Excellent | API key via AWS Bedrock or direct |
| OpenAI GPT-4 | $$$ | Slow | Very Good | OpenAI API key |
| Replicate | $ | Medium | Good | Replicate API key + model ID |
| Ollama | Free | Medium | Fair | Local install + model download |

**Recommendation: Anthropic Claude** (best reasoning for creative analysis)

### Step 2: Create Provider Implementation

Example: `src/director/providers/anthropic_provider.py`

```python
from .base import LLMProvider
from anthropic import Anthropic
import json
from typing import Dict, List, Any, Optional

class AnthropicProvider(LLMProvider):
    """Claude-based creative director via Anthropic API."""
    
    def __init__(self, model: str = "claude-3-sonnet-20240229"):
        """Initialize Anthropic client."""
        self.client = Anthropic()
        self.model = model
    
    def generate_concepts(
        self,
        movie_metadata: Dict[str, Any],
        scene_index: List[Dict[str, Any]],
        transcript: Dict[str, Any],
        creative_memory: str = "",
        user_topic: Optional[str] = None,
        num_concepts: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate creative concepts using Claude."""
        
        # Build prompt from metadata/scenes/transcript
        scene_summary = "\n".join([
            f"- Scene {s['scene_id']} ({s['start_sec']:.1f}s-{s['end_sec']:.1f}s): {s.get('transcript', '')[:100]}"
            for s in scene_index[:5]
        ])
        
        prompt = f"""You are a brilliant creative director for video essays and documentary analysis.

Movie: {movie_metadata.get('title', 'Untitled')}
Duration: {movie_metadata.get('duration_sec', 'unknown')} seconds

Scenes:
{scene_summary}

Transcript (excerpt):
{transcript.get('segments', [{}])[0].get('text', '')[:300]}

{f'Previous concepts (avoid repetition): {creative_memory}' if creative_memory else ''}
{f'User topic: {user_topic}' if user_topic else ''}

Generate {num_concepts} diverse, specific, analytical concepts for a video essay about this content.
Each concept should explore a different angle (theme, psychology, narrative structure, visual metaphor, etc).
Return as JSON array with objects containing: title, thesis, hook, why_interesting, tone, why_specific_to_this_film, structure, visual_strategy, estimated_duration_sec.
"""
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse Claude response
        response_text = message.content[0].text
        
        # Extract JSON from response (Claude might add text before/after)
        try:
            # Try to parse as-is
            concepts = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                concepts = json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse Claude response: {response_text}")
        
        # Validate and clean
        if not isinstance(concepts, list):
            concepts = [concepts]
        
        return concepts[:num_concepts]
    
    def refine_concept(
        self, concept: Dict[str, Any], feedback: str
    ) -> Dict[str, Any]:
        """Refine a concept based on feedback."""
        # Similar pattern: build prompt, call Claude, parse response
        pass
    
    def generate_production_plan(
        self,
        concept: Dict[str, Any],
        scene_index: List[Dict[str, Any]],
        transcript: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate detailed production plan from concept."""
        # Similar pattern: build prompt, call Claude, parse response
        pass
```

### Step 3: Update Orchestrator to Use Real Provider

In `src/app/orchestrator.py`, line ~45, replace:

```python
# OLD:
provider = MockLLMProvider()

# NEW:
from director.providers.anthropic_provider import AnthropicProvider
provider = AnthropicProvider()
```

### Step 4: Set API Keys

For Anthropic:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python src/main.py run --project-id <id>
```

For OpenAI:
```bash
export OPENAI_API_KEY="sk-..."
python src/main.py run --project-id <id>
```

### Step 5: Test

```bash
# Run with real provider
CREATIVE_DIRECTOR_ENABLED=true python src/main.py init --title "Test with Real LLM" --source tests/fixtures/test_speech.mp4
CREATIVE_DIRECTOR_ENABLED=true python src/main.py run --project-id <id>

# Check generated concepts
cat data/<project-id>/memory/concepts.jsonl | python -m json.tool

# Check director plan
cat data/<project-id>/director_plan.json | python -m json.tool
```

### Step 6: Handle Errors Gracefully

Add retry logic and fallback:

```python
import time
from typing import Optional

class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-3-sonnet-20240229", max_retries: int = 3):
        self.client = Anthropic()
        self.model = model
        self.max_retries = max_retries
    
    def _call_with_retry(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call Claude with exponential backoff retry."""
        for attempt in range(self.max_retries):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # exponential backoff
                    print(f"LLM call failed (attempt {attempt+1}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"LLM failed after {self.max_retries} retries: {e}")
```

## Understanding the Critic

The `ConceptCritic` evaluates concepts on 6 dimensions (0.0-1.0 scale):

1. **Originality** — Is the concept unique and fresh?
2. **Thesis Strength** — Is the core claim compelling and specific?
3. **Evidence Strength** — Are supporting details grounded in the film?
4. **Visual Potential** — Can this be communicated through visual media?
5. **Audience Curiosity** — Will viewers want to watch this analysis?
6. **Feasibility** — Can this be produced with available resources?

Scoring is **deterministic heuristic-based** (no ML):
- Checks text length and keyword presence
- Detects vagueness patterns ("some", "certain", "aspects")
- Evaluates specificity to movie title/scenes
- Validates completeness of structure

This keeps criticism fast (<1ms per concept) and testable.

## Understanding the Memory

Concepts are stored in `data/<project-id>/memory/concepts.jsonl`:

```jsonl
{"timestamp": "2024-08-07T22:03:31", "title": "The Architect Within", "thesis": "...", "movie_title": "Test Movie", "themes": ["psychology", "architecture"]}
{"timestamp": "2024-08-07T22:04:12", "title": "Character as Metaphor", "thesis": "...", "movie_title": "Test Movie", "themes": ["symbolism", "identity"]}
```

**One JSON object per line** (JSONL format):
- Append-only for durability
- No parsing overhead
- Easy to query/analyze

The `CreativeMemory.get_concepts_summary()` retrieves recent concepts to **inform the LLM and avoid repetition**:

```python
memory = CreativeMemory(memory_dir)
summary = memory.get_concepts_summary(limit=5)
# Returns: "Previous concepts on this film: 1) The Architect (psychology), 2) Character as Metaphor (symbolism)"
# Pass to LLM to avoid generating similar ideas
```

## Testing Real Providers

When you implement a real provider, add integration tests:

```python
# tests/test_anthropic_provider.py
import pytest
from director.providers.anthropic_provider import AnthropicProvider

@pytest.mark.integration
@pytest.mark.requires_anthropic
def test_anthropic_generates_concepts(anthropic_api_key):
    """Test real Anthropic API (requires key)."""
    provider = AnthropicProvider()
    concepts = provider.generate_concepts(...)
    
    assert len(concepts) >= 1
    assert all("thesis" in c for c in concepts)
    assert all(len(c.get("thesis", "")) > 20 for c in concepts)
```

Mark with `@pytest.mark.requires_anthropic` to skip unless env var is set:

```bash
# Skip expensive tests by default
pytest tests/

# Run with real providers (requires API keys)
ANTHROPIC_API_KEY=... pytest -m requires_anthropic
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'anthropic'"

```bash
pip install anthropic
```

### Issue: "ANTHROPIC_API_KEY not found"

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
echo $ANTHROPIC_API_KEY  # verify
python src/main.py run --project-id <id>
```

### Issue: "JSON parse error" from LLM response

LLMs sometimes output text before/after JSON. Add extraction:

```python
import re
json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
if json_match:
    concepts = json.loads(json_match.group())
```

### Issue: LLM generating generic concepts

Try more specific prompts. Example for Claude:

```python
prompt = f"""You are an expert video essay creator. Generate SPECIFIC, ORIGINAL concepts that:
1) Reference specific scenes, dialogue, or visual moments from the film
2) Propose novel analytical angles not usually explored
3) Include concrete production suggestions (duration, structure, visual approach)
4) Are philosophically grounded, not superficial

Film: {title}
Scenes: {scenes_summary}

Generate {num_concepts} concepts. Each must be distinct and avoid generic phrases like 'explores themes of' or 'examines the role of'."""
```

## Next Steps After Real LLM Integration

1. **A/B Test Different Providers** — Compare Claude vs GPT-4 on same input
2. **Add Human Feedback Loop** — Critic feedback → human curation → regenerate
3. **Implement Caching** — Cache LLM responses to reduce API calls
4. **Add Cost Tracking** — Monitor API usage and costs
5. **Ensemble Voting** — Multiple providers generate concepts, vote on best
6. **Prompt Engineering** — Optimize prompts for quality/cost tradeoff

## References

- **Anthropic API Docs**: https://docs.anthropic.com/
- **OpenAI API Docs**: https://platform.openai.com/docs/
- **Replicate Docs**: https://replicate.com/docs/
- **Ollama**: https://ollama.ai/

## Questions?

See `PROJECT_STATUS.md` for complete architecture overview and `tests/test_creative_director*.py` for example usage.
