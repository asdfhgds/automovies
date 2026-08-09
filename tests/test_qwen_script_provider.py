import pytest

from script.providers import QwenScriptProvider


class FakeQwen:
    def __init__(self, result):
        self.result = result

    def generate_json(self, prompt, required_keys):
        assert "scene-1" in prompt
        assert required_keys == ["sections"]
        return self.result


def test_qwen_script_provider_returns_grounded_script():
    provider = QwenScriptProvider(FakeQwen({"sections": [
        {"id": "hook", "type": "hook", "narration": "A choice becomes a trap.", "scene_ids": ["scene-1"]},
        {"id": "analysis", "type": "analysis", "narration": "The exchange turns chance into control.", "scene_ids": ["scene-1"]},
    ]}))
    script = provider.generate_script("Chance is control.", [{"scene_id": "scene-1", "start_sec": 0, "end_sec": 5, "transcript": "I chose this."}])
    assert script["provider"] == "qwen"
    assert script["word_count"] > 0


def test_qwen_script_provider_rejects_invented_scene_ids():
    provider = QwenScriptProvider(FakeQwen({"sections": [
        {"id": "analysis", "type": "analysis", "narration": "Unsupported claim.", "scene_ids": ["invented"]},
    ]}))
    with pytest.raises(ValueError, match="outside selected evidence"):
        provider.generate_script("A thesis", [{"scene_id": "scene-1"}])
