"""TTS stage — synthesize narration audio."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict

from .contracts import PipelineStage, StageConfig, StageResult, STAGE_CONTRACTS
from ..manifest import ProjectManifest


class TTSStage(PipelineStage):
    """Synthesize narration audio."""
    
    def run(self, stage_config: StageConfig) -> StageResult:
        self.mark_running()
        
        try:
            project_id = self.manifest.project_id
            project_dir = self.artifact_root / project_id
            
            # Get config
            provider = self.config.get("provider")
            model = self.config.get("model")
            voice = self.config.get("voice")
            emotion = self.config.get("emotion")
            pace = self.config.get("pace")
            
            # Check if script exists
            script_path = project_dir / "script.json"
            if not script_path.exists():
                return StageResult(
                    success=False,
                    stage_name=self.contract.name,
                    error="script.json not found",
                )
            
            # Run TTS using existing adapter
            from audio.tts_adapter import synthesize_voice, _tts_config
            from generation.provider_factory import get_tts_provider
            from utils.strict import require_real_tts, tts_strict_mode_enabled
            
            tts_config = _tts_config()
            if provider:
                tts_config["provider"] = provider
            if model:
                tts_config["model"] = model
            if voice:
                tts_config["voice"] = voice
            if emotion:
                tts_config["emotion"] = emotion
            if pace:
                tts_config["pace"] = pace
            
            tts_provider_obj = get_tts_provider(tts_config)
            
            if tts_strict_mode_enabled():
                tts_provider_obj = require_real_tts(tts_provider_obj, "TTS")
            
            synthesize_voice(project_dir)
            
            # Check output
            audio_path = project_dir / "audio" / "voice.wav"
            if not audio_path.exists():
                return StageResult(
                    success=False,
                    stage_name=self.contract.name,
                    error="Voice audio not generated",
                )
            
            pname = getattr(tts_provider_obj, "name", "unknown")
            tts_real = bool(pname not in ("mock", "unknown")) and not bool(getattr(tts_provider_obj, "mock", False))
            
            # Register artifacts
            artifact_ids = []
            
            artifact_id = self.register_output(
                artifact_type="tts_audio",
                relative_path=f"{project_id}/audio/voice.wav",
                metadata={
                    "provider": pname,
                    "model": getattr(tts_provider_obj, "model", None),
                    "real": tts_real,
                },
            )
            artifact_ids.append(artifact_id)
            
            tts_meta_path = project_dir / "audio" / "tts_meta.json"
            if tts_meta_path.exists():
                artifact_id = self.register_output(
                    artifact_type="tts_meta",
                    relative_path=f"{project_id}/audio/tts_meta.json",
                )
                artifact_ids.append(artifact_id)
            
            return StageResult(
                success=True,
                stage_name=self.contract.name,
                output_artifact_ids=artifact_ids,
                metrics={
                    "provider": pname,
                    "model": getattr(tts_provider_obj, "model", None),
                    "real": tts_real,
                },
            )
            
        except Exception as e:
            return StageResult(
                success=False,
                stage_name=self.contract.name,
                error=str(e),
            )


TTSStage = TTSStage