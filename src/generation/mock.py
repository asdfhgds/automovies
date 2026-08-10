"""Mock implementations of generation providers for local development."""
import json
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib
from .base import ScriptProvider, TTSProvider, ImageProvider, VideoProvider


class MockScriptProvider(ScriptProvider):
    """Mock script generator - produces placeholder scripts."""
    
    def generate_script(
        self,
        thesis: str,
        selected_scenes: list,
        movie_context: Optional[Dict[str, Any]] = None,
        target_duration: int = 300,
        tone: str = "analytical",
        structure: str = "classic",
    ) -> Dict[str, Any]:
        """Generate a mock script."""
        scene_ids = [s.get("scene_id", f"scene_{i}") for i, s in enumerate(selected_scenes)]
        
        sections = [
            {
                "type": "hook",
                "narration": f"[MOCK HOOK] {thesis}",
                "scene_ids": scene_ids[:1] if scene_ids else [],
                "duration_sec": 15,
            },
            {
                "type": "analysis",
                "narration": f"[MOCK ANALYSIS] Examining key evidence.",
                "scene_ids": scene_ids[1:-1] if len(scene_ids) > 2 else [],
                "duration_sec": target_duration - 30,
            },
            {
                "type": "conclusion",
                "narration": "[MOCK CONCLUSION] Key takeaway.",
                "scene_ids": scene_ids[-1:] if scene_ids else [],
                "duration_sec": 15,
            },
        ]
        
        return {
            "sections": sections,
            "total_duration_sec": target_duration,
            "tone": tone,
            "structure": structure,
            "mock": True,
        }


class MockTTSProvider(TTSProvider):
    """Mock TTS provider - creates silent audio placeholders."""

    name = "mock"

    def synthesize(
        self,
        text: str,
        voice: str = "default",
        language: str = "en",
        emotion: str = "neutral",
        speaking_rate: float = 1.0,
        pitch: float = 1.0,
        output_path: Optional[Path] = None,
        narration: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate mock audio file."""
        from .tts_common import NarrationProperties

        props = NarrationProperties.from_dict(narration)
        if output_path is None:
            output_path = Path("mock_audio.wav")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Estimate duration: rough 150 words/minute = 2.5 words/second
        word_count = len(text.split())
        duration_sec = max(1, word_count / 2.5) * (1 / (speaking_rate * props.pace))
        
        # Create a minimal WAV file header (silent audio)
        # WAV format: 44100 Hz, 16-bit mono, ~duration_sec seconds
        sample_rate = 44100
        channels = 1
        bits_per_sample = 16
        num_samples = int(sample_rate * duration_sec)
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        subchunk2_size = num_samples * channels * bits_per_sample // 8
        chunk_size = 36 + subchunk2_size
        
        wav_data = bytearray()
        wav_data.extend(b'RIFF')
        wav_data.extend(chunk_size.to_bytes(4, 'little'))
        wav_data.extend(b'WAVE')
        wav_data.extend(b'fmt ')
        wav_data.extend((16).to_bytes(4, 'little'))  # subchunk1 size
        wav_data.extend((1).to_bytes(2, 'little'))   # audio format (PCM)
        wav_data.extend(channels.to_bytes(2, 'little'))
        wav_data.extend(sample_rate.to_bytes(4, 'little'))
        wav_data.extend(byte_rate.to_bytes(4, 'little'))
        wav_data.extend(block_align.to_bytes(2, 'little'))
        wav_data.extend(bits_per_sample.to_bytes(2, 'little'))
        wav_data.extend(b'data')
        wav_data.extend(subchunk2_size.to_bytes(4, 'little'))
        wav_data.extend(b'\x00' * subchunk2_size)  # silent audio
        
        output_path.write_bytes(bytes(wav_data))
        
        return {
            "audio_path": output_path,
            "duration_sec": duration_sec,
            "sample_rate": sample_rate,
            "voice": voice,
            "language": language,
            "provider": self.name,
            "model": "mock",
            "device": "cpu",
            "generation_time_sec": 0.0,
            "model_load_time_sec": 0.0,
            "supported": {
                "emotion": False,
                "pace": True,
                "pitch": False,
                "energy": False,
                "dramatic_intensity": False,
            },
            "mock": True,
        }


class MockImageProvider(ImageProvider):
    """Mock image provider - creates placeholder images."""
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1920,
        height: int = 1080,
        seed: int = -1,
        style: str = "default",
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Generate a mock image (PNG placeholder)."""
        if output_path is None:
            output_path = Path("mock_image.png")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a minimal PNG placeholder (1x1 gray pixel, then scale to desired size)
        # PNG header + minimal image data
        png_data = (
            b'\x89PNG\r\n\x1a\n'  # PNG signature
            b'\x00\x00\x00\rIHDR'  # IHDR chunk
            b'\x00\x00\x00\x01'  # width: 1
            b'\x00\x00\x00\x01'  # height: 1
            b'\x08\x02'  # bit depth: 8, color type: 2 (RGB)
            b'\x00\x00\x00'  # compression, filter, interlace
            b'\x90\x77\x53\xde'  # CRC
            b'\x00\x00\x00\x0cIDAT'  # IDAT chunk
            b'\x08\xd7c\xf8\x0f\x00\x00\x01\x01\x00\x00'
            b'\x18\xdd\x8d\xb4'  # image data and CRC
            b'\x00\x00\x00\x00IEND'  # IEND chunk
            b'\xaeB`\x82'  # CRC
        )
        output_path.write_bytes(png_data)
        
        return {
            "image_path": output_path,
            "width": width,
            "height": height,
            "seed": seed if seed >= 0 else hash(prompt) % (2**31),
            "style": style,
            "mock": True,
        }


class MockVideoProvider(VideoProvider):
    """Mock video provider - creates placeholder video files."""
    
    def generate_video(
        self,
        prompt: str,
        reference_image: Optional[Path] = None,
        duration: float = 5.0,
        aspect_ratio: str = "16:9",
        seed: int = -1,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Generate a mock video file."""
        if output_path is None:
            output_path = Path("mock_video.mp4")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse aspect ratio
        w_str, h_str = aspect_ratio.split(":")
        w, h = int(w_str), int(h_str)
        
        # Calculate dimensions (keep reasonable max)
        scale = 1920 / w
        width = int(w * scale)
        height = int(h * scale)
        if width > 1920:
            width, height = 1920, int(1920 * h / w)
        
        # Create a minimal MP4 file (ftypisom + minimal mdat)
        # This is not a valid video but has correct file signature
        mp4_data = (
            b'\x00\x00\x00\x20ftypisom'  # ftyp box
            b'\x00\x00\x00\x00isomiso2avc1mp41'
            b'\x00\x00\x00\x0emdat'  # mdat box with minimal data
            b'\x00' * 6
        )
        output_path.write_bytes(mp4_data)
        
        return {
            "video_path": output_path,
            "duration_sec": duration,
            "width": width,
            "height": height,
            "fps": 30,
            "seed": seed if seed >= 0 else hash(prompt) % (2**31),
            "mock": True,
        }
