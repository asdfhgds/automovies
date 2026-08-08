"""Base provider interfaces for all generation capabilities."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path


class ScriptProvider(ABC):
    """Interface for script/narration generation."""
    
    @abstractmethod
    def generate_script(
        self,
        thesis: str,
        selected_scenes: list,
        movie_context: Optional[Dict[str, Any]] = None,
        target_duration: int = 300,
        tone: str = "analytical",
        structure: str = "classic",
    ) -> Dict[str, Any]:
        """
        Generate a script/narration plan.
        
        Args:
            thesis: Creative thesis/concept from director
            selected_scenes: List of scene objects with timestamps
            movie_context: Metadata about source movie
            target_duration: Target video duration in seconds
            tone: Tone of narration (analytical, humorous, emotional, etc.)
            structure: Script structure (classic, climax-first, parallel, etc.)
        
        Returns:
            Dictionary with structure:
            {
                "sections": [
                    {
                        "type": "hook",
                        "narration": "...",
                        "scene_ids": [],
                        "duration_sec": 15
                    },
                    ...
                ]
            }
        """
        pass


class TTSProvider(ABC):
    """Interface for text-to-speech generation."""
    
    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str = "default",
        language: str = "en",
        emotion: str = "neutral",
        speaking_rate: float = 1.0,
        pitch: float = 1.0,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice: Voice/speaker identifier
            language: Language code
            emotion: Emotional tone
            speaking_rate: Speech speed multiplier
            pitch: Pitch adjustment
            output_path: Path to save audio file
        
        Returns:
            Dictionary with:
            {
                "audio_path": Path,
                "duration_sec": float,
                "sample_rate": int
            }
        """
        pass


class ImageProvider(ABC):
    """Interface for image generation."""
    
    @abstractmethod
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
        """
        Generate an image.
        
        Args:
            prompt: Positive prompt describing desired image
            negative_prompt: Negative prompt for things to avoid
            width: Image width in pixels
            height: Image height in pixels
            seed: Random seed for reproducibility (-1 for random)
            style: Visual style (cinematic, artistic, photorealistic, etc.)
            output_path: Path to save image file
        
        Returns:
            Dictionary with:
            {
                "image_path": Path,
                "width": int,
                "height": int,
                "seed": int
            }
        """
        pass


class VideoProvider(ABC):
    """Interface for video generation."""
    
    @abstractmethod
    def generate_video(
        self,
        prompt: str,
        reference_image: Optional[Path] = None,
        duration: float = 5.0,
        aspect_ratio: str = "16:9",
        seed: int = -1,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate video content.
        
        Args:
            prompt: Description of desired video content
            reference_image: Optional image to guide generation
            duration: Video duration in seconds
            aspect_ratio: Aspect ratio (16:9, 1:1, etc.)
            seed: Random seed for reproducibility
            output_path: Path to save video file
        
        Returns:
            Dictionary with:
            {
                "video_path": Path,
                "duration_sec": float,
                "width": int,
                "height": int,
                "fps": int
            }
        """
        pass
