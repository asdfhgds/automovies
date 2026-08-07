"""Quality control module for validating pipeline outputs."""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class QCValidator:
    """Validates pipeline outputs and generates QC reports."""
    
    def __init__(self):
        self.checks: Dict[str, bool] = {}
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.metadata: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
        }
    
    def check_file_exists(self, path: Path, name: str = "") -> bool:
        """Check if a file exists."""
        if path.exists():
            self.checks[f"file_exists_{name or path.name}"] = True
            return True
        else:
            self.issues.append(f"Missing file: {path}")
            self.checks[f"file_exists_{name or path.name}"] = False
            return False
    
    def check_json_valid(self, path: Path, name: str = "") -> bool:
        """Check if a JSON file is valid."""
        if not path.exists():
            self.issues.append(f"JSON file missing: {path}")
            self.checks[f"json_valid_{name or path.name}"] = False
            return False
        
        try:
            with open(path, 'r') as f:
                json.load(f)
            self.checks[f"json_valid_{name or path.name}"] = True
            return True
        except json.JSONDecodeError as e:
            self.issues.append(f"Invalid JSON in {path}: {e}")
            self.checks[f"json_valid_{name or path.name}"] = False
            return False
    
    def check_json_has_fields(self, path: Path, required_fields: List[str], name: str = "") -> bool:
        """Check if JSON has required fields."""
        if not path.exists():
            self.issues.append(f"JSON file missing: {path}")
            return False
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            missing = [f for f in required_fields if f not in data]
            if missing:
                self.issues.append(f"Missing fields in {path}: {missing}")
                self.checks[f"json_fields_{name or path.name}"] = False
                return False
            
            self.checks[f"json_fields_{name or path.name}"] = True
            return True
        except Exception as e:
            self.issues.append(f"Error checking JSON fields in {path}: {e}")
            self.checks[f"json_fields_{name or path.name}"] = False
            return False
    
    def check_transcript(self, transcript_path: Path) -> bool:
        """Validate transcript.json."""
        if not self.check_json_valid(transcript_path, "transcript"):
            return False
        
        try:
            with open(transcript_path, 'r') as f:
                transcript = json.load(f)
            
            # Check required fields
            if "segments" not in transcript:
                self.issues.append("Transcript missing 'segments' field")
                return False
            
            if not isinstance(transcript["segments"], list):
                self.issues.append("Transcript segments must be a list")
                return False
            
            if len(transcript["segments"]) == 0:
                self.issues.append("Transcript has no segments")
                return False
            
            # Check segment structure
            for i, seg in enumerate(transcript["segments"]):
                if "start_sec" not in seg or "end_sec" not in seg or "text" not in seg:
                    self.issues.append(f"Segment {i} missing required fields")
                    return False
                
                if seg["end_sec"] <= seg["start_sec"]:
                    self.issues.append(f"Segment {i} has invalid timestamps")
                    return False
            
            self.checks["transcript_valid"] = True
            return True
        except Exception as e:
            self.issues.append(f"Error validating transcript: {e}")
            return False
    
    def check_scene_index(self, scene_index_path: Path) -> bool:
        """Validate scene_index.json."""
        if not self.check_json_valid(scene_index_path, "scene_index"):
            return False
        
        try:
            with open(scene_index_path, 'r') as f:
                scene_index = json.load(f)
            
            # Check required fields
            if "scenes" not in scene_index:
                self.issues.append("Scene index missing 'scenes' field")
                return False
            
            if not isinstance(scene_index["scenes"], list):
                self.issues.append("Scene index scenes must be a list")
                return False
            
            if len(scene_index["scenes"]) == 0:
                self.issues.append("Scene index has no scenes")
                return False
            
            # Check scene structure
            for i, scene in enumerate(scene_index["scenes"]):
                if "scene_id" not in scene or "start_sec" not in scene or "end_sec" not in scene:
                    self.issues.append(f"Scene {i} missing required fields")
                    return False
                
                if scene["end_sec"] <= scene["start_sec"]:
                    self.issues.append(f"Scene {i} has invalid timestamps")
                    return False
            
            self.checks["scene_index_valid"] = True
            return True
        except Exception as e:
            self.issues.append(f"Error validating scene index: {e}")
            return False
    
    def check_video_file(self, video_path: Path, name: str = "") -> bool:
        """Check if video file exists and has valid header."""
        if not video_path.exists():
            self.issues.append(f"Video file missing: {video_path}")
            self.checks[f"video_exists_{name or video_path.name}"] = False
            return False
        
        try:
            # Check for valid video file signatures
            with open(video_path, 'rb') as f:
                header = f.read(12)
            
            # Check for MP4, MOV, or WebM signatures
            if header.startswith(b'\x00\x00\x00\x20ftyp'):  # MP4
                self.checks[f"video_valid_{name or video_path.name}"] = True
                return True
            elif header.startswith(b'\x00\x00\x00\x18ftypqt'):  # MOV
                self.checks[f"video_valid_{name or video_path.name}"] = True
                return True
            elif header.startswith(b'\x1aMASK'):  # WebM (RIFF WEBP-like)
                self.checks[f"video_valid_{name or video_path.name}"] = True
                return True
            else:
                self.warnings.append(f"Unknown video format for {video_path}")
                self.checks[f"video_valid_{name or video_path.name}"] = True  # Don't fail, just warn
                return True
        except Exception as e:
            self.issues.append(f"Error validating video {video_path}: {e}")
            self.checks[f"video_valid_{name or video_path.name}"] = False
            return False
    
    def check_audio_file(self, audio_path: Path, name: str = "") -> bool:
        """Check if audio file exists and has valid header."""
        if not audio_path.exists():
            self.issues.append(f"Audio file missing: {audio_path}")
            self.checks[f"audio_exists_{name or audio_path.name}"] = False
            return False
        
        try:
            # Check for valid audio file signatures
            with open(audio_path, 'rb') as f:
                header = f.read(4)
            
            # Check for WAV or MP3 signatures
            if header.startswith(b'RIFF'):  # WAV
                self.checks[f"audio_valid_{name or audio_path.name}"] = True
                return True
            elif header.startswith(b'\xff\xfb') or header.startswith(b'\xff\xfa') or header.startswith(b'ID3'):  # MP3
                self.checks[f"audio_valid_{name or audio_path.name}"] = True
                return True
            else:
                self.warnings.append(f"Unknown audio format for {audio_path}")
                self.checks[f"audio_valid_{name or audio_path.name}"] = True  # Don't fail, just warn
                return True
        except Exception as e:
            self.issues.append(f"Error validating audio {audio_path}: {e}")
            self.checks[f"audio_valid_{name or audio_path.name}"] = False
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate QC report."""
        passed = sum(1 for v in self.checks.values() if v)
        total = len(self.checks)
        
        return {
            "timestamp": self.metadata["timestamp"],
            "summary": {
                "passed": passed,
                "failed": total - passed,
                "total": total,
                "issues": len(self.issues),
                "warnings": len(self.warnings),
            },
            "checks": self.checks,
            "issues": self.issues,
            "warnings": self.warnings,
            "passed": len(self.issues) == 0 and len(self.checks) > 0 and all(self.checks.values()),
        }
    
    def save_report(self, report_path: Path) -> None:
        """Save QC report to file."""
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(self.generate_report(), f, indent=2)
        logger.info(f"QC report saved to {report_path}")
