"""Timeline and editing models for video assembly."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import Enum
from pathlib import Path


class TrackType(str, Enum):
    """Types of timeline tracks."""
    VOICE = "voice"
    VIDEO = "video"
    MUSIC = "music"
    SFX = "sfx"
    TEXT = "text"


@dataclass
class TimelineItem:
    """Single item on a timeline track."""
    type: str  # "clip", "generated", "effect", "text"
    start_sec: float
    duration_sec: float
    content_path: Optional[Path] = None
    content_text: Optional[str] = None  # for text tracks
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def end_sec(self) -> float:
        return self.start_sec + self.duration_sec
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.content_path:
            data["content_path"] = str(self.content_path)
        return data


@dataclass
class TimelineTrack:
    """A single track in the timeline."""
    track_type: TrackType
    items: List[TimelineItem] = field(default_factory=list)
    volume: float = 1.0  # For audio tracks
    mute: bool = False
    
    def add_item(self, item: TimelineItem) -> None:
        """Add an item to the track."""
        self.items.append(item)
        self.items.sort(key=lambda x: x.start_sec)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.track_type.value,
            "items": [item.to_dict() for item in self.items],
            "volume": self.volume,
            "mute": self.mute,
        }


@dataclass
class Timeline:
    """Complete video timeline."""
    total_duration_sec: float
    tracks: List[TimelineTrack] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_track(self, track_type: TrackType) -> TimelineTrack:
        """Create and add a new track."""
        track = TimelineTrack(track_type=track_type)
        self.tracks.append(track)
        return track
    
    def get_track(self, track_type: TrackType) -> Optional[TimelineTrack]:
        """Get first track of a given type."""
        for track in self.tracks:
            if track.track_type == track_type:
                return track
        return None
    
    def validate(self) -> List[str]:
        """Validate timeline integrity."""
        errors = []
        
        # Check for overlapping items on same track
        for track in self.tracks:
            for i, item1 in enumerate(track.items):
                for item2 in track.items[i+1:]:
                    if item1.start_sec < item2.start_sec < item1.end_sec:
                        errors.append(
                            f"Overlap on {track.track_type} track: "
                            f"[{item1.start_sec}, {item1.end_sec}] overlaps [{item2.start_sec}, {item2.end_sec}]"
                        )
        
        # Check that all items are within total duration
        for track in self.tracks:
            for item in track.items:
                if item.end_sec > self.total_duration_sec:
                    errors.append(
                        f"Item extends beyond total duration: "
                        f"{item.end_sec} > {self.total_duration_sec}"
                    )
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_duration_sec": self.total_duration_sec,
            "tracks": [track.to_dict() for track in self.tracks],
            "metadata": self.metadata,
        }


class TimelineBuilder:
    """Helper class to build timelines."""
    
    def __init__(self, total_duration_sec: float):
        self.timeline = Timeline(total_duration_sec=total_duration_sec)
    
    def add_voiceover(self, audio_path: Path, start_sec: float, duration_sec: float) -> None:
        """Add voiceover to timeline."""
        track = self.timeline.get_track(TrackType.VOICE)
        if not track:
            track = self.timeline.add_track(TrackType.VOICE)
        
        item = TimelineItem(
            type="audio_clip",
            start_sec=start_sec,
            duration_sec=duration_sec,
            content_path=audio_path,
        )
        track.add_item(item)
    
    def add_video_clip(self, clip_path: Path, start_sec: float, duration_sec: float) -> None:
        """Add video clip to timeline."""
        track = self.timeline.get_track(TrackType.VIDEO)
        if not track:
            track = self.timeline.add_track(TrackType.VIDEO)
        
        item = TimelineItem(
            type="video_clip",
            start_sec=start_sec,
            duration_sec=duration_sec,
            content_path=clip_path,
        )
        track.add_item(item)
    
    def add_music(self, audio_path: Path, start_sec: float, duration_sec: float, volume: float = 0.5) -> None:
        """Add background music to timeline."""
        track = self.timeline.get_track(TrackType.MUSIC)
        if not track:
            track = self.timeline.add_track(TrackType.MUSIC)
            track.volume = volume
        
        item = TimelineItem(
            type="audio_clip",
            start_sec=start_sec,
            duration_sec=duration_sec,
            content_path=audio_path,
        )
        track.add_item(item)
    
    def add_subtitle(self, text: str, start_sec: float, duration_sec: float) -> None:
        """Add subtitle text."""
        track = self.timeline.get_track(TrackType.TEXT)
        if not track:
            track = self.timeline.add_track(TrackType.TEXT)
        
        item = TimelineItem(
            type="subtitle",
            start_sec=start_sec,
            duration_sec=duration_sec,
            content_text=text,
        )
        track.add_item(item)
    
    def build(self) -> Timeline:
        """Build and validate the timeline."""
        errors = self.timeline.validate()
        if errors:
            for error in errors:
                print(f"WARNING: {error}")
        return self.timeline
