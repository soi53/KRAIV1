import sys
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# Ensure the current directory is in sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

class VideoInfo(BaseModel):
    """Information about a video being processed."""
    id: str
    original_name: str
    original_path: str
    size: int
    duration_ms: Optional[int] = None
    extension: str
    upload_time: datetime = Field(default_factory=datetime.now)
    trimmed: bool = False
    trimmed_path: Optional[str] = None
    trim_start_ms: Optional[int] = None
    trim_end_ms: Optional[int] = None
    
    # File path to use for processing (either original or trimmed)
    @property
    def active_video_path(self) -> str:
        """Return the path of the video that should be used for processing."""
        return self.trimmed_path if self.trimmed else self.original_path

class SubtitleSegment(BaseModel):
    """A single segment/line of a subtitle."""
    index: int
    start_time_ms: int
    end_time_ms: int
    text: str
    translated_text: Optional[str] = None
    
    # Format start time as SRT format (HH:MM:SS,mmm)
    def format_start_time(self) -> str:
        """Format the start time as HH:MM:SS,mmm for SRT format."""
        hours, remainder = divmod(self.start_time_ms // 1000, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = self.start_time_ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    
    # Format end time as SRT format (HH:MM:SS,mmm)
    def format_end_time(self) -> str:
        """Format the end time as HH:MM:SS,mmm for SRT format."""
        hours, remainder = divmod(self.end_time_ms // 1000, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = self.end_time_ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

class SubtitleFile(BaseModel):
    """Information about a subtitle file."""
    file_id: str
    segments: List[SubtitleSegment]
    language: str
    original_path: Optional[str] = None
    extracted_path: Optional[str] = None
    
    def get_subtitle_text(self, translated: bool = False) -> str:
        """
        Get the full subtitle text as a string.
        
        Args:
            translated (bool): Whether to get the translated text instead of original.
            
        Returns:
            str: The concatenated subtitle text.
        """
        if translated:
            return "\n".join([segment.translated_text for segment in self.segments if segment.translated_text])
        else:
            return "\n".join([segment.text for segment in self.segments])

class TranslationTask(BaseModel):
    """Information about a translation task (for future async processing)."""
    task_id: str
    file_id: str
    source_language: str
    target_language: str
    status: str = "pending"  # pending, processing, completed, failed
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    result_path: Optional[str] = None
    error_message: Optional[str] = None 