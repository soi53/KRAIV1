import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import json
import tempfile
import time
from loguru import logger
import re
import chardet

# Ensure the current directory is in sys.path for imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from schemas import SubtitleSegment, SubtitleFile

# 자막 파일 확장자 목록
SUBTITLE_EXTENSIONS = ['.srt', '.vtt', '.ass', '.ssa', '.sub']

def detect_encoding(file_path: str) -> str:
    """
    파일의 인코딩을 감지합니다.
    
    Args:
        file_path (str): 파일 경로
        
    Returns:
        str: 감지된 인코딩, 기본값은 'utf-8'
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(4096)  # 파일의 일부만 읽어 효율성 높임
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'
            logger.info(f"파일 인코딩 감지: {file_path} -> {encoding}")
            return encoding
    except Exception as e:
        logger.error(f"인코딩 감지 실패: {str(e)}, 기본값 utf-8 사용")
        return 'utf-8'

def read_subtitle_file(file_path: str) -> List[SubtitleSegment]:
    """
    Read a subtitle file and parse it into a list of SubtitleSegment objects.
    
    Args:
        file_path (str): Path to the subtitle file
        
    Returns:
        List[SubtitleSegment]: List of subtitle segments
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        # 파일 인코딩 감지
        encoding = detect_encoding(file_path)
        
        if file_extension == '.srt':
            return read_srt_file(file_path, encoding)
        elif file_extension == '.vtt':
            return read_vtt_file(file_path, encoding)
        elif file_extension in ['.ass', '.ssa']:
            return read_ass_file(file_path, encoding)
        else:
            logger.error(f"Unsupported subtitle format: {file_extension}")
            return []
    except Exception as e:
        logger.error(f"Error reading subtitle file: {str(e)}")
        return []

def read_srt_file(file_path: str, encoding: str = 'utf-8') -> List[SubtitleSegment]:
    """
    Parse an SRT file into a list of SubtitleSegment objects.
    
    Args:
        file_path (str): Path to the SRT file
        encoding (str): 파일 인코딩
        
    Returns:
        List[SubtitleSegment]: List of subtitle segments
    """
    segments = []
    
    try:
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
            
        # Remove BOM if present
        if content.startswith('\ufeff'):
            content = content[1:]
        
        # Split the content by double newline (segment separator in SRT)
        raw_segments = re.split(r'\n\s*\n', content.strip())
        
        current_index = 0
        for raw_segment in raw_segments:
            if not raw_segment.strip():
                continue
                
            lines = raw_segment.strip().split('\n')
            if len(lines) < 2:
                continue
                
            # Try to find segment index
            try:
                segment_idx = int(lines[0].strip())
            except ValueError:
                current_index += 1
                segment_idx = current_index
            
            # Parse timestamp line
            timestamp_line = lines[1].strip()
            if '-->' not in timestamp_line:
                logger.warning(f"Invalid timestamp format in segment {segment_idx}: {timestamp_line}")
                continue
                
            time_parts = timestamp_line.split('-->')
            if len(time_parts) != 2:
                logger.warning(f"Invalid timestamp format in segment {segment_idx}: {timestamp_line}")
                continue
                
            start_time = time_parts[0].strip()
            end_time = time_parts[1].strip()
            
            # Convert timestamps to milliseconds
            start_ms = timestr_to_ms(start_time)
            end_ms = timestr_to_ms(end_time)
            
            # Combine remaining lines as text
            text = '\n'.join(lines[2:]).strip()
            
            # Create a subtitle segment
            segment = SubtitleSegment(
                index=segment_idx,
                start_time=start_time,
                end_time=end_time,
                start_time_ms=start_ms,
                end_time_ms=end_ms,
                text=text,
                translated_text=None
            )
            
            segments.append(segment)
            current_index = segment_idx
            
        logger.info(f"Read {len(segments)} segments from SRT file: {file_path}")
        return segments
        
    except Exception as e:
        logger.error(f"Error parsing SRT file: {str(e)}")
        return []

def write_subtitle_file(segments: List[SubtitleSegment], 
                       output_path: str, 
                       format: str = 'srt',
                       language: str = None) -> bool:
    """
    Write subtitle segments to a file in the specified format.
    
    Args:
        segments (List[SubtitleSegment]): List of subtitle segments
        output_path (str): Path to save the subtitle file
        format (str): Format of the subtitle file ('srt', 'vtt', 'ass')
        language (str): Optional language code for the subtitle
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if format.lower() == 'srt':
            return write_srt_file(segments, output_path)
        elif format.lower() == 'vtt':
            return write_vtt_file(segments, output_path, language)
        elif format.lower() == 'ass':
            return write_ass_file(segments, output_path, language)
        else:
            logger.error(f"Unsupported subtitle format: {format}")
            return False
    except Exception as e:
        logger.error(f"Error writing subtitle file: {str(e)}")
        return False

def write_srt_file(segments: List[SubtitleSegment], output_path: str) -> bool:
    """
    Write subtitle segments to an SRT file.
    
    Args:
        segments (List[SubtitleSegment]): List of subtitle segments
        output_path (str): Path to save the SRT file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments):
                # Use translated text if available, otherwise use original
                text = segment.translated_text if segment.translated_text else segment.text
                
                # Format segment in SRT format
                f.write(f"{i+1}\n")
                f.write(f"{ms_to_timestr(segment.start_time_ms)} --> {ms_to_timestr(segment.end_time_ms)}\n")
                f.write(f"{text}\n\n")
                
        logger.info(f"Successfully wrote {len(segments)} segments to SRT file: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error writing SRT file: {str(e)}")
        return False

def burn_subtitles_to_video(video_path: str, subtitle_path: str, output_path: str, font_size: int = 24, position: str = "bottom", font_name: str = None, encoding: str = "utf-8") -> Tuple[bool, str]:
    """
    Burn subtitles into a video using FFmpeg.
    
    Args:
        video_path (str): Path to the input video
        subtitle_path (str): Path to the subtitle file
        output_path (str): Path to save the output video
        font_size (int): Font size for the subtitles
        position (str): Position of the subtitles ('bottom', 'top', 'middle')
        font_name (str): Optional font name to use
        encoding (str): Optional encoding for the subtitle file
        
    Returns:
        Tuple[bool, str]: (Success status, output message or error)
    """
    try:
        # Check if input files exist
        if not os.path.isfile(video_path):
            return False, f"Input video not found: {video_path}"
        
        if not os.path.isfile(subtitle_path):
            return False, f"Subtitle file not found: {subtitle_path}"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get subtitle file extension
        subtitle_ext = os.path.splitext(subtitle_path)[1].lower()
        
        # Prepare subtitle filter based on file type
        if subtitle_ext == '.srt':
            subtitle_filter = f"subtitles='{subtitle_path.replace(':', '\\:')}':force_style='FontSize={font_size}"
        elif subtitle_ext == '.ass' or subtitle_ext == '.ssa':
            subtitle_filter = f"ass='{subtitle_path.replace(':', '\\:')}'"
        else:
            # For other formats, convert to SRT first
            temp_dir = tempfile.mkdtemp()
            temp_srt = os.path.join(temp_dir, 'temp.srt')
            
            # Read and write subtitle to ensure encoding is correct
            segments = read_subtitle_file(subtitle_path)
            write_srt_file(segments, temp_srt)
            
            subtitle_path = temp_srt
            subtitle_filter = f"subtitles='{subtitle_path.replace(':', '\\:')}':force_style='FontSize={font_size}"
        
        # Add position to style if specified
        if position == "top":
            subtitle_filter += ",MarginV=20,Alignment=6"  # Top alignment
        elif position == "middle":
            subtitle_filter += ",MarginV=0,Alignment=8"   # Middle alignment
        else:  # Default: bottom
            subtitle_filter += ",MarginV=30,Alignment=2"  # Bottom alignment
        
        # Add font name if specified
        if font_name:
            subtitle_filter += f",FontName={font_name}"
            
        # Add encoding parameter for non-UTF8 subtitles
        if encoding and encoding.lower() != "utf-8":
            subtitle_filter += f"':charenc={encoding}"
        else:
            subtitle_filter += "'"
        
        # Construct FFmpeg command
        command = [
            "ffmpeg",
            "-i", video_path,
            "-vf", subtitle_filter,
            "-c:v", "libx264",  # Use H.264 codec for video
            "-crf", "18",       # Quality setting (lower is better)
            "-c:a", "copy",     # Copy audio stream without re-encoding
            "-y",               # Overwrite output file if it exists
            output_path
        ]
        
        logger.info(f"Running FFmpeg command: {' '.join(command)}")
        
        # Run FFmpeg command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Check if the process was successful
        if process.returncode != 0:
            logger.error(f"FFmpeg Error: {stderr.decode()}")
            return False, f"FFmpeg Error: {stderr.decode()}"
        
        # Check if output file was created
        if not os.path.isfile(output_path):
            return False, "Output file was not created for an unknown reason"
        
        logger.info(f"Successfully burned subtitles to video: {output_path}")
        return True, output_path
        
    except Exception as e:
        logger.error(f"Error burning subtitles: {str(e)}")
        return False, f"Error: {str(e)}" 