import os
import sys
import re
from datetime import datetime
from loguru import logger

# Ensure the current directory is in sys.path for imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from schemas import SubtitleSegment, SubtitleFile
from utils.time_converter import ms_to_time_with_ms

def extract_subtitles(video_path, output_dir, model_size="medium", language="ko", 
                     prompt="", temperature=0, verbose=False):
    """
    Extract subtitles from a video using local Whisper model.
    
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save the output SRT file.
        model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
        language (str): Language code (e.g., 'ko' for Korean, 'en' for English).
        prompt (str): Initial prompt to guide the transcription.
        temperature (float): Sampling temperature between 0 and 1.
        verbose (bool): Whether to log detailed information.
        
    Returns:
        str: Path to the extracted SRT file, or None if extraction failed.
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract file ID from the video path
        filename = os.path.basename(video_path)
        file_id = filename.split('_', 1)[1].split('.')[0]  # Extract ID part
        
        # Log details
        logger.info(f"Extracting subtitles from video: {video_path}")
        logger.info(f"Model size: {model_size}, Language: {language}")
        
        # Verify that the input video exists
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
            
        # Create SRT filename
        srt_filename = f"subtitle_{file_id}.srt"
        srt_path = os.path.join(output_dir, srt_filename)
        
        # Use ffmpeg to extract audio first (to improve performance)
        logger.info("Extracting audio from video...")
        import subprocess
        audio_path = os.path.join(output_dir, f"audio_{file_id}.wav")
        
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", video_path, 
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", 
            audio_path
        ]
        
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        
        # Use whisper module to transcribe
        logger.info(f"Transcribing audio with whisper model: {model_size}...")
        
        try:
            import whisper
        except ImportError:
            logger.error("Whisper module not installed. Please install it with 'pip install openai-whisper'")
            return None
            
        # Load model
        model = whisper.load_model(model_size)
        
        # Set transcription options
        options = {
            "language": language if language != "auto" else None,
            "task": "transcribe",
            "initial_prompt": prompt if prompt else None,
            "temperature": temperature,
            "verbose": verbose
        }
        
        # Filter out None values
        options = {k: v for k, v in options.items() if v is not None}
        
        # Transcribe audio
        logger.info("Starting transcription...")
        result = model.transcribe(audio_path, **options)
        
        # Convert result to SRT format
        logger.info("Converting transcription to SRT format...")
        with open(srt_path, "w", encoding="utf-8") as srt_file:
            for i, segment in enumerate(result["segments"], 1):
                # Get start and end time in seconds
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"].strip()
                
                # Convert to SRT format (HH:MM:SS,mmm)
                start_srt = format_time_srt(start_time)
                end_srt = format_time_srt(end_time)
                
                # Write to SRT file
                srt_file.write(f"{i}\n")
                srt_file.write(f"{start_srt} --> {end_srt}\n")
                srt_file.write(f"{text}\n\n")
        
        # Clean up temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        logger.success(f"Subtitles extracted and saved to: {srt_path}")
        return srt_path
        
    except Exception as e:
        logger.error(f"Error extracting subtitles: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def format_time_srt(time_seconds):
    """Convert time in seconds to SRT format (HH:MM:SS,mmm)"""
    hours = int(time_seconds // 3600)
    minutes = int((time_seconds % 3600) // 60)
    seconds = int(time_seconds % 60)
    milliseconds = int((time_seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def parse_srt_file(srt_path):
    """
    Parse an SRT file into a list of SubtitleSegment objects.
    
    Args:
        srt_path (str): Path to the SRT file.
        
    Returns:
        list: List of SubtitleSegment objects.
    """
    try:
        if not os.path.exists(srt_path):
            logger.error(f"SRT file not found: {srt_path}")
            return []
        
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content by double newline (each subtitle block)
        blocks = content.strip().split('\n\n')
        segments = []
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue  # Skip invalid blocks
            
            # Parse index
            try:
                index = int(lines[0])
            except ValueError:
                logger.warning(f"Invalid subtitle index: {lines[0]}")
                continue
            
            # Parse timestamp
            timestamp_line = lines[1]
            timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', timestamp_line)
            if not timestamp_match:
                logger.warning(f"Invalid timestamp format: {timestamp_line}")
                continue
            
            start_time_str, end_time_str = timestamp_match.groups()
            
            # Convert timestamp to milliseconds
            def time_str_to_ms(time_str):
                h, m, s_ms = time_str.split(':')
                s, ms = s_ms.split(',')
                return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
            
            start_time_ms = time_str_to_ms(start_time_str)
            end_time_ms = time_str_to_ms(end_time_str)
            
            # Parse text (could be multiple lines)
            text = '\n'.join(lines[2:])
            
            # Create SubtitleSegment
            segment = SubtitleSegment(
                index=index,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                text=text
            )
            segments.append(segment)
        
        logger.info(f"Parsed {len(segments)} subtitle segments from: {srt_path}")
        return segments
        
    except Exception as e:
        logger.error(f"Error parsing SRT file: {str(e)}")
        return []

def parse_uploaded_subtitle(file, output_dir):
    """
    Parse an uploaded subtitle file (SRT/TXT).
    
    Args:
        file: The uploaded file object.
        output_dir (str): Directory to save the processed file.
        
    Returns:
        tuple: (List of SubtitleSegment objects, file path)
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = file.name
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # Save the uploaded file
        saved_filename = f"uploaded_subtitle_{timestamp}{file_ext}"
        saved_path = os.path.join(output_dir, saved_filename)
        
        with open(saved_path, "wb") as f:
            f.write(file.getbuffer())
        
        logger.info(f"Uploaded subtitle file saved to: {saved_path}")
        
        # Process based on file type
        if file_ext == '.srt':
            # Parse SRT file
            segments = parse_srt_file(saved_path)
            return segments, saved_path
        elif file_ext == '.txt':
            # Basic TXT format: Assume each line is a subtitle
            with open(saved_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            segments = []
            for i, line in enumerate(lines):
                text = line.strip()
                if text:  # Skip empty lines
                    # For TXT, we don't have timing information, so use placeholders
                    # Each segment is 5 seconds starting from 0
                    start_time_ms = i * 5000
                    end_time_ms = start_time_ms + 5000
                    
                    segment = SubtitleSegment(
                        index=i+1,
                        start_time_ms=start_time_ms,
                        end_time_ms=end_time_ms,
                        text=text
                    )
                    segments.append(segment)
            
            logger.info(f"Created {len(segments)} subtitle segments from TXT file")
            return segments, saved_path
        else:
            logger.error(f"Unsupported subtitle file format: {file_ext}")
            return [], None
            
    except Exception as e:
        logger.error(f"Error processing uploaded subtitle: {str(e)}")
        return [], None

def save_subtitles_to_file(segments, output_path, file_format='srt', translated=False):
    """
    Save subtitle segments to a file.
    
    Args:
        segments (list): List of SubtitleSegment objects.
        output_path (str): Path to save the output file.
        file_format (str): Output format ('srt' or 'txt').
        translated (bool): Whether to use translated_text instead of text.
        
    Returns:
        str: Path to the saved file, or None if saving failed.
    """
    try:
        # Ensure the directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if file_format.lower() == 'srt':
                # SRT format
                for segment in segments:
                    text = segment.translated_text if translated and segment.translated_text else segment.text
                    f.write(f"{segment.index}\n")
                    f.write(f"{ms_to_time_with_ms(segment.start_time_ms)} --> {ms_to_time_with_ms(segment.end_time_ms)}\n")
                    f.write(f"{text}\n\n")
            else:
                # TXT format (simple text, one subtitle per line)
                for segment in segments:
                    text = segment.translated_text if translated and segment.translated_text else segment.text
                    f.write(f"{text}\n")
        
        logger.info(f"Subtitle {'translation' if translated else 'text'} saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving subtitles to file: {str(e)}")
        return None 