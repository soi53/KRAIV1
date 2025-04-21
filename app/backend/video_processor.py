import os
import sys
import ffmpeg
from loguru import logger

# Ensure the current directory is in sys.path for imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def get_video_duration(video_path):
    """
    Get the duration of a video file in milliseconds using ffmpeg.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
        int: Duration of the video in milliseconds.
        None: If an error occurs.
    """
    try:
        # Get video information using ffmpeg probe
        probe = ffmpeg.probe(video_path)
        
        # Get format information
        format_info = probe.get('format', {})
        
        # Get duration in seconds (as a string) and convert to milliseconds
        duration_str = format_info.get('duration', '0')
        duration_ms = int(float(duration_str) * 1000)
        
        logger.info(f"Video duration extracted: {duration_ms}ms ({float(duration_str):.2f} seconds)")
        return duration_ms
        
    except ffmpeg.Error as e:
        logger.error(f"Error probing video file: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while getting video duration: {str(e)}")
        return None

def trim_video(video_path, start_ms, end_ms, output_dir):
    """
    Trim a video to the specified start and end times.
    
    Args:
        video_path (str): Path to the input video file.
        start_ms (int): Start time in milliseconds.
        end_ms (int): End time in milliseconds.
        output_dir (str): Directory to save the output file.
        
    Returns:
        str: Path to the trimmed video file.
        None: If an error occurs.
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert milliseconds to seconds for ffmpeg
        start_sec = start_ms / 1000
        duration_sec = (end_ms - start_ms) / 1000
        
        # Extract filename and generate output filename
        filename = os.path.basename(video_path)
        name_without_ext, ext = os.path.splitext(filename)
        output_filename = f"trimmed_{name_without_ext}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Check input file for audio streams
        probe = ffmpeg.probe(video_path)
        has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
        logger.info(f"Input file has audio: {has_audio}")
        
        # Prepare ffmpeg command
        input_stream = ffmpeg.input(video_path, ss=start_sec, t=duration_sec)
        
        # Get video stream
        video_stream = input_stream.video
        
        # Get audio stream if it exists
        if has_audio:
            audio_stream = input_stream.audio
            # Output with both video and audio
            output_args = {'c:v': 'copy'}
            if has_audio:
                # Use AAC codec for audio with 128k bitrate
                output_args.update({'c:a': 'aac', 'b:a': '128k'})
            
            output = ffmpeg.output(video_stream, audio_stream, output_path, **output_args)
        else:
            # Output with video only
            output = ffmpeg.output(video_stream, output_path, c='copy')
        
        # Log the ffmpeg command that will be executed
        cmd = ffmpeg.compile(output)
        logger.debug(f"ffmpeg command: {' '.join(cmd)}")
        
        # Run the ffmpeg command
        ffmpeg.run(output, overwrite_output=True, quiet=True)
        
        logger.info(f"Video trimmed successfully: {output_path}")
        return output_path
        
    except ffmpeg.Error as e:
        error_message = str(e.stderr.decode('utf-8') if hasattr(e, 'stderr') else str(e))
        logger.error(f"FFmpeg error while trimming video: {error_message}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while trimming video: {str(e)}")
        return None

def combine_video(video_path, audio_path, subtitle_path=None, output_dir="/data/results", target_language=None):
    """
    Combine video, audio, and optional subtitles into a final output video.
    
    Args:
        video_path (str): Path to the input video file.
        audio_path (str): Path to the TTS audio file.
        subtitle_path (str, optional): Path to the subtitle file (SRT format).
        output_dir (str): Directory to save the output file.
        target_language (str, optional): Target language code for naming.
        
    Returns:
        str: Path to the combined video file.
        None: If an error occurs.
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract filename and generate output filename
        filename = os.path.basename(video_path)
        name_without_ext, ext = os.path.splitext(filename)
        
        # If we have a target language, include it in the filename
        lang_suffix = f"_{target_language}" if target_language else ""
        output_filename = f"result_{name_without_ext}{lang_suffix}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Start with the video input
        input_video = ffmpeg.input(video_path)
        input_audio = ffmpeg.input(audio_path)
        
        # Extract video stream from input
        video_stream = input_video.video
        
        # If subtitles are provided, add them as a video filter
        if subtitle_path and os.path.exists(subtitle_path):
            logger.info(f"Adding subtitles from: {subtitle_path}")
            # Add subtitles using the subtitles filter
            video_stream = video_stream.filter('subtitles', subtitle_path)
        
        # Use the provided audio
        audio_stream = input_audio.audio
        
        # Output with combined streams
        output = ffmpeg.output(
            video_stream, 
            audio_stream, 
            output_path,
            **{
                'vcodec': 'libx264',  # Use H.264 codec for video
                'acodec': 'aac',      # Use AAC codec for audio
                'strict': 'experimental',
                'b:a': '192k'         # Audio bitrate
            }
        )
        
        # Log the ffmpeg command that will be executed
        cmd = ffmpeg.compile(output)
        logger.debug(f"ffmpeg command: {' '.join(cmd)}")
        
        # Run the ffmpeg command
        ffmpeg.run(output, overwrite_output=True, quiet=True)
        
        logger.info(f"Video, audio, and subtitles combined successfully: {output_path}")
        return output_path
        
    except ffmpeg.Error as e:
        error_message = str(e.stderr.decode('utf-8') if hasattr(e, 'stderr') else str(e))
        logger.error(f"FFmpeg error while combining video: {error_message}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while combining video: {str(e)}")
        return None 