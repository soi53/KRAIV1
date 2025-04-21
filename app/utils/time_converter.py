import re

def validate_time_format(time_str):
    """
    Validate if the time string is in MM:SS format.
    
    Args:
        time_str (str): Time string to validate.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    if not time_str:
        return False
    return bool(re.match(r'^\d+:\d{2}$', time_str))

def time_to_ms(time_str):
    """
    Convert time string in MM:SS format to milliseconds.
    
    Args:
        time_str (str): Time string in MM:SS format (e.g., '5:30').
        
    Returns:
        int: Time in milliseconds, or None if invalid format.
    """
    if not time_str or not validate_time_format(time_str):
        return None
    
    try:
        minutes, seconds = time_str.split(':')
        return (int(minutes) * 60 + int(seconds)) * 1000
    except ValueError:
        return None

def ms_to_time(ms):
    """
    Convert milliseconds to MM:SS format.
    
    Args:
        ms (int): Time in milliseconds.
        
    Returns:
        str: Time string in MM:SS format.
    """
    if ms is None:
        return ""
    
    total_seconds = ms // 1000
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"

def ms_to_time_with_ms(ms):
    """
    Convert milliseconds to HH:MM:SS,mmm format (SRT format).
    
    Args:
        ms (int): Time in milliseconds.
        
    Returns:
        str: Time string in HH:MM:SS,mmm format.
    """
    if ms is None:
        return ""
    
    hours, remainder = divmod(ms // 1000, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}" 