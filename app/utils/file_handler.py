import os
import uuid
import shutil
from datetime import datetime
from loguru import logger
import sys

# Ensure the current directory is in sys.path for imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Helper function to generate a unique ID
def generate_unique_id():
    """Generate a unique identifier based on timestamp and UUID."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = f"{timestamp}_{str(uuid.uuid4())[:8]}"
    return unique_id

def validate_video_file(file):
    """
    Validate the uploaded video file format and size.
    
    Args:
        file: The StreamlitUploadedFile object.
        
    Returns:
        tuple: (is_valid, message)
            is_valid (bool): True if file is valid, False otherwise.
            message (str): Validation message or error.
    """
    # Check if file exists
    if file is None:
        return False, "No file uploaded."
    
    # Extract file extension without the dot
    file_name = file.name
    file_ext = os.path.splitext(file_name)[1].lower()[1:]
    
    # Validate file format
    allowed_formats = ['mp4', 'avi']
    if file_ext not in allowed_formats:
        return False, f"Invalid file format. Allowed formats: {', '.join(allowed_formats)}"
    
    # Validate file size (2GB limit)
    max_size_bytes = 2 * 1024 * 1024 * 1024  # 2GB
    if file.size > max_size_bytes:
        max_size_gb = max_size_bytes / (1024 * 1024 * 1024)
        return False, f"File size exceeds the limit of {max_size_gb}GB."
    
    return True, "File is valid."

def save_uploaded_file(file, directory):
    """
    Save an uploaded file to the specified directory with a unique name.
    
    Args:
        file: The StreamlitUploadedFile object.
        directory (str): Directory path to save the file to.
        
    Returns:
        dict: Information about the saved file including:
            - id: Unique identifier for the file
            - original_name: Original filename
            - path: Path to the saved file
            - size: File size in bytes
            - extension: File extension
    """
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Generate unique ID for the file
    file_id = generate_unique_id()
    
    # Extract file extension
    file_name = file.name
    file_ext = os.path.splitext(file_name)[1].lower()
    
    # Create a unique filename
    saved_filename = f"original_{file_id}{file_ext}"
    saved_path = os.path.join(directory, saved_filename)
    
    # Save the file
    with open(saved_path, "wb") as f:
        f.write(file.getbuffer())
    
    logger.info(f"File '{file_name}' uploaded and saved as '{saved_filename}'")
    
    # Return information about the saved file
    return {
        "id": file_id,
        "original_name": file_name,
        "path": saved_path,
        "size": file.size,
        "extension": file_ext
    }

def delete_file(file_path):
    """
    Delete a file at the specified path.
    
    Args:
        file_path (str): Path to the file to delete.
        
    Returns:
        bool: True if deletion was successful, False otherwise.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"File deleted: {file_path}")
            return True
        else:
            logger.warning(f"File not found for deletion: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {str(e)}")
        return False

def clean_temporary_files(file_id):
    """
    Clean up all temporary files associated with a specific file ID.
    
    Args:
        file_id (str): The unique identifier for the files to clean up.
    
    Returns:
        int: Number of files deleted.
    """
    # Define directories to clean
    directories = [
        "/data/uploads",
        "/data/processed"
    ]
    
    # DO NOT clean /data/results as those are final output files
    # that users might want to keep
    
    deleted_count = 0
    
    for directory in directories:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if file_id in filename:
                    file_path = os.path.join(directory, filename)
                    if delete_file(file_path):
                        deleted_count += 1
    
    logger.info(f"Cleaned up {deleted_count} temporary files for file ID: {file_id}")
    return deleted_count 