import sys
import os
from loguru import logger

# Ensure the current directory is in sys.path for imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 상대 경로로 설정 모듈 가져오기
from config import settings

# Define the directory for log files within the mapped /data volume
log_dir = "/data/logs"
os.makedirs(log_dir, exist_ok=True) # Ensure the log directory exists
log_file_path = os.path.join(log_dir, "app_{time}.log")

# Remove default handler to prevent duplicate logs in console
logger.remove()

# Add a handler for console output
# Use the LOG_LEVEL from settings.py
logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL.upper(),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)

# Add a handler for file output
# Use the LOG_LEVEL from settings.py
# Rotate the log file when it reaches 10 MB or after 7 days
# Keep up to 5 rotated log files
# Compress rotated files
logger.add(
    log_file_path,
    level=settings.LOG_LEVEL.upper(),
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    encoding="utf-8"
)

# Export the configured logger instance for use in other modules
__all__ = ["logger"] 