import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This will search for a .env file in the current directory or parent directories
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env') # Construct path relative to this file
load_dotenv(dotenv_path=dotenv_path)

# --- General Settings ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- External API Settings ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WHISPER_API_URL = os.getenv("WHISPER_API_URL")

# --- You can add other configuration variables here as needed ---

# --- Validation (Optional but recommended) ---
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    # Depending on the application's requirements, you might want to raise an error here
    # raise ValueError("Missing required environment variable: OPENAI_API_KEY")

if not WHISPER_API_URL:
    print("Warning: WHISPER_API_URL environment variable not set.")
    # raise ValueError("Missing required environment variable: WHISPER_API_URL") 