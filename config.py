import os 
from dotenv import load_dotenv
load_dotenv()

LLM_CONFIG = {
    "model": os.getenv('GEMINI_MODEL'),  # e.g., "gemini-2.0-flash"
    "api_key": os.getenv('GEMINI_API'), 
    "temperature": 0.4,  # Lowered for more deterministic, accurate outputs
    "max_tokens": 500,
    "top_p": 0.8,  # Slightly tightened for better focus
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5,
    "base_url": "https://generativelanguage.googleapis.com/v1beta",
    "api_type": "google"
}