from config import *
import ast
from tqdm import tqdm

# Initialize Gemini API
MODEL_NAME = "gemini-1.5-flash"

# API parameters
REQUESTS_PER_MINUTE = 15	
REQUESTS_PER_DAY = 1500
TOKENS_PER_MINUTE = 1048576
INPUT_TOKENS = TOKENS_PER_MINUTE

# Model parameters
TEMPERATURE = 0.1
TOP_P = 0.9
TOP_K = 64
MAX_OUTPUT_TOKENS = 8192
RESPONSE_MIME_TYPE = "application/json"

def gemini_configurations():
    generation_config = {
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "max_output_tokens": MAX_OUTPUT_TOKENS,
    "response_mime_type": RESPONSE_MIME_TYPE,
    }

    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
    ]

    return generation_config, safety_settings