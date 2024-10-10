from config import *
import ast
from tqdm import tqdm
from transformers import AutoTokenizer

# Model name 
MODEL_NAME = "qwen2"
MODEL_VERSION = "7b-instruct-q8_0"

# Model parameters
MAX_TOKENS = 131072 # 128k tokens
OUTPUT_TOKENS = 350 # 350 tokens
INPUT_TOKENS = MAX_TOKENS - OUTPUT_TOKENS

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

def get_tokenizer():
    return tokenizer
