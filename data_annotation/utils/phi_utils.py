from config import *
import ast
from tqdm import tqdm
from transformers import AutoTokenizer

# Model name
MODEL_NAME = "phi3"
MODEL_VERSION = "3.8b-mini-128k-instruct-q8_0" # 3.8b-mini-128k-instruct-q8_0 | 14b-medium-128k-instruct-q6_K

# Model parameters
MAX_TOKENS = 131072 # 128k tokens
OUTPUT_TOKENS = 350 # 350 tokens
INPUT_TOKENS = MAX_TOKENS - OUTPUT_TOKENS

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

def get_tokenizer():
    return tokenizer