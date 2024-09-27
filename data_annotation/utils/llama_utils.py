from config import *
import ast
from tqdm import tqdm
from transformers import AutoTokenizer

# Model name 
MODEL_NAME = "llama3.1"
MODEL_VERSION = "8b-instruct-q6_K" # 8b-instruct-q8_0 | 8b-instruct-q6_K 

# Model parameters
MAX_TOKENS = 131072 # 128k tokens
OUTPUT_TOKENS = 350 # 350 tokens
INPUT_TOKENS = MAX_TOKENS - OUTPUT_TOKENS

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

def get_tokenizer():
    return tokenizer