from config import *
import ast
from tqdm import tqdm
from transformers import AutoTokenizer

# Model name 
MODEL_NAME = "mistral-nemo"
MODEL_VERSION = "12b-instruct-2407-q5_K_S" # 12b-instruct-2407-q5_K_S | 12b-instruct-2407-q4_K_M

# Model parameters
MAX_TOKENS = 131072 # 128k tokens
OUTPUT_TOKENS = 350 # 350 tokens
INPUT_TOKENS = MAX_TOKENS - OUTPUT_TOKENS

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407")

def get_tokenizer():
    return tokenizer
