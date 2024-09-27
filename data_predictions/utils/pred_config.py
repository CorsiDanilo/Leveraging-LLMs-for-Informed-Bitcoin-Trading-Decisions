# General configurations
PATH_TO_SAVE_RESULTS = "checkpoints"
PATH_TO_SAVE_PLOTS = "plots"
RANDOM_SEED = 42

# Dataset parameters
DATASET_TYPE = 'daily' #  daily | hourly
LLM_MODEL = "gemini-1.5-flash" # gemini-1.5-flash | phi3_3.8b-mini-128k-instruct-q8_0 | mistral-nemo_12b-instruct-2407-q5_K_S_ | llama3.1_8b-instruct-q6_K | qwen2_7b-instruct-q8_0
TARGET_FEATURE = 'pct_price_change' # pct_price_change | trend

# Model parameters
MODEL_NAME = 'transformer_base' # lstm_base | transformer_base | autoformer | informer | prob_transformer

# Data parameters
USE_LLM_FEATURES = True # Decide whether to use the Gemini features or not
SCALE_DATA = True # Choose if the data should be scaled or not
USE_WANDB = False # Configure wandb

# Split date
DATE = '2023-06-30'
SPLIT_DATE = DATE if DATASET_TYPE == 'daily' else DATE + '00:00:00'

# Model parameters
OUTPUT_SIZE = 1 if TARGET_FEATURE == 'pct_price_change' else 3

## Hyperparameters
BATCH_SIZE = 64
EMB_DIM = 64
EPOCHS = 500
LR = 1e-3
REG = 1e-3
DROPOUT = 0.2

## LSTM Hyperparameters
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 4

## Transformer Hyperparameters
TF_MODEL_DIM = 128
TF_NUM_HEADS = 8
TF_NUM_LAYERS = 4