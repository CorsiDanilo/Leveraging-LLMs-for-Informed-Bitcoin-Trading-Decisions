import os
import datetime

# Check if PyTorch is installed
# If not, show error and proceed
try:
    import torch
except Exception as e:
    print("PyTorch is not installed.")
    
# General configurations
DATA_DIR = "hf_data"
DEMO_DIR = "demo"

# Dataset configurations
# Datasets directories
PRICE_BLOCKCHAIN_DATASET_DIR = "price_blockchain"
INDEXES_DATASET_DIR = "indexes"
SOCIAL_DATASET_DIR = "social"
REDDIT_DATASET_DIR = "reddit"
NEWS_DATASET_DIR = "news"
MERGED_DATASET_DIR = "merged"
ANNOTATED_DATASET_DIR = "annotated"
DEMO_DATASET_DIR = "demo"
DEMO_TODAY_DATASET_DIR = "today"
DEMO_TRADING_DATASET_DIR = "trading"
DATA_EXPLORATORY_ANALYSIS_DIR = "data_exploratory_analysis"
DATA_PREDICTIONS_DIR = "data_predictions"
BACKTEST_DIR = "backtest"

# Datasets paths
PRICE_BLOCKCHAIN_DATASET_PATH = os.path.join(DATA_DIR, PRICE_BLOCKCHAIN_DATASET_DIR)
INDEXES_DATASET_PATH = os.path.join(DATA_DIR, INDEXES_DATASET_DIR)
SOCIAL_DATASET_PATH = os.path.join(DATA_DIR, SOCIAL_DATASET_DIR)
REDDIT_DATASET_PATH = os.path.join(SOCIAL_DATASET_PATH, REDDIT_DATASET_DIR)
NEWS_DATASET_PATH = os.path.join(DATA_DIR, NEWS_DATASET_DIR)
MERGED_DATASET_PATH = os.path.join(DATA_DIR, MERGED_DATASET_DIR)
ANNOTATED_DATASET_PATH = os.path.join(DATA_DIR, ANNOTATED_DATASET_DIR)
DEMO_DATASET_PATH = os.path.join(DATA_DIR, DEMO_DIR)
DEMO_TODAY_DATASET_PATH = os.path.join(DEMO_DATASET_PATH, DEMO_TODAY_DATASET_DIR)
DEMO_TRADING_DATASET_PATH = os.path.join(DEMO_DATASET_PATH, DEMO_TRADING_DATASET_DIR)
DATA_EXPLORATORY_ANALYSIS_PATH = os.path.join(DATA_EXPLORATORY_ANALYSIS_DIR)
DATA_PREDICTIONS_PATH = os.path.join(DATA_PREDICTIONS_DIR)
BACKTEST_PATH = os.path.join(BACKTEST_DIR)

# Dataset parameters
START_DATE = "2016-01-01" # Default: 2019-01-01 (YYY-MM-DD) | First day available: 2016-01-01
END_DATE = "2024-06-31" # Default: 2024-02-29 (YYY-MM-DD) | Today: datetime.date.today().strftime("%Y-%m-%d") | Last day available: 2024-06-30
TIMESPAN = str(int(END_DATE.split("-")[0]) - int(START_DATE.split("-")[0])+1)+"years" # Count the number of years between the start and end date, add 1 to include the last year
PERCENTAGE_CHANGE_THRESHOLD = 2 # Threshold to consider a price change as positive or negative