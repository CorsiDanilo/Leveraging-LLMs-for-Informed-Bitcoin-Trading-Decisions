import datetime

# Select the minimum and maximum date for the date picker 
MIN_DATE = datetime.date(2016, 1, 1)
MAX_DATE = datetime.date(2024, 6, 30)

# Model names
MODEL_NAMES = ["gemini", "llama", "phi", "mistral", "qwen"]

# Strategies
STRATEGIES = {
    1: "Invest all",
    2: "Dollar cost averaging", 
    3: "Fixed investment"
}