# Sentiment
trend_mapping = {'down': 0, 'same': 1, 'up': 2}
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
action_mapping = {'sell': 0, 'hold': 1, 'buy': 2}

# Trading
STRATEGIES = {
    1: "invest_all_every_day",
    2: "fixed_dollar_cost_averaging",
    3: "percentage_dollar_cost_averaging",
    4: "fixed_investment_following_llm_opinion",
    5: "percentage_investment_following_llm_opinion"
}

NUM_STRATEGIES = [1, 2, 3, 4, 5]

ACCOUNTS = [1, 2, 3, 4, 5]