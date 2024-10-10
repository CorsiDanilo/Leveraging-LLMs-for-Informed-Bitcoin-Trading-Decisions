from rateninja import RateNinja

# Root directory
ROOT = '../'

# func_kwargs folder name
FUNC_KWARGS_FOLDER_NAME = 'func_kwargs'

# Instructions
INSTRUCTIONS = '''
###### INSTRUCTIONS ######
You are an experienced cryptocurrency trader, specializing in Bitcoin. 
What I ask you to do is to consider the following row containing news items and Reddit posts with their respective comments about Bitcoin.
The content includes titles, leadtexts, and bodies for news items, titles, texts, and comments for Reddit posts.
Your task is to say, based on this information, what is the general sentiment of these infromations, whether they are "positive", "neutral" or "negative" for Bitcoin, and what would an experienced trader do, whether they would “buy” “hold” or “sell.”
You have to produce a json file with the following structure without any additional information or explanation:
{"reasoning": "[REASONING_TEXT]", "sentiment": "[SENTIMENT_CLASS]", "action": "[ACTION]", "action_score": [ACTION_SCORE]}   
Where:
[REASONING_TEXT] is a string containing the reasoning that led you to make the [SENTIMENT_CLASS] and [ACTION] decisions. This must be objective, not personal and must contain a couple of short sentences since it must contain only the reasoning that led you to make the [SENTIMENT_CLASS] and [ACTION] decisions.
[SENTIMENT_CLASS] is a string containing the sentiment of the information taken as input having the following values: 'positive', 'neutral', 'negative'.
[ACTION] is a string containing the action you plan to take having the following values: 'buy', 'hold', 'sell'.
[ACTION_SCORE] is an integer between 1 and 10 indicating the confidence level of the action you plan to take (1 means you are not confident at all, 10 means you are very confident).
'''

# Example of the input and output
'''
###### INSTRUCTIONS ######
You are an experienced cryptocurrency trader, specializing in Bitcoin. 
What I ask you to do is to consider the following row containing news items and Reddit posts with their respective comments about Bitcoin.
The content includes titles, leadtexts, and bodies for news items, titles, texts, and comments for Reddit posts.
Your task is to say, based on this information, what is the general sentiment of these infromations, whether they are "positive", "neutral" or "negative" for Bitcoin, and what would an experienced trader do, whether they would “buy” “hold” or “sell.”
You have to produce a JSON file with the following structure without any additional information or explanation:
{"reasoning": "[REASONING_TEXT]", "sentiment": "[SENTIMENT_CLASS]", "action": "[ACTION]", "action_score": [ACTION_SCORE]}   
Where:
[REASONING_TEXT] is a string containing the reasoning that led you to make the [SENTIMENT_CLASS] and [ACTION] decisions. This must be objective, not personal and must contain a couple of short sentences since it must contain only the reasoning that led you to make the [SENTIMENT_CLASS] and [ACTION] decisions.
[SENTIMENT_CLASS] is a string containing the sentiment of the information taken as input having the following values: 'positive', 'neutral', 'negative'.
[ACTION] is a string containing the action you plan to take having the following values: 'buy', 'hold', 'sell'.
[ACTION_SCORE] is an integer between 1 and 10 indicating the confidence level of the action you plan to take (1 means you are not confident at all, 10 means you are very confident).

###### INPUT ######
### BEGINNING OF THE NEWS ### 
### NEWS 1 ###
### TITLE OF NEWS 1 ###
<title>
### LEADTEXT OF NEWS 1 ###
<leadtext>
### BODY OF NEWS 1 ###
<body>
...
### END OF THE NEWS ###
### BEGINNING OF THE REDDIT POSTS ###
### POST 1 ###
### TITLE OF POST 1 ###
<title>
### TEXT OF POST 1 ###
<text>
### BEGINNING OF THE COMMENTS OF POST 1 ###
### COMMENT 1.1 ###
### TEXT OF COMMENT 1.1 ###
<text>
...
### END OF THE COMMENTS OF POST 1 ###
...
### END OF THE REDDIT POSTS ###

###### OUTPUT ######
{"reasoning": "[REASONING_TEXT]", "sentiment": "[SENTIMENT_CLASS]", "action": "[ACTION]", "action_score": [ACTION_SCORE]}   
'''

# RateNinja configurations
RATENINJA = RateNinja(
    max_call_count=15, # 15 calls for each minute
    per_seconds=65, # 1 minute + 5 seconds of margin
    greedy=False,  
    progress_bar=True, 
    max_retries=0,
    max_workers=300, # concurrent calls
)