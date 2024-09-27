from config import *
import ast
from tqdm import tqdm
import json

# Tokenizer
tokenizer = None
live_annotation = False

def tokenize_text(input_text):
    return len(tokenizer.tokenize(input_text))

def trim_reddit_posts(reddit_posts, max_reddit_text_length):
    print(f"-- Trimming Reddit posts to fit the limit ({max_reddit_text_length} tokens for the reddit posts)")
    print(f"-- Total posts: {len(reddit_posts)}")

    # Order the Reddit posts based on their score (post[2]) from highest to lowest
    reddit_posts_sorted = sorted(reddit_posts, key=lambda x: x[2], reverse=True)

    # Prepare the beginning and ending markers
    beginning_marker = "### BEGINNING OF THE REDDIT POSTS ###\n"
    ending_marker = "### END OF THE REDDIT POSTS ###\n"

    # Initialize the text with the beginning marker
    tmp_reddit_text = beginning_marker

    # Function to get the current token count including the ending marker
    def current_token_count():
        return tokenize_text(tmp_reddit_text + ending_marker)

    # Insert the first posts until the limit is reached
    new_reddit_posts = []
    for p in reddit_posts_sorted:
        reddit_text_to_append, _ = extract_reddit_posts([p])
        if current_token_count() + tokenize_text(reddit_text_to_append) > max_reddit_text_length:
            break

        tmp_reddit_text += reddit_text_to_append
        new_reddit_posts.append(p)

    print(f"-- Total posts after trimming: {len(new_reddit_posts)}, posts removed: {len(reddit_posts) - len(new_reddit_posts)}")

    # Extract the Reddit posts and comments from the trimmed list
    reddit_text, _ = extract_reddit_posts(new_reddit_posts)
    reddit_text = beginning_marker + reddit_text + ending_marker

    # Check the length of the Reddit posts after trimming
    if tokenize_text(reddit_text) > max_reddit_text_length:
        print(f"-- Error: The number of tokens exceeds the limit ({tokenize_text(reddit_text)}/{max_reddit_text_length})")
        # Retry trimming by reducing posts further
        return trim_reddit_posts(new_reddit_posts[:-1], max_reddit_text_length)

    return reddit_text

def extract_news(news, news_type):
    news_text = ""
    if news_type == "cointelegraph_news":
            for i, n in enumerate(news):
                try:
                    title = n[3]
                    leadtext = n[5]
                    body = n[6]
                except Exception as e:
                    print(f"Error: {e}")
                    title = "none_title"
                    leadtext = "none_leadtext"
                    body = "none_body"
                    
                news_text += f"### NEWS {i+1} ###\n"
                news_text += f"### TITLE OF NEWS {i+1} ###\n"
                news_text += title + "\n"
                news_text += f"### LEADTEXT OF NEWS {i+1} ###\n"
                news_text += leadtext + "\n"
                news_text += f"### BODY OF NEWS {i+1} ###\n"
                news_text += body + "\n"

    elif news_type == "bitcoin_news":
            for i, n in enumerate(news):
                try:
                    title = n[2]
                    text = n[5]
                except Exception as e:
                    print(f"Error: {e}")
                    title = "none_title"
                    text = "none_text"
                    
                news_text += f"### NEWS {i+1} ###\n"
                news_text += f"### TITLE OF NEWS {i+1} ###\n"
                news_text += title + "\n"
                news_text += f"### TEXT OF NEWS {i+1} ###\n"
                news_text += text + "\n"
    else:
        print(f"-- Error: Invalid news type")
        return "none_news"

    return news_text

def extract_reddit_comments(p, i):
    reddit_comments_text = f"### BEGINNING OF THE COMMENTS OF POST {i+1} ###\n"
    try:
        # Retrieve the comments from the post
        if live_annotation == False:
            reddit_comments = p[8] if p[8] != '[]' else []
        else:
            reddit_comments = ast.literal_eval(p[8]) if p[8] != '[]' else []
    except ValueError as e:
        print(f"Error: {e}")
        return "none_comments"

    for j, c in enumerate(reddit_comments):
        try:
            text = c[5]
        except Exception as e:
            print(f"Error: {e}")
            text = "none_text"
        reddit_comments_text += f"### COMMENT {i+1}.{j+1} ###\n"
        reddit_comments_text += f"### TEXT OF COMMENT {i+1}.{j+1} ###\n"
        reddit_comments_text += text + "\n"
    reddit_comments_text += f"### END OF THE COMMENTS OF POST {i+1} ###\n"

    return reddit_comments_text

def extract_reddit_posts(reddit_posts):
    reddit_posts_text = ""
    for i, p in enumerate(reddit_posts):
        try:
            title = p[1]
            text = p[5]
        except Exception as e:
            print(f"Error: {e}")
            title = "none_title"
            text = "none_text"

        reddit_posts_text += f"### POST {i+1} ###\n"
        reddit_posts_text += f"### TITLE OF POST {i+1} ###\n"
        reddit_posts_text += title + "\n"
        reddit_posts_text += f"### TEXT OF POST {i+1} ###\n"
        reddit_posts_text += text + "\n"

        # Extract the comments
        reddit_comments_text = extract_reddit_comments(p, i)
        
        # Append the comments to the posts
        reddit_posts_text += reddit_comments_text

    return reddit_posts_text, reddit_posts

def check_response(response_json):
    # Get the values from the response
    try:
        reasoning_text = response_json['reasoning']
        sentiment_class = response_json['sentiment']
        action_class = response_json['action']
        action_score = response_json['action_score']
    except Exception as e:
        print(f"Error: {e}")
        return ("none_reasoning", "none_sentiment", "none_action", "none_score")

    # Lowercase the sentiment and action classes
    sentiment_class = sentiment_class.lower()
    action_class = action_class.lower()

    # Check values
    if not isinstance(reasoning_text, str):
        print(f"Error: Invalid reasoning text")
        return ("none_reasoning", "none_sentiment", "none_action", "none_score")
    if sentiment_class not in ['positive', 'neutral', 'negative']:
        print(f"Error: Invalid sentiment class")
        return ("none_reasoning", "none_sentiment", "none_action", "none_score")
    if action_class not in ['buy', 'hold', 'sell']:
        print(f"Error: Invalid action class")
        return ("none_reasoning", "none_sentiment", "none_action", "none_score")
    if not isinstance(action_score, int) or action_score < 1 or action_score > 10:
        print(f"Error: Invalid action score")
        return ("none_reasoning", "none_sentiment", "none_action", "none_score")

    return reasoning_text, sentiment_class, action_class, action_score

def populate_func_kwargs_loop(model_name, func_kwargs, count, merged_dataset, opinion_dataset, max_tokens, instructions):
    for index, row in tqdm(merged_dataset.iterrows(), total=merged_dataset.shape[0]):
        # Check if the current row in opinion_dataset dataset is already populated by reasoning_text, sentiment_class, action_class and action_score values or has NaN values
        if opinion_dataset.loc[index, 'reasoning_text'] == 'none_reasoning_text' or opinion_dataset.loc[index, 'sentiment_class'] == 'none_sentiment_class' or opinion_dataset.loc[index, 'action_class'] == 'none_action_class' or opinion_dataset.loc[index, 'action_score'] == 'none_action_score':
            # Add the input text header
            input_text = ""
            input_text += "\n###### INPUT ######\n"

            # Extract the news
            cointelegraph_news = ast.literal_eval(row['cointelegraph']) if row['cointelegraph'] != '[]' else []
            bitcoin_news = ast.literal_eval(row['bitcoin_news']) if row['bitcoin_news'] != '[]' else []
            news_text = ""
            news_text += "### BEGINNING OF THE COINTELEGRAPH NEWS ###\n"
            news_text += extract_news(cointelegraph_news, "cointelegraph_news")
            news_text += "### END OF THE COINTELEGRAPH NEWS ###\n"
            news_text += "### BEGINNING OF THE BITCOIN NEWS ###\n"
            news_text += extract_news(bitcoin_news, "bitcoin_news")
            news_text += "### END OF THE BITCOIN NEWS ###\n"
            news_text += "### END OF THE NEWS ###\n"

            # Extract the Reddit posts and comments
            reddit_posts = ast.literal_eval(row['reddit']) if row['reddit'] != '[]' else []
            reddit_text = ""
            reddit_text += "### BEGINNING OF THE REDDIT POSTS ###\n"
            reddit_text_to_append, reddit_posts = extract_reddit_posts(reddit_posts)
            reddit_text += reddit_text_to_append
            reddit_text += "### END OF THE REDDIT POSTS ###\n"

            # Check the model name to determine the input text length
            if model_name == "phi3" or model_name == "llama3.1" or model_name == "mistral-nemo" or model_name == "qwen2":
                news_length = tokenize_text(news_text)
                reddit_length = tokenize_text(reddit_text)
                instructions_length = tokenize_text(instructions)

                # Calculate the total length of the input text
                input_text_length = news_length + reddit_length + instructions_length

                # Check if the total length exceeds the limit
                if input_text_length > max_tokens:
                    print(f"-- Error: The number of tokens exceeds the limit ({input_text_length}/{max_tokens})")
                    max_reddit_text_length = max_tokens - news_length - instructions_length
                    
                    # Trim the Reddit posts to fit the limit 
                    reddit_text = trim_reddit_posts(reddit_posts, max_reddit_text_length)

                    print(f"-- Total length after trimming: {tokenize_text(news_text) + tokenize_text(reddit_text) + tokenize_text(instructions)}")
                    
            # Append the input text and the index to the func_kwargs list
            input_text += news_text + reddit_text + instructions

            # Append the input text and the index to the func_kwargs list
            func_kwargs.append(
                {"index": index, "input_text": input_text}
            )

            # Count the number of row affected
            count += 1

    return func_kwargs, count

def populate_func_kwargs(model_name, merged_dataset, opinion_dataset, max_tokens, instructions, model_tokenizer=None, test=False, live=False): 
    # For each row in the dataset, populate the func_kwargs list with the input text and the index of each row
    func_kwargs = []
    count = 0

    # If test is True, only take the first 10 rows
    if test:
        rows = 10
        print(f"-- Test mode enabled (using only the first {rows} rows)")
        merged_dataset = merged_dataset.head(10)

    # Set the tokenizer
    if model_tokenizer is not None:
        global tokenizer
        tokenizer = model_tokenizer
        print(f"-- Tokenizer set to {model_name}")

    if live:
        print(f"-- Live annotation enabled")
        global live_annotation
        live_annotation = live

    # Call the populate_func_kwargs_loop function
    func_kwargs, count = populate_func_kwargs_loop(model_name, func_kwargs, count, merged_dataset, opinion_dataset, max_tokens, instructions)

    # Print the number of rows affected
    print(f"-- {count}/{opinion_dataset.shape[0]} rows affected")

    return func_kwargs