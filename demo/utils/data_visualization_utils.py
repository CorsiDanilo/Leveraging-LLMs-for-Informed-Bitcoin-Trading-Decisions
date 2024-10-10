import pandas as pd
import ast
import os
import plotly.graph_objs as go
import requests
import datetime

from config import *
from demo.demo_config import *
from demo.scripts.live_data_annotation import *

def live_data_annotation():
    # Retrieve the data
    retrieve_cointelegraph_news()
    retrieve_bitcoin_news()
    retrieve_reddit_data()
    
    # Annotate the data
    annotate_data_with_llm_offline("llama")
    annotate_data_with_llm_offline("mistral")
    annotate_data_with_llm_offline("phi")
    annotate_data_with_llm_offline("qwen")

    annotate_data_with_gemini()

    # Merge the opinions
    merge_opinions()

def get_today_data():
    # Annotate the data
    if RETRIEVE_DATA:
        live_data_annotation()

    # Load the CSV file
    today_news_and_posts_path = os.path.join(DEMO_TODAY_DATASET_PATH, "today_news_and_reddit_data.csv")
    today_news_and_posts = pd.read_csv(today_news_and_posts_path)

    # Extract news and Reddit data
    cointelegraph_news = ast.literal_eval(today_news_and_posts['cointelegraph'].iloc[0])
    bitcoin_news = ast.literal_eval(today_news_and_posts['bitcoin_news'].iloc[0])
    reddit = ast.literal_eval(today_news_and_posts['reddit'].iloc[0])

    # Extract the news and Reddit data
    cointelegraph_news_list = []
    for n in cointelegraph_news:
        id = n[0]
        link = "https://cointelegraph.com/news/"+n[1]
        score = n[2]
        title = n[3]
        published = n[4]
        leadtext = n[5]
        body = n[6]
        cointelegraph_news_list.append([link, title, published, body])

    bitcoin_news_list = []
    for n in bitcoin_news:
        id = n[0]
        published = n[1]
        title = n[2]
        link = "https://news.bitcoin.com/"+n[3]
        author = n[4]
        body = n[5]
        bitcoin_news_list.append([link, title, published, body])
    
    reddit_list = []
    for p in reddit:
        comments = ast.literal_eval(p[8]) # comments
        comments_list = []
        for c in comments:
            user = c[0]
            score = c[1]
            published = c[2]
            link = c[3]
            text = c[4]
            code = c[5]
            comments_list.append([user, score, published, text])

        user = p[0]
        title = p[1]
        score = p[2]
        published = p[3]
        link = p[4]
        text = p[5]
        url = p[6]
        code = p[7]

        reddit_list.append([user, title, score, published, link, text, comments_list])

    # Display the news
    display = "<h1>Data:</h1>"
    display += "<h2><details><summary>News</summary>"
    display += f"<div style='padding-left: 20px;'>"
    display += "<h3><details><summary>Cointelegraph</summary>"
    display += f"<div style='padding-left: 20px;'>"
    for i, n in enumerate(cointelegraph_news_list):
        link = n[0]
        title = n[1]
        published = n[2]
        body = n[3]

        display += "<blockquote>"
        display += f"<p><b><a href='{link}' target='_blank'>{title}</a></b></p>"
        display += f"<p>Published: {published}</p>"
        display += f"<p>Body: {body}</p>"
        display += "</blockquote>"
        display += "<hr>"

    display += "</div>"
    display += "</details></h3>"

    display += "<h3><details><summary>Bitcoin News</summary>"
    display += f"<div style='padding-left: 20px;'>"
    for i, n in enumerate(bitcoin_news_list):
        link = n[0]
        title = n[1]
        published = n[2]
        body = n[3]

        display += "<blockquote>"
        display += f"<p><b><a href='{link}' target='_blank'>{title}</a></b></p>"
        display += f"<p>Published: {published}</p>"
        display += f"<p>Body: {body}</p>"
        display += "</blockquote>"
        display += "<hr>"

    display += "</div>"
    display += "</div>"

    display += "</details></h3>"
    display += "</details></h2>"

    # Display the Reddit posts and comments
    display += "<h2><details><summary>Social Media</summary>"
    display += f"<div style='padding-left: 20px;'>"
    display += "<h3><details><summary>Reddit</summary>"
    display += f"<div style='padding-left: 20px;'>"
    for i, p in enumerate(reddit_list):
        user = p[0]
        title = p[1]
        score = p[2]
        published = p[3]
        link = p[4]
        text = p[5]

        display += "<blockquote>"
        display += f"<p><b><a href='{link}' target='_blank'>{title}</a></p>"
        display += f"<p>User: {user}</p>"
        display += f"<p>Score: {score}</p>"
        display += f"<p>Published: {published}</p>"
        display += f"<p>Text: {text}</p>"
        display += "<details><summary>Comments</summary>"
        display += f"<div style='padding-left: 20px;'>"

        for j, c in enumerate(p[6]):
            user = c[0]
            score = c[1]
            published = c[2]
            text = c[3]

            # Convert the markdown to text
            text = text.replace("\n", "<br>")

            display += "<blockquote>"
            display += f"<p>User: {user}</p>"
            display += f"<p>Score: {score}</p>"
            display += f"<p>Published: {published}</p>"
            display += f"<p>Text: {text}</p>"
            display += "</blockquote>"
        
        display += "</div>"
        display += "</details>"
        display += "</blockquote>"

        display += "<hr>"
    display += "</div>"
    display += "</details></h3>"
    display += "</div>"
    display += "</details></h2>"

    return display

def get_today_llm_opinions():
    today_llm_opinions_path = os.path.join(DEMO_TODAY_DATASET_PATH, "today_llm_news_and_reddit_data_opinions.csv")
    today_llm_opinions = pd.read_csv(today_llm_opinions_path)

    # Extract the model's opinion
    phi_opinion = {
        "action_class": today_llm_opinions['phi_action_class'].iloc[0],
        "action_score": today_llm_opinions['phi_action_score'].iloc[0],
        "sentiment_class": today_llm_opinions['phi_sentiment_class'].iloc[0],
        "reasoning_text": today_llm_opinions['phi_reasoning_text'].iloc[0]
    }

    mistral_opinion = {
        "action_class": today_llm_opinions['mistral_action_class'].iloc[0],
        "action_score": today_llm_opinions['mistral_action_score'].iloc[0],
        "sentiment_class": today_llm_opinions['mistral_sentiment_class'].iloc[0],
        "reasoning_text": today_llm_opinions['mistral_reasoning_text'].iloc[0]
    }

    llama_opinion = {
        "action_class": today_llm_opinions['llama_action_class'].iloc[0],
        "action_score": today_llm_opinions['llama_action_score'].iloc[0],
        "sentiment_class": today_llm_opinions['llama_sentiment_class'].iloc[0],
        "reasoning_text": today_llm_opinions['llama_reasoning_text'].iloc[0]
    }

    qwen_opinion = {
        "action_class": today_llm_opinions['qwen_action_class'].iloc[0],
        "action_score": today_llm_opinions['qwen_action_score'].iloc[0],
        "sentiment_class": today_llm_opinions['qwen_sentiment_class'].iloc[0],
        "reasoning_text": today_llm_opinions['qwen_reasoning_text'].iloc[0]
    }

    gemini_opinion = {
        "action_class": today_llm_opinions['gemini_action_class'].iloc[0],
        "action_score": today_llm_opinions['gemini_action_score'].iloc[0],
        "sentiment_class": today_llm_opinions['gemini_sentiment_class'].iloc[0],
        "reasoning_text": today_llm_opinions['gemini_reasoning_text'].iloc[0]
    }

    # Compute mean opinion
    sentiment_classes = [phi_opinion["sentiment_class"], mistral_opinion["sentiment_class"], llama_opinion["sentiment_class"], qwen_opinion["sentiment_class"], gemini_opinion["sentiment_class"]]
    mean_sentiment_class = max(set(sentiment_classes), key=sentiment_classes.count) # Get the most common sentiment class
    
    action_classes = [phi_opinion["action_class"], mistral_opinion["action_class"], llama_opinion["action_class"], qwen_opinion["action_class"], gemini_opinion["action_class"]]
    mean_action_class = max(set(action_classes), key=action_classes.count) # Get the most common action class
    
    action_scores = [phi_opinion["action_score"], mistral_opinion["action_score"], llama_opinion["action_score"], qwen_opinion["action_score"], gemini_opinion["action_score"]]
    mean_action_score = sum(action_scores) / len(action_scores) # Get the mean action score

    mean_opinion = {
        "action_class": mean_action_class,
        "action_score": mean_action_score,
        "sentiment_class": mean_sentiment_class,
    }

    models_opinions = {
        "phi": phi_opinion,
        "mistral": mistral_opinion,
        "llama": llama_opinion,
        "qwen": qwen_opinion,
    }

    # Define functions to get color based on sentiment and action
    def get_action_color(action_class):
        if action_class == "buy":
            return "green"
        elif action_class == "hold":
            return "orange"
        elif action_class == "sell":
            return "red"
        else:
            return "white"  # Default color for unknown sentiment

    def get_sentiment_color(sentiment_class):
        if sentiment_class == "positive":
            return "green"
        elif sentiment_class == "neutral":
            return "orange"
        elif sentiment_class == "negative":
            return "red"
        else:
            return "white"  # Default color for unknown action

    def get_action_score_color(action_score):
        if action_score >= 7:
            return "green"
        elif action_score >= 4:
            return "orange"
        elif action_score >= 0:
            return "red"
        else:
            return "white"

    # Display the mean opinion
    display = "<h1>Overall</h1>"
    action_class = mean_opinion["action_class"]
    action_score = mean_opinion["action_score"]
    sentiment_class = mean_opinion["sentiment_class"]

    display += f"<h2 style='text-align: center; color:{get_action_score_color(action_score)}'>Action: {action_class}</h2>"
    display += f"<h3 style='text-align: center; color:{get_action_score_color(action_score)}'>Confidence score: {action_score}/10</h3>"
    display += f"<h3 style='text-align: center; color:{get_sentiment_color(sentiment_class)}';>Sentiment: {sentiment_class}</h3>"	
    display += "<hr>"

    # # Display the best model's opinion (gemini)
    display += "<h2>Best Model (Gemini)</h2>"
    action_class = gemini_opinion["action_class"]
    action_score = gemini_opinion["action_score"]
    sentiment_class = gemini_opinion["sentiment_class"]
    reasoning_text = gemini_opinion["reasoning_text"]

    display += f"<h3 style='text-align: center; color:{get_action_color(action_class)}'>Action: {action_class}</h3>"
    display += f"<h3 style='text-align: center; color:{get_action_score_color(action_score)}'>Confidence score: {action_score}/10</h3>"
    display += f"<h3 style='text-align: center; color:{get_sentiment_color(sentiment_class)}'>Sentiment: {sentiment_class}</h3>"
    display += f"<h3 style='text-align: center'>Reasoning:</h3>"
    display += f"<h3 style='text-align: center'>{reasoning_text}</h3>"
    display += "<hr>"

    # Display the model's opinion
    display += "<h2><details><summary>Models</summary>"

    for model, opinion in models_opinions.items():
        action_class = opinion["action_class"]
        action_score = opinion["action_score"]
        sentiment_class = opinion["sentiment_class"]
        reasoning_text = opinion["reasoning_text"]

        display += f"<h2 style='text-align: center'>Model: {model.upper()}</h2>"
        display += f"<h3 style='text-align: center; color:{get_action_color(action_class)}'>Action: {action_class}</h3>"
        display += f"<h3 style='text-align: center; color:{get_action_score_color(action_score)}'>Confidence score: {action_score}/10</h3>"
        display += f"<h3 style='text-align: center; color:{get_sentiment_color(sentiment_class)}'>Sentiment: {sentiment_class}</h3>"
        display += f"<h3 style='text-align: center'>Reasoning:</h3>"
        display += f"<h3 style='text-align: center'>{reasoning_text}</h3>"
        display += "<hr>"

    display += "</details>"
    display += "</h2>"

    return display

def get_bitcoin_price_graph():
    # Calculate the timestamp for 1 year ago
    today = datetime.datetime.now()
    one_year_ago = today - datetime.timedelta(days=365)

    # Convert datetime to Unix timestamp
    start_date = int(one_year_ago.timestamp())
    end_date = int(today.timestamp())

    # Define the URL for CoinGecko API
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"

    # Define the parameters
    params = {
        'vs_currency': 'usd',
        'from': start_date,
        'to': end_date
    }

    # Fetch the data
    response = requests.get(url, params=params)
    data = response.json()

    # Extract prices
    prices = data['prices']
    
    # Plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[datetime.datetime.fromtimestamp(p[0]/1000) for p in prices], y=[p[1] for p in prices], mode='lines', name='Bitcoin Price'))
    fig.update_layout(title='Bitcoin Price Chart', xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_white')

    return fig