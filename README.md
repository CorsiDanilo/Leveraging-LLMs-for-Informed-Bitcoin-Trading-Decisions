# Leveraging LLMs for Informed Bitcoin Trading Decisions: Prompting with Social and News Data Reveals Promising Predictive Abilities

## Description
This thesis investigates the potential of leveraging Large Language Models (LLMs) to support Bitcoin traders. Specifically, we analyze the correlation between Bitcoin price movements and sentiment expressed in news headlines, posts, and comments on social media.
We build a novel, large-scale dataset that aggregates various features related to Bitcoin and its price over time, spanning from 2016 to 2024, and includes data from news outlets, social media posts, and comments.
Using this dataset, we tried to evaluate the effectiveness of deep learning models by trying to predict some target features. Having found bad results, we decided to change approach and use LLM-based predictions on real data through standard classification tasks, as well as backtesting and demo trading accounts with different investment strategies.
We build interactive interfaces to annotate real-time data via LLMs, perform custom backtesting, and visualize demo trading account performances.
Our approach leverages the extended context capabilities of recent LLMs through simple prompting to generate outputs such as textual reasoning, sentiment, recommended trading actions, and confidence scores. Our findings reveal that LLMs represent a powerful tool for assisting trading decisions, opening up promising avenues for future research.

## Dataset
The dataset can be found here: https://huggingface.co/datasets/danilocorsi/Bitcoin-Price-Blockchain-Sentiment-Indexes-News-Reddit-Post-Comments-Dataset
