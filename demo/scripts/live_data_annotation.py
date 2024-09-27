from config import *
from datetime import datetime as dt

# Full command: python -m demo.scripts.live_data_annotation
# Full command: C:\Users\User1\Desktop\Sentiment-Augmented-Bitcoin-Price-Prediction\.venv\Scripts\python.exe -m demo.scripts.live_data_annotation

TODAY = dt.now().strftime('%Y-%m-%d')

######################################################
# Coitelegraph News                                  #
######################################################
                        
def retrieve_cointelegraph_news():
    print("Retrieving Coitelegraph news...")
    import subprocess
    import json

    # Parameters
    short = "en"
    slug = "bitcoin"
    order = "postPublishedTime"
    offset = str(0) # From the last news published
    length = str(100) # Up to 100 news

    # Define the curl command with parameters
    curl_command = [
        "curl",
        "https://conpletus.cointelegraph.com/v1/",
        "--compressed",
        "-X", "POST",
        "-H", "Accept-Encoding: gzip, deflate",
        "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "-H", "Accept: application/graphql-response+json, application/graphql+json, application/json, text/event-stream, multipart/mixed",
        "-H", "Accept-Language: en-US,en;q=0.5",
        "-H", "Accept-Encoding: gzip, deflate, br, zstd",
        "-H", "Referer: https://cointelegraph.com/",
        "-H", "baggage: ",
        "-H", "sentry-trace: ",
        "-H", "content-type: application/json",
        "-H", "Origin: https://cointelegraph.com",
        "-H", "Connection: keep-alive",
        "-H", "Sec-Fetch-Dest: empty",
        "-H", "Sec-Fetch-Mode: cors",
        "-H", "Sec-Fetch-Site: same-site",
        "-H", "Priority: u=4",
        "-H", "TE: trailers",
        "--data-raw", '{"operationName":"TagPageQuery","query":"query TagPageQuery($short: String, $slug: String!, $order: String, $offset: Int!, $length: Int!) {\\n  locale(short: $short) {\\n    tag(slug: $slug) {\\n      id\\n      slug\\n      avatar\\n      createdAt\\n      updatedAt\\n      redirectRelativeUrl\\n      alternates {\\n        short\\n        domain\\n        id\\n        code\\n        __typename\\n      }\\n      tagTranslates {\\n        id\\n        title\\n        metaTitle\\n        pageTitle\\n        description\\n        metaDescription\\n        keywords\\n        __typename\\n      }\\n      posts(order: $order, offset: $offset, length: $length) {\\n        data {\\n          id\\n          slug\\n          views\\n          postTranslate {\\n            id\\n            title\\n            avatar\\n            published\\n            publishedHumanFormat\\n            leadText\\n            author {\\n              id\\n              slug\\n              authorTranslates {\\n                id\\n                name\\n                __typename\\n              }\\n              __typename\\n            }\\n            __typename\\n          }\\n          category {\\n            id\\n            slug\\n            __typename\\n          }\\n          author {\\n            id\\n            slug\\n            authorTranslates {\\n              id\\n              name\\n              __typename\\n            }\\n            __typename\\n          }\\n          postBadge {\\n            id\\n            label\\n            postBadgeTranslates {\\n              id\\n              title\\n              __typename\\n            }\\n            __typename\\n          }\\n          showShares\\n          showStats\\n          __typename\\n        }\\n        postsCount\\n        __typename\\n      }\\n      __typename\\n    }\\n    __typename\\n  }\\n}","variables":{"cacheTimeInMS":300000,"length":'+length+',"offset":'+offset+',"order":"'+order+'","short":"'+short+'","slug":"'+slug+'"}}',
    ]

    # Execute the curl command
    result = subprocess.run(curl_command, capture_output=True, text=True).stdout

    # Turn result.stdout into json
    data = json.loads(result)

    # Save in a list all the news
    news = data["data"]["locale"]["tag"]["posts"]["data"]

    import pandas as pd

    # List to hold dictionaries
    news_list = []

    # Loop through news items and create dictionaries
    for n in news:
        id = n["id"] if "id" in n else 'none_id'
        slug = n["slug"] if "slug" in n else 'none_slug'
        views = n["views"] if "views" in n else 'none_views'
        title = n["postTranslate"]["title"] if "postTranslate" in n else 'none_title'
        published = n["postTranslate"]["published"] if "postTranslate" in n else 'none_published'
        leadtext = n["postTranslate"]["leadText"] if "postTranslate" in n else 'none_leadtext'

        news_dict = {
            "id": id,
            "slug": slug,
            "views": views,
            "title": title,
            "published": published,
            "leadtext": leadtext,
        }

        news_list.append(news_dict)

    # Create DataFrame from list of dictionaries
    df = pd.DataFrame(news_list)

    # Convert "2024-07-17T15:30:28+01:00" to "2024-07-17 15:30:28"
    df["published"] = pd.to_datetime(df["published"])
    df["published"] = df["published"].dt.strftime("%Y-%m-%d %H:%M:%S")

    from datetime import datetime as dt

    # Retrieve today's date
    today = TODAY

    # Filter the DataFrame to only show news published today
    cointelegraph_today = df[df["published"].str.contains(today)]

    ######################################################
    # [ADDON] Get "body" field of Coitelegraph news      #
    ######################################################
    print("Retrieving Coitelegraph news body...")
    # Add new column "body" to cointelegraph_with_body_with_sentiment between "leadtext" and "sentiment"
    cointelegraph_with_body = cointelegraph_today.copy()
    cointelegraph_with_body.insert(6, "body", 'none_body')

    def find_text_to_remove(body):
        # Extract all the <p ...>...</p> tags that contains <strong>...</strong> with <em>Related:</em> inside them
        # Find all the <p> tags
        text_to_remove = []
        p_tags = body.find_all('p')
        for p in p_tags:
            # Find all the <strong> tags inside the <p> tag
            strong_tags = p.find_all('strong')
            for strong in strong_tags:
                # Find all the <em> tags inside the <strong> tag
                em_tags = strong.find_all('em')
                for em in em_tags:
                    # Check if the <em> tag contains the text "Related:"
                    if "Related:" in em.get_text():
                        text_to_remove.append(p)
                        
        # Extract all the <p ...>...</p> tags that contains <strong>...</strong> with <em>Magazine:</em> inside them
        p_tags = body.find_all('p')
        for p in p_tags:
            # Find all the <strong> tags inside the <p> tag
            strong_tags = p.find_all('strong')
            for strong in strong_tags:
                # Find all the <em> tags inside the <strong> tag
                em_tags = strong.find_all('em')
                for em in em_tags:
                    # Check if the <em> tag contains the text "Magazine:"
                    if "Magazine:" in em.get_text():
                        text_to_remove.append(p)

        # Extract all the <p ...>...</p> tags that contains <strong>...</strong> with <em>Recent:</em> inside them
        p_tags = body.find_all('p')
        for p in p_tags:
            # Find all the <strong> tags inside the <p> tag
            strong_tags = p.find_all('strong')
            for strong in strong_tags:
                # Find all the <em> tags inside the <strong> tag
                em_tags = strong.find_all('em')
                for em in em_tags:
                    # Check if the <em> tag contains the text "Recent:"
                    if "Recent:" in em.get_text():
                        text_to_remove.append(p)

        return text_to_remove

    import requests
    from bs4 import BeautifulSoup

    def retrieve_html(url):
        soup = None

        # With Charles Proxy/HTTP Toolkit
        headers = {
            "Host": "cointelegraph.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://cointelegraph.com/tags/bitcoin",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Priority": "u=0, i",
            "TE": "trailers"
        }

        # With Requestly Proxy
        # Retrieve the cookie from the secrets/cookies.json file
        # with open(os.path.join('secrets/cookies.json')) as file:
        #     cookie = json.load(file)['cookie']

        # headers = {
        #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:130.0) Gecko/20100101 Firefox/130.0",
        #     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8",
        #     "Accept-Language": "en-US,en;q=0.5",
        #     "Accept-Encoding": "gzip, deflate, br, zstd",
        #     "Connection": "keep-alive",
        #     "Cookie": cookie,
        #     "Upgrade-Insecure-Requests": "1",
        #     "Sec-Fetch-Dest": "document",
        #     "Sec-Fetch-Mode": "navigate",
        #     "Sec-Fetch-Site": "none",
        #     "Sec-Fetch-User": "?1",
        #     "Priority": "u=0, i"
        # }

        # Get the body of the article from the URL
        response = requests.get(url, headers=headers)
        
        # Retry if the status code is not 200
        if response.status_code != 200:
            count = 0
            while response.status_code != 200 and count < 5:
                response = requests.get(url, headers=headers)
                count += 1
        else:
            # Parse the HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

        return soup

    def get_cointelegraph_body(url, cus_class):
        import requests

        try:
            # Get the body of the article from the URL
            soup = retrieve_html(url)

            if soup == None:
                print(f"Failed to get the content of the article: {url}")
                return 'none_body'

            # Find the body of the article
            body = soup.find('div', class_=cus_class)

            # Clean the body
            text_to_remove = find_text_to_remove(body)
        except Exception as e:
            print(e)
            print(f"Failed to find the body of the article: {url}")
            return 'none_body'

        # Get the text of the body
        body_text = body.get_text()

        # Remove the text that contains "Related:"
        for text in text_to_remove:
            body_text = body_text.replace(text.get_text(), '')
        
        if body_text == '':      
            print(f"Failed to find the body of the article in the URL: {url}")
            return 'none_body'
        return body_text

    from tqdm import tqdm
    from time import sleep
    import os

    # For each news open cointelegrap.com/news/{slug} and save the content
    for index, row in tqdm(cointelegraph_with_body.iterrows(), total=len(cointelegraph_with_body)):
        try:
            # Get the slug
            slug = row['slug']

            # Open the URL
            url = f"https://cointelegraph.com/news/{slug}"
            
            # Get the body
            cus_class = 'post-content relative'
            body = get_cointelegraph_body(url, cus_class)

            # Save the body into the dataset
            cointelegraph_with_body.at[index, 'body'] = body
        except Exception as e:
            print(e)
            cointelegraph_with_body.at[index, 'body'] = 'none_body'

    # Save the dataset to a CSV file
    output_file = os.path.join(DEMO_TODAY_DATASET_PATH, 'today_cointelegraph_with_body.csv')
    cointelegraph_with_body.to_csv(output_file, index=False)

######################################################
# Bitcoin News                                      #
######################################################

def retrieve_bitcoin_news():
    print("Retrieving Bitcoin news...")
    import subprocess
    import json

    from datetime import datetime as dt

    # Retrieve today's date
    today = TODAY

    # Define the parameters
    start = 0 # Default: 0
    MAX_LENGHT = 100 # Max: 100
    offset = str(start) # From the last news published
    per_page = str(MAX_LENGHT)

    data = []
    while True:
        try:
            print(f"Fetching data from {start} to {start + MAX_LENGHT}")
            # Define the curl command with parameters
            curl_command = [
                "curl",
                f"https://api.news.bitcoin.com/wp-json/bcn/v1/posts?offset={offset}&per_page={per_page}",
                "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
                "-H", "Accept: application/json, text/plain, */*",
                "-H", "Accept-Language: en-US,en;q=0.5",
                "-H", "Referer: https://news.bitcoin.com/",
                "-H", "Origin: https://news.bitcoin.com",
                "-H", "Connection: keep-alive",
                "-H", "Sec-Fetch-Dest: empty",
                "-H", "Sec-Fetch-Mode: cors",
                "-H", "Sec-Fetch-Site: same-site",
                "-H", "TE: trailers"
            ]

            # Execute the curl command
            result = subprocess.run(curl_command, capture_output=True, text=True).stdout

            # Turn result.stdout into json
            data.append(json.loads(result)['posts'])

            print(f"Data fetched from {start} to {start + MAX_LENGHT}")
            print(f"last post published at {data[-1][-1]['date']}")

            # Check if the last post is published today
            if dt.strptime(data[-1][-1]['date'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d') != today:
                print(f"Last post published at {data[-1][-1]['date']} is not today")
                break

            # Increase the offset
            start += MAX_LENGHT
            offset = str(start)
        except Exception as e:
            print(e)
            break

    import pandas as pd
    from datetime import datetime as dt

    # List to hold dictionaries
    news_list = []

    # Loop through news items and create dictionaries
    for i in range(len(data)):
        for d in data[i]:
            if dt.strptime(d["date"], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d') == today:

                id = d["id"] if "id" in d else 'none_id'
                date = d["date"] if "date" in d else 'none_date'
                title = d["title"] if "title" in d else 'none_title'
                slug = d["slug"] if "slug" in d else 'none_slug'
                author = d["author"]["name"] if "author" in d else 'none_author'

                news_dict = {
                    "id": id,
                    "date": date,
                    "title": title,
                    "slug": slug,
                    "author": author
                }

                news_list.append(news_dict)

    # Create DataFrame from list of dictionaries
    df = pd.DataFrame(news_list)

    ######################################################
    # [ADDON] Get "body" field of Bitcoin news          #
    ######################################################
    print("Retrieving Bitcoin news body...")
    # Add new column "body" to bitcoin_news_with_body_with_sentiment between "leadtext" and "sentiment"
    bitcoin_news_with_body = df.copy()
    bitcoin_news_with_body.insert(5, "body", 'none_body')

    import re
    from html import unescape

    def clean_html_content(html_content):
        # Remove script tags and their contents
        html_content = re.sub(r'<script.*?</script>', '', html_content, flags=re.DOTALL)
        
        # Remove HTML tags
        html_content = re.sub(r'<[^>]+>', '', html_content)
        
        # Unescape HTML entities
        html_content = unescape(html_content)
        
        # Remove extra whitespace and blank lines
        lines = [line.strip() for line in html_content.split('\n') if line.strip()]
        cleaned_content = '\n'.join(lines)
        
        return cleaned_content

    import subprocess
    import json
    from tqdm import tqdm

    for index, row in tqdm(bitcoin_news_with_body.iterrows(), total=len(bitcoin_news_with_body)):
        # Get the slug
        slug = row['slug']
        # Open the URL
        url = f'https://news.bitcoin.com/{slug}'

        try:
            # Define the curl command with parameters
            curl_command = [
                "curl",
                f"https://api.news.bitcoin.com/wp-json/bcn/v1/post?slug={slug}",
                "--compressed",
                "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
                "-H", "Accept: application/json, text/plain, */*",
                "-H", "Accept-Language: en-US,en;q=0.5",
                "-H", "Accept-Encoding: gzip, deflate",
                "-H", "Referer: https://news.bitcoin.com/",
                "-H", "Origin: https://news.bitcoin.com",
                "-H", "Connection: keep-alive",
                "-H", "Sec-Fetch-Dest: empty",
                "-H", "Sec-Fetch-Mode: cors",
                "-H", "Sec-Fetch-Site: same-site",
                "-H", "TE: trailers"
            ]

            # Execute the curl command
            result = subprocess.run(curl_command, capture_output=True, text=True).stdout

            # Extract the body from the result
            body = json.loads(result)['content']

            # Clean the body
            cleaned_body = clean_html_content(body)

            # Save the body into the dataset
            bitcoin_news_with_body.at[index, 'body'] = body
        except Exception as e:
            print(e)
            # Set the body to None
            bitcoin_news_with_body.at[index, 'body'] = 'none_body'

    ######################################################
    # [ADDON] Clear Bitcoin News body                    #
    ###################################################### 
    print("Cleaning Bitcoin news body...")
    # Convert the body column to string
    bitcoin_news_with_body['body'] = bitcoin_news_with_body['body'].astype(str)
    
    import re

    def clean_html(html_text):
        # Remove HTML tags
        clean_text = re.sub('<[^<]+?>', '', html_text)
        
        # Remove extra whitespace
        clean_text = re.sub('\s+', ' ', clean_text).strip()
        
        return clean_text

    for index, row in tqdm(bitcoin_news_with_body.iterrows(), total=len(bitcoin_news_with_body)):
        # Clean the body
        body = row['body']
        cleaned_body = clean_html(body)

        # Save the cleaned body
        bitcoin_news_with_body.at[index, 'body'] = cleaned_body

    # Save the dataset to a CSV file
    output_file = os.path.join(DEMO_TODAY_DATASET_PATH, 'today_bitcoin_news_with_body.csv')
    bitcoin_news_with_body.to_csv(output_file, index=False)

######################################################
# Reddit Data                                       #
######################################################

def retrieve_reddit_data():
    print("Retrieving Reddit data...")
    from datetime import datetime as dt
    from datetime import timedelta
    import json
    from tqdm import tqdm
    import praw
    import os
    import pandas as pd

    RETRIEVE_SUBREDDITS = False # Set to True to retrieve the subreddits
    if RETRIEVE_SUBREDDITS:
        # Load reddit_daily_grouped.csv 
        df = pd.read_csv(os.path.join(SOCIAL_DATASET_PATH, 'reddit', 'reddit_daily_grouped.csv'))

        import ast

        # Save in a list all the links from the reddit column
        links = []

        # Iterate over the rows of the dataframe
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            # Get the data from the reddit column
            reddit_posts = ast.literal_eval(row['reddit']) if row['reddit'] != '[]' else []
            for p in reddit_posts:
                link = p[4]
                links.append(link)

        # Print the first 5 links
        print(links[:5])

        # Extract the subreddit from the link
        subreddits = [link.split('/')[4] for link in links]

        # Save the list of subreddits in a file
        subreddits_file = os.path.join(SOCIAL_DATASET_PATH, 'reddit', 'subreddits.txt')
        with open(subreddits_file, 'w') as f:
            for subreddit in subreddits:
                f.write(f"{subreddit}\n")

    # Load the list of subreddits from the file
    subreddits_file = os.path.join(SOCIAL_DATASET_PATH, 'reddit', 'subreddits.txt')

    with open(subreddits_file, 'r') as file:
        subreddits = file.read().splitlines()

    # Count the number of times each subreddit appears
    subreddit_counts = pd.Series(subreddits).value_counts()

    from matplotlib import pyplot as plt

    # Plot in percentage the number of times each subreddit appears
    # Show only the percentage of the top 5 subreddits and the rest as 'Other'
    top_subreddits = subreddit_counts.index[:5]
    subreddit_counts_top = subreddit_counts[top_subreddits]
    subreddit_counts_other = subreddit_counts[~subreddit_counts.index.isin(top_subreddits)].sum()
    subreddit_counts_top['Others'] = subreddit_counts_other

    # Plot the percentage of the top 5 subreddits and the rest as 'Others'
    subreddit_counts_top.plot.pie(autopct='%1.1f%%', figsize=(10, 10))
    plt.ylabel('')

    # Save the most popular subreddit in a list
    top_subreddits = subreddit_counts.head(4).index.tolist()

    # Retrieve today's date
    today = TODAY

    # Import reddit credentials from twitter.json
    with open(os.path.join('secrets/reddit.json')) as file:
        creds = json.load(file)

    # Initialize the reddit API
    reddit = praw.Reddit(client_id=creds["client_id"],
        client_secret=creds["client_secret"],
        user_agent=creds["user_agent"])

    # Retrieve all the today's reddit posts from the top subreddits
    # Retrieve: author, title, score, created, link, text, url, code, comments
    posts = []
    comments = []
    for subreddit in tqdm(top_subreddits, total=len(top_subreddits)):
        for post in reddit.subreddit(subreddit).top(time_filter='day', limit=None):
            post_created = dt.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')
            # Check if the post was created today and append it to the list
            if post_created.split(' ')[0] == today:
                code = post.permalink.split('/')[4]
                post_url = f"https://www.reddit.com{post.permalink}"

                # Retrieve the comments
                post.comments.replace_more(limit=0)
                comments_obj = post.comments.list()
                for comment in comments_obj:
                    comment_url = f"https://www.reddit.com{comment.permalink}"
                    comment_created = dt.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S')

                    # Check the values
                    comment_author = comment.author if comment.author != None else 'none_author'
                    comment_score = comment.score if comment.score != None else 'none_score'
                    comment_created = comment_created if comment_created != None else 'none_created'
                    comment_url = comment_url if comment_url != None else 'none_url'
                    comment_body = comment.body if comment.body != None else 'none_body'

                    comments.append([str(comment_author), comment_score, comment_created, comment_url, comment_body, code])

                # Check the values
                post_author = post.author if post.author != None else 'none_user'
                post_title = post.title if post.title != None else 'none_title'
                post_score = post.score if post.score != None else 'none_score'
                post_created = post_created if post_created != None else 'none_created'
                post_url = post_url if post_url != None else 'none_url' 
                post_text = post.selftext if post.selftext != None else 'none_text'
                post_url = post.url if post.url != None else 'none_url'
                post_code = code if code != None else 'none_code'

                posts.append([post_author, post_title, post_score, post_created, post_url, post_text, post_url, code])

    # Create a dataframe with the posts
    df_posts = pd.DataFrame(posts, columns=['author', 'title', 'score', 'created', 'link', 'text', 'url', 'code'])
    df_comments = pd.DataFrame(comments, columns=['author', 'score', 'created', 'link', 'body', 'code'])

    # Save the posts in a csv file
    posts_output_file = os.path.join(DEMO_TODAY_DATASET_PATH, 'today_reddit_posts.csv')
    df_posts.to_csv(posts_output_file, index=False)

    # Save the comments in a csv file
    comments_output_file = os.path.join(DEMO_TODAY_DATASET_PATH, 'today_reddit_comments.csv')
    df_comments.to_csv(comments_output_file, index=False)

    ### [ADDON] Cleaning Reddit posts/comments 
    print("Cleaning Reddit posts and comments...")
    from models.fasttext import fasttext_lang_model

    # Load the datasets
    reddit_posts = pd.read_csv(os.path.join(DEMO_TODAY_DATASET_PATH, 'today_reddit_posts.csv'))
    reddit_comments = pd.read_csv(os.path.join(DEMO_TODAY_DATASET_PATH, 'today_reddit_comments.csv'))

    # Delete the reddit_posts column where 'title' or 'text' contains only links
    rows_to_delete = reddit_posts[(reddit_posts['title'].str.contains('http')) | (reddit_posts['text'].str.contains('http'))].index
    print(f"Rows to delete: {rows_to_delete}")
    reddit_posts = reddit_posts.drop(rows_to_delete)

    # Delete the reddit_comments column where 'body' contains only links
    rows_to_delete = reddit_comments[reddit_comments['body'].str.contains('http')].index
    print(f"Rows to delete: {rows_to_delete}")
    reddit_comments = reddit_comments.drop(rows_to_delete)

    # Delete the reddit_posts column where 'title' or 'text' contains '[removed]' or '[deleted]'
    rows_to_delete = reddit_posts[(reddit_posts['title'].str.contains('\[removed\]|\[deleted\]') | reddit_posts['text'].str.contains('\[removed\]|\[deleted\]'))].index
    print(f"Rows to delete: {rows_to_delete}")
    reddit_posts = reddit_posts.drop(rows_to_delete)

    # Delete the reddit_comments column where 'body' contains '[removed]' or '[deleted]'
    rows_to_delete = reddit_comments[reddit_comments['body'].str.contains('\[removed\]|\[deleted\]')].index
    print(f"Rows to delete: {rows_to_delete}")
    reddit_comments = reddit_comments.drop(rows_to_delete)

    def check_language(dataset, column):
        new_dataset = []
        for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
            if len(column) == 1:
                # Remove "\n" from the text
                if(fasttext_model.predict_lang(str(row[column[0]]).replace("\n", " ")) == 'en'):
                    new_dataset.append(row)
            else:
                append = True
                for col in column:
                    # Remove "\n" from the text
                    if(fasttext_model.predict_lang(str(row[col]).replace("\n", " ")) != 'en'):
                        append = False
                        break
                if append:
                    new_dataset.append(row)

        return new_dataset

    # Load the fasttext model
    fasttext_model = fasttext_lang_model()

    # Create a new dataset that have the same column as the original dataset but only the rows where the language is 'en'
    reddit_posts_en = []
    reddit_comments_en = []

    # For each row in the dataset, predict the language of the title, if the language is not English, predict the language of the text
    # Select only the rows where the language is 'en'
    reddit_posts_en = check_language(reddit_posts, ['title', 'text'])
    reddit_comments_en = check_language(reddit_comments, ['body'])

    # Create a DataFrame with the new dataset
    reddit_posts_en = pd.DataFrame(reddit_posts_en, columns=reddit_posts.columns)
    reddit_comments_en = pd.DataFrame(reddit_comments_en, columns=reddit_comments.columns)

    # Show the rows that have been deleted
    print(f"Number of rows deleted in reddit_posts: {reddit_posts.shape[0] - reddit_posts_en.shape[0]}")
    print(f"Number of rows deleted in reddit_comments: {reddit_comments.shape[0] - reddit_comments_en.shape[0]}")

    # Create a new column called "comments" in the reddit_posts_en DataFrame
    # Save in the "comments" column all the comments that have the same "code" as the post
    # [[author, score, created, link, body, code], ...]
    reddit_posts_en['comments'] = reddit_posts_en['code'].apply(lambda x: reddit_comments_en[reddit_comments_en['code'] == x].values.tolist())

    # Save the cleaned posts and comments in a csv file
    posts_output_file = os.path.join(DEMO_TODAY_DATASET_PATH, 'today_reddit_posts_and_comments_en.csv')
    reddit_posts_en.to_csv(posts_output_file, index=False)

    # Concatenate news/post/comments
    # Load the datasets
    cointelegraph = pd.read_csv(os.path.join(DEMO_TODAY_DATASET_PATH, 'today_cointelegraph_with_body.csv'))
    bitcoin_news = pd.read_csv(os.path.join(DEMO_TODAY_DATASET_PATH, 'today_bitcoin_news_with_body.csv'))
    reddit_posts_and_comments = pd.read_csv(os.path.join(DEMO_TODAY_DATASET_PATH, 'today_reddit_posts_and_comments_en.csv'))

    # Convert nan to 'none'
    cointelegraph = cointelegraph.where(pd.notnull(cointelegraph), 'none')
    bitcoin_news = bitcoin_news.where(pd.notnull(bitcoin_news), 'none')
    reddit_posts_and_comments = reddit_posts_and_comments.where(pd.notnull(reddit_posts_and_comments), 'none')

    # Create a new dataset with one row having the following columns: 'timestamp', 'cointelegraph', 'reddit'
    # The 'timestamp' column will have the current today's date
    # The 'cointelegraph' column will have today's news from cointelegraph: [[id, slug, views, title, published, leadtext, body], ...]
    # The 'reddit' column will have today's posts and comments from reddit: [[author, title, score, created, link, text, url, code, comments], ...]
    data = {
        'timestamp': [today],
        'cointelegraph': [cointelegraph.values.tolist()],
        'bitcoin_news': [bitcoin_news.values.tolist()],
        'reddit': [reddit_posts_and_comments.values.tolist()]
    }

    # Create a DataFrame with the new dataset
    df = pd.DataFrame(data)

    # Save the dataset in a csv file
    output_file = os.path.join(DEMO_TODAY_DATASET_PATH, 'today_news_and_reddit_data.csv')
    df.to_csv(output_file, index=False)

######################################################
# Annotate data with LLM                             #
######################################################
from data_annotation.utils.config import *
from data_annotation.utils.utils import *

def annotate_data_with_llm_offline(model_name):
    import pandas as pd

    if model_name == "phi":
        print("Annotating data with Phi...")
        from data_annotation.utils.phi_utils import MODEL_NAME, MODEL_VERSION, MAX_TOKENS, OUTPUT_TOKENS, INPUT_TOKENS, get_tokenizer
    elif model_name == "mistral":
        print("Annotating data with Mistral...")
        from data_annotation.utils.mistral_utils import MODEL_NAME, MODEL_VERSION, MAX_TOKENS, OUTPUT_TOKENS, INPUT_TOKENS, get_tokenizer
    elif model_name == "llama":
        print("Annotating data with Llama...")
        from data_annotation.utils.llama_utils import MODEL_NAME, MODEL_VERSION, MAX_TOKENS, OUTPUT_TOKENS, INPUT_TOKENS, get_tokenizer
    elif model_name == "qwen":
        print("Annotating data with Qwen...")
        from data_annotation.utils.qwen_utils import MODEL_NAME, MODEL_VERSION, MAX_TOKENS, OUTPUT_TOKENS, INPUT_TOKENS, get_tokenizer
    else:
        raise Exception("Model name not found")

    # Set the dataset name and the model name
    FULL_DATASET_NAME = MODEL_NAME+"_"+MODEL_VERSION
    FULL_MODEL_NAME = MODEL_NAME+":"+MODEL_VERSION

    # Set the paths
    DATASET_TYPE = "today"

    DATASET_NAME = "news_and_reddit_data"
    ANNOTATED_DATASET_NAME = DATASET_TYPE + "_" + FULL_DATASET_NAME + "_" + DATASET_NAME +"_opinion.csv"

    # Set the paths
    ORIGINAL_DATASET_PATH = os.path.join(DEMO_TODAY_DATASET_PATH, 'today_news_and_reddit_data.csv')
    ANNOTATED_DATASET_PATH = os.path.join(DEMO_TODAY_DATASET_PATH, ANNOTATED_DATASET_NAME)
    OUTPUT_DATASET_PATH = os.path.join(DEMO_TODAY_DATASET_PATH, "merged_" + ANNOTATED_DATASET_NAME)

    # Read daily dataset from the file
    original_dataset = pd.read_csv(ORIGINAL_DATASET_PATH)
    
    # Create a new dataset with row_index, reasoning_text and sentiment_class columns starting from the merged_daily dataset
    # Copy the index from the merged_daily dataset to the new dataset
    llm_opinion = original_dataset.copy()   
    # Drop the columns from the new dataset except the index
    llm_opinion.drop(columns=original_dataset.columns, inplace=True)
    # Add the reasoning_text and sentiment_class columns to the new dataset
    llm_opinion['reasoning_text'] = 'none_reasoning_text'
    llm_opinion['sentiment_class'] = 'none_sentiment_class'
    llm_opinion['action_class'] = 'none_action_class'
    llm_opinion['action_score'] = 'none_action_score'

    # Load tokenizer
    tokenizer = get_tokenizer()

    # For each row in the dataset, populate the func_kwargs list with the input text and the index of each row
    func_kwargs = populate_func_kwargs(
            model_name=MODEL_NAME, 
            merged_dataset=original_dataset, 
            opinion_dataset=llm_opinion, 
            max_tokens=INPUT_TOKENS, 
            instructions=INSTRUCTIONS, 
            model_tokenizer=tokenizer, 
            test=False,
            live=True
        )

    import ollama

    for query in tqdm(func_kwargs, total=len(func_kwargs)):
        index = query['index']
        input_text = query['input_text']

        try:
            response = ollama.generate(
                model=FULL_MODEL_NAME,
                prompt=input_text,
                format="json",
            )

            response_json = ast.literal_eval(response['response'])

            # Check the response
            reasoning_text, sentiment_class, action_class, action_score = check_response(response_json)
        except Exception as e:
            print(e)

            # Set the default values
            reasoning_text = 'none_reasoning_text'
            sentiment_class = 'none_sentiment_class'
            action_class = 'none_action_class'
            action_score = 'none_action_score'

        # Update the sentiment dataset
        llm_opinion.loc[index, 'reasoning_text'] = reasoning_text
        llm_opinion.loc[index, 'sentiment_class'] = sentiment_class
        llm_opinion.loc[index, 'action_class'] = action_class
        llm_opinion.loc[index, 'action_score'] = action_score

        # Save temporary results
        llm_opinion.to_csv(os.path.join(DEMO_TODAY_DATASET_PATH, ANNOTATED_DATASET_NAME), index=False)
            
    # Append llm_opinion to the original dataset
    original_dataset_with_llm_opinion = pd.concat([original_dataset, llm_opinion], axis=1)
    original_dataset_with_llm_opinion

    # Save the daily dataset
    original_dataset_with_llm_opinion.to_csv(os.path.join(OUTPUT_DATASET_PATH))

######################################################
# Annotate data with Gemini                          #
######################################################

def annotate_data_with_gemini(trading_annotation=False):
    print("Annotating data with Gemini...")
    import google.generativeai as genai
    import pandas as pd

    from data_annotation.utils.gemini_utils import MODEL_NAME, INPUT_TOKENS, gemini_configurations

    # Import reddit credentials from twitter.json
    with open(os.path.join('secrets/gemini.json')) as file:
        creds = json.load(file)  

    # Select the Google API key
    google_api_key = creds['GOOGLE_API_KEY_1']

    # Set up the API key
    genai.configure(api_key=google_api_key)

    # Model configurations
    generation_config, safety_settings = gemini_configurations()

    model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    safety_settings=safety_settings,
    generation_config=generation_config,
    )

    # Set the paths
    DATASET_TYPE = "today"

    DATASET_NAME = "news_and_reddit_data"
    FULL_DATASET_NAME = MODEL_NAME+"_"+DATASET_NAME
    ANNOTATED_DATASET_NAME = DATASET_TYPE + "_" + FULL_DATASET_NAME +"_opinion.csv"

    # Set the paths
    ORIGINAL_DATASET_PATH = os.path.join(DEMO_TODAY_DATASET_PATH, 'today_news_and_reddit_data.csv')
    ANNOTATED_DATASET_PATH = os.path.join(DEMO_TODAY_DATASET_PATH, ANNOTATED_DATASET_NAME)
    OUTPUT_DATASET_PATH = os.path.join(DEMO_TODAY_DATASET_PATH, "merged_" + ANNOTATED_DATASET_NAME)

    # Read daily dataset from the file
    original_dataset = pd.read_csv(ORIGINAL_DATASET_PATH)

    # Create a new dataset with row_index, reasoning_text and sentiment_class columns starting from the merged_daily dataset
    # Copy the index from the merged_daily dataset to the new dataset
    llm_opinion = original_dataset.copy()   
    # Drop the columns from the new dataset except the index
    llm_opinion.drop(columns=original_dataset.columns, inplace=True)
    # Add the reasoning_text and sentiment_class columns to the new dataset
    llm_opinion['reasoning_text'] = 'none_reasoning_text'
    llm_opinion['sentiment_class'] = 'none_sentiment_class'
    llm_opinion['action_class'] = 'none_action_class'
    llm_opinion['action_score'] = 'none_action_score'

    # For each row in the dataset, populate the func_kwargs list with the input text and the index of each row
    func_kwargs = populate_func_kwargs(
            model_name=MODEL_NAME, 
            merged_dataset=original_dataset, 
            opinion_dataset=llm_opinion, 
            max_tokens=INPUT_TOKENS, 
            instructions=INSTRUCTIONS, 
            model_tokenizer=None, 
            test=False,
            live=True
        )


    def call_gemini_api(index, input_text):
        try:
            # Generate the reasoning and sentiment
            response = model.generate_content(input_text)
            response_json = ast.literal_eval(response.text)

            # Check the response
            reasoning_text, sentiment_class, action_class, action_score = check_response(response_json)
        
        except Exception as e:
            print(e)

            # Set the default values
            reasoning_text = 'none_reasoning_text'
            sentiment_class = 'none_sentiment_class'
            action_class = 'none_action_class'
            action_score = 'none_action_score'

        # Update the sentiment dataset
        llm_opinion.loc[index, 'reasoning_text'] = reasoning_text
        llm_opinion.loc[index, 'sentiment_class'] = sentiment_class
        llm_opinion.loc[index, 'action_class'] = action_class
        llm_opinion.loc[index, 'action_score'] = action_score

        # Save temporary results
        llm_opinion.to_csv(os.path.join(DEMO_TODAY_DATASET_PATH, ANNOTATED_DATASET_NAME), index=False)
            
        return index

    # Execute the API call
    results, errors = RATENINJA(call_gemini_api, func_args=None, func_kwargs=func_kwargs)

    # Check if trading_annotation is True
    if trading_annotation:
        dataset_type = "daily"
        path = os.path.join(DEMO_TODAY_DATASET_PATH,  dataset_type + "_" + FULL_DATASET_NAME +"_opinion.csv")

        # Read values
        reasoning_text = llm_opinion['reasoning_text'].values[0]
        sentiment_class = llm_opinion['sentiment_class'].values[0]
        action_class = llm_opinion['action_class'].values[0]
        action_score = llm_opinion['action_score'].values[0]
        
        # Retrieve today's date
        today = TODAY

        if not os.path.exists(path):
            daily_original_dataset_with_llm_opinion = pd.DataFrame([[today, reasoning_text, sentiment_class, action_class, action_score]], columns=['timestamp', 'reasoning_text', 'sentiment_class', 'action_class', 'action_score'])
            daily_original_dataset_with_llm_opinion.to_csv(path, index=False)

            print(f"Saved annotated data with Gemini for trading at {today}")
        else:
            daily_original_dataset_with_llm_opinion = pd.read_csv(path)

            # Create a new DataFrame with the row to be added
            new_row = pd.DataFrame({
                'timestamp': [today],
                'reasoning_text': [reasoning_text],
                'sentiment_class': [sentiment_class],
                'action_class': [action_class],
                'action_score': [action_score]
            })

            daily_original_dataset_with_llm_opinion = pd.concat([daily_original_dataset_with_llm_opinion, new_row], ignore_index=True)

            daily_original_dataset_with_llm_opinion.to_csv(path, index=False)

            print(f"Updated annotated data with Gemini for trading at {today}")

        # Save func_kwargs in a .csv file using pandas
        path = os.path.join(DEMO_TODAY_DATASET_PATH, dataset_type + '_func_kwargs_' + FULL_DATASET_NAME + '.csv')

        # Check if the file exists
        if not os.path.exists(path):
            # Create a new DataFrame with timestamp and the row to be added
            func_kwargst_df = pd.DataFrame([[today, func_kwargs]], columns=['timestamp', 'func_kwargs'])

            # Save the DataFrame to a .csv file
            func_kwargst_df.to_csv(path, index=False)

            print(f"Saved input_text in {path}")
        else:
            # Load the file
            func_kwargst_df = pd.read_csv(path)

            # Create a new DataFrame with the row to be added
            new_row = pd.DataFrame({
                'timestamp': [today],
                'func_kwargs': [func_kwargs]
            })

            # Concatenate the new row to the original DataFrame
            func_kwargst_df = pd.concat([func_kwargst_df, new_row], ignore_index=True)

            # Save the DataFrame to a .csv file
            func_kwargst_df.to_csv(path, index=False)

            print(f"Updated func_kwargst in {path}")

    # Append llm_opinion to the original dataset
    original_dataset_with_llm_opinion = pd.concat([original_dataset, llm_opinion], axis=1)

    # Save the daily dataset
    original_dataset_with_llm_opinion.to_csv(os.path.join(OUTPUT_DATASET_PATH))

######################################################
# Merge opinions                                     #
######################################################

def merge_opinions():
    print("Merging opinions...")
    import pandas as pd
    import os

    # Load the datasets
    phi_opinion = pd.read_csv(os.path.join(DEMO_TODAY_DATASET_PATH, "today_phi3_3.8b-mini-128k-instruct-q8_0_news_and_reddit_data_opinion.csv"))
    mistral_opinion = pd.read_csv(os.path.join(DEMO_TODAY_DATASET_PATH, "today_mistral-nemo_12b-instruct-2407-q5_K_S_news_and_reddit_data_opinion.csv"))
    llama_opinion = pd.read_csv(os.path.join(DEMO_TODAY_DATASET_PATH, "today_llama3.1_8b-instruct-q6_K_news_and_reddit_data_opinion.csv"))
    qwen_opinion = pd.read_csv(os.path.join(DEMO_TODAY_DATASET_PATH, "today_qwen2_7b-instruct-q8_0_news_and_reddit_data_opinion.csv"))
    gemini_opinion = pd.read_csv(os.path.join(DEMO_TODAY_DATASET_PATH, "today_gemini-1.5-flash_news_and_reddit_data_opinion.csv"))

    # Create a new dataset with the following columns: 
    # 'timestamp', 'phi_reasoning_text', 'phi_sentiment_class', 'phi_action_class', 'phi_action_score', 'mistral_reasoning_text', 'mistral_sentiment_class', 'mistral_action_class', 'mistral_action_score', 'llama_reasoning_text', 'llama_sentiment_class', 'llama_action_class', 'llama_action_score', 'qwen_reasoning_text', 'qwen_sentiment_class', 'qwen_action_class', 'qwen_action_score', 'gemini_reasoning_text', 'gemini_sentiment_class', 'gemini_action_class', 'gemini_action_score'
    # The 'timestamp' column will have the current today's date

    columns = ['timestamp', 
            'phi_reasoning_text', 'phi_sentiment_class', 'phi_action_class', 'phi_action_score', 
            'mistral_reasoning_text', 'mistral_sentiment_class', 'mistral_action_class', 'mistral_action_score', 
            'llama_reasoning_text', 'llama_sentiment_class', 'llama_action_class', 'llama_action_score', 
            'qwen_reasoning_text', 'qwen_sentiment_class', 'qwen_action_class', 'qwen_action_score', 
            'gemini_reasoning_text', 'gemini_sentiment_class', 'gemini_action_class', 'gemini_action_score']
    
    # Retrieve today's date
    from datetime import datetime as dt
    today = TODAY

    data = {
        'timestamp': [today],
        'phi_reasoning_text': phi_opinion['reasoning_text'].values.tolist(),
        'phi_sentiment_class': phi_opinion['sentiment_class'].values.tolist(),
        'phi_action_class': phi_opinion['action_class'].values.tolist(),
        'phi_action_score': phi_opinion['action_score'].values.tolist(),
        'mistral_reasoning_text': mistral_opinion['reasoning_text'].values.tolist(),
        'mistral_sentiment_class': mistral_opinion['sentiment_class'].values.tolist(),
        'mistral_action_class': mistral_opinion['action_class'].values.tolist(),
        'mistral_action_score': mistral_opinion['action_score'].values.tolist(),
        'llama_reasoning_text': llama_opinion['reasoning_text'].values.tolist(),
        'llama_sentiment_class': llama_opinion['sentiment_class'].values.tolist(),
        'llama_action_class': llama_opinion['action_class'].values.tolist(),
        'llama_action_score': llama_opinion['action_score'].values.tolist(),
        'qwen_reasoning_text': qwen_opinion['reasoning_text'].values.tolist(),
        'qwen_sentiment_class': qwen_opinion['sentiment_class'].values.tolist(),
        'qwen_action_class': qwen_opinion['action_class'].values.tolist(),
        'qwen_action_score': qwen_opinion['action_score'].values.tolist(),
        'gemini_reasoning_text': gemini_opinion['reasoning_text'].values.tolist(),
        'gemini_sentiment_class': gemini_opinion['sentiment_class'].values.tolist(),
        'gemini_action_class': gemini_opinion['action_class'].values.tolist(),
        'gemini_action_score': gemini_opinion['action_score'].values.tolist()
    }

    # Create a DataFrame with the new dataset
    df = pd.DataFrame(data)

    # Save the dataset in a csv file
    output_file = os.path.join(DEMO_TODAY_DATASET_PATH, 'today_llm_news_and_reddit_data_opinions.csv')
    df.to_csv(output_file, index=False)

    print("Merging opinions done!")
    print("Return to demo")

if __name__ == "__main__":
    # Retrieve the data
    retrieve_cointelegraph_news()
    retrieve_bitcoin_news()
    retrieve_reddit_data()

    # Annotate data
    annotate_data_with_gemini(trading_annotation=True)

    exit(0)