######################################################
# General configurations                             #
######################################################
RETRIEVE_DATA = False # Set to True to retrieve the data

######################################################
# Trading configurations                             #
######################################################
ACCOUNT = 3 # 1 | 2 | 3 | 4 | 5
INITIAL_BALANCE = 99999 # Initial balance in USD
SYMBOL = "tTESTBTC:TESTUSD" # tTESTBTC:TESTUSD
TYPE = "EXCHANGE MARKET" # EXCHANGE MARKET
DATE = "2024-09-16 23:50:00" # Date to start retrieve the data

# URLs
# https://trading.bitfinex.com/t
# https://docs.bitfinex.com/docs/introduction

######################################################
# Trading methods                                    #
######################################################
import os, json, hmac, hashlib, requests
import pandas as pd

from config import *
from bfxapi import Client

def get_account_keys(account):
    # Import reddit credentials from twitter.json
    with open(os.path.join('secrets/bitfinex.json')) as file:
        creds = json.load(file)

    # Retrieve the credentials
    api_key = creds[f'api_key_{account}']
    api_secret = creds[f'api_secret_{account}']

    # Authenticate the client
    client = Client(api_key, api_secret)

    # Define the endpoint
    api = "https://api.bitfinex.com/v2"

    return api_key, api_secret, client, api

def build_authentication_headers(api_key, api_secret, endpoint, payload = None):
    from datetime import datetime

    nonce = str(round(datetime.now().timestamp() * 1_000))

    message = f"/api/v2/{endpoint}{nonce}"

    if payload != None:
        message += json.dumps(payload)

    signature = hmac.new(
        key=api_secret.encode("utf8"),
        msg=message.encode("utf8"),
        digestmod=hashlib.sha384
    ).hexdigest()

    return {
        "bfx-apikey": api_key,
        "bfx-nonce": nonce,
        "bfx-signature": signature
    }

def get_current_bitcoin_price():
    # Retrieve current symbol price
    url = f"https://api-pub.bitfinex.com/v2/ticker/tBTCUSD"

    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers)

    # Retrieve last_price
    last_price = json.loads(response.text)[6]

    return last_price