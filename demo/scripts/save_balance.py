import pandas as pd
import requests
import os
import argparse

from demo.demo_config import *
from shared.constants import ACCOUNTS

if __name__ == "__main__":
    def save_balance(account):
        print(f"Saving the balance for account {account}")
        # Retrieve the USD and BTC balance
        endpoint = "auth/r/wallets"
        payload = {}

        api_key, api_secret, _, api = get_account_keys(account)

        credentials = build_authentication_headers(api_key, api_secret, endpoint, payload)

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "bfx-nonce": credentials["bfx-nonce"],
            "bfx-apikey": credentials["bfx-apikey"],
            "bfx-signature": credentials["bfx-signature"]
        }

        response = requests.post(f"{api}/{endpoint}", json=payload, headers=headers)

        balance = response.json()

        current_balance = {}
        for i in range(len(balance)):
            current_balance[balance[i][1]] = balance[i][2]

        try:
            total_usd = float(current_balance['TESTUSD'])
        except:
            total_usd = 0
        try:
            total_btc = float(current_balance['TESTBTC'])
        except:
            total_btc = 0

        total_usd_in_btc = total_usd / get_current_bitcoin_price()
        total_btc_in_usd = total_btc * get_current_bitcoin_price()

        total_usd_balance = total_usd + total_btc * get_current_bitcoin_price()
        total_btc_balance = total_btc + total_usd / get_current_bitcoin_price()

        # Get the timestamp
        timestamp = pd.Timestamp.now()

        # Create a DataFrame with the balance if it does not exist, if the balance exists, load it and append the new balance
        path = os.path.join(DEMO_TRADING_DATASET_PATH, f'trading_balance_{account}.csv')

        if not os.path.exists(path):
            df_balance = pd.DataFrame([[timestamp, total_usd, total_btc, total_usd_in_btc, total_btc_in_usd, total_usd_balance, total_btc_balance]], columns=['timestamp', 'usd', 'btc', 'usd_in_btc', 'btc_in_usd', 'usd_balance', 'btc_balance'])
            df_balance.to_csv(path, index=False)
        else:
            df_balance = pd.read_csv(path)

            # Create a new DataFrame with the row to be added
            new_row = pd.DataFrame({
                'timestamp': [timestamp],
                'usd': [total_usd],
                'btc': [total_btc],
                'usd_in_btc': [total_usd_in_btc],
                'btc_in_usd': [total_btc_in_usd],
                'usd_balance': [total_usd_balance],
                'btc_balance': [total_btc_balance]
            })

            df_balance = pd.concat([df_balance, new_row], ignore_index=True)

            df_balance.to_csv(path, index=False)

        print(f"Balance saved for account {account}")

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Save the balance before the order')
    parser.add_argument('--account', type=int, help=f"{ACCOUNTS}", required=True, choices=ACCOUNTS)
    args = parser.parse_args()
    
    account = args.account

    # Save the balance before the order
    save_balance(account)

    # Close the script
    exit(0)