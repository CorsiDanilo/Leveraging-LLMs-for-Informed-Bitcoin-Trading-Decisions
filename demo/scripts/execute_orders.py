import pandas as pd
import requests
import os
import argparse

from demo.demo_config import *
from shared.constants import STRATEGIES, ACCOUNTS, NUM_STRATEGIES

if __name__ == "__main__":
    print("Executing the orders")
    print("Parsing the arguments")
    ##############################
    # Parse the arguments        #
    ##############################

    def parse_arguments():
        verify = False

        # Parse the arguments
        parser = argparse.ArgumentParser(description='Make an order based on a given strategy.')
        parser.add_argument('--account', type=int, help=f"{ACCOUNTS}", required=True, choices=ACCOUNTS)
        parser.add_argument('--strategy', type=int, help=f"{STRATEGIES}", choices=NUM_STRATEGIES, required=True)
        parser.add_argument('--percentage', type=float, help='e.g. 0.10 for 10%')
        parser.add_argument('--amount', type=float, help='e.g. 1000 for 1000 USD')
        args = parser.parse_args()

        account = args.account
        strategy = args.strategy
        percentage = None
        amount = None

        if strategy > 1:
            if args.percentage is not None and args.amount is not None:
                print("You can only provide either the percentage or the amount")
                verify = True
                return verify, account, strategy, percentage, amount

            if args.percentage is not None and (args.percentage <= 0 or args.percentage >= 1):
                print("The percentage must be between 0 and 1")
                verify = True
                return verify, account, strategy, percentage, amount

            if args.amount is not None and args.amount <= 0:
                print("The amount must be greater than 0")
                verify = True
                return verify, account, strategy, percentage, amount

            if args.percentage is not None:
                percentage = args.percentage
            if args.amount is not None:
                amount = args.amount

        # Print the arguments
        print(f"Account: {account}")
        print(f"Strategy: {STRATEGIES[strategy]}")
        if percentage is not None:
            print(f"Percentage: {percentage}")
        if amount is not None:
            print(f"Amount: {amount}")

        return verify, account, strategy, percentage, amount

    verify, account, strategy, percentage, raw_amount = parse_arguments()

    if verify:
        exit()

    ##############################
    # Retrieve the data          #
    ##############################

    # Load Gemini opinion
    llm_opinion = pd.read_csv(os.path.join(DEMO_TODAY_DATASET_PATH, 'merged_today_gemini-1.5-flash_news_and_reddit_data_opinion.csv'))

    # Retrieve the action_class
    action_class = llm_opinion['action_class'].values

    # TODO: Just for testing purposes
    # action_class = "buy" # buy | sell | hold

    ##############################
    # Execute the orders         #
    ##############################
    if action_class == "hold":
        print("No action needed (action_class: hold)")
        exit()
    
    print("Executing the orders")

    def get_account_balance():
        # Retrieve the USD and BTC balance
        endpoint = "auth/r/wallets"
        payload = {}

        # Retrieve the credentials
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

        # Retrieve only 8 decimals
        total_usd = round(total_usd, 8)
        total_btc = round(total_btc, 8)
        total_usd_in_btc = round(total_usd_in_btc, 8)
        total_btc_in_usd = round(total_btc_in_usd, 8)

        return total_usd, total_btc, total_usd_in_btc, total_btc_in_usd
    
    def convert_quantity_to_btc(quantity):
        quantity = quantity / get_current_bitcoin_price()

        # Retrieve only 8 decimals
        quantity = round(quantity, 8)

        return quantity

    def get_daily_budget(choice, amount, total_usd, percentage):
        if choice == 0: # Fixed amount
            return amount
        elif choice == 1: # Percentage of the total balance
            amount = total_usd * percentage
            return amount

    def get_orders():
        orders = []

        def invest_all():
            if action_class == "buy":
                # Get the balance before the order
                total_usd, total_btc, total_usd_in_btc, _ = get_account_balance()
                
                # Check if the account has enough balance
                if total_usd <= 0:
                    error = "Not enough balance"
                    print(error)
                    orders.append(error)
                    return orders

                # Select all the USD balance (minus 2% for the fee)
                total_usd_in_btc *= 0.98

                # Print the balance
                print("Balance BEFORE the BUY order:")
                print(f"Total USD: {total_usd}")
                print(f"Total BTC: {total_btc}")
                print(f"Total amount in USD: {total_usd}")
                print(f"Total amount in BTC: {total_usd_in_btc}")

                # Buy
                payload = {
                    "type": TYPE,
                    "symbol": SYMBOL,
                    "amount": str(total_usd_in_btc)
                }

                orders.append(make_order(payload))

                # Check the balance after the order
                total_usd, total_btc, _, _ = get_account_balance()

                # Print the balance
                print("\nBalance AFTER the BUY order:")
                print(f"Total USD: {total_usd}")
                print(f"Total BTC: {total_btc}")

            elif action_class == "sell":
                # Get the balance before the order
                total_usd, total_btc, _, _ = get_account_balance()

                # Check if the account has enough balance
                if total_btc <= 0:
                    error = "Not enough balance"
                    print(error)
                    orders.append(error)
                    return orders

                # Select all the BTC balance (minus 1% for the fee)
                total_btc *= 0.99

                # Compute the amount in USD
                total_usd = total_btc * get_current_bitcoin_price()

                # Print the balance
                print("Balance BEFORE the SELL order:")
                print(f"Total USD: {total_usd}")
                print(f"Total BTC: {total_btc}")
                print(f"Total amount in USD: {total_usd}")
                print(f"Total amount in BTC: {total_btc}")

                # Sell
                payload = {
                    "type": TYPE,
                    "symbol": SYMBOL,
                    "amount": f"-{total_btc}"
                }

                orders.append(make_order(payload))

                # Check the balance after the order
                total_usd, total_btc, _, _ = get_account_balance()

                # Print the balance
                print("\nBalance AFTER the SELL order:")
                print(f"Total USD: {total_usd}")
                print(f"Total BTC: {total_btc}")

        def dollar_cost_averaging(choice):
            if action_class == "buy":
                # Get the balance before the order
                total_usd, total_btc, _, _ = get_account_balance()

                # Get the daily budget
                amount = get_daily_budget(choice, raw_amount, total_usd, percentage)

                # Check if the account has enough balance
                if total_usd < amount:
                    error = "Not enough balance"
                    print(error)
                    orders.append(error)
                    return orders

                # Convert the amount to BTC
                quantity_in_btc = convert_quantity_to_btc(amount)

                # Print the balance
                print("Balance BEFORE the BUY order:")
                print(f"Total USD: {total_usd}")
                print(f"Total BTC: {total_btc}")
                print(f"Total amount in USD: {amount}")
                print(f"Total amount in BTC: {quantity_in_btc}")
                    
                # Buy
                payload = {
                    "type": TYPE,
                    "symbol": SYMBOL,
                    "amount": str(quantity_in_btc)
                }

                orders.append(make_order(payload))

                # Check the balance after the order
                total_usd, total_btc, _, _ = get_account_balance()
                
                # Print the balance
                print("\nBalance AFTER the BUY order:")
                print(f"Total USD: {total_usd}")
                print(f"Total BTC: {total_btc}")

        def fixed_investment(choice):
            if action_class == "buy":
                # Get the balance before the order
                total_usd, total_btc, _, _ = get_account_balance()

                # Get the daily budget
                amount = get_daily_budget(choice, raw_amount, total_usd, percentage)

                # Check if the account has enough balance
                if total_usd < amount:
                    error = "Not enough balance"
                    print(error)
                    orders.append(error)
                    return orders

                # Convert the amount to BTC
                quantity_in_btc = convert_quantity_to_btc(amount)

                # Print the balance
                print(f"Total USD: {total_usd}")
                print(f"Total BTC: {total_btc}")
                print(f"Total amount in USD: {amount}")
                print(f"Total amount in BTC: {quantity_in_btc}")
                
                payload = {
                    "type": TYPE,
                    "symbol": SYMBOL,
                    "amount": str(quantity_in_btc)
                }

                orders.append(make_order(payload))

                # Check the balance after the order
                total_usd, total_btc, _, _ = get_account_balance()

                # Print the balance
                print("\nBalance AFTER the BUY order:")
                print(f"Total USD: {total_usd}")
                print(f"Total BTC: {total_btc}")

            elif action_class == "sell":
                # Get the balance before the order
                total_usd, total_btc, _, _ = get_account_balance()

                # Get the daily budget
                amount = get_daily_budget(choice, raw_amount, total_usd, percentage)

                # Convert the amount to BTC
                quantity_in_btc = convert_quantity_to_btc(amount)

                # Check if the account has enough balance
                if total_btc < quantity_in_btc:
                    error = "Not enough balance"
                    print(error)
                    orders.append(error)
                    return orders

                # Print the balance
                print(f"Total USD: {total_usd}")
                print(f"Total BTC: {total_btc}")
                print(f"Total amount in USD: {amount}")
                print(f"Total amount in BTC: {quantity_in_btc}")

                payload = {
                    "type": TYPE,
                    "symbol": SYMBOL,
                    "amount": f"-{quantity_in_btc}"
                }

                orders.append(make_order(payload))
                
                # Check the balance after the order
                total_usd, total_btc, _, _ = get_account_balance()

                # Print the balance
                print("\nBalance AFTER the SELL order:")
                print(f"Total USD: {total_usd}")
                print(f"Total BTC: {total_btc}")

        if strategy == 1:
            print("Invest all")
            invest_all()
        elif strategy == 2:
            print("Dollar cost averaging - Fixed amount")
            dollar_cost_averaging(0)
        elif strategy == 3:
            print("Dollar cost averaging - Percentage of the total balance")
            dollar_cost_averaging(1)
        elif strategy == 4:
            print("Fixed investment - Fixed amount")
            fixed_investment(0)
        elif strategy == 5:
            print("Fixed investment - Percentage of the total balance")
            fixed_investment(1)
        return orders

    def make_order(payload):
        endpoint = "auth/w/order/submit"

        # Retrieve the credentials
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
        response = response.json()

        return response

    # Execute the orders
    orders = get_orders()

    for order in orders:
        try:
            if order[6] == "SUCCESS":
                print(f"Order made: \n{order}")

                # Parse the order
                mts = order[0]
                order_type = order[1]
                message_id = order[2]

                # Order data
                order_data = order[4][0]
                order_id = order_data[0]
                gid = order_data[1]
                cid = order_data[2]
                symbol = order_data[3]
                mts_create = order_data[4]
                mts_update = order_data[5]
                amount = order_data[6]
                amount_orig = order_data[7]
                order_type = order_data[8]
                type_prev = order_data[9]
                mts_tif = order_data[10]
                flags = order_data[12]
                status = order_data[13]
                price = order_data[16]
                price_avg = order_data[17]
                price_trailing = order_data[18]
                price_aux_limit = order_data[19]
                notify = order_data[23]
                hidden = order_data[24]
                placed_id = order_data[25]
                routing = order_data[28]
                meta = order_data[31]

                code = order[5]
                status = order[6]
                text = order[7]

                # Append the order to the trading_executed_orders_{account}.csv file
                order = {
                    "mts": mts,
                    "order_type": order_type,
                    "message_id": message_id,
                    "order_id": order_id,
                    "gid": gid,
                    "cid": cid,
                    "symbol": symbol,
                    "mts_create": mts_create,
                    "mts_update": mts_update,
                    "amount": amount,
                    "amount_orig": amount_orig,
                    "order_type": order_type,
                    "type_prev": type_prev,
                    "mts_tif": mts_tif,
                    "flags": flags,
                    "status": status,
                    "price": price,
                    "price_avg": price_avg,
                    "price_trailing": price_trailing,
                    "price_aux_limit": price_aux_limit,
                    "notify": notify,
                    "hidden": hidden,
                    "placed_id": placed_id,
                    "routing": routing,
                    "meta": meta,
                    "code": code,
                    "status": status,
                    "text": text
                }

                path = os.path.join(DEMO_TRADING_DATASET_PATH, f"trading_executed_orders_{account}.csv")

                # Check if the file exists
                if not os.path.exists(path):
                    with open(path, "w") as f:
                        f.write("mts,order_type,message_id,order_id,gid,cid,symbol,mts_create,mts_update,amount,amount_orig,order_type,type_prev,mts_tif,flags,status,price,price_avg,price_trailing,price_aux_limit,notify,hidden,placed_id,routing,meta,code,status,text\n")

                # Append the order to the file
                with open(path, "a") as f:
                    f.write(f"{mts},{order_type},{message_id},{order_id},{gid},{cid},{symbol},{mts_create},{mts_update},{amount},{amount_orig},{order_type},{type_prev},{mts_tif},{flags},{status},{price},{price_avg},{price_trailing},{price_aux_limit},{notify},{hidden},{placed_id},{routing},{meta},{code},{status},{text}\n")
        except:
            print("Order not made:\n", order)

            # Retrieve today's date
            from datetime import datetime
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save the order to the trading_failed_orders_{account}.csv file
            path = os.path.join(DEMO_TRADING_DATASET_PATH, f"trading_failed_orders_{account}.csv")

            # Check if the file exists
            if not os.path.exists(path):
                with open(path, "w") as f:
                    f.write("date,order\n")
                    f.write(f"{date},{order}\n")

            else:
                # Append the order to the file
                with open(path, "a") as f:
                    f.write(f"{date},{order}\n")
                    
    exit(0)