import pandas as pd
import os
import plotly.graph_objs as go
import json
import requests

from config import *
from demo.demo_config import DATE, INITIAL_BALANCE, get_current_bitcoin_price, build_authentication_headers, get_account_keys

def retrieve_orders(account):
    # Retrieve all the trades
    endpoint = "auth/r/trades/hist"

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

    # Retrieve the orders
    trades = json.loads(response.text)

    # Create a DataFrame with the trades
    df_trades = pd.DataFrame(trades, columns=['id', 'symbol', 'mts', 'order_id', 'exec_amount', 'exec_price', 'order_type', 'order_price', 'maker', 'fee', 'fee_currency', 'cid'])

    # Group the trades by order_id
    # Select all the columns and apply the following aggregations:
    # 'exec_amount': 'sum', 'exec_price': 'mean', 'order_price': 'mean','fee': 'sum', 'mts': 'max'
    df_orders = df_trades.groupby('order_id').agg({'exec_amount': 'sum', 'exec_price': 'mean', 'order_price': 'mean', 'fee': 'sum', 'mts': 'max'})
    
    # Add the remaining columns
    df_orders['symbol'] = df_trades.groupby('order_id')['symbol'].first()
    df_orders['order_type'] = df_trades.groupby('order_id')['order_type'].first()
    df_orders['fee_currency'] = df_trades.groupby('order_id')['fee_currency'].first()

    # Rename the columns
    df_orders.columns = ['exec_amount', 'exec_price', 'order_price', 'fee', 'mts', 'symbol', 'order_type', 'fee_currency']

    # Sort the DataFrame by 'mts'
    df_orders.sort_values('mts', inplace=True, ascending=False)

    # Reset the index
    df_orders.reset_index(inplace=True)

    # Add column 'side' to the DataFrame
    df_orders['side'] = df_orders['exec_amount'].apply(lambda x: 'buy' if x > 0 else 'sell')

    # Convert mts to datetime
    df_orders['mts'] = pd.to_datetime(df_orders['mts'], unit='ms')

    # Convert fee from btc to usd
    for i in range(len(df_orders)):
        if df_orders['fee_currency'].values[i] == 'TESTBTC':
            df_orders['fee'].values[i] = df_orders['fee'].values[i] * df_orders['exec_price'].values[i]
            df_orders['fee_currency'].values[i] = 'TESTUSD'

    # Rename the columns
    df_orders.columns = ['order_id', 'exec_amount_btc', 'exec_price', 'order_price', 'fee', 'timestamp', 'symbol', 'order_type', 'fee_currency', 'side']

    # Compute the amount in usd and add it to the DataFrame as a new column 'amount_usd'
    df_orders['exec_amount_usd'] = df_orders['exec_amount_btc'] * df_orders['exec_price']

    # Select the trades from a certain date
    df_orders = df_orders[df_orders['timestamp'] >= DATE]

    # Save the trades
    df_orders.to_csv(os.path.join(DEMO_TRADING_DATASET_PATH, f'trading_orders_{account}.csv'), index=False)

def get_orders(account):
    # Get the orders
    retrieve_orders(account)

    # Load the CSV file
    df_orders_path = os.path.join(DEMO_TRADING_DATASET_PATH, f"trading_orders_{account}.csv")
    df_orders = pd.read_csv(df_orders_path)

    return df_orders

def get_balance(account):
    # Load the CSV file
    df_balance_path = os.path.join(DEMO_TRADING_DATASET_PATH, f'trading_balance_{account}.csv')
    df_balance = pd.read_csv(df_balance_path)

    # Select the balance from a certain date
    df_balance = df_balance[df_balance['timestamp'] >= DATE]

    return df_balance

def get_trading_data(df_orders, account):
    # Show the account balance
    total_usd_balance, total_btc_balance, total_usd, total_btc, total_usd_in_btc, total_btc_in_usd = get_account_balance(account)
    pnl, percentage_pnl = compute_profits_and_losses(account)
    fees = compute_fees(df_orders)

    display = f"""
    <div style="display: flex; flex-direction: row;">
        <div style="flex: 1;">
            <h1>Total balance</h1>
            <h2>{total_usd_balance} $ / {total_btc_balance} BTC</h2>
        </div>

        <div style="flex: 1;">
            <h1>PnL</h1>
            <h2 style="color: {'green' if pnl > 0 else 'red'}">{pnl} $ | {percentage_pnl} %</h2>
        </div>

        <div style="flex: 1;">
            <h1>Fees paid</h1>
            <h2>{fees} $</h2>
        </div>
    </div>

    <hr>
    <h1>Assets</h1>
    <div style="display: flex; flex-direction: row;">
        <div style="flex: 1;">
            <h2>USD: {total_usd} $ / {total_usd_in_btc} BTC</h2>
        </div>

        <div style="flex: 1;">
            <h2>BTC: {total_btc} BTC / {total_btc_in_usd} $</h2>
        </div>
    </div>
    """

    # Add a bar to separate the sections
    display += "<hr>"

    # Convert the data into an HTML table
    display += """
    <h1><details><summary>Trades</summary>
    <div style="overflow-x:auto;">
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th>Timestamp</th>
                <th>Order ID</th>
                <th>Symbol</th>
                <th>Side</th>
                <th>Amount (USD)</th>
                <th>Amount (BTC)</th>
                <th>BTC price (USD)</th>
                <th>Fee (USD)</th>
            </tr>
        """

    for row in df_orders.iterrows():
        timestamp = row[1]['timestamp']
        order_id = row[1]['order_id']
        symbol = row[1]['symbol']
        side = row[1]['side']
        exec_amount_usd = round(row[1]['exec_amount_usd'], 2)
        exec_amount_btc = round(row[1]['exec_amount_btc'], 6)
        exec_price = round(row[1]['exec_price'], 2)
        fee = round(row[1]['fee'], 2)

        display += f"""
        <tr>
            <td>{timestamp}</td>
            <td>{order_id}</td>
            <td>{symbol}</td>
            <td>{side}</td>
            <td>{exec_amount_usd}</td>
            <td>{exec_amount_btc}</td>
            <td>{exec_price}</td>
            <td>{fee}</td>
        </tr>
        """

    display += """
        </table>
    </div>
    </details></h1>
    """

    return display  

def get_bitcoin_prices(start_date, end_date):
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

    # [[timestamp, price], [timestamp, price], ...]
    # Convert the timestamp to datetime
    for i in range(len(prices)):
        prices[i][0] = pd.to_datetime(prices[i][0], unit='ms')

    return prices

def show_buy_sell_operations_with_bitcoin_price(df_orders, df_balance):
    # Define the start and end date
    start_date = DATE
    end_date = datetime.datetime.now()

    # Retrieve only the YYYY-MM-DD part of the date
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    # Transform the date to timestamp
    start_date = int(datetime.datetime.combine(start_date, datetime.datetime.min.time()).timestamp())
    end_date = int(datetime.datetime.combine(end_date, datetime.datetime.min.time()).timestamp())

    # Get the Bitcoin prices
    prices = get_bitcoin_prices(start_date, end_date)

    # Check if investments are provided and set the number of rows
    rows = 2

    layout = dict(
        hoversubplots="axis",
        title="Bitcoin Price and Portfolio value over Time",
        xaxis_title="Days",
        template="plotly_white",
        hovermode="x unified",
        grid=dict(rows=rows, columns=1),
        height=800
    )

    data = [
            go.Scatter(x=[p[0] for p in prices], y=[p[1] for p in prices], mode='lines', name='Bitcoin Price', xaxis="x", yaxis="y1"),
            go.Scatter(x=df_balance['timestamp'], y=df_balance['usd_balance'], name='USD Balance', xaxis="x", yaxis="y2"),
    ]

    fig = go.Figure(data=data, layout=layout)

    # Add buy and sell operations to the plot
    buy_added = False
    sell_added = False

    for index, row in df_orders.iterrows():
        # Find the closest price to the timestamp
        # Convert current date to pd.to_datetime
        current_date = pd.to_datetime(row['timestamp'])
        closest_price = min(prices, key=lambda x: abs(x[0] - current_date))[1]

        if row['side'] == 'buy':
            showlegend = not buy_added
            fig.add_trace(go.Scatter(
                x=[row['timestamp']], 
                y=[closest_price], 
                mode='markers', 
                name='Buy', 
                marker=dict(color='green', size=10),
                showlegend=showlegend
            ))
            buy_added = True
        elif row['side'] == 'sell':
            showlegend = not sell_added
            fig.add_trace(go.Scatter(
                x=[row['timestamp']], 
                y=[closest_price], 
                mode='markers', 
                name='Sell', 
                marker=dict(color='red', size=10),
                showlegend=showlegend
            ))
            sell_added = True

    return fig

def compute_fees(df_orders):
    total_fees = df_orders['fee'].sum()

    # Use only 2 decimals
    total_fees = round(total_fees, 2)

    return total_fees

def compute_profits_and_losses(account):
    # Compute the profit and loss with respect to the initial balance
    total_usd_balance, _, _, _, _, _ = get_account_balance(account)

    # Compute the total profit (final balance - initial balance)
    pnl = total_usd_balance - INITIAL_BALANCE
    percentage_pnl = pnl / INITIAL_BALANCE * 100

    # Use only 2 decimals
    pnl = round(pnl, 2)
    percentage_pnl = round(percentage_pnl, 2)

    return pnl, percentage_pnl

def get_account_balance(account):
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

    total_usd = float(current_balance['TESTUSD'])
    total_btc = float(current_balance['TESTBTC'])

    total_usd_in_btc = total_usd / get_current_bitcoin_price()
    total_btc_in_usd = total_btc * get_current_bitcoin_price()

    total_usd_balance = total_usd + total_btc * get_current_bitcoin_price()
    total_btc_balance = total_btc + total_usd / get_current_bitcoin_price()

    # Use only 2 decimals
    total_usd_balance = round(total_usd_balance, 2)
    total_btc_balance = round(total_btc_balance, 2)
    total_usd = round(total_usd, 2)
    total_btc = round(total_btc, 2)
    total_usd_in_btc = round(total_usd_in_btc, 2)
    total_btc_in_usd = round(total_btc_in_usd, 2)

    return total_usd_balance, total_btc_balance, total_usd, total_btc, total_usd_in_btc, total_btc_in_usd