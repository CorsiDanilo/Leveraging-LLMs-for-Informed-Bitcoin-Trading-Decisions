import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import gradio as gr
import datetime

from config import *
from backtest.backtest_config import *
from backtest.utils.strategies import *

def run_backtest_and_display_plot(start_date,
            end_date,
            initial_capital,
            consider_commission_rate,
            commission_rate,
            model_name,
            strategy,
            fixed_daily_budget,
            percentage_daily_budget,
            daily_budget_selected):

    # Set the path to the dataset
    MERGED_GEMINI_OPINION_DATASET_PATH = os.path.join(ANNOTATED_DATASET_PATH, "merged_no_text_daily_gemini-1.5-flash_opinion.csv")
    MERGED_PHI_OPINION_DATASET_PATH = os.path.join(ANNOTATED_DATASET_PATH, "merged_no_text_daily_phi3_3.8b-mini-128k-instruct-q8_0_opinion.csv")
    MERGED_MISTRAL_OPINION_DATASET_PATH = os.path.join(ANNOTATED_DATASET_PATH, "merged_no_text_daily_mistral-nemo_12b-instruct-2407-q5_K_S_opinion.csv")
    MERGED_LLAMA_OPINION_DATASET_PATH = os.path.join(ANNOTATED_DATASET_PATH, "merged_no_text_daily_llama3.1_8b-instruct-q6_K_opinion.csv")
    MERGED_QWEN_OPINION_DATASET_PATH = os.path.join(ANNOTATED_DATASET_PATH, "merged_no_text_daily_qwen2_7b-instruct-q8_0_opinion.csv")

    # Load the dataset
    def load_dataset(dataset_path):
        dataset = pd.read_csv(dataset_path) # Load the dataset
        dataset = dataset[:-1] # Remove the last row

        return dataset

    # Load the datasets
    if model_name == "gemini":
        dataset = load_dataset(MERGED_GEMINI_OPINION_DATASET_PATH)
    elif model_name == "phi":
        dataset = load_dataset(MERGED_PHI_OPINION_DATASET_PATH)
    elif model_name == "mistral":
        dataset = load_dataset(MERGED_MISTRAL_OPINION_DATASET_PATH)
    elif model_name == "llama":
        dataset = load_dataset(MERGED_LLAMA_OPINION_DATASET_PATH)
    elif model_name == "qwen":
        dataset = load_dataset(MERGED_QWEN_OPINION_DATASET_PATH)
    else:
        raise ValueError("Invalid model name!")

    # Check start and end date of the dataset
    start_date_dataset = datetime.datetime.strptime(dataset['timestamp'].min(), "%Y-%m-%d").date()
    end_date_dataset = datetime.datetime.strptime(dataset['timestamp'].max(), "%Y-%m-%d").date()
    
    # Filter the dataset based on the start and end date
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp']).dt.date
    dataset = dataset[(dataset['timestamp'] >= start_date) & (dataset['timestamp'] <= end_date)]

    # Reset the index
    dataset.reset_index(drop=True, inplace=True)
    
    if strategy == STRATEGIES[1]:
        orders, portfolio_value = invest_all(dataset, initial_capital, commission_rate, consider_fee=consider_commission_rate)
        fig = plot_data(portfolio_value, dataset, orders)
    elif strategy == STRATEGIES[2]:
        orders, investments, portfolio_value  = dollar_cost_averaging(dataset, initial_capital, fixed_daily_budget, percentage_daily_budget, daily_budget_selected, commission_rate, consider_fee=consider_commission_rate)
        fig = plot_data(portfolio_value, dataset, orders, investments)
    elif strategy == STRATEGIES[3]:
        orders, investments, portfolio_value = fixed_investment(dataset, initial_capital, fixed_daily_budget, percentage_daily_budget, daily_budget_selected, commission_rate, consider_fee=consider_commission_rate)
        fig = plot_data(portfolio_value, dataset, orders, investments)
    else:
        raise ValueError("Invalid strategy!")

    return fig, orders, portfolio_value

def plot_data(portfolio_value, dataset, orders, investments=None):
    # Check if investments are provided and set the number of rows
    rows = 3 if investments else 2

    layout = dict(
        hoversubplots="axis",
        title="Bitcoin Price, Portfolio value, and Investments over Time",
        xaxis_title="Days",
        template="plotly_white",
        hovermode="x unified",
        grid=dict(rows=rows, columns=1),
        height=800
    )

    if investments:
        data = [
            go.Scatter(x=dataset['timestamp'], y=dataset['close'], name='Bitcoin Price', xaxis="x", yaxis="y"),
            go.Scatter(x=dataset['timestamp'], y=portfolio_value, name='Portfolio value', xaxis="x", yaxis="y2"),
            go.Scatter(x=dataset['timestamp'], y=investments, name='Investments', xaxis="x", yaxis="y3")
        ]
    else:
        data = [
            go.Scatter(x=dataset['timestamp'], y=dataset['close'], name='Bitcoin Price', xaxis="x", yaxis="y"),
            go.Scatter(x=dataset['timestamp'], y=portfolio_value, name='Portfolio value', xaxis="x", yaxis="y2")
        ]

    fig = go.Figure(data=data, layout=layout)

    # Add buy and sell operations to the plot
    buy_added = False
    sell_added = False

    for order in orders:
        if order[1] == 'buy':
            showlegend = not buy_added
            fig.add_trace(go.Scatter(
                x=[order[0]], 
                y=[order[4]], 
                mode='markers', 
                name='Buy', 
                marker=dict(color='green', size=10),
                showlegend=showlegend
            ))
            buy_added = True
        elif order[1] == 'sell':
            showlegend = not sell_added
            fig.add_trace(go.Scatter(
                x=[order[0]], 
                y=[order[4]], 
                mode='markers', 
                name='Sell', 
                marker=dict(color='red', size=10),
                showlegend=showlegend
            ))
            sell_added = True

    # Show the plot
    return fig

def display_orders(orders):
    # Create a table to display the orders: Timestamp, Action, Amount, Avg. close price, Pct. price change, Fee
    display = """
    <h1><details><summary>Orders</summary>
    <table border='1' style='border-collapse: collapse; width: 100%;'>
    """

    display += """
        <tr>
            <th>Timestamp</th>
            <th>Action</th>
            <th>Amount (USD)</th>
            <th>Amount (BTC)</th>
            <th>Avg. BTC Close Price</th>
            <th>Pct. BTC Price Change</th>
            <th>Fee (USD)</th>
        </tr>
    """
    
    for order in orders:
        timestamp = order[0]
        action = order[1]
        amount_usd = round(order[2], 2)
        amount_btc = round(order[3], 6)
        avg_btc_close_price = round(order[4], 2)
        pct_btc_price_change = round(order[5], 2)
        fee_usd = round(order[6], 2)

        display += f"""
        <tr>
            <td>{timestamp}</td>
            <td>{action}</td>
            <td>{amount_usd}</td>
            <td>{amount_btc}</td>
            <td>{avg_btc_close_price}</td>
            <td>{pct_btc_price_change}</td>
            <td>{fee_usd}</td>
        </tr>
        """

    display += "</table></details></h1>"

    return display

def display_stats(orders, initial_capital, portfolio_value):  
    def compute_profits_and_losses():
        # Get the final balance
        final_balance = portfolio_value[-1]

        # Compute the total profit (final balance - initial balance)
        pnl = final_balance - initial_capital
        percentage_pnl = pnl / initial_capital * 100

        # Use only 2 decimals
        pnl = round(pnl, 2)
        percentage_pnl = round(percentage_pnl, 2)
        final_balance = round(final_balance, 2)

        return pnl, percentage_pnl, final_balance

    def compute_fees():
        total_fees = 0
        for order in orders:
            fee_usd = order[6]
            total_fees += fee_usd

        # Use only 2 decimals
        total_fees = round(total_fees, 2)

        return total_fees

    # Show the account balance
    pnl, percentage_pnl, final_balance = compute_profits_and_losses()
    fees = compute_fees()

    display = f"""
    <div style="display: flex; flex-direction: row;">
        <div style="flex: 1;">
            <h1>Balance</h1>
            <h2>{final_balance} $</h2>
        </div>

        <div style="flex: 1;">
            <h1>PnL</h1>
            <h2 style="color: {'green' if pnl > 0 else 'red'}">{pnl} $ | {percentage_pnl} %</h2>
        </div>

        <div style="flex: 1;">
            <h1>Fees paid</h1>
            <h2>-{fees} $</h2>
        </div>
    </div>
    """

    # Add a bar to separate the sections
    display += "<hr>"

    return display

# Define the validation functions
def validate_dates(start_date, end_date):
    if not (MIN_DATE <= start_date <= MAX_DATE):
        return False, f"Start date must be between {MIN_DATE} and {MAX_DATE}."
    if not (MIN_DATE <= end_date <= MAX_DATE):
        return False, f"End date must be between {MIN_DATE} and {MAX_DATE}."
    if start_date >= end_date:
        return False, "End date must be after the start date."
    
    return True, ""

def validate_initial_capital(initial_capital):
    if initial_capital <= 0:
        return False, "Initial capital must be greater than 0."
    return True, ""

def validate_commission_rate(consider_commission_rate, commission_rate):
    if not consider_commission_rate:
        return True, ""
    if commission_rate <= 0:
        return False, "Commission rate must be greater than 0."
    return True, ""

def validate_daily_budget(strategy, fixed_daily_budget, percentage_daily_budget, daily_budget_selected):
    if not strategy == STRATEGIES[1]:
        if daily_budget_selected == "Fixed amount":
            if fixed_daily_budget <= 0:
                return False, "Fixed daily budget must be greater than 0."
        elif daily_budget_selected == "Percentage of capital":
            if percentage_daily_budget <= 0 or percentage_daily_budget > 1:
                return False, "Percentage of initial capital must be between 0 and 1."
    return True, ""

def validate_inputs(start_date, end_date, initial_capital, consider_commission_rate, commission_rate, model_name, strategy, fixed_daily_budget, percentage_daily_budget, daily_budget_selected=None):
    # Validate inputs
    valid, message = validate_dates(start_date, end_date)
    if not valid:
        return False, message

    valid, message = validate_initial_capital(initial_capital)
    if not valid:
        return False, message

    valid, message = validate_commission_rate(consider_commission_rate, commission_rate)
    if not valid:
        return False, message

    valid, message = validate_daily_budget(strategy, fixed_daily_budget, percentage_daily_budget, daily_budget_selected)
    if not valid:
        return False, message

    return True, ""

# Define the function to run the backtest and return the plot
def validate_and_run_backtest(start_date, end_date, initial_capital, consider_commission_rate, commission_rate, model_name, strategy, fixed_daily_budget, percentage_daily_budget, daily_budget_selected):
    start_date = datetime.datetime.fromtimestamp(start_date).date()
    end_date = datetime.datetime.fromtimestamp(end_date).date()
    
    # Validate inputs
    valid, message = validate_inputs(start_date, end_date, initial_capital, consider_commission_rate, commission_rate, model_name, strategy, fixed_daily_budget, percentage_daily_budget, daily_budget_selected)
    
    if not valid:
        return gr.update(visible=True), gr.update(visible=False), "❌ " + message, None # Show error message, hide plot
    
    # If inputs are valid, run the backtest
    fig, orders, portfolio_value = run_backtest_and_display_plot(start_date, end_date, initial_capital, consider_commission_rate, commission_rate, model_name, strategy, fixed_daily_budget, percentage_daily_budget, daily_budget_selected)
    
    orders_display = display_orders(orders)

    stats_display = display_stats(orders, initial_capital, portfolio_value)
    
    return gr.update(visible=False), gr.update(visible=True), None, fig, gr.update(visible=True), orders_display, gr.update(visible=True), stats_display