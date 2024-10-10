def invest_all(dataset, initial_budget, fee, consider_fee):
    '''
    This function simulates the "invest all" strategy. 
    The strategy consists of investing all the available money in the portfolio when the LLM model suggests to buy and sell
    '''
    orders = []
    portfolio_value = []
    usd_balance = initial_budget
    btc_balance = 0.0

    # Initialize the fee
    if not consider_fee:
        fee = 0.0

    # Iterate over the dataset
    for i, row in dataset.iterrows():
        # Get the timestamp, action, close price and price change for the current day
        timestamp = dataset.loc[i, 'timestamp']
        action = dataset.loc[i, 'action_class']
        close_price = dataset.loc[i, 'close']
        pct_price_change = dataset.loc[i, 'pct_price_change']

        # Update the budget based on the action
        if action == "buy":
            if usd_balance > 0: # Check if there are USD to invest
                if consider_fee: # Check if the fee should be considered
                    usd_balance -= usd_balance*fee

                # Compute the amount of BTC bought
                btc_bought = usd_balance/close_price 

                # Compute the amount spent on fees
                fee_usd = usd_balance*fee 

                # Append the order to the orders list
                orders.append([timestamp, action, usd_balance, btc_bought, close_price, pct_price_change, fee_usd])

                # Update the balance
                usd_balance = 0.0
                btc_balance += btc_bought

                # Compute the portfolio value
                portfolio_value.append(usd_balance + btc_balance*close_price)
            else:
                # Append the last portfolio value
                portfolio_value.append(usd_balance + btc_balance*close_price)
        if action == "sell":
            if btc_balance > 0: # Check if there are BTC to sell
                # Compute the amount of USD received from selling the BTC
                usd_received = btc_balance*close_price

                if consider_fee: # Check if the fee should be considered
                    usd_received -= usd_received*fee

                # Compute the amount of BTC to sell
                btc_sold = usd_received / close_price

                # Compute the amount spent on fees
                fee_usd = usd_received*fee

                # Append the order to the orders list
                orders.append([timestamp, action, usd_received, btc_sold, close_price, pct_price_change, fee_usd])

                # Update the balance
                usd_balance += usd_received
                btc_balance = 0.0

                # Compute the portfolio value
                portfolio_value.append(usd_balance + btc_balance*close_price)
            else:
                # Append the last portfolio value
                portfolio_value.append(usd_balance + btc_balance*close_price)
        else: # If the action is "hold" do not update the balance
            portfolio_value.append(usd_balance + btc_balance*close_price)
    
    return orders, portfolio_value

def dollar_cost_averaging(dataset, initial_budget, fixed_daily_budget, percentage_daily_budget, daily_budget_selected, fee, consider_fee):
    '''
    This function simulates the "dollar cost averaging" strategy.
    The strategy consists of buying a fixed amount of money every time the LLM model suggests to buy.
    '''
    orders = []
    investments = []
    portfolio_value = []
    invested = 0
    usd_balance = initial_budget
    btc_balance = 0.0

    # Initialize the fee
    if not consider_fee:
        fee = 0.0

    # Iterate over the dataset
    for i, row in dataset.iterrows():
        # Check if the daily budget is fixed or a percentage of the initial budget
        if daily_budget_selected == "Fixed amount":
            daily_budget = fixed_daily_budget
        elif daily_budget_selected == "Percentage of capital":
            daily_budget = initial_budget*percentage_daily_budget

        # Get the timestamp, action, close price and price change for the current day
        timestamp = dataset.loc[i, 'timestamp']
        action = dataset.loc[i, 'action_class']
        close_price = dataset.loc[i, 'close']
        pct_price_change = dataset.loc[i, 'pct_price_change']

        # Update the budget based on the action
        if action == "buy":
            if consider_fee: # Check if the fee should be considered
                daily_budget -= daily_budget*fee

            # Compute the amount spent on fees
            fee_usd = daily_budget*fee

            if usd_balance > daily_budget + fee_usd: # Check if there are USD to invest
                # Compute the amount of BTC bought
                btc_bought = daily_budget/close_price

                # Append the order to the orders list
                orders.append([timestamp, action, daily_budget, btc_bought, close_price, pct_price_change, fee_usd])

                # Update the balance
                usd_balance -= daily_budget + fee_usd
                btc_balance += btc_bought

                # Compute the invested amount
                invested += daily_budget + fee_usd
                investments.append(invested)

                # Compute the portfolio value
                portfolio_value.append(usd_balance + btc_balance*close_price)
            else: # If there are not enough USD to invest, do not update the balance
                # Append the last invested amount and the current portfolio value
                investments.append(investments[-1] if len(investments) > 0 else 0)
                portfolio_value.append(usd_balance + btc_balance*close_price)
        else: # If the action is "sell" or "hold" do not update the balance
            # Append the last invested amount and the current portfolio value
            investments.append(investments[-1] if len(investments) > 0 else 0)
            portfolio_value.append(usd_balance + btc_balance*close_price)
    
    return orders, investments, portfolio_value

def fixed_investment(dataset, initial_budget, fixed_daily_budget, percentage_daily_budget, daily_budget_input, fee, consider_fee):
    '''
    This function simulates the "fixed investment" strategy.
    The strategy consists of investing a fixed amount of money every time the LLM model suggests to buy or sell.
    '''
    orders = []
    investments = []
    portfolio_value = []
    invested = 0
    usd_balance = initial_budget
    btc_balance = 0.0

    # Initialize the fee
    if not consider_fee:
        fee = 0.0

    # Iterate over the dataset
    for i, row in dataset.iterrows():
        if daily_budget_input == "Fixed amount":
            daily_budget = fixed_daily_budget
        elif daily_budget_input == "Percentage of capital":
            daily_budget = initial_budget*percentage_daily_budget

        # Get the timestamp, action, close price and price change for the current day
        timestamp = dataset.loc[i, 'timestamp']
        action = dataset.loc[i, 'action_class']
        close_price = dataset.loc[i, 'close']
        pct_price_change = dataset.loc[i, 'pct_price_change']

        # Update the budget based on the action
        if action == "buy":
            if consider_fee: # Check if the fee should be considered
                daily_budget -= daily_budget*fee

            # Compute the amount spent on fees
            fee_usd = daily_budget*fee

            if usd_balance > daily_budget + fee_usd:
                # Compute the amount of BTC bought
                btc_bought = daily_budget/close_price

                # Append the order to the orders list
                orders.append([timestamp, action, daily_budget, btc_bought, close_price, pct_price_change, fee_usd])

                # Update the balance
                usd_balance -= daily_budget + fee_usd
                btc_balance += btc_bought

                # Compute the invested amount
                invested += daily_budget + fee_usd
                investments.append(invested)

                # Compute the portfolio value
                portfolio_value.append(usd_balance + btc_balance*close_price)
            else: # If there are not enough USD to invest, do not update the balance
                # Append the last invested amount and the current portfolio value
                investments.append(investments[-1] if len(investments) > 0 else 0)
                portfolio_value.append(usd_balance + btc_balance*close_price)
        elif action == "sell":
            if consider_fee: # Check if the fee should be considered
                daily_budget -= daily_budget*fee

            # Compute the amount spent on fees
            fee_usd = daily_budget*fee

            # Compute the btc_balance in USD
            btc_balance_usd = btc_balance*close_price

            # Check if the balance is enough to sell 
            if btc_balance_usd > daily_budget + fee_usd:
                # Compute the amount of BTC to sell
                btc_sold = (daily_budget + fee_usd) / close_price

                # Append the order to the orders list
                orders.append([timestamp, action, daily_budget, btc_sold, close_price, pct_price_change, fee_usd])

                # Update the balance
                usd_balance += daily_budget
                btc_balance -= btc_sold

                # Compute the portfolio value
                portfolio_value.append(usd_balance + btc_balance*close_price)

                # Compute the invested amount
                invested -= 0
                investments.append(invested)
            else: # If there are not enough BTC to sell, do not update the balance
                # Append the last invested amount and the current portfolio value
                investments.append(investments[-1] if len(investments) > 0 else 0)
                portfolio_value.append(usd_balance + btc_balance*close_price)
        else: # If the action is "hold" do not update the balance
            # Append the last invested amount and the current portfolio value
            investments.append(investments[-1] if len(investments) > 0 else 0)
            portfolio_value.append(usd_balance + btc_balance*close_price)
    
    return orders, investments, portfolio_value