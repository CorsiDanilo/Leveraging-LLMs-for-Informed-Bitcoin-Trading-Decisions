import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import gradio as gr
import datetime

from backtest.backtest_config import *
from backtest.utils.backtest_utils import *

# Define Gradio components using the updated API
with gr.Blocks() as interface:
    gr.Markdown("# LLM Opinion Backtest")
    with gr.Column():
        with gr.Row():
            start_date_input = gr.DateTime(label="Start Date", value="2023-01-01", include_time=False)
            end_date_input = gr.DateTime(label="End Date", value="2023-12-31", include_time=False)
            
        initial_capital_input = gr.Number(label="Initial Capital", value=100000, minimum=0)

        consider_commission_rate_input = gr.Checkbox(label="Consider Commission Rate", value=False)
        commission_rate_input = gr.Number(label="Commission Rate", value=0.01, minimum=0, visible=False)
        
        model_name_input = gr.Dropdown(
            choices=MODEL_NAMES,
            label="Model Name",
            value="gemini",
            allow_custom_value=False
        )
        
        strategy_input = gr.Radio(
            choices=[strategy for strategy in STRATEGIES.values()],
            label="Strategy",
            value=STRATEGIES[1]
        )

        daily_budget_input = gr.Radio(
            choices=["Fixed amount", "Percentage of capital"],
            label="Daily budget type",
            value="Fixed amount",
            visible=False
        )
        
        fixed_daily_budget_input = gr.Number(label="Daily Budget", value=1000, visible=False, minimum=0.01)
        percentage_daily_budget_input = gr.Number(label="Daily Percentage", value=0.10, minimum=0.01, maximum=1, visible=False)

        # Update visibility of commission_rate_input based on checkbox
        def update_commission_rate_visibility(consider_commission_rate):
            return gr.update(visible=consider_commission_rate)

        consider_commission_rate_input.change(
            fn=update_commission_rate_visibility,
            inputs=consider_commission_rate_input,
            outputs=commission_rate_input
        )

        # Update visibility of daily_budget_input based on strategy
        def update_daily_budget_visibility(strategy):
            if strategy == STRATEGIES[1]:
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value="Fixed amount")
            elif strategy == STRATEGIES[2] or strategy == STRATEGIES[3]:
                return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(value="Fixed amount")

        def update_daily_budget_type_visibility(daily_budget_type, strategy):
            if daily_budget_type == "Fixed amount" and (strategy == STRATEGIES[2] or strategy == STRATEGIES[3]):
                return gr.update(visible=True), gr.update(visible=False)
            elif daily_budget_type == "Percentage of capital" and (strategy == STRATEGIES[2] or strategy == STRATEGIES[3]):
                return gr.update(visible=False), gr.update(visible=True)
            elif strategy == STRATEGIES[1]:
                return gr.update(visible=False), gr.update(visible=False)

        strategy_input.change(
            fn=update_daily_budget_visibility,
            inputs=strategy_input,
            outputs=[daily_budget_input, fixed_daily_budget_input, percentage_daily_budget_input, daily_budget_input]
        )

        daily_budget_input.change(
            fn=update_daily_budget_type_visibility,
            inputs=[daily_budget_input, strategy_input],
            outputs=[fixed_daily_budget_input, percentage_daily_budget_input]
        )
        
        submit_button = gr.Button("Submit")

        with gr.Row():
            output_message = gr.Textbox(label="Validation Message", visible=False)

        with gr.Row():
            output_plot = gr.Plot(label="Backtest Results", visible=False)

        with gr.Row():
            stats_output = gr.HTML(label="Backtest Statistics", visible=False)

        with gr.Row():
            orders_table = gr.HTML(label="Orders Table", visible=False)

    # Events
    submit_button.click(
        fn=validate_and_run_backtest,
        inputs=[
            start_date_input,
            end_date_input,
            initial_capital_input,
            consider_commission_rate_input,
            commission_rate_input,
            model_name_input,
            strategy_input,
            fixed_daily_budget_input,
            percentage_daily_budget_input,
            daily_budget_input
        ],
        outputs=[
            output_message, 
            output_plot, 
            output_message, 
            output_plot, 
            orders_table, 
            orders_table,
            stats_output,
            stats_output
        ]
    )

if __name__ == "__main__":
    # Launch the interface
    interface.launch()
