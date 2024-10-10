import gradio as gr

from demo.utils.data_visualization_utils import *
from demo.utils.trading_demo_utils import *

# Define a function that updates the Markdown components on button click
def update_analysis_content():
    data_md = get_today_data()
    opinion_md = get_today_llm_opinions()

    price_graph = get_bitcoin_price_graph()

    return opinion_md, data_md, price_graph

# Define a function that updates the Markdown components on button click
def update_trading_account_content():
    for i in range(1, 6):
        print(f"Retrieving data for account {i}...")
        orders = get_orders(i)
        data_md = get_trading_data(orders, i)
        balance = get_balance(i)
        price_graph = show_buy_sell_operations_with_bitcoin_price(orders, balance)

        if i == 1:
            data_md_1 = data_md
            price_graph_1 = price_graph
        elif i == 2:
            data_md_2 = data_md
            price_graph_2 = price_graph
        elif i == 3:
            data_md_3 = data_md
            price_graph_3 = price_graph
        elif i == 4:
            data_md_4 = data_md
            price_graph_4 = price_graph
        elif i == 5:
            data_md_5 = data_md
            price_graph_5 = price_graph

    print("Data retrieved successfully!, returning...")

    return data_md_1, price_graph_1, data_md_2, price_graph_2, data_md_3, price_graph_3, data_md_4, price_graph_4, data_md_5, price_graph_5

with gr.Blocks() as demo:
    gr.Markdown("# Bitcoin Sentiment Analysis Demo")

    with gr.Tabs():
        with gr.Tab("Today's analysis"):
            # Button to trigger the update
            update_button = gr.Button("Retrieve data")

            # Bitcoin Price Graph
            price_graph = gr.Plot()
           
            # Layout for left (model opinions) and right (news/posts) sections
            with gr.Row():
                # Left Section (Model Opinions)
                with gr.Column(variant='panel', scale=1):
                    opinion_md = gr.Markdown("")
               
                # Right Section (News/Posts Data)
                with gr.Column(variant='panel', scale=1):
                    data_md = gr.Markdown("")
            
            # Set the button click to trigger the content update
            update_button.click(update_analysis_content, outputs=[opinion_md, data_md, price_graph])

        with gr.Tab("Trading Analysis"):         
            # Button to trigger the update
            update_button = gr.Button("Retrieve data")

            with gr.Tab("Invest All"):  
                # Layout with two rows, one for the graph and one for the trading data
                price_graph_1 = gr.Plot()

                with gr.Row():
                    data_md_1 = gr.HTML()

            with gr.Tab("Dollar Cost Averaging (Fixed Amount)"):  
                # Layout with two rows, one for the graph and one for the trading data
                price_graph_2 = gr.Plot()

                with gr.Row():
                    data_md_2 = gr.HTML()

            with gr.Tab("Dollar Cost Averaging (Percentage of Capital)"):  
                # Layout with two rows, one for the graph and one for the trading data
                price_graph_3 = gr.Plot()

                with gr.Row():
                    data_md_3 = gr.HTML()

            with gr.Tab("Fixed Investment (Fixed Amount)"):  
                # Layout with two rows, one for the graph and one for the trading data
                price_graph_4 = gr.Plot()

                with gr.Row():
                    data_md_4 = gr.HTML()

            with gr.Tab("Fixed Investment (Percentage of Capital)"): 
                # Layout with two rows, one for the graph and one for the trading data
                price_graph_5 = gr.Plot()

                with gr.Row():
                    data_md_5 = gr.HTML()

            # Set the button click to trigger the content update
            update_button.click(
                update_trading_account_content, 
                outputs=[
                    data_md_1, price_graph_1,
                    data_md_2, price_graph_2,
                    data_md_3, price_graph_3,
                    data_md_4, price_graph_4,
                    data_md_5, price_graph_5
                ]
            )

if __name__ == "__main__":
    demo.launch()