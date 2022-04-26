# Imports
import pandas as pd


# Getting the stock list from a csv
def getting_data_from_csv(file_name: str):
    direction = "r'../tools/"
    file_name = file_name + ".csv'"
    complete_dir = direction + file_name
    df = pd.read_csv(complete_dir)
    index_tickers = []
    index_tickers.extend(df['Company'].str.split(' ').str[-1])
    return index_tickers


# Retrieving List of World Major Stock Indices from Yahoo! Finance
def world_major_index_list():
    df_list = pd.read_html('https://finance.yahoo.com/world-indices/')
    major_stocks_idx = df_list[0]
    return major_stocks_idx
