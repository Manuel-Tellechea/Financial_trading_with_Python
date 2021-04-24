import pandas as pd
import matplotlib
import plotly.graph_objects as go
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

# Load the data
apple_data = pd.read_csv('AAPL.csv', index_col='Date', parse_dates=True)

# Define the candlestick data
candlestick = go.Candlestick(
    x=apple_data.index,
    open=apple_data['Open'],
    high=apple_data['High'],
    low=apple_data['Low'],
    close=apple_data['Close'])

# Create a candlestick figure
fig = go.Figure(data=[candlestick])
fig.update_layout(title='Apple prices')

# Show the plot
fig.show()
