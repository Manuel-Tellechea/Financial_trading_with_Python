import pandas as pd
import matplotlib
import plotly.graph_objects as go
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

# Load the data
apple_data = pd.read_csv('AAPL.csv', index_col='Date', parse_dates=True)

# Resample the data to daily by calculating the mean values
apple_daily = apple_data.resample('D').mean()

# Print the top 5 rows
print(apple_daily.head(5))

# Calculate daily returns
apple_data['daily_return'] = apple_data['Close'].pct_change() * 100

# Plot the histogram
apple_data['daily_return'].hist(bins=100, color='red')
plt.ylabel('Frequency')
plt.xlabel('Daily return')
plt.title('Daily return histogram')
plt.show()

# Calculate SMA
apple_data['sma_50'] = apple_data['Close'].rolling(window=50).mean()

# Plot the SMA
plt.plot(apple_data['sma_50'], color='green', label='SMA_50')
# Plot the close price
plt.plot(apple_data['Close'], color='red', label='Close')

# Customize and show the plot
plt.title('Simple moving averages')
plt.legend()
plt.show()
