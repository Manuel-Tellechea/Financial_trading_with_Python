import talib
import bt
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# Get the data
price_data = yf.download('AAPL', '2019-1-1', '2019-12-27')

# Calculate the RSI
stock_RSI = talib.RSI(price_data['Close']).to_frame()

# Create the same DataFrame structure as RSI
signal = stock_RSI.copy()
signal[stock_RSI.isnull()] = 0

# Construct the signal
signal[stock_RSI < 30] = 1
signal[stock_RSI > 70] = -1
signal[(stock_RSI >= 30) & (stock_RSI <= 70)] = 0

# Merge data into one DataFrame
combined_df = bt.merge(signal, price_data)
combined_df.columns = ['signal', 'Price']

# Plot the signal with price
combined_df.plot(secondary_y=['signal'])

# Plot the RSI
stock_RSI.plot()
plt.title('RSI')

# Define the strategy
bt_strategy = bt.Strategy('RSI_MeanReversion',
                          [bt.algos.WeighTarget(signal),
                           bt.algos.Rebalance()])

# Create the backtest and run it
bt_backtest = bt.Backtest(bt_strategy, price_data)
bt_result = bt.run(bt_backtest)

# Plot the backtest result
bt_result.plot(title='Backtest result')
plt.show()

