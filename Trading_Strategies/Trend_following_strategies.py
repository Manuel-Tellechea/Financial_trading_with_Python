import bt
import talib
import matplotlib
import matplotlib.pyplot as plt
import yfinance as yf
matplotlib.use('TkAgg')

# Added the Data
# power = yf.Ticker("POWERGRID.NS")
# df = power.history(start="2020-01-01", end='2020-09-04')
aapl = yf.download('AAPL', '2019-1-1','2019-12-27')
print(aapl.head())

# Calculate the indicators
EMA_short = talib.EMA(aapl['Close'], timeperiod=10).to_frame()
EMA_long = talib.EMA(aapl['Close'], timeperiod=40).to_frame()

# Create the Signal DataFrame
signal = EMA_long.copy()
signal[EMA_long.isnull()] = 0

# Construct the signal
signal[EMA_short > EMA_long] = 1
signal[EMA_short < EMA_long] = -1

# Merge the data
combined_df = bt.merge(signal, aapl['Close'], EMA_short, EMA_long)
combined_df.columns = ['signal', 'Price', 'EMA_short', 'EMA_long']
# Plot the signal, price and MAs
combined_df.plot(secondary_y=['signal'])
plt.show()

# Define the strategy
bt_strategy = bt.Strategy('EMA_crossover',
                          [bt.algos.WeighTarget(signal),
                           bt.algos.Rebalance()])

# Create the backtest and run it
bt_backtest = bt.Backtest(bt_strategy, aapl['Close'])
bt_result = bt.run(bt_backtest)

# Plot the backtest result
bt_result.plot(title='Backtest result')
plt.show()
