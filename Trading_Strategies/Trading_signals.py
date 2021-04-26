import bt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

price_data = bt.get('aapl', start='2019-11-1', end='2020-12-1')

# Calculate SMA
sma = price_data.rolling(20).mean()

# Define the signal-based strategy
bt_strategy = bt.Strategy('AboveEMA',
                          [bt.algos.SelectWhere(price_data > sma),
                           bt.algos.WeighEqually(),
                           bt.algos.Rebalance()])

# Create a backtest and run it
bt_backtest = bt.Backtest(bt_strategy, price_data)
bt_result = bt.run(bt_backtest)

# Plot the backtest result
bt_result.plot(title='Backtest Result')
plt.show()
