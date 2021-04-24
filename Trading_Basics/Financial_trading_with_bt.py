import bt
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

# Download historical prices
bt_data = bt.get('fb, amzn, goog, nflx, aapl',
                 start='2020-6-1', end='2020-12-1')
# Define the strategy
bt_strategy = bt.Strategy('Trade_Weekly',
                         [bt.algos.RunWeekly(),
                          bt.algos.SelectAll(),
                          bt.algos.WeighEqually(),
                          bt.algos.Rebalance()])
# Create a backtest
bt_test = bt.Backtest(bt_strategy, bt_data)
# Run the backtest
bt_res = bt.run(bt_test)
# Plot the test result
bt_res.plot(title="Backtest result")
plt.show()


