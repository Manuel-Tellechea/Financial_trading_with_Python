import talib
import bt
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


# Benchmarking
def buy_and_hold(ticker, name, start='2020-2-1', end='2020-11-1'):
    # Get the data
    price_data = bt.get(ticker, start=start, end=end)
    # Define the benchmark strategy
    bt_strategy = bt.Strategy(name,
                              [bt.algos.RunOnce(),
                               bt.algos.SelectAll(),
                               bt.algos.WeighEqually(),
                               bt.algos.Rebalance()])
    # Return the backtest
    return bt.Backtest(bt_strategy, price_data)


# Create benchmark strategy backtest
benchmark = buy_and_hold('tsla', name='benchmark')

# Run all backtests and plot the resutls
bt_results = bt.run(benchmark)
bt_results.plot(title='Strategy benchmarking')
plt.show()


# Get all backtest stats
resInfo = bt_results.stats
print(resInfo.index)

# Get daily, monthly and yearly returns
print('Daily return: %.4F' % resInfo.loc['daily_mean'])
print('Monthly return: %.4F' % resInfo.loc['monthly_mean'])
print(('Yearly return: %.4F' % resInfo.loc['yearly_mean']))

# Plot the weekly return histogram
bt_results.plot_histogram(bins=50, freq='w')
plt.show()

# Get the compound annual growth rate
print('Compound annual growth rate: %.4f'% resInfo.loc['cagr'])

# Get the lookback returns
lookback_returns = bt_results.display_lookback_returns()
print(lookback_returns)
