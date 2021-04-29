import talib
import bt
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


# Benchmarking
def buy_and_hold(ticker, name, start='2018-1-1', end='2021-1-1'):
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

# Get all backtest stats
resInfo = bt_results.stats

# Get the Sharpe ratios from the backtest stats
print('Sharpe ratio daily: %.2f' % resInfo.loc['daily_sharpe'])
print('Sharpe ratio monthly: %.2f' % resInfo.loc['monthly_sharpe'])
print('Sharpe ratio yearly: %.2f' % resInfo.loc['yearly_sharpe'])

# Calculate Sharpe Ratio manually
# Obtain annual return
annual_return = resInfo.loc['yearly_mean']

# Obtain annual volatility
volatility = resInfo.loc['yearly_vol']

# Calculate Sharpe Ratio manually
sharpe_ratio = annual_return / volatility
print('Sharpe ratio annually: %.2f' % sharpe_ratio)

# Obtain Sortino ratio from bt backtest
print('Sortino ratio daily: %.2f' % resInfo.loc['daily_sortino'])
print('Sortino ratio monthly: %.2f' % resInfo.loc['monthly_sortino'])
print('Sortino ratio yearly: %.2f' % resInfo.loc['yearly_sortino'])
