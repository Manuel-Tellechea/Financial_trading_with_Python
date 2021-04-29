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

# Get all backtest stats
resInfo = bt_results.stats

# Get the max drawdown
max_drawdown = resInfo.loc['max_drawdown']
print('Maximum drawdown: %.2f' % max_drawdown)

# Get the average drawdown
average_drawdown = resInfo.loc['avg_drawdown']
print('Average drawdown: %.2f' % average_drawdown)

# Get the average drawdown days
average_drawdown_days = resInfo.loc['avg_drawdown_days']
print('Average drawdown days: %.0f' % average_drawdown_days)

# Get the CAGR
cagr = resInfo.loc['cagr']

# Calculate Calmar ratio mannually
calmar_calc = cagr / max_drawdown * (-1)
print('Calmar Ratio calculated: %.2f' % calmar_calc)

# Get the Calmar ratio
calmar = resInfo.loc['calmar']
print('Calmar Ratio: %.2f' % calmar)
