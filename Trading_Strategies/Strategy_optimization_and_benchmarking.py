import talib
import bt
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def signal_strategy(ticker, period, name, start='2020-2-1', end='2020-11-1'):
    # Get the data and calculate SMA
    price_data = bt.get(ticker, start=start, end=end)
    sma = price_data.rolling(period).mean()
    # Define the signal-based strategy
    bt_strategy = bt.Strategy(name,
                              [bt.algos.SelectWhere(price_data > sma),
                               bt.algos.WeighEqually(),
                               bt.algos.Rebalance()])
    # Return the backtest
    return bt.Backtest(bt_strategy, price_data)


sma10 = signal_strategy('tsla', period=10, name='SMA10')
sma30 = signal_strategy('tsla', period=30, name='SMA30')
sma50 = signal_strategy('tsla', period=50, name='SMA50')

# Run backtest and compare results
bt_results = bt.run(sma10, sma30, sma50)
bt_results.plot(title='Strategy optimization')
plt.show()


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

# Create benchmark strategy backtest
benchmark = buy_and_hold('tsla', name='benchmark')

# Run all backtests and plot the resutls
bt_results = bt.run(sma10, sma30, sma50, benchmark)
bt_results.plot(title='Strategy benchmarking')
plt.show()
