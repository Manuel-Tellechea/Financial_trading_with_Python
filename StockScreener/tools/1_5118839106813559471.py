import pandas as pd
import numpy as np
from scipy.stats import skew
import plotly.graph_objects as go
import time
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')


# data = yf.download('V', start="2000-01-01", end="2021-03-01")
data = pd.read_csv('is_data_SPY.csv', index_col='Date')

start_time = time.time()


def long(data, stop):

    # Data transformations
    df_open = data.Open.values
    df_close = data.Close.values
    df_high = data.High.values
    df_low = data.Low.values

    # Indicators

    # Trades
    buy_list = []
    buy_date = []
    sell_list = []
    sell_date = []
    trailing_stop = np.zeros(len(data))
    flag_long = False

    for i in range(0, len(data)):

        # Entry condition
        if df_open[i] > df_close[i] and flag_long == False and i + 1 < len(data):
            buy_list.append(df_open[i + 1])
            buy_date.append(data.index[i + 1])
            stop_price = df_open[i + 1] * (1 - stop)
            buy_stop = data.index[i + 1]
            flag_long = True

        # Stop loss exit
        if flag_long and data.index[i] >= buy_stop and i + 1 < len(data):
            if df_open[i] > trailing_stop[i - 1]:
                if df_low[i] <= trailing_stop[i - 1]:
                    sell_list.append(trailing_stop[i - 1])
                    sell_date.append(data.index[i])
                    flag_long = False
                else:
                    pass
            else:
                sell_list.append(df_open[i])
                sell_date.append(data.index[i])
                flag_long = False

        # if flag_long and data.Date[i] > buy_date_1 and df_close[i] > df_open[i] and i + days < len(data):
        #     sell_list.append(df_open[i + 1])
        #     sell_date.append(data.Date[i + 1])
        #     flag_long = False

        if flag_long:
            trailing = df_close[i] - (df_close[i] * stop)
            stop_loss_price = max(trailing, stop_price)
            trailing_stop[i] = stop_loss_price

    if len(buy_list) > len(sell_list) and len(buy_date) > len(sell_date):
        sell_date.append(str(buy_date[-1]))
        sell_list.append(float(np.float32(buy_list[-1:])))

    trades = pd.DataFrame()
    trades['buy_date'] = buy_date
    trades['buy_price'] = buy_list
    trades['sell_date'] = sell_date
    trades['sell_price'] = sell_list

    return trades


def statistics(trades, cap, lev, risk, iterate=False):

    # Returns
    trades['pct'] = ((trades.sell_price.values / trades.buy_price.values) - 1) * lev
    trades['accum'] = (((trades.pct + 1).cumprod()) * cap)  # * (1 - (2 * fee))
    trades['ret%'] = ((trades.accum[-1:] / cap) - 1) * 100
    ret = float(trades['ret%'][-1:])
    trades['CAGR'] = ((trades.accum[-1:] / cap) ** (252 / len(data)) - 1) * 100
    cagr = float(trades.CAGR[-1:])

    # Profit factor
    trades['gains'] = np.where(trades.pct > 0, trades.pct, 0)
    trades['loses'] = np.where(trades.pct <= 0, trades.pct, 0)
    trades['profit'] = trades.gains.cumsum()
    trades['loss'] = trades.loses.cumsum()
    trades['PF'] = trades.profit[-1:] / abs(trades.loss[-1:])
    pf = float(trades.PF[-1:])

    # Number of trades
    trades['trade'] = 1
    trades['tot_trades'] = trades.trade.cumsum()
    tot_trades = int(trades.tot_trades[-1:])

    # Maximum Drawdown
    trades['roll'] = trades.accum.cummax()
    trades['trade_DD'] = ((trades.accum / trades.roll) - 1) * 100
    trades['max_DD'] = trades.trade_DD.min()
    max_dd = float(trades.max_DD[-1:])

    # Mar ratio
    trades['MAR'] = trades.CAGR / abs(trades.max_DD)
    mar = float(trades.MAR[-1:])

    # Hit ratio
    trades['pos_trade'] = np.where(trades.pct > 0, 1, 0)
    trades['neg_trade'] = np.where(trades.pct < 0, 1, 0)
    trades['pos_trades'] = trades.pos_trade.cumsum()
    trades['neg_trades'] = trades.neg_trade.cumsum()
    trades['hit_ratio'] = (trades.pos_trades[-1:] / trades.tot_trades[-1:]) * 100
    hit_ratio = float(round(trades.hit_ratio[-1:]))

    # Annualized volatility
    trades['stdev'] = trades.pct.std() * lev * 100
    std = float(trades.stdev[-1:])

    # Other metrics
    m_crit = abs((1 - 1 * (risk + 1)) / (1 - 1 * (1 - (max_dd / 100))))
    start_date = trades.buy_date.iloc[0]
    end_date = trades.sell_date.iloc[-1]
    skewness = skew(trades.pct)

    # Summary
    if iterate is False:
        # data['ret'] = (((data.Close[-1:] / data.Close[0]) ** (252 / len(data))) - 1) * 100
        # data['roll'] = data.Close.cummax()
        # data['DD'] = ((data.Close / data.roll) - 1) * 100
        # data['max_dd'] = data.DD.min()
        # data['MAR'] = (-1) * (data.ret[-1:] / data.max_dd[-1:])
        # print(data.ret[-1:], data.max_dd[-1:], data.MAR[-1:])
        # data['return'] = ((data['Close'].pct_change(1) + 1).cumprod()) * cap

        print()
        print('-' * 25, "Strategy Statistics")
        print()
        print("Start date", '.' * 14, start_date)
        print("End date", '.' * 16, end_date)
        print("Percentage return", '.' * 7, ret)
        print("CAGR", '.' * 20, cagr)
        print("Returns volatility", '.' * 6, std)
        print("Trades returns skewness", '.', skewness)
        print("Profit Factor", '.' * 11, pf)
        print("The number of trades", '.' * 4, tot_trades)
        print("The maximum Drawdown", '.' * 4, abs(max_dd))
        print("MAR ratio (CAGR/MAXDD)", '.' * 2, mar)
        print("% positive trades:", '.' * 6, hit_ratio)
        print("M Criterion %:", '.' * 10, round(m_crit * 100))
        print('-' * 45)
        # fig = go.Figure([go.Scatter(x=trades['sell_date'], y=trades['accum'], name='Strategy'),
        #                  go.Scatter(x=data.index, y=data['return'], name='Benchmark')])
        # fig.update_layout(title="Capital Curve", template="plotly_dark")
        # fig2 = go.Figure([go.Scatter(x=trades['sell_date'], y=trades['trade_DD'], line=dict(color='orange'))])
        # fig2.update_layout(title="Drawdowns", template="plotly_dark")
        # fig3 = go.Figure(data=[go.Histogram(x=trades.pct)])
        # fig3.update_layout(title="Trade returns distribution", template="plotly_dark")
        # fig.show()
        # fig2.show()
        # fig3.show()
        # trades.to_csv('trades.csv')

    else:
        return trades[['ret%', 'CAGR', 'PF', 'tot_trades', 'max_DD', 'MAR']][-1:]


strat = long(data, 0.005)
statistics(trades=strat, cap=1000, lev=1, risk=0.02, iterate=False)
print(len(data))
print("--- %s seconds ---" % (time.time() - start_time))
