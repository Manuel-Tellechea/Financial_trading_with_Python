# Imports
from tools.indicators_library import Indicators
from tools.tools import world_major_index_list
from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si
from pandas import ExcelWriter
import yfinance as yf
import pandas as pd
import datetime
import time
import numpy as np
import sys
import warnings

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=sys.maxsize)
yf.pdr_override()
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

start_time = time.time()

# Getting the stock list from a csv
df = pd.read_csv(r'../tools/russell3000.csv')
tickers = []
tickers.extend(df['Company'].str.split(' ').str[-1])
tickers = tickers[:int(len(tickers)*0.4)]
tickers = tickers[:50]

# Getting the stocks list from Yahoo Finance
# tickers = si.tickers_dow()
# tickers = tickers[:int(len(tickers)*0.4)]
# tickers = tickers[:50]
# # tickers = ['AVB', 'EVLO', 'VRNA', 'EMBC', 'TBI', 'KR', 'HTA', 'UVV']
# tickers = [item.replace(".", "-") for item in tickers]   # Yahoo Finance uses dashes instead of dots

# Setting the benchmark to compare
# If you need to know the world major index symbols uncomment next line
# world_index = world_major_index_list()
index_name = '^RUA'  # Russell 3000 (^RUA), Dow Jones Industrial (^DJI), Russell 3000 (^RUA), NASDAQ (^IXIC)

# Setting timeframe valuation period
START_DATE = datetime.datetime.now() - datetime.timedelta(days=3650)
END_DATE = datetime.date.today()

# Setting the values to show in output
exportList = pd.DataFrame(columns=['Stock', 'Latest_Price', 'SMA21', 'SMA63', 'SMA200',
                                   '52 Week Low', '52 week High', 'Pattern', 'Score'])

# Index Returns
returns_compared = []
index_df = pdr.get_data_yahoo(index_name, START_DATE, END_DATE)
index_close = index_df['Close'].values.astype(np.float32)

# Calculating Index returns
indicator_index = Indicators(index_df)
index_return = indicator_index.returns()
index_return = index_return[-1]

# Find the conditions
for ticker in tickers:
    try:
        stock_df = pdr.get_data_yahoo(ticker, START_DATE, END_DATE)
        indicators = Indicators(stock_df)

        # Calculating returns relative to the market (returns compared)
        stock_return = indicators.returns()
        stock_return = stock_return[-1]
        relative_return = stock_return / index_return
        relative_return = round(relative_return, 2)
        returns_compared.extend([relative_return])

        # Calculating SMA21, SMA63 and SMA200
        moving_average_21 = indicators.sma(21)
        moving_average_21 = moving_average_21[-1]
        moving_average_63 = indicators.sma(63)
        moving_average_63 = moving_average_63[-1]
        moving_average_200 = indicators.sma(200)
        # Storage value of a 20 days ago of a SMA200
        try:
            moving_average_200_20 = moving_average_200[-20]
        except Exception:
            moving_average_200_20 = 0

        moving_average_200 = moving_average_200[-1]

        # Calculating New 52 Weeks High
        high_of_52week = indicators.rolling_max(260)
        high_of_52week = round(high_of_52week[-1], 2)

        # Calculating New 52 Weeks Low
        low_of_52week = indicators.rolling_min(260)
        low_of_52week = round(low_of_52week[-1], 2)

        # Last Prices
        current_close = indicators.last_values()
        current_close = current_close[3][-1]

        print(f'Ticker: {ticker}; Returns Compared against Russell 3000 Index: {relative_return}\n')

        # Looking for engulfing pattern
        bullish_engulfing, bearish_engulfing, pattern = indicators.engulfing_pattern()
        bullish_engulfing = bullish_engulfing[-1]
        bearish_engulfing = bearish_engulfing[-1]
        pattern = pattern[-1]

        # Condition 1: Engulfing Pattern
        condition_1 = bullish_engulfing

        # Condition 2: Bearish Pattern
        condition_2 = bearish_engulfing

        # Conditions 3 & 4: Price below or above 21 SMA
        condition_3 = current_close < moving_average_21
        condition_4 = current_close > moving_average_21

        # Condition 5 & 6: 200 SMA trending up or dow for at least 1 month
        condition_5 = moving_average_200 < moving_average_200_20
        condition_6 = moving_average_200 > moving_average_200_20

        # If all conditions above are true, add stock to exportList
        if (
                (condition_1 and condition_3 and condition_5) or (condition_2 and condition_4 and condition_6)
        ):
            exportList = exportList.append({'Stock': ticker, "Latest_Price": current_close,
                                            "SMA21": moving_average_21,
                                            "SMA63": moving_average_63,
                                            "SMA200": moving_average_200,
                                            "52 Week Low": low_of_52week,
                                            "52 week High": high_of_52week,
                                            "Pattern": pattern, "Score": relative_return}, ignore_index=True)
            print(ticker + " made the Screener requirements")

    except Exception as e:
        print(e)
        print(f"Could not gather data on {ticker}")
    time.sleep(1)

# Creating dataframe of only top 30%
exportList['Returns Score'] = exportList['Score'].rank(pct=True) * 100
# exportList = exportList[exportList['Returns Score'] >= exportList['Returns Score'].quantile(.65)]
exportList = exportList.sort_values(by='Returns Score', ascending=False)

print('\n', exportList)
writer = ExcelWriter("../ScreenOutput.xlsx")
exportList.to_excel(writer, "Sheet1")
writer.save()
print("--- %s seconds ---" % (time.time() - start_time))