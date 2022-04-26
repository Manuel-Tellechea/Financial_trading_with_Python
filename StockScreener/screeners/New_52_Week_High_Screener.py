# Imports
from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si
from pandas import ExcelWriter
from tools.indicator_lib import rolling_max, rolling_min, sma, returns
import yfinance as yf
import pandas as pd
import datetime
import time
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


yf.pdr_override()
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

# Getting the stocks list
tickers = si.tickers_sp500()
# tickers = random.sample(tickers, 50)
# tickers = ['ACER', 'ACET', 'AMPH', 'APA', 'BRCC', 'BSM', 'CALM', 'EDRY', 'DMLP', 'CVE']

# tickers = tickers[:int(len(tickers)*0.2)]
# tickers = tickers[:25]
tickers = [item.replace(".", "-") for item in tickers]   # Yahoo Finance uses dashes instead of dots
index_name = '^GSPC'  # S&P 500
start_date = datetime.datetime.now() - datetime.timedelta(days=3650)
end_date = datetime.date.today()

exportList = pd.DataFrame(columns=['Stock', 'Latest_Price', 'Return Compared', 'SMA21', 'SMA63', 'SMA200',
                                   '52 Week Low', '52 week High', 'PE_Ratio'])

returns_compared = []

# Index Returns
index_df = pdr.get_data_yahoo(index_name, start_date, end_date)
index_close = index_df['Close'].values.astype(np.float32)
# index_df['Pct Change'] = index_df['Adj Close'].pct_change()
# index_return = (index_df['Pct Change'] + 1).cumprod()[-1]

# Calculating Index returns
index_return = returns(index_close)

# Find top 30% performing stocks (relative to the S&P 500)
for ticker in tickers:
    try:
        df = pdr.get_data_yahoo(ticker, start_date, end_date)
        Close = df['Close'].values.astype(np.float32)
        High = df['High'].values.astype(np.float32)
        Low = df['Low'].values.astype(np.float32)

        # Calculating returns relative to the market (returns compared)
        stock_return = returns(Close)
        relative_return = stock_return / index_return
        relative_return = round(relative_return[-1], 2)
        returns_compared.extend([relative_return])

        # Calculating SMA21, SMA63 and SMA200
        moving_average_21 = sma(Close, 21)
        moving_average_21 = moving_average_21[-1]
        moving_average_63 = sma(Close, 63)
        moving_average_63 = moving_average_63[-1]
        moving_average_200 = sma(Close, 200)
        # Storage value of a 20 days ago of a SMA200
        try:
            moving_average_200_20 = moving_average_200[-20]
        except Exception:
            moving_average_200_20 = 0

        moving_average_200 = moving_average_200[-1]

        # Calculating New 52 Weeks High
        high_of_52week = rolling_max(High, 260)
        high_of_52week = high_of_52week[-1]

        # Calculating New 52 Weeks Low
        low_of_52week = rolling_min(Low, 260)
        low_of_52week = low_of_52week[-1]

        # Last Prices
        currentClose = round(Close[-1], 2)
        currentHigh = round(High[-1], 2)
        currentLow = round(Low[-1], 2)

        # PER Ratio
        pe_ratio = float(si.get_quote_table(ticker)['PE Ratio (TTM)'])

        # PEG Ratio
        # peg_ratio = float(si.get_stats_valuation(ticker)[1][4])

        print(f'Ticker: {ticker}; Returns Compared against S&P 500: {relative_return}\n')

        # # Condition 1: Current Price > 52 weeks high
        # condition_1 = currentHigh >= high_of_52week

        # Condition 2: 61 SMA and > 200 SMA
        condition_2 = moving_average_63 > moving_average_200

        # Condition 3: Current Price is within 25% of 52 week high
        condition_3 = currentClose >= (.75 * high_of_52week)

        # Condition 4: 200 SMA trending up for at least 1 month
        condition_4 = moving_average_200 > moving_average_200_20

        # If all conditions above are true, add stock to exportList
        if (
                condition_2 and condition_3 and condition_4
        ):
            exportList = exportList.append({'Stock': ticker, "Latest_Price": currentClose,
                                            "Return Compared": relative_return, "SMA21": moving_average_21,
                                            "SMA63": moving_average_63, "SMA200": moving_average_200,
                                            "52 Week Low": low_of_52week, "52 week High": high_of_52week,
                                            "PE_Ratio": pe_ratio}, ignore_index=True)
            print(ticker + " made the Minervini requirements")

    except Exception as e:
        print(e)
        print(f"Could not gather data on {ticker}")
    time.sleep(1)

# Creating dataframe of only top 30%
exportList['Return Score'] = exportList['Return Compared'].rank(pct=True) * 100
exportList = exportList[exportList['Return Score'] >= exportList['Return Score'].quantile(.65)]
exportList = exportList.sort_values(by='Return Score', ascending=False)

print('\n', exportList)
writer = ExcelWriter("../ScreenOutputZero.xlsx")
exportList.to_excel(writer, "Sheet1")
writer.save()
