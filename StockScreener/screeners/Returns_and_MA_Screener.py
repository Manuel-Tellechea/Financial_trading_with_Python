# Imports
from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si
from pandas import ExcelWriter
from datetime import timedelta
import yfinance as yf
import pandas as pd
import datetime
import time

yf.pdr_override()
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

# Getting the stocks list
tickers = si.tickers_sp500()
tickers = tickers[:5]
tickers = [item.replace(".", "-") for item in tickers]   # Yahoo Finance uses dashes instead of dots
index_name = '^GSPC'  # S&P 500
start_date = datetime.datetime.now() - datetime.timedelta(days=365)
end_date = datetime.date.today()

exportList = pd.DataFrame(columns=['Stock', 'Latest_Price', 'Return Compared', 'SMA21', 'SMA61', 'SMA200',
                                   '52 Week Low', '52 week High', 'PE_Ratio'])

returns_compared = []

# Index Returns
index_df = pdr.get_data_yahoo(index_name, start_date, end_date)
index_df['Pct Change'] = index_df['Adj Close'].pct_change()
index_return = (index_df['Pct Change'] + 1).cumprod()[-1]

# Find top 30% performing stocks (relative to the S&P 500)
for ticker in tickers:
    try:
        df = pdr.get_data_yahoo(ticker, start_date, end_date)

        # Calculating returns relative to the market (returns compared)
        df['Pct Change'] = df['Adj Close'].pct_change()
        stock_return = (df['Pct Change'] + 1).cumprod()[-1]

        relative_return = round((stock_return / index_return), 2)
        returns_compared.extend([relative_return])

        df['SMA21'] = round(df['Adj Close'].rolling(window=21).mean(), 2)
        df['SMA61'] = round(df['Adj Close'].rolling(window=61).mean(), 2)
        df['SMA200'] = round(df['Adj Close'].rolling(window=200).mean(), 2)
        currentClose = df["Adj Close"][-1]
        moving_average_21 = df["SMA21"][-1]
        moving_average_61 = df["SMA61"][-1]
        moving_average_200 = df["SMA200"][-1]
        low_of_52week = round(min(df["Low"][-(52*5):]), 2)
        high_of_52week = round(max(df["High"][-(52*5):]), 2)
        # moving_average_200_20 = df["SMA_200"][-20]
        pe_ratio = float(si.get_quote_table(ticker)['PE Ratio (TTM)'])
        # peg_ratio = float(si.get_stats_valuation(ticker)[1][4])

        print(f'Ticker: {ticker}; Returns Compared against S&P 500: {relative_return}\n')

        # Condition 1: Current Price > 61 SMA and > 200 SMA
        condition_1 = currentClose > moving_average_61 > moving_average_200

        # Condition 2: 61 SMA and > 200 SMA
        condition_2 = moving_average_61 > moving_average_200

        # # Condition 3: 200 SMA trending up for at least 1 month
        # condition_3 = moving_average_200 > moving_average_200_20

        # Condition 4: 21 SMA> 61 SMA and 61 SMA> 200 SMA
        condition_4 = moving_average_21 > moving_average_61 > moving_average_200

        # # Condition 5: PE Ratio less than 40
        # condition_5 = pe_ratio < 40

        # Condition 6: Current Price is at least 30% above 52 week low
        condition_6 = currentClose >= (1.3 * low_of_52week)

        # Condition 7: Current Price is within 25% of 52 week high
        condition_7 = currentClose >= (.75 * high_of_52week)

        # If all conditions above are true, add stock to exportList
        if (
                condition_1 and condition_2 and condition_4 and condition_6 and condition_7
        ):
            exportList = exportList.append({'Stock': ticker, "Latest_Price": currentClose,
                                            "Return Compared": relative_return, "SMA21": moving_average_21,
                                            "SMA61": moving_average_61, "SMA200": moving_average_200,
                                            "52 Week Low": low_of_52week, "52 week High": high_of_52week,
                                            "PE_Ratio": pe_ratio}, ignore_index=True)
            print(ticker + " made the Minervini requirements")

    except Exception as e:
        print(e)
        print(f"Could not gather data on {ticker}")
    time.sleep(1)

# Creating dataframe of only top 30%
exportList['Return Score'] = exportList['Return Compared'].rank(pct=True) * 100
exportList = exportList[exportList['Return Score'] >= exportList['Return Score'].quantile(.70)]
exportList = exportList.sort_values(by='Return Score', ascending=False)

print('\n', exportList)
writer = ExcelWriter("../ScreenOutputZero.xlsx")
exportList.to_excel(writer, "Sheet1")
writer.save()
