# Imports
from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si
from pandas import ExcelWriter
import yfinance as yf
import pandas as pd
import datetime as dt
import time

yf.pdr_override()

# Variables
tickers = si.tickers_sp500()

start_date = dt.datetime.now() - dt.timedelta(days=365)
end_date = dt.date.today()

tickers = [item.replace(".", "-") for item in tickers]  # Yahoo Finance uses dashes instead of dots
index_name = '^GSPC'  # S&P 500

# Index Returns
sp500_df = pdr.get_data_yahoo(index_name, start_date, end_date)
sp500_df['Pct Change'] = sp500_df['Adj Close'].pct_change()
sp500_return = (sp500_df['Pct Change'] + 1).cumprod()[-1]

return_list = []

final_df = pd.DataFrame(columns=['Ticker', 'Latest_Price', 'Score', 'PE_Ratio', 'PEG_Ratio', '21 Day MA',
                                 '61 Day Ma', '200 Day MA', '52 Week Low', '52 week High'])

# Find top 30% performing stocks (relative to the S&P 500)
for ticker in tickers:
    # Download historical data as CSV for each stock (makes the process faster)
    df = pdr.get_data_yahoo(ticker, start_date, end_date)
    df.to_csv(f'{ticker}.csv')
    # df.to_csv(f'stock_data/{ticker}.csv')

    # Calculating returns relative to the market (returns multiple)
    df['Pct Change'] = df['Adj Close'].pct_change()
    stock_return = (df['Pct Change'] + 1).cumprod()[-1]

    returns_compared = round((stock_return / sp500_return), 2)
    return_list.extend([returns_compared])

    print(f'Ticker: {ticker}; Returns Compared against S&P 500: {returns_compared}\n')
    time.sleep(1)

# Creating dataframe of only top 30%
best_performers = pd.DataFrame(list(zip(tickers, return_list)), columns=['Ticker', 'Returns Compared'])
best_performers['Score'] = best_performers['Returns Compared'].rank(pct=True) * 100
best_performers = best_performers[best_performers['Score'] >= best_performers['Score'].quantile(0.7)]

# Checking Minervini conditions of top 30% of stocks in given list
for ticker in best_performers['Ticker']:
    try:
        df = pd.read_csv(f'{ticker}.csv', index_col=0)
        # df.pd.read_csv(f'stock_data/{ticker}.csv', index_col=0)
        moving_averages = [21, 61, 200]
        for ma in moving_averages:
            df['SMA_' + str(ma)] = round(df['Adj Close'].rolling(window=ma).mean(), 2)

        # Storing required values
        latest_price = df["Adj Close"][-1]
        pe_ratio = float(si.get_quote_table(ticker)['PE RATIO (TTM)'])
        peg_ratio = float(si.get_stats_valuation(ticker)[1][4])
        moving_average_21 = df['SMA_21'][-1]
        moving_average_61 = df['SMA_61'][-1]
        moving_average_200 = df['SMA_200'][-1]
        low_52week = round(min(df['Low'][-(52*5):]), 2)
        high_52week = round(max(df['High'][-(52*5):]), 2)
        score = round(best_performers[best_performers['Ticker'] == ticker]['Score'].tolist()[0])

        try:
            moving_average_200_20 = df["SMA_200"][-20]
        except Exception:
            moving_average_200_20 = 0

        # Condition 1: Current Price > 61 SMA and > 200 SMA
        condition_1 = latest_price > moving_average_21 > moving_average_61 > moving_average_200

        # Condition 2: 61 SMA and > 200 SMA
        condition_2 = moving_average_61 > moving_average_200

        # Condition 3: 200 SMA trending up for at least 1 month
        condition_3 = moving_average_200 > moving_average_200_20

        # Condition 4: Current Price is at least 30% above 52 week low
        condition_4 = latest_price >= (1.3 * low_52week)

        # Condition 5: Current Price is within 25% of 52 week high
        condition_5 = latest_price >= (0.75 * high_52week)

        # Condition 6: PE Ratio less than 40
        condition_6 = pe_ratio < 40

        # Condition 7: PEG Ratio less than 2
        condition_7 = peg_ratio < 2

        # If all conditions above are true, add stock to exportList
        if (condition_1 and condition_2 and condition_3 and condition_4 and condition_5 and condition_6
                and condition_7):
            final_df = final_df.append({'Ticker': ticker, 'Latest_Price': latest_price, "Score": score,
                                        'PE_Ratio': pe_ratio, 'PEG_Ratio': peg_ratio,
                                        "200 Day MA": moving_average_200,
                                        "52 Week Low": low_52week, "52 week High": high_52week},
                                       ignore_index=True)
            print(ticker + " made the Minervini requirements")

    except Exception as e:
        print(f"{e} for {ticker}")

final_df = final_df.sort_values(by='Score', ascending=False)
print('\n', final_df)
writer = ExcelWriter("../ScreenOutputZero.xlsx")
final_df.to_excel(writer, "Sheet1")
writer.save()
