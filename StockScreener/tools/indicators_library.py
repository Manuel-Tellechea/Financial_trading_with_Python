# Imports
from yahoo_fin import stock_info as si
from pandas_datareader import data as pdr
import numpy as np
import yfinance as yf
import pandas as pd
import datetime
import time
import sys

# start_time = time.time()
#
# np.set_printoptions(threshold=sys.maxsize)
#
# yf.pdr_override()
# pd.set_option('display.max_columns', 20)
# pd.set_option('display.width', 2000)
#
# # Constants
# TICKER = '^GSPC'  # S&P 500
# START_DATE = datetime.datetime.now() - datetime.timedelta(days=3650)
# END_DATE = datetime.date.today()
# data = pdr.get_data_yahoo(TICKER, START_DATE, END_DATE)


class Indicators:
    def __init__(self, df):
        self.open = df['Open'].values.astype(np.float32)
        self.high = df['High'].values.astype(np.float32)
        self.low = df['Low'].values.astype(np.float32)
        self.close = df['Close'].values.astype(np.float32)
        self.adj_close = df['Adj Close'].values.astype(np.float32)
        self.volume = df['Volume'].values.astype(np.float32)

    def lag(self, lag_value, arr=None):
        """
            Documentation:
             Compute a lagged version of a time series, shifting the time base back by a given number of observations.

            :param arr: An array time series
            :param lag_value: The number of lags (in units of observations).

            :return: lag array  of the time series.
            """

        if arr is None:
            arr = self.close

        lagged = np.roll(arr, lag_value)
        for i in range(lag_value):
            lagged[i] = np.NaN
        return lagged

    def returns(self, arr=None):
        """
        calculates the moving correlation between 2 vectors for a given period, omiting all nans in the datas.
        uses the nancorr function.

        :param arr: first array
        :return: correlation omitting all the nan values
        """

        if arr is None:
            arr = self.close

        arr_shifted = np.hstack((np.zeros(1) * np.nan, arr[:-1]))
        arr_shifted[arr_shifted == 0] = np.nan
        return_value = np.log(arr / arr_shifted)

        return return_value

    def sma(self, window, arr=None):
        """
        This version of ma omits all nan in the calculation process.

        ma stands for the moving average of a price series.
        :param window: int value, its the length of the moving window, minimum value for the window may be 2.
        :param arr: np.array with dimension 1, should be np.float32 numbers.
        :return: np.array vector where the first #window-elements are np.nan, then you have the moving average of the arr.
        """

        if arr is None:
            arr = self.close

        ma_values_close = np.array([np.nanmean(arr[i:i + window]) for i in range(len(arr) - window + 1)])
        moving_average = np.append(np.zeros(window - 1) + np.nan, ma_values_close)

        return moving_average.astype(np.float32)

    def std(self, window, arr=None):
        """
        This version of std omits all nan in the calculation process.
        std stands for the moving standard deviation of a price series. similar to the ma function.

        :param window: int value, its the length of the moving window, minimum value for the window may be 2.
        :param arr: np.array with dimension 1, should be np.float32 numbers.
        :return: np.array vector where the first #window-elements are np.nan, then you have the std of the arr.
        """

        if arr is None:
            arr = self.close

        np_std_values_close = np.array([np.nanstd(arr[i:i + window]) for i in range(len(arr) - window + 1)])
        std_close = np.append(np.zeros(window - 1) + np.nan, np_std_values_close).astype(np.float32)

        return std_close

    def ema(self, period, arr=None):
        """
            Documentation:
            An exponential moving average (EMA) is a type of moving average (MA) that places a greater weight and significance
            on the most recent data points. The exponential moving average is also referred to as the exponentially weighted
            moving average. An exponentially weighted moving average reacts more significantly to recent price changes than
            a simple moving average (SMA), which applies an equal weight to all observations in the period.


            EMA(t) = Value t * (Smoothing/(1+days)) + EMA t-1 (1-(Smoothing/(1+days)))

            While there are many possible choices for the smoothing factor, the most common choice is:

            Smoothing = 2
            That gives the most recent observation more weight. If the smoothing factor is increased,
            more recent observations have more influence on the EMA.

            :param period: integer value, the bigger the value the slower the calculation will but also less impacted by
            outlayers, picking a good value according to the temporality is a most.
            :param arr: numpy array, dtype=np.float32, this is the close time series you want to use

            :return: numpy array, dtype=np.float32, this is the value for every moment of the calculation.
            """

        if arr is None:
            arr = self.close

        ema_calc = np.zeros(len(arr))
        ema_calc[period - 1] = np.mean(arr[0:period])
        for i in range(period, len(ema_calc)):
            ema_calc[i] = (arr[i] - ema_calc[i - 1]) * (2 / (period + 1)) + ema_calc[i - 1]
        ema_calc[:period - 1] = np.nan

        ema_calc = ema_calc.astype(np.float32)
        return ema_calc

    def smoothed_ma(self, window, arr=None):
        """
            Documentation:
            Smoothed moving average is a moving average that deals with a longer period, allowing for an
            easier price calculation and viewing and represents the combination of simple moving average
            and exponential moving average. A smoothed moving average does not refer to a fixed period,
            but rather collects and enrolls all available data from the past. To calculate today’s moving
            average, you have to subtract the yesterday’s smoothed moving average from today’s price. After
            that, you have to add the result to yesterday’s price.

            The SMMA formula

            The formula to calculate the SMMA is

            SMMA = (SMMA# – SMMA* + CLOSE)/N
            Where

            SMMA# – Previous bar’s smoothed sum

            SMMA* – Previous bar’s smoothed moving average
            CLOSE – Present closing price

            N – Period of smoothing

            :param window: integer value, the bigger the value the slower the calculation will but also less impacted by
            outlayers, picking a good value according to the temporality is a most.
            :param arr: numpy array, dtype=np.float32, this is the close time series you want to use

            :return: numpy array, dtype=np.float32, this is the value for every moment of the calculation.
         """

        if arr is None:
            arr = self.close

        alpha = 1 / window
        smma_calc = np.zeros(len(arr))
        smma_calc[window - 1] = np.nanmean(arr[0:window])
        for i in range(window, len(smma_calc)):
            smma_calc[i] = (arr[i] - smma_calc[i - 1]) * alpha + smma_calc[i - 1]
            smma_calc[:window - 1] = np.nan
        smma_calc = smma_calc.astype(np.float32)
        return smma_calc

    def wma(self, window, arr=None):
        """
                Documentation:
                Weighted moving averages assign a heavier weighting to more current data points since
                they are more relevant than data points in the distant past. The sum of the weighting
                should add up to 1 (or 100 percent). In the case of the simple moving average, the
                weightings are equally distributed, which is why they are not shown in the table above.

                The weighted average is calculated by multiplying the given price by its associated
                weighting and totaling the values. The formula for the WMA is as follows:

                WMA = (value_1 x n + value_2 x (n-1) + ... value_n) / ((n x (n+1))/2)

                n = time period

                :param window: integer value, the bigger the value the slower the calculation will but also less impacted by
                outlayers, picking a good value according to the temporality is a most.
                :param arr: numpy array, dtype=np.float32, this is the close time series you want to use

                :return: numpy array, dtype=np.float32, this is the value for every moment of the calculation.
             """

        if arr is None:
            arr = self.close

        wma_calc = np.zeros(len(arr) - window + 1)
        for i in range(window - 1, len(arr)):
            wma_calc[i - window + 1] = np.sum(np.arange(1, window + 1, 1) * arr[i - window + 1:i + 1]) \
                                       / np.sum(np.arange(1, window + 1))
        return wma_calc.astype(np.float32)

    def engulfing_pattern(self):
        """
        The bullish engulfing pattern consists of two candlesticks, the first black and the second white. The size of the
        black candlestick is not that important, but it should not be a doji which would be relatively easy to engulf. The
        second should be a long white candlestick – the bigger it is, the more bullish. The white body must totally engulf
        the body of the first black candlestick. Ideally, though not necessarily, the white body would engulf the shadows
        as well. Although shadows are permitted, they are usually small or nonexistent on both candlesticks.

        The bearish engulfing pattern consists of two candlesticks: the first is white and the second black. The size of the
        white candlestick is relatively unimportant, but it should not be a doji, which would be relatively easy to engulf.
        The second should be a long black candlestick. The bigger it is, the more bearish the reversal. The black body must
        totally engulf the body of the first white candlestick. Ideally, the black body should engulf the shadows as well,
        but this is not a requirement. Shadows are permitted, but they are usually small or nonexistent on both
        candlesticks.

        The bullish engulfing pattern is a strong reversal signal when it appears at the bottom. The bearish
        engulfing is a strong reversal signal when it appears at the top.

        :param self: numpy arrays, dtype=np.float32, those are the open, high, low and close price time series.
        :return: numpy arrays, dtype=np.float32, whit the True values for engulfing patterns presence.
        """

        open_shifted = np.hstack((np.zeros(1) * np.nan, self.open[:-1]))
        open_shifted[open_shifted == 0] = np.nan
        high_shifted = np.hstack((np.zeros(1) * np.nan, self.high[:-1]))
        high_shifted[open_shifted == 0] = np.nan
        low_shifted = np.hstack((np.zeros(1) * np.nan, self.low[:-1]))
        low_shifted[low_shifted == 0] = np.nan
        close_shifted = np.hstack((np.zeros(1) * np.nan, self.close[:-1]))
        close_shifted[close_shifted == 0] = np.nan

        bullish_engulfing = np.zeros(len(self.close))
        bearish_engulfing = np.zeros(len(self.close))
        pattern = []

        for i in range(len(self.close)):
            if self.close[i] > open_shifted[i] > close_shifted[i] > self.open[i]:
                bullish_engulfing[i] = 1
                pattern.append("Bullish Engulfing")
            elif self.open[i] > close_shifted[i] > open_shifted[i] > self.close[i]:
                bearish_engulfing[i] = 1
                pattern.append("Bearish Engulfing")

        return bullish_engulfing, bearish_engulfing, pattern

    def rolling_max(self, window: int, arr=None):
        """
        rolling_max stands the maximum value of a price series within a moving window
        :param arr: np.array with dimension 1, should be np.float32 numbers.
        :param window:int value, its the length of the moving window, minimum value for the window may be 2.
        :return: np.array vector with the values for the roll_max.
        """

        if arr is None:
            arr = self.high

        ma_values_close = np.array([arr[i:i + window].max() for i in range(len(arr) - window + 1)])
        roll_max = np.append(np.zeros(window - 1) + np.nan, ma_values_close).astype(np.float32)
        return roll_max

    def rolling_min(self, window: int, arr=None):
        """
        rolling_min stands the minimum value of a price series within a moving window
        :param arr: np.array with dimension 1, should be np.float32 numbers.
        :param window:int value, its the length of the moving window, minimum value for the window may be 2.
        :return: np.array vector with the values for the roll_min.
        """

        if arr is None:
            arr = self.low

        ma_values_close = np.array([arr[i:i + window].min() for i in range(len(arr) - window + 1)])
        roll_min = np.append(np.zeros(window - 1) + np.nan, ma_values_close).astype(np.float32)
        return roll_min

    def last_values(self):
        values = [self.open, self.high, self.low, self.close, self.adj_close, self.volume]
        return values


# indicators = Indicators(data)
#
# # Calculating Index returns
# sp500_returns = indicators.returns()
# sp500_std = indicators.std(21)
# print(sp500_std[:30])
#
#
# print("--- %s seconds ---" % (time.time() - start_time))
