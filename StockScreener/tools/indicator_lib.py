import numpy as np
import pandas as pd
from numba import njit, jit
from pandas_datareader import data as pdr


@njit
def rolling_max(arr: np.array, window: int):
    """
    rolling_max stands the maximum value of a price series within a moving window
    :param arr: np.array with dimension 1, should be np.float32 numbers.
    :param window:int value, its the length of the moving window, minimum value for the window may be 2.
    :return: np.array vector with the values for the roll_max.
    """
    ma_values_close = np.array([arr[i:i + window].max() for i in range(len(arr) - window + 1)])
    roll_max = np.append(np.zeros(window - 1) + np.nan, ma_values_close).astype(np.float32)
    return roll_max


@njit
def rolling_min(arr: np.array, window: int):
    """
    rolling_min stands the minimum value of a price series within a moving window
    :param arr: np.array with dimension 1, should be np.float32 numbers.
    :param window:int value, its the length of the moving window, minimum value for the window may be 2.
    :return: np.array vector with the values for the roll_min.
    """
    ma_values_close = np.array([arr[i:i + window].min() for i in range(len(arr) - window + 1)])
    roll_min = np.append(np.zeros(window - 1) + np.nan, ma_values_close).astype(np.float32)
    return roll_min


@njit
def sma(arr: np.array, window: int):
    """
    This version of ma omits all nan in the calculation process.

    ma stands for the moving average of a price series.
    :param arr: np.array with dimension 1, should be np.float32 numbers.
    :param window: int value, its the length of the moving window, minimum value for the window may be 2.
    :return: np.array vector where the first #window-elements are np.nan, then you have the moving average of the arr.
    """
    ma_values_close = np.array([np.nanmean(arr[i:i + window]) for i in range(len(arr) - window + 1)])
    moving_average = np.append(np.zeros(window - 1) + np.nan, ma_values_close)

    return moving_average.astype(np.float32)


@njit
def returns(arr: np.array):
    """
    calculates the moving correlation between 2 vectors for a given period, omiting all nans in the datas.
    uses the nancorr function.

    :param arr: first array
    :return: correlation omitting all the nan values
    """
    arr_shifted = np.hstack((np.zeros(1) * np.nan, arr[:-1]))
    arr_shifted[arr_shifted == 0] = np.nan
    return_value = np.log(arr / arr_shifted)

    return return_value


@njit
def lag(arr: np.array, lag_value: int):
    """
        Documentation:
         Compute a lagged version of a time series, shifting the time base back by a given number of observations.

        :param arr: An array time series
        :param lag_value: The number of lags (in units of observations).

        :return: lag array  of the time series.
        """
    lagged = np.roll(arr, lag_value)
    for i in range(lag_value):
        lagged[i] = np.NaN

    return lagged


@njit
def engulfing_pattern(open, high, low, close):
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

    :param open: numpy array, dtype=np.float32, this is the open price time series you want to use.
    :param high: numpy array, dtype=np.float32, this is the high price time series you want to use.
    :param low: numpy array, dtype=np.float32, this is the low price time series you want to use.
    :param close: numpy array, dtype=np.float32, this is the close price time series you want to use.
    :return: numpy arrays, dtype=np.float32, whit the True values for engulfing patterns presence.
    """

    lagged_open = lag(open, 1)
    lagged_high = lag(high, 1)
    lagged_low = lag(low, 1)
    lagged_close = lag(close, 1)
    bullish_engulfing = np.zeros(len(close))
    bearish_engulfing = np.zeros(len(close))
    pattern = []

    for i in range(len(close)):
        if close[i] > lagged_open[i] > lagged_close[i] > open[i]:
            bullish_engulfing[i] = 1
            pattern.append("Bullish Engulfing")
        elif open[i] > lagged_close[i] > lagged_open[i] > close[i]:
            bearish_engulfing[i] = 1
            pattern.append("Bearish Engulfing")

    return bullish_engulfing, bearish_engulfing, pattern


def basic_calculation(ticker, start_date, end_date):
    df = pdr.get_data_yahoo(ticker, start_date, end_date)
    stock_open = df['Open'].values.astype(np.float32)
    stock_high = df['High'].values.astype(np.float32)
    stock_low = df['Low'].values.astype(np.float32)
    stock_close = df['Close'].values.astype(np.float32)
    stock_volume = df['Volume'].values.astype(np.float32)

    # Calculating SMA21, SMA63 and SMA200
    moving_average_21 = sma(stock_close, 21)
    moving_average_21 = moving_average_21[-1]
    moving_average_63 = sma(stock_close, 63)
    moving_average_63 = moving_average_63[-1]
    moving_average_200 = sma(stock_close, 200)
    # Storage value of a 20 days ago of a SMA200
    try:
        moving_average_200_20 = moving_average_200[-20]
    except Exception:
        moving_average_200_20 = 0

    moving_average_200 = moving_average_200[-1]

    # Calculating New 52 Weeks High
    high_of_52week = rolling_max(stock_high, 260)
    high_of_52week = high_of_52week[-1]

    # Calculating New 52 Weeks Low
    low_of_52week = rolling_min(stock_low, 260)
    low_of_52week = low_of_52week[-1]

    # Last Prices
    current_close = round(stock_close[-1], 2)

    basics = [moving_average_21, moving_average_63, moving_average_200, moving_average_200_20, high_of_52week,
              low_of_52week, current_close]

    return basics
