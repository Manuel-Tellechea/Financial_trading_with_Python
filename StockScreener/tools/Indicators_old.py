import numpy as np
from numba import njit, jit
import pandas as pd


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def standarize(arr, periods):
    """
            Documentation:
            The standardized function determines that given a series of data,
            it assumes that they are distributed as a normal function, determines
            the mean and its standard deviation and through the formula of
            z-score = x-Mean / StDev determines the indicator Z
            The formula to calculate the Z is
            Z = x - μ  / σ
                x -> values ​​to normalize
                μ -> Arithmetic mean of the distribution
                σ -> Standard deviation of the distribution.
            if periodos > 1:
                 μ it will be the moving average, σ t will be the moving standard deviation,
                 both will depend on the value of periods
            if periods == 0
                μ and σ are calculated over the entire data series
            :param arr: numpy array, dtype=np.float32, this is the data series you want to standardize
            :param periods: integer value, calculate moving mean and standard deviation moving
            :return: numpy array, dtype=np.float32, this is the value for every moment of the calculation.
     """
    if periods > 1:
        np_ma_values = np.array([arr[i:i + periods].mean() for i in range(len(arr) - periods + 1)])
        np_ma = np.append(np.zeros(periods - 1) + np.nan, np_ma_values)
        np_std_values = np.array([arr[i:i + periods].std() for i in range(len(arr) - periods + 1)])
        np_std = np.append(np.zeros(periods - 1) + np.nan, np_std_values)
        # to ensure np_std doesn't contain zeros
        np_std = np.where(np_std == 0, np.nan, np_std)
        z_column = (arr - np_ma) / np_std
    if periods == 0:
        n = len(arr)
        np_ma = np.array([np.nanmean(arr[0:i + 1]) for i in range(n)])
        np_std = np.array([np.nanstd(arr[0:i + 1]) for i in range(n)])
        # to ensure np_std doesn't contain zeros
        np_std = np.where(np_std == 0, np.nan, np_std)
        z_column = (arr - np_ma) / np_std
        z_column[0:51] = np.nan
    z_column = z_column.astype(np.float32)
    return z_column


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def intermarket_standarize(arr, periods):
    """
        USE ME ONLY AFTER MERGING DATA
        Same as standarize, but this one omits nan values for Intermarkets

        Documentation:
        The standardized function determines that given a series of data,
        it assumes that they are distributed as a normal function, determines
        the mean and its standard deviation and through the formula of
        z-score = x-Mean / StDev determines the indicator Z
        The formula to calculate the Z is
        Z = x - μ  / σ
            x -> values ​​to normalize
            μ -> Arithmetic mean of the distribution
            σ -> Standard deviation of the distribution.
        if periodos > 1:
             μ it will be the moving average, σ t will be the moving standard deviation,
             both will depend on the value of periods
        if periods == 0
            μ and σ are calculated over the entire data series
        :param arr: numpy array, dtype=np.float32, this is the data series you want to standardize
        :param periods: integer value, calculate moving mean and standard deviation moving
        :return: numpy array, dtype=np.float32, this is the value for every moment of the calculation.
     """
    if periods > 1:
        np_ma_values = np.array([np.nanmean(arr[i:i + periods]) for i in range(len(arr) - periods + 1)])
        np_ma = np.append(np.zeros(periods - 1) + np.nan, np_ma_values)
        np_std_values = np.array([np.nanstd(arr[i:i + periods]) for i in range(len(arr) - periods + 1)])
        np_std = np.append(np.zeros(periods - 1) + np.nan, np_std_values)
        # to ensure np_std doesn't contain zeros
        np_std = np.where(np_std == 0, np.nan, np_std)
        z_column = (arr - np_ma) / np_std

    if periods == 0:
        n = len(arr)
        np_ma = np.array([np.nanmean(arr[0:i + 1]) for i in range(n)])
        np_std = np.array([np.nanstd(arr[0:i + 1]) for i in range(n)])
        # to ensure np_std doesn't contain zeros
        np_std = np.where(np_std == 0, np.nan, np_std)
        z_column = (arr - np_ma) / np_std
        z_column[0:51] = np.nan

    z_column = z_column.astype(np.float32)
    return z_column


# @njit('float32[:](float32[:], float32[:], int32)', cache=True, nogil=True)
@njit
def gap(open, close, periods=0):
    """
    Documentation:
    Price charts often have blank spaces known as gaps, which represent times when no shares were traded within a
    particular price range. Normally this occurs between the close of the market on one day and the next day's open.
    There are two primary kinds of gaps - up gaps and down gaps. The strategy seeks to measure the gaps in price with a
    standardization of values.
    :param open: numpy array, dtype=np.float32, this is the open time series you want to use.
    :param close: numpy array, dtype=np.float32, this is the close time series you want to use.
    :param periods: integer value, the bigger the value the slower the calculation will but also less impacted by
    outlayers, picking a good value according to the temporality is a most.
    :return: numpy array, dtype=np.float32, this is the value for every moment of the calculation.
    """
    close_shifted = np.hstack((np.zeros(1) * np.nan, close[:-1]))
    np_gap = (open - close_shifted) / close_shifted
    result = standarize(arr=np_gap.astype(np.float32), periods=np.int32(periods))
    result = result.astype(np.float32)
    return result


# @njit('float32[:](float32[:], float32[:], float32[:])', cache=True, nogil=True)
@njit
def tr(high: np.array, low: np.array, close: np.array):
    """
    The average true range (ATR) is a technical analysis indicator that measures market volatility by decomposing
    the entire range of an asset price for that period.

    tr has 3 components, we always save the bigger one as the tr value for that candle, the 3 components are:
    1. current high - current low.
    2. absolute value of the current high - previous close.
    3. absolute value of the current low - previous close.

    :param high:  np.array with the column high. inside values may be np.float32.
    :param low:   np.array with the column low. inside values may be np.float32.
    :param close: np.array with the column close. inside values may be np.float32.
    :return: np.array vector with the values for the true_range.
    """
    true_range = np.zeros(len(high))
    true_range[0] = np.nan

    for i in range(1, len(true_range)):
        high_minus_low = high[i] - low[i]
        abs_high_minus_prev_close = np.fabs(high[i] - close[i - 1])
        abs_low_minus_prev_close = np.fabs(low[i] - close[i - 1])
        true_range[i] = np.amax(np.array([high_minus_low, abs_high_minus_prev_close, abs_low_minus_prev_close]))

    return true_range.astype(np.float32)


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def ma(arr: np.array, window: int):
    """
    ma stands for the moving average of a price series.
    :param arr: np.array with dimension 1, should be np.float32 numbers.
    :param window: int value, its the length of the moving window, minimum value for the window may be 2.
    :return: np.array vector where the first #window-elements are np.nan, then you have the moving average of the arr.
    """
    ma_values_close = np.array([arr[i:i + window].mean() for i in range(len(arr) - window + 1)])
    moving_average = np.append(np.zeros(window - 1) + np.nan, ma_values_close)

    return moving_average.astype(np.float32)


@njit
def nan_ma(arr: np.array, window: int):
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


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def std(arr: np.array, window: int):
    """
    std stands for the moving standard deviation of a price series. similar to the ma function.

    :param arr: np.array with dimension 1, should be np.float32 numbers.
    :param window: int value, its the length of the moving window, minimum value for the window may be 2.
    :return: np.array vector where the first #window-elements are np.nan, then you have the std of the arr.
    """
    np_std_values_close = np.array([arr[i:i + window].std() for i in range(len(arr) - window + 1)])
    std_close = np.append(np.zeros(window - 1) + np.nan, np_std_values_close).astype(np.float32)

    return std_close


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def nan_std(arr: np.array, window: int):
    """
    This version of std omits all nan in the calculation process.
    std stands for the moving standard deviation of a price series. similar to the ma function.
    :param arr: np.array with dimension 1, should be np.float32 numbers.
    :param window: int value, its the length of the moving window, minimum value for the window may be 2.
    :return: np.array vector where the first #window-elements are np.nan, then you have the std of the arr.
    """
    np_std_values_close = np.array([np.nanstd(arr[i:i + window]) for i in range(len(arr) - window + 1)])
    std_close = np.append(np.zeros(window - 1) + np.nan, np_std_values_close).astype(np.float32)

    return std_close


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def ema(arr, period):
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

        :param arr: numpy array, dtype=np.float32, this is the close time series you want to use
        :param period: integer value, the bigger the value the slower the calculation will but also less impacted by
        outlayers, picking a good value according to the temporality is a most.

        :return: numpy array, dtype=np.float32, this is the value for every moment of the calculation.
        """

    ema_calc = np.zeros(len(arr))
    ema_calc[period - 1] = np.mean(arr[0:period])
    for i in range(period, len(ema_calc)):
        ema_calc[i] = (arr[i] - ema_calc[i - 1]) * (2 / (period + 1)) + ema_calc[i - 1]
    ema_calc[:period - 1] = np.nan

    ema_calc = ema_calc.astype(np.float32)
    return ema_calc


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def smma(arr, period):
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

        :param period:
        :param arr: numpy array, dtype=np.float32, this is the close time series you want to use
        :param window: integer value, the bigger the value the slower the calculation will but also less impacted by
        outlayers, picking a good value according to the temporality is a most.

        :return: numpy array, dtype=np.float32, this is the value for every moment of the calculation.
     """

    alpha = 1 / period
    smma_calc = np.zeros(len(arr))
    smma_calc[period - 1] = np.nanmean(arr[0:period])
    for i in range(period, len(smma_calc)):
        smma_calc[i] = (arr[i] - smma_calc[i - 1]) * alpha + smma_calc[i - 1]
        smma_calc[:period - 1] = np.nan
    smma_calc = smma_calc.astype(np.float32)
    return smma_calc


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def wma(arr, window):
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

            :param arr: numpy array, dtype=np.float32, this is the close time series you want to use
            :param window: integer value, the bigger the value the slower the calculation will but also less impacted by
            outlayers, picking a good value according to the temporality is a most.

            :return: numpy array, dtype=np.float32, this is the value for every moment of the calculation.
         """

    wma_calc = np.zeros(len(arr) - window + 1)
    for i in range(window - 1, len(arr)):
        wma_calc[i - window + 1] = np.sum(np.arange(1, window + 1, 1) * arr[i - window + 1:i + 1]) \
                                   / np.sum(np.arange(1, window + 1))
    return wma_calc.astype(np.float32)


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
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


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
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


# @njit('(float32[:], int32, int32)', cache=True, nogil=True)
@njit
def bband(arr, window, std_param):
    """
    Bollinger Bands are volatility bands placed above and below a moving average. Volatility is based on
    the standard deviation, which changes as volatility increases and decreases.
    The bands automatically widen when volatility increases and contract when volatility decreases.

    The bands has three components:
    * Middle Band = 20-day simple moving average (SMA)
    * Upper Band = 20-day SMA + (20-day standard deviation of price x 2)
    * Lower Band = 20-day SMA - (20-day standard deviation of price x 2)

    :param arr: the array input, usually the close price
    :param window: the rolling period for the calculations
    :param std_param: the standard deviation multiplier
    :return: np.array vector with the middle, upper and lower bands calculation
    """
    # MA
    ma_values_close = np.array([arr[i:i + window].mean() for i in range(len(arr) - window + 1)])
    ma_close = np.append(np.zeros(window - 1) + np.nan, ma_values_close).astype(np.float32)

    # STD
    np_std_values_close = np.array([arr[i:i + window].std() for i in range(len(arr) - window + 1)])
    std_close = np.append(np.zeros(window - 1) + np.nan, np_std_values_close).astype(np.float32)

    upper_band = ma_close + std_param * std_close
    lower_band = ma_close - std_param * std_close
    middle_band = (upper_band + lower_band) / 2

    return upper_band.astype(np.float32), lower_band.astype(np.float32), middle_band.astype(np.float32)


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def rsi(close, window):
    """
    Documentation:
    The Relative Strength Index is a momentum oscillator that measures the speed and change of price movements.
    RSI = 100 - (100 / (1 + RS))
    RS = Average gain / Average loss

    The very first calculations for average gain and average loss are simple 14-period averages:

        - First Average Gain = Sum of Gains over the past 14 periods / 14.
        - First Average Loss = Sum of Losses over the past 14 periods / 14.

    The second, and subsequent, calculations are based on the prior averages and the current gain loss:

        - Average Gain = [(previous Average Gain) x 13 + current Gain] / 14.
        - Average Loss = [(previous Average Loss) x 13 + current Loss] / 14.


    Interpretation:
    RSI oscillates between zero and 100 and is considered overbought when above 70 and oversold when below 30.

    Example:
    Observe the last 14 closing prices of a stock.
    Determine whether the current day’s closing price is higher or lower than the previous day.
    Calculate the average gain and loss over the last 14 days.
    Compute the relative strength (RS): (AvgGain/AvgLoss)
    Compute the relative strength index (RSI): (100–100 / ( 1 + RS))

    The RSI will then be a value between 0 and 100.

    :param close: numpy array, dtype=np.float32, this is the close time series you want to use
    :param window: integer value, the bigger the value the slower the calculation will but also less impacted by
    outlayers, picking a good value according to the temporality is a most.

    :return: numpy array, dtype=np.float32, this is the value for the rsi.
    """
    shifted = np.roll(close, 1)[1:]
    shifted = np.append(np.zeros(1) + np.nan, shifted)

    delta = close - shifted
    gain, loss = np.copy(delta), np.copy(delta)
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = np.absolute(loss)

    avg_gain = smma(gain.astype(np.float32), np.int32(window))
    avg_loss = smma(loss.astype(np.float32), np.int32(window))

    n = len(avg_loss)
    rsi_value = np.zeros(n) + np.nan
    for i in range(n):
        if avg_loss[i] == 0:
            rsi_value[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_value[i] = 100 - (100 / (1 + rs))

    rsi_value[0: window] = np.nan

    return rsi_value.astype(np.float32)


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def roc(close, window):
    """
    Documentation:
    The Rate-of-Change (ROC) indicator is a pure momentum oscillator that measures the percent change in price
    from one period to the next. The ROC calculation compares the current price with the price “n” periods ago. The
    plot forms an oscillator that fluctuates above and below the zero line as the Rate-of-Change moves from positive
    to negative.

    ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100

    The RSI will then be a value between 0 and 100.

    Interpretation:
    In general, prices are rising as long as the Rate-of-Change remains positive. Conversely, prices are falling
    when the Rate-of-Change is negative. ROC expands into positive territory as an advance accelerates. ROC dives
    deeper into negative territory as a decline accelerates.

    Example:
    As for example imagine close=[2, 1, 4, 5, 6, 8, 5, 6] and window = 5, then the ROC will be
    ROC = [(6 - 4) / 4] * 100
    ROC = 50

    :param close: numpy array, dtype=np.float32, this is the close time series you want to use
    :param window: integer value, the bigger the value the slower the calculation will but also less impacted by
    outlayers, picking a good value according to the temporality is a most.

    :return: numpy array, dtype=np.float32, whit the value for the rate of change.
    """
    roc_calc = (close[window:] - close[:-window]) / close[:-window] * 100
    roc_calc = np.concatenate((np.array([np.nan] * window), roc_calc)).astype(np.float32)

    return roc_calc


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def streak(close, periods=0):
    """
    Documentation:
    The strategy seeks measure the streaks in a price series with standardization of values.

    :param close: numpy array, dtype=np.float32, this is the close time series you want to use
    :param periods: integer value, the bigger the value the slower the calculation will but also less impacted by
    outlayers, picking a good value according to the temporality is a most.

    :return: numpy array, dtype=np.float32, this is the value of the ratio for every moment of the calculation.
    """
    curr_streak = np.zeros(len(close))
    streaks = int(0)

    for i in range(len(close)):
        if i == 0:
            continue
        if close[i] > close[i - 1]:
            streaks = np.maximum(0, streaks)
            streaks += 1
            curr_streak[i] = streaks
        elif close[i] < close[i - 1]:
            streaks = np.minimum(0, streaks)
            streaks += -1
            curr_streak[i] = streaks
        else:
            streaks = 0
            curr_streak[i] = streaks

    result = standarize(curr_streak.astype(np.float32), periods=np.int32(periods))

    return result.astype(np.float32)


# @njit('float32[:](float32[:], int32, int32)', cache=True, nogil=True)
@njit
def velocity(close, n_candle, period):
    """
    Documentation:
    The strategy seeks measure the rate of change in price with standardization of values and capture the acceleration
    or lack of acceleration in the trend. The speed represented by momentum is a more sensitive measure than the
    direction of a trendline.

    :param close: numpy array, dtype=np.float32, this is the close time series you want to use
    :param n_candle: integer, this is the window of valuation
    :param period: integer value, the bigger the value the slower the calculation will but also less impacted by
    outlayers, picking a good value according to the temporality is a most.

    :return: numpy array, dtype=np.float32, this is the value of the ratio for every moment of the calculation.
    """
    change_price = roc(close, n_candle)
    vel = standarize(change_price, period)

    return vel.astype(np.float32)


# @jit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def oscilador_body(column: np.array, periods: int):
    """
    Documentation:
    The oscilador body indicator is a pure momentum oscillator that measures the percent change in the
    current price between the difference of the maximum and minimum price of N periods ago. The
    plot forms an oscillator that fluctuates above a the zero(value between 0 and 100)

    oscilador_body = ((column - min) / (max - min)) * 100

    Example:
    As for example imagine column is price close=[2, 1, 9, 5, 6, 8, 5] and periods = 5, then the oscilador body will be
    oscilador_body = [(6 - 1) / (9 - 1)] * 100
    oscilador_body = 62.5

    :param column: numpy array, dtype=np.float32, this is the close time series you want to use
    :param periods: integer value, the bigger the value the slower the calculation will but also less impacted by
    outlayers, picking a good value according to the temporality is a most.

    :return: numpy array, dtype=np.float32, whit the value for the rate of change.
    """
    if periods >= 1:
        np_max_values = np.array([column[i:i + periods].max() for i in range(len(column) - periods + 1)])
        np_max = np.append(np.zeros(periods - 1) + np.nan, np_max_values)
        np_min_values = np.array([column[i:i + periods].min() for i in range(len(column) - periods + 1)])
        np_min = np.append(np.zeros(periods - 1) + np.nan, np_min_values)
        np_max_min = np_max - np_min
        np_max_min = np.where(np_max_min == 0, np.nan, np_max_min)
        oscilador = np.true_divide((column - np_min), (np_max_min)) * 100

    if periods == 0:
        n = len(column)
        np_max = np.array([np.max(column[0:i + 1]) for i in range(n)])
        np_min = np.array([np.min(column[0:i + 1]) for i in range(n)])
        np_max_min = np_max - np_min
        np_max_min = np.where(np_max_min == 0, np.nan, np_max_min)
        oscilador = np.true_divide((column - np_min), (np_max_min)) * 100
        oscilador[0:51] = np.nan

    return oscilador.astype(np.float32)


# @njit('float32[:](float32[:], int32, int32)', cache=True, nogil=True)
@njit
def percent_b(close, window, std_param):
    """
    Documentation:
    %B quantifies a security's price relative to the upper and lower Bollinger Band. There are six basic relationship
    levels:

    %B is below 0 when price is below the lower band
    %B equals 0 when price is at the lower band
    %B is between 0 and .50 when price is between the lower and middle band (20-day SMA)
    %B is between .50 and 1 when price is between the upper and middle band (20-day SMA)
    %B equals 1 when price is at the upper band
    %B is above 1 when price is above the upper band

    Calculation: %B = (Price - Lower Band)/(Upper Band - Lower Band)

    :param close: np.array, the close prices of a series
    :param window: integer, the window period for the moving average
    :param std_param: the standard deviation multiplier
    :return: numpy array, dtype=np.float32, this is the calculation value.
    """
    bollinger_bands = bband(close, window, std_param)
    per_b = ((close - bollinger_bands[1]) / (bollinger_bands[0] - bollinger_bands[1])) * 100

    return per_b.astype(np.float32)


# @njit('float32[:](float32[:], int32, int32)', cache=True, nogil=True)
@njit
def acceleration(close, n_candle, periods):
    """
    Documentation:

    The strategy seeks to measure the speed of change in price with a standardization of values.

    Interpretation:
    The speed represented by momentum is a more sensitive measure than the direction of a trendline.

    :param close: numpy array, dtype=np.float32, this is the close time series you want to use
    :param n_candle: integer value, the bigger the value the slower the calculation will but also less impacted by
    outlayers, picking a good value according to the temporality is a most.
    :param periods: integer value, the bigger the value the slower the calculation will but also less impacted by
    outlayers, picking a good value according to the temporality is a most.

    :return: numpy array, dtype=np.float32, this is the value for every moment of the calculation.
    """
    roc_close = roc(close, n_candle)
    roc_close_shifted = np.hstack((np.zeros(1) * np.nan, roc_close[:-1]))
    accel = roc_close - roc_close_shifted

    return standarize(accel.astype(np.float32), periods=np.int32(periods)).astype(np.float32)


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def volume(np_volume: np.array, periods: int):
    """
    Indicator that measure extreme values in volume with standardization of values
    :param arr: np.array with dimension 1, should be np.float32 numbers.
    :param window: int value, its the length of the moving window, minimum value for the window may be 2.
    :return: return: numpy array, dtype=np.float32, this is the value for every moment of the calculation .
    """
    z_volume = standarize(np_volume, periods)

    return z_volume.astype(np.float32)


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def cumsum(arr, window):
    """

               Documentation:
               Returns a vector whose elements are the cumulative sums, products of the elements of
               the argument.

               :param arr: numpy array, dtype=np.float32, this is the close time series you want to use
               :param window: integer value.

               :return: numpy array, dtype=np.float32, this is the value for every moment of the calculation.
            """

    values = np.array([arr[i:i + window].sum() for i in range(len(arr) - window + 1)])
    cumsum_result = np.append(np.zeros(window - 1) + np.nan, values)
    return cumsum_result.astype(np.float32)


# @njit('float32[:](float32[:], float32[:], float32[:], int32)', cache=True, nogil=True)
@njit
def adx(high, low, close, window):
    """
    Documentation:
    The Average Directional Index (ADX), Minus Directional Indicator (-DI) and Plus Directional Indicator (+DI)
    represent a group of directional movement indicators that form a trading system developed by Welles Wilder.
    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI) are derived from smoothed
    averages of these differences and measure trend direction over time. These two indicators are often
    collectively referred to as the Directional Movement Indicator (DMI).

    The Average Directional Index (ADX) is in turn derived from the smoothed averages of the difference between
    +DI and -DI; it measures the strength of the trend (regardless of direction) over time.

    Directional movement is calculated by comparing the difference between two consecutive lows with the
    difference between their respective highs.

    Directional movement is positive (plus) when the current high minus the prior high is greater than the prior
    low minus the current low. This so-called Plus Directional Movement (+DM) then equals the current high minus
    the prior high, provided it is positive. A negative value would simply be entered as zero.

    Directional movement is negative (minus) when the prior low minus the current low is greater than the
    current high minus the prior high. This so-called Minus Directional Movement (-DM) equals the prior low
    minus the current low, provided it is positive. A negative value would simply be entered as zero.

    Interpretation:
    The Average Directional Index (ADX) is used to measure the strength or weakness of a trend, not the actual
    direction. Directional movement is defined by +DI and -DI. In general, the bulls have the edge when +DI is
    greater than -DI, while the bears have the edge when -DI is greater. Crosses of these directional indicators
    can be combined with ADX for a complete trading system.

    Example:
    The calculation example below is based on a 14-period indicator setting, as recommended by Wilder.

    - Calculate the True Range (TR), Plus Directional Movement (+DM) and Minus Directional Movement (-DM) for
    each period.

    - Smooth these periodic values using Wilder's smoothing techniques. These are explained in detail in the
    next section.

    - Divide the 14-day smoothed Plus Directional Movement (+DM) by the 14-day smoothed True Range to find the
    14-day Plus Directional Indicator (+DI14). Multiply by 100 to move the decimal point two places. This +DI14
    is the green Plus Directional Indicator line (+DI) that is plotted along with the ADX line.

    - Divide the 14-day smoothed Minus Directional Movement (-DM) by the 14-day smoothed True Range to find the
    14-day Minus Directional Indicator (-DI14). Multiply by 100 to move the decimal point two places. This -DI14
    is the red Minus Directional Indicator line (-DI) that is plotted along with the ADX line.

    - The Directional Movement Index (DX) equals the absolute value of +DI14 less -DI14 divided by the sum of
    +DI14 and -DI14. Multiply the result by 100 to move the decimal point over two places.

    - After all these steps, it is time to calculate the Average Directional Index (ADX) line. The first ADX
    value is simply a 14-day average of DX. Subsequent ADX values are smoothed by multiplying the previous
    14-day ADX value by 13, adding the most recent DX value and dividing this total by 14.

    :param high: numpy array, dtype=np.float32, this is the close time series you want to use
    :param low: numpy array, dtype=np.float32, this is the close time series you want to use
    :param close: numpy array, dtype=np.float32, this is the close time series you want to use
    :param window: integer value, the bigger the value the slower the calculation will but also less impacted by
    outlayers, picking a good value according to the temporality is a most.

    :return: numpy array, dtype=np.float32, whit the value for the adx.
    """
    up = np.zeros(len(high))
    dw = np.zeros(len(high))
    for i in range(1, len(high)):
        up[i] = high[i] - high[i - 1]
        dw[i] = low[i - 1] - low[i]

    # Direction movement
    posdm = np.zeros(len(high))
    negdm = np.zeros(len(high))
    posdm = np.where((up > dw) & (up > 0), up, 0)
    negdm = np.where((dw > up) & (dw > 0), dw, 0)

    pdm = np.zeros(len(high))
    ndm = np.zeros(len(high))
    pdm[window + 1] = np.sum(posdm[:window + 2])
    for i in range(window + 2, len(high)):
        pdm[i] = pdm[i - 1] - (pdm[i - 1] / window) + posdm[i - 1]
    ndm[window + 1] = np.sum(negdm[:window + 1])
    for i in range(window + 2, len(high)):
        ndm[i] = ndm[i - 1] - (ndm[i - 1] / window) + negdm[i - 1]

    # Direction index
    pdi = np.zeros(len(high))
    ndi = np.zeros(len(high))
    tr_arr = np.zeros(len(high))
    tr_arr[0] = 0
    atr_calc = np.zeros(len(high))
    for i in range(1, len(tr_arr)):
        hl = high[i] - low[i]
        hpc = np.fabs(high[i] - close[i - 1])
        lpc = np.fabs(low[i] - close[i - 1])
        tr_arr[i] = np.amax(np.array([hl, hpc, lpc]))

    atr_calc[window + 1] = np.sum(tr_arr[:window + 1])
    for i in range(window + 2, len(high)):
        atr_calc[i] = atr_calc[i - 1] - (atr_calc[i - 1] / window) + tr_arr[i - 1]

    pdi = 100 * (pdm / atr_calc)
    ndi = 100 * (ndm / atr_calc)

    diff = np.zeros(len(high))
    summ = np.zeros(len(high))
    diff = np.fabs(pdi - ndi)
    summ = pdi + ndi

    dx = np.zeros(len(high))
    dx = 100 * (diff / summ)

    adx_calc = np.zeros(len(high))
    adx_calc[window * 2] = np.mean(dx[window + 2:window * 2])
    for i in range(window * 2 + 1, len(high)):
        adx_calc[i] = ((adx_calc[i - 1] * (window - 1)) + dx[i]) / window

    return adx_calc.astype(np.float32), pdi.astype(np.float32), ndi.astype(np.float32)


# @njit('float32[:](float32[:], float32[:])', cache=True, nogil=True)
@njit
def e_ratio(high, low):
    """
    TODO:

    Documentation:
    The engulfing ratio is the quotient between the difference in the high and the low of the current candle and the
    difference between the high and the low of the previous candle.

    Calculation:
    First we get the differences between the high and the low of the current candle:
    HL = High[t] - Low[t]
    Then we get the same difference but between the high and the low of the previous candle:
    HPL = High[t-1] - Low[t-1]
    After that we proceed to calculate the actual ratio between these differences:
    E ratio = HL[t] / HPL[t]

    Interpretation:
    If the engulfing ratio is greater than 1, then we can say that the current candle is engulfing the previous candle,
    because the difference between the High and the Low from the current candle is greater than de difference between
    the High and the Low from the previous candle, and we can say that we have a bullish momentum in that case.

    :param high: np.array, the High prices from the series
    :param low: np.array, the Low prices from the series
    :return: np.array, the e_ratio calculation
    """

    hl = np.fabs(high - low)
    hpl = np.hstack((np.zeros(1) * np.nan, np.fabs(high[:-1] - low[:-1])))

    hpl = np.where(hpl == 0, 1, hpl)  # if hpl=0 -> hpl=1

    e_ratio_o = np.divide(hl, hpl)

    return e_ratio_o.astype(np.float32)


# @njit('float32[:](float32[:], float32[:], float32[:], int32)', cache=True, nogil=True)
@njit
def cci(high, low, close, window):
    """
    TODO:

    Documentation:
    The Commodity Channel Index (CCI) is a versatile indicator that can be used to identify a new
    trend or warn of extreme conditions. Lambert originally developed CCI to identify cyclical turns in commodities,
    but the indicator can be successfully applied to indices, ETFs, stocks and other securities. In general,
    CCI measures the current price level relative to an average price level over a given period of time.

    Calculation:
    CCI = (Typical Price  -  N-period SMA of TP) / (.015 x Mean Deviation)

    Typical Price (TP) = (High + Low + Close)/3

    Constant = .015

    There are four steps to calculating the Mean Deviation:
    First, subtract the most recent N-period average of the typical price from each period's typical price.
    Second, take the absolute values of these numbers.
    Third, sum the absolute values.
    Fourth, divide by the total number of periods (N).

    Interpretation:
    CCI measures the difference between a security's price change and its average price change. High
    positive readings indicate that prices are well above their average, which is a show of strength. Low negative
    readings indicate that prices are well below their average, which is a show of weakness.

    The Commodity Channel Index (CCI) can be used as either a coincident or leading indicator. As a coincident
    indicator, surges above +100 reflect strong price action that can signal the start of an uptrend. Plunges below
    -100 reflect weak price action that can signal the start of a downtrend.

    :param high: numpy array, this is the high price of the candle
    :param low: numpy array, the low price of the candle
    :param close: numpy array, the closing price of the candle
    :param window: integer, the parameter for the moving averages in the calculation
    :return: np.array the values of the CCI
    """
    tp = np.zeros(len(high))
    for i in range(0, len(high)):
        tp[i] = (high[i] + low[i] + close[i]) / 3

    atp = np.zeros(len(high))  # average typical price
    md = np.zeros(len(high))  # mean deviation
    cci_value = np.zeros(len(high))
    for i in range(window - 1, len(high)):
        atp[i] = np.sum(tp[i - (window - 1):i + 1]) / window
        md[i] = np.sum(np.fabs(atp[i] - tp[i - (window - 1):i + 1])) / window
        if md[i] == 0:
            md[i] = np.NaN
        cci_value[i] = (tp[i] - atp[i]) / (0.015 * md[i])
    return cci_value.astype(np.float32)


# @njit('float32[:](float32[:], float32[:], int32, int32)', cache=True, nogil=True)
@njit
def ao(high, low, fast_sma, slow_sma):
    """

    Documentation:
    The AO indicator is a good indicator for measuring the market dynamics, it reflects specific
    changes in the driving force of the market, which helps to identify the strength of the trend, including the
    points of its formation and reversal.

    The Awesome Oscillator detects the prevalence of bullish or bearish forces in the market by comparing the recent
    momentum (5 bars) with the wider trend (34 bars): the 34-period SMA is subtracted from the 5-period SMA.
    Moreover, the 34-period and 5-period simple moving averages are calculated not by closing prices, as is the case
    of many indicators, but by the midpoints of the bars (arithmetic averages of the highs and lows for the chosen
    timeframe).

    Calculation: Awesome Oscillator is a 34-period simple moving average, plotted through the central points of the
    bars (H+L)/2, and subtracted from the 5-period simple moving average, graphed across the central points of the
    bars (H+L)/2.

    MEDIAN PRICE = (HIGH+LOW)/2
    AO = SMA(MEDIAN PRICE, 5)-SMA(MEDIAN PRICE, 34)

    where:
    SMA — Simple Moving Average.

    Interpretation:
    The oscillator builds its bars above and below the zero line depending on whether the 5-period MA
    is above or below the 34-period MA. In the first case, AO will have a positive value, and its bars will line up
    above the zero line. In the second case, we will see the oscillator bars below the zero level. As the trend
    increases, the moving averages will diverge more, and the oscillator bars will stretch more up or down (with
    bullish and bearish trends, respectively).

    :param high: np.array, the high prices of a series
    :param low: np.array, the low prices of a series
    :param fast_sma: integer, the window period for the fast moving average in the oscillator
    :param slow_sma: integer, the window period for the slow moving average in the oscillator
    :return:
    """

    med_price = np.zeros(len(high))
    for i in range(0, len(high)):
        med_price[i] = (high[i] - low[i]) / 2
    ma_fast = ma(med_price.astype(np.float32), np.int32(fast_sma))
    ma_slow = ma(med_price.astype(np.float32), np.int32(slow_sma))
    ao_data = ma_fast - ma_slow
    return ao_data.astype(np.float32)


@njit
def ao_fix(high, low, fast_sma, slow_sma):
    """

    Documentation:
    The AO indicator is a good indicator for measuring the market dynamics, it reflects specific
    changes in the driving force of the market, which helps to identify the strength of the trend, including the
    points of its formation and reversal.

    The Awesome Oscillator detects the prevalence of bullish or bearish forces in the market by comparing the recent
    momentum (5 bars) with the wider trend (34 bars): the 34-period SMA is subtracted from the 5-period SMA.
    Moreover, the 34-period and 5-period simple moving averages are calculated not by closing prices, as is the case
    of many indicators, but by the midpoints of the bars (arithmetic averages of the highs and lows for the chosen
    timeframe).

    Calculation: Awesome Oscillator is a 34-period simple moving average, plotted through the central points of the
    bars (H+L)/2, and subtracted from the 5-period simple moving average, graphed across the central points of the
    bars (H+L)/2.

    MEDIAN PRICE = (HIGH+LOW)/2
    AO = SMA(MEDIAN PRICE, 5)-SMA(MEDIAN PRICE, 34)

    where:
    SMA — Simple Moving Average.

    Interpretation:
    The oscillator builds its bars above and below the zero line depending on whether the 5-period MA
    is above or below the 34-period MA. In the first case, AO will have a positive value, and its bars will line up
    above the zero line. In the second case, we will see the oscillator bars below the zero level. As the trend
    increases, the moving averages will diverge more, and the oscillator bars will stretch more up or down (with
    bullish and bearish trends, respectively).

    :param high: np.array, the high prices of a series
    :param low: np.array, the low prices of a series
    :param fast_sma: integer, the window period for the fast moving average in the oscillator
    :param slow_sma: integer, the window period for the slow moving average in the oscillator
    :return:
    """

    med_price = np.zeros(len(high))
    for i in range(0, len(high)):
        med_price[i] = (high[i] + low[i]) / 2
    ma_fast = ma(med_price.astype(np.float32), np.int32(fast_sma))
    ma_slow = ma(med_price.astype(np.float32), np.int32(slow_sma))
    ao_data = ma_fast - ma_slow
    return ao_data.astype(np.float32)


# @njit('float32[:](float32[:], float32[:], float32[:], int32)', cache=True, nogil=True)
@njit
def atr(high, low, close, window):
    """
    The Average True Range (ATR) is an indicator that measures volatility,  A volatility formula based only on the
    high-low range would fail to capture volatility from gap or limit moves. Wilder created Average True Range to
    capture this “missing” volatility. It is important to remember that ATR does not provide an indication of price
    direction, just volatility.

    We calculate the ATR by first getting the True range:
    Method 1: Current High less the current Low
    Method 2: Current High less the previous Close (absolute value)
    Method 3: Current Low less the previous Close (absolute value)

    After we got the true range for the series we apply a simple moving average for the true range and by that way we
    get the Average True range of the series

    :param high: The high price in a candle
    :param low: The low price of the candle
    :param close: The closing price for the period
    :param window: the rolling window for the simple moving average
    :return: np.array with the ATR calculations
    """
    true_range = tr(high.astype(np.float32), low.astype(np.float32), close.astype(np.float32))
    np_atr_values = np.array([true_range[i:i + window].mean() for i in range(len(true_range) - window + 1)])
    average_true_range = np.append(np.zeros(window - 1) + np.nan, np_atr_values)

    return average_true_range.astype(np.float32)


# @njit('(float32[:], float32[:], float32[:], int32, int32, int32)', cache=True, nogil=True)
@njit
def keltner_channel(high, low, close, ema_period, multiplier, atr_period):
    """
    Documentation:
    Keltner Channels are volatility-based envelopes set above and below an exponential moving average. Instead of using
    the standard deviation, Keltner Channels use the Average True Range (ATR) to set channel distance. The channels are
    typically set two Average True Range values above and below the 20-day EMA. The exponential moving average dictates
    direction and the Average True Range sets channel width. Keltner Channels are a trend following indicator used to
    identify reversals with channel breakouts and channel direction. Channels can also be used to identify overbought
    and oversold levels when the trend is flat.

    Calculation: There are three steps to calculating Keltner Channels. First, select the length for the exponential
    moving average. Second, choose the time periods for the Average True Range (ATR). Third, choose the multiplier for
    the Average True Range.

    Middle Line: 20-day exponential moving average
    Upper Channel Line: 20-day EMA + (2 x ATR(10))
    Lower Channel Line: 20-day EMA - (2 x ATR(10))

    Interpretation:
    Indicators based on channels, bands and envelopes are designed to encompass most price action. Therefore, moves
    above or below the channel lines warrant attention because they are relatively rare. Trends often start with strong
    moves in one direction or another. A surge above the upper channel line shows extraordinary strength, while a plunge
    below the lower channel line shows extraordinary weakness. Such strong moves can signal the end of one trend and the
    beginning of another.

    :param high: np.array, the high prices of a series
    :param low: np.array, the low prices of a series
    :param close: np.array, the close prices of a series
    :param ema_period: integer, the window period for the moving average
    :param multiplier: integer, the window period for the moving average
    :param atr_period: integer, the window period for the ATR
    :return: 3 numpy array, dtype=np.float32, this is the calculation value.
    """
    # EMA
    middle_line = ema(close.astype(np.float32), np.int32(ema_period))
    upper_channel = middle_line + (multiplier * atr(high, low, close, atr_period))
    lower_channel = middle_line - (multiplier * atr(high, low, close, atr_period))

    return middle_line.astype(np.float32), upper_channel.astype(np.float32), lower_channel.astype(np.float32)


# @njit('float32[:](float32[:], float32[:], float32[:], float32[:])', cache=True, nogil=True)
@njit
def adl(high, low, close, volume):
    """
    Documentation:
    The Accumulation Distribution Line is a volume-based indicator designed to measure the cumulative flow of money
    into and out of a security. As with cumulative indicators, the Accumulation Distribution Line is a running total of
    each period's Money Flow Volume. First, a multiplier is calculated based on the relationship of the close to the
    high-low range. Second, the Money Flow Multiplier is multiplied by the period's volume to come up with a Money Flow
    Volume. A running total of the Money Flow Volume forms the Accumulation Distribution Line.

    Calculation: There are three steps to calculating the Accumulation Distribution Line (ADL). First, calculate the
    Money Flow Multiplier. Second, multiply this value by volume to find the Money Flow Volume. Third, create a running
    total of Money Flow Volume to form the Accumulation Distribution Line (ADL).

    1. Money Flow Multiplier = [(Close  -  Low) - (High - Close)] /(High - Low)

    2. Money Flow Volume = Money Flow Multiplier x Volume for the Period

    3. ADL = Previous ADL + Current Period's Money Flow Volume

    Interpretation:
    The Money Flow Multiplier fluctuates between +1 and -1. As such, it holds the key to the Money Flow Volume and the
    Accumulation Distribution Line. The multiplier is positive when the close is in the upper half of the high-low range
    and negative when in the lower half. This makes sense, as buying pressure is stronger than selling pressure when
    prices close in the upper half of the period's range (and vice versa). The Accumulation Distribution Line rises when
    the multiplier is positive and falls when the multiplier is negative.

    :param high: np.array, the high prices of a series
    :param low: np.array, the low prices of a series
    :param close: np.array, the close prices of a series
    :param volume: np.array, the volume of a series
    :return: np.array with the ADL calculations
    """

    money_flow_multiplier = ((close - low) - (high - close)) / ((high - low) + 0.000001)
    money_flow_volume = money_flow_multiplier * volume
    accumulation_distribution_line = np.zeros(len(close))
    accumulation_distribution_line[0] = money_flow_volume[0]
    for i in range(1, len(close) - 1):
        accumulation_distribution_line[i] = accumulation_distribution_line[i - 1] + money_flow_volume[i]

    return accumulation_distribution_line.astype(np.float32)


# @njit('float32[:](float32[:], float32[:], float32[:], float32[:], int32, int32)', cache=True, nogil=True)
@njit
def chaikin_oscillator(high, low, close, volume, short_ema, long_ema):
    """
        Documentation:
        The Chaikin Oscillator measures the momentum of the Accumulation Distribution Line using the MACD formula.
        (This makes it an indicator of an indicator.) The Chaikin Oscillator is the difference between the 3-day and
        10-day EMAs of the Accumulation Distribution Line.

        Calculation:
        There are three steps to calculating the Accumulation Distribution Line (ADL). First, calculate the Money Flow
        Multiplier. Second, multiply this value by volume to find Money Flow Volume. Third, create a running total of
        Money Flow Volume to form the Accumulation Distribution Line (ADL). Fourth, take the difference between two
        moving averages to calculate the Chaikin Oscillator.

        1. Money Flow Multiplier = [(Close  -  Low) - (High - Close)] /(High - Low)

        2. Money Flow Volume = Money Flow Multiplier x Volume for the Period

        3. ADL = Previous ADL + Current Period's Money Flow Volume

        4. Chaikin Oscillator = (3-day EMA of ADL)  -  (10-day EMA of ADL)

        Interpretation:
        Like other momentum indicators, this indicator is designed to anticipate directional changes in the Accumulation
        Distribution Line by measuring the momentum behind the movements.

        :param high: np.array, the High prices from the series
        :param low: np.array, the Low prices from the series
        :param close: np.array, the Close prices from the series
        :param volume: np.array, the Low prices from the series
        :param short_ema: integer, the window period for the short moving average in the oscillator
        :param long_ema: integer, the window period for the long moving average in the oscillator
        :return: np.array, the chaikin oscillator calculation
        """
    adl_value = adl(high, low, close, volume)
    short_ema_value = ema(adl_value.astype(np.float32), np.int32(short_ema))
    long_ema_value = ema(adl_value.astype(np.float32), np.int32(long_ema))
    chaikin = short_ema_value - long_ema_value

    return chaikin.astype(np.float32)


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def max_close(column: np.array, periods: int):
    """
    max_close calculates if today's close is greater than N candles back
    :param column: np.array with dimension 1, should be np.float32 numbers.
    :param periods:int value, its the length of the moving window, minimum value for the window may be 2.
    :return: np.array vector with the values for the np_close_max.
    """
    ma_value_close = np.array(
        [True if max(column[i:i + periods]) == column[i + periods - 1] else False for i in
         range(len(column) - periods + 1)])
    np_close_max = np.append(np.zeros(periods - 1) + np.nan, ma_value_close)
    return np_close_max.astype(np.float32)


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def min_close(column: np.array, periods: int):
    """
    min_close calculates if today's close is less than N candles back
    :param column: np.array with dimension 1, should be np.float32 numbers.
    :param periods:int value, its the length of the moving window, minimum value for the window may be 2.
    :return: np.array vector with the values for the np_close_min.
    """
    ma_value_close = np.array(
        [True if min(column[i:i + periods]) == column[i + periods - 1] else False for i in
         range(len(column) - periods + 1)])
    np_close_min = np.append(np.zeros(periods - 1) + np.nan, ma_value_close)
    return np_close_min.astype(np.float32)


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def higher_highs(column: np.array, window: int):
    """
    higher_highs calculates number of new highs (higher highs) in a window of N Bars
    :param column: np.array with dimension 1, should be np.float32 numbers.
    :param window:int value, its the length of the moving window, minimum value for the window may be 2.
    :return: np.array vector with the values for the np_max_high.
    """
    total_high_periods = 0
    high_porcentual = []

    for i in range(len(column) - window + 1):

        for n in range(len(column[i:i + window])):

            if max(column[i:i + window][:n + 1]) == column[i:i + window][n]:
                total_high_periods += 1

        high_porcentual.append((total_high_periods * 100) / window)
        total_high_periods = 0

    high_porcentual = np.array(high_porcentual)
    np_max_high = np.append(np.zeros(window - 1) + np.nan, high_porcentual)

    return np_max_high.astype(np.float32)


# @njit('float32[:](float32[:], float32[:], int32)', cache=True, nogil=True)
@njit
def positive_candle(column: np.array, column1: np.array, window: int):
    """
    The Positive Candles% indicator is constructed, which is not more than the number of
    positive closes (close greater than open) in a stream of N Bars. Example 7 Positive closes
    in a 10 bar window equals a Positive Candle% of 70%
    :param column: np.array with dimension 1, should be np.float32 numbers(close).
    :param column1: np.array with dimension 1, should be np.float32 numbers(open).
    :param window:int value, its the length of the moving window, minimum value for the window may be 2.
    :return: np.array vector with the values for the np_max_high.
    """
    total_candle_periods = 0
    candle_porcentual = []

    for i in range(len(column) - window + 1):

        for n in range(len(column[i:i + window])):

            if column[i + n] > column1[i + n]:
                total_candle_periods += 1

        candle_porcentual.append((total_candle_periods * 100) / window)
        total_candle_periods = 0

    candle_porcentual = np.array(candle_porcentual)
    np_max_candle = np.append(np.zeros(window - 1) + np.nan, candle_porcentual)

    return np_max_candle.astype(np.float32)


# @njit('(float32[:], int32)', cache=True, nogil=True)
@njit
def linear_regression(arr, window):
    """
            Documentation:
            Linear regression is used to fit linear models for a window of limited periods,
            by the least squares method.

            Method
            For each (x,y) point calculate x2 and xy

            Sum all x, y, x2 and xy, which gives us Σx, Σy, Σx2 and Σxy

            Calculate Slope m:

            m =  N Σ(xy) − Σx ΣyN Σ(x2) − (Σx)2

            (N is the number of points.)

            Calculate Intercept b:

            b =  Σy − m ΣxN


            :param arr: numpy array, dtype=np.float32, this is the close time series you want to use
            :param window: integer value, the bigger the value the slower the calculation will but also less impacted by
            outlayers, picking a good value according to the temporality is a most.

            :return: numpy array, dtype=np.float32, the slope and r squared to mesure the goodness of fit.
            """

    n = len(arr)

    np_index = np.arange(0, n, 1)

    np_slope = np.zeros(n) + np.nan
    np_r_square = np.zeros(n) + np.nan

    for i in range(window, n):
        x = np_index[i - window:i]
        y = arr[i - window:i]
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        if np.sum((x - mean_x) ** 2) == 0 or \
                np.sum((y - mean_y) ** 2) == 0:
            np_slope[i] = np.nan
            np_r_square[i] = np.nan
        else:
            slope = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x) ** 2)
            intercept = mean_y - slope * mean_x

            y_predict = intercept + slope * x

            r_square = np.sum((y_predict - mean_y) ** 2) / np.sum((y - mean_y) ** 2)

            np_slope[i] = slope
            np_r_square[i] = int(r_square * 100)

    return np_slope, np_r_square


# @njit('float32[:](float32[:], float32[:], float32[:], int32)', cache=True, nogil=True)
@njit
def strength_oscillator(high, low, close, period_ma):
    """Documentation:
    Oscillator that shows the component of strength of the trend.

    Interpretation:
    As the trend increases, the average change in closing prices becomes larger relative to the high-low range for de
    day. In a sideways market, both the change in the closes and daily range will get smaller, but the net price change
    over period n should be close to zero.

    Calculation:
    Strength Oscillator = Average(Close(t) - Close(t -1), n) / Average(High(t) - Low(t), n)

    :param high: numpy array, dtype=np.float32, this is the high time series you want to use
    :param low: numpy array, dtype=np.float32, this is the low time series you want to use
    :param close: numpy array, dtype=np.float32, this is the close time series you want to use
    :param period_ma: integer value, the bigger the value the slower the calculation will but also less impacted by
    outlayers, picking a good value according to the temporality is a most.

    :return: numpy array, dtype=np.float32, this is the value of the ratio for every moment of the calculation."""
    close_shifted = np.hstack((np.zeros(1) * np.nan, close[:-1]))
    dif_close = close - close_shifted
    numerator = ma(dif_close.astype(np.float32), np.int32(period_ma))
    dif_range = high - low

    denominator = ma(dif_range.astype(np.float32), window=np.int32(period_ma))
    so = numerator / denominator

    return so.astype(np.float32)


# @njit('float32[:](float32[:], float32[:], float32[:], int32)', cache=True, nogil=True)
@njit
def percent_r(high, low, close, period):
    """
    Documentation:
    Williams %R is a momentum indicator that reflects the level of the close relative to the highest high for
    the look-back period. The Williams %R may be used to find entry and exit points in the market.

    Interpretation:
    %R is a type of momentum indicator that moves between 0 and -100 and measures overbought and oversold levels.
    Low readings (below -80) indicate that price is near its low for the given time period. High readings
    (above -20) indicate that price is near its high for the given time period.

    Example:
    Assume that the highest high equals 110, the lowest low equals 100 and the close equals 108. The high-low range
    is 10 (110 - 100), which is the denominator in the %R formula. The highest high less the close equals 2
    (110 - 108), which in turn is divided by 10, resulting in 0.20. Multiply this number by -100 to get -20 for %R.
    If the close was 103, Williams %R would be -70 (((110-103)/10) x -100).

    :param high: numpy array, dtype=np.float32, this is the high time series you want to use
    :param low: numpy array, dtype=np.float32, this is the low time series you want to use
    :param close: numpy array, dtype=np.float32, this is the close time series you want to use
    :param period: integer value, the bigger the value the slower the calculation will but also less impacted by
    outlayers, picking a good value according to the temporality is a most.

    :return: numpy array, dtype=np.float32, this is the value of the ratio for every moment of the calculation.
    """
    highest_high = rolling_max(high, period)
    lowest_low = rolling_min(low, period)
    buying_power = highest_high - close
    range_window = highest_high - lowest_low

    percent_r_value = (buying_power / range_window) * -100

    return percent_r_value.astype(np.float32)


# @njit('(float32[:], int32)', cache=True, nogil=True)
@njit
def regression_spread(arr, window):
    """
    Documentation:
    Indicator mesure dispersions between a market value and its theoretical value expressed in a linear regression
    for a defined time window. The Z - Score Regression is constructed to measure the difference between both values
    between both values (real-theoretical).

    :param arr: numpy array, dtype=np.float32
    :param window: integer value.
    :return: two numpy array, dtype=np.float32, first the regreassion spread standarized, and seccond the r square of
    the regressions in the window  time period.
    """

    n = len(arr)

    np_index = np.arange(0, n, 1)

    np_regression_spread = np.zeros(n) + np.nan
    np_r_square = np.zeros(n) + np.nan

    for i in range(window, n):
        x = np_index[i - window:i]
        y = arr[i - window:i]
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        numerator1 = np.where(np.sum((x - mean_x) ** 2) == 0, np.nan, np.sum((x - mean_x) ** 2))
        numerator2 = np.where(np.sum((y - mean_y) ** 2) == 0, np.nan, np.sum((y - mean_y) ** 2))

        slope = np.sum((x - mean_x) * (y - mean_y)) / numerator1
        intercept = mean_y - slope * mean_x

        y_predict = intercept + slope * x
        predict = np.where((intercept + slope * i) == 0, np.nan, intercept + slope * i)

        r_square = np.sum((y_predict - mean_y) ** 2) / numerator2

        spread = (arr[i] - predict) / predict * 100
        np_regression_spread[i] = spread
        np_r_square[i] = int(r_square * 100)

    np_regression_spread = standarize(np_regression_spread.astype(np.float32), np.int32(window))
    return np_regression_spread, np_r_square


# @njit('float32[:](float32[:], float32[:], int32)', cache=True, nogil=True)
@njit
def n_consecutive_sessions(arr1: np.array, arr2: np.array, window: int):
    """
    The n_consecutive_sessions indicators calculates if el arr1 is above the arr2 in N bars consecutive'
    :param arr1: np.array with dimension 1, should be np.float32 numbers.
    :param arra2: np.array with dimension 1, should be np.float32 numbers.
    :param window:int value, its the length of the moving window, minimum value for the window may be 2.
    :return: np.array vector with the values for the np_max_high.
    """

    sessions_periods = []
    number_of_sessions = 0

    for i in range(window - 1, len(arr1)):

        for j in range(window):

            if arr1[i - j] > arr2[i - j]:
                number_of_sessions += 1

        if number_of_sessions == window:

            sessions_periods.append(True)

        else:
            sessions_periods.append(False)
        number_of_sessions = 0

    np_sessions_periods = np.append(np.zeros(window - 1) + np.nan, sessions_periods)
    return np_sessions_periods.astype(np.float32)


# @njit('(float32[:], int32, int32)', cache=True, nogil=True)
@njit
def divergence_index(close, slow_ma, fast_ma):
    """
    Documentation:
    The divergence index takes the volatility-adjusted difference between two moving averages, 10 and 40 bars
    by default and uses a Standard Deviation in the change in values to introduce a volatility measure.

    Interpretation:
    The trading rules require a band around zero to trigger entries. Buy when the DI moves below the lower band
    while in an uptrend. Sell when the DI moves above the upper band while in a downtrend. Exit longs and shorts
    when the DI crosses zero.

    Example:
    DI(t) = [fast average(t) - slow average(t)] / std(difference [price(t) - price(t-1)], slow period)^2
    Band(t) = std((DI(t), slow period)
    Upper Band(t) = factor x Band(t)
    Lower Band(t) = -factor x Band(t)
    where factor = 1.0

    The standard deviation allows the bands to widen or narrow according to the pattern of fluctuations.

    :param close: numpy array, dtype=np.float32, this is the close time series you want to use.
    :param slow_ma: integer value, the bigger the value the slower the calculation will but also less impacted by
    outlayers, picking a good value according to the temporality is a most.
    :param fast_ma: integer value, the bigger the value the slower the calculation will but also less impacted by
    outlayers, picking a good value according to the temporality is a most.

    :return: 3 numpy array, dtype=np.float32, this is the value of the ratio for every moment of the calculation.
    """
    faster_ma = ma(close.astype(np.float32), np.int32(fast_ma))
    slower_ma = ma(close.astype(np.float32), np.int32(slow_ma))
    close_shifted = np.hstack((np.zeros(1) * np.nan, close[:-1]))
    difference = close - close_shifted
    numerator = faster_ma - slower_ma
    denominator = std(difference.astype(np.float32), np.int32(slow_ma)) ** 2
    di = numerator / denominator

    band = std(di.astype(np.float32), np.int32(slow_ma))
    factor = 1
    upper_band = factor * band
    lower_band = -factor * band

    return di.astype(np.float32), upper_band.astype(np.float32), lower_band.astype(np.float32)


# @njit('float32[:](float32[:])', cache=True, nogil=True)
@njit
def only_streak(close):
    """
    Documentation:
    The strategy seeks measure the streaks in a price series without the standardization of values.

    :param close: numpy array, dtype=np.float32, this is the close time series you want to use
    :param periods: integer value, the bigger the value the slower the calculation will but also less impacted by
    outlayers, picking a good value according to the temporality is a most.
    :return: numpy array, dtype=np.float32, this is the value of the ratio for every moment of the calculation.
    """
    curr_streak = np.zeros(len(close))
    streaks = int(0)
    for i in range(len(close)):
        if i == 0:
            continue
        if close[i] > close[i - 1]:
            streaks = np.maximum(0, streaks)
            streaks += 1
            curr_streak[i] = streaks
        elif close[i] < close[i - 1]:
            streaks = np.minimum(0, streaks)
            streaks += -1
            curr_streak[i] = streaks
        else:
            streaks = 0
            curr_streak[i] = streaks

    return curr_streak.astype(np.float32)


# @njit('float32[:](float32[:], int32, int32, int32, int32)', cache=True, nogil=True)
@njit
def connors_rsi(close, period_rsi, period_updown, period_look_back, period_roc=1):
    """
    Documentation:
    ConnorsRSI is a momentum oscillator developed by Larry Connors and the team at Connors Research. It's used for
    identifying overbought/oversold conditions in shorter trading timeframes. The traditional 14-period RSI indicator
    developed by Welles Wilder reacts too slowly to be useful for short-term trading; Connors Research sought to
    improve on this indicator, making it more suitable for shorter timeframes.
    ConnorsRSI combines the momentum measurement of RSI with components that measure the duration of the trend and the
    magnitude of the price change, to create a more reliable short-term RSI indicator.

    Calculation:
    ConnorsRSI is calculated by taking the average of its three components.

    ConnorsRSI(3,2,100) = (RSI(3) + RSI(Streak,2) + PercentRank(100)) / 3

    Relative Strength Index
    The first component is a simple 3-period RSI of price. This component measures price momentum on a scale of 0-100.

    Up/Down Streak Length
    The second component is a 2-period RSI of the up/down streak length. It measures the duration of the trend.
    The up/down streak is essentially the number of days in a row that the security's closing price has been higher
    (up) orlower (down) than the previous day's close. If a stock closes above its previous close three days in a row,
    then the up/down streak is +3. If it has closed below its previous close for 2 days, then its streak is -2. If
    it does not change price between one period and the next, then the streak is reset to 0.
    Applying the 2-period RSI to this streak value converts it to a bound oscillator where values must be in the range
    of 0-100.

    Magnitude of Price Change
    The third component ranks the most recent period's price change against the price change of the other periods in the
    specified timeframe (100 periods by default).

    Essentially you determine the percentage of previous price changes that are lower than the most recent one. For
    example, if you specify a 20-day timeframe, and 7 of those 20 price change values are lower than today's price
    change, then 7 / 20 = 0.35 = 35%.

    Again, defining this as a percentage restricts the component to a scale of 0-100. If today's price change was large
    and positive, the value of this component will be closer to 100; large negative price changes will result in a value
    closer to 0.
    :param close: numpy array, dtype=np.float32, the close time series
    :param period_rsi: integer, the period for the RSI component
    :param period_updown: integer, the streak period
    :param period_look_back: integer, the window for the Rate of change to compare the magnitude
    :param period_roc: integer, the period for the Rate of change calculations
    :return: connors: np.array, dtype=np.float32, the Connors_RSI final result
    """
    lon = len(close)
    # Normal RSI
    rsi_calc = rsi(close.astype(np.float32), np.int32(period_rsi))

    # Streak
    stck = only_streak(close.astype(np.float32))
    stck_calc = np.zeros(lon)
    stck_calc = rsi(stck.astype(np.float32), np.int32(period_updown))

    # Rate of change
    roc_calc = roc(close.astype(np.float32), np.int32(period_roc))
    # se divide entre 0.01 para que el resultado de connor_x este en una escala porcentual
    connor = [(roc_calc[i:i + period_look_back] < roc_calc[i + period_look_back - 1]).sum() / (0.01 * period_look_back)
              for i in range(len(close) - period_look_back + 1)]
    connor_x = np.append(np.zeros(period_look_back - 1) * np.nan, connor)
    connors = (rsi_calc + stck_calc + connor_x) / 3
    return connors.astype(np.float32)


# @njit('float32[:](float32[:], int32, int32)', cache=True, nogil=True)
@njit
def change_ratio(arr, roc_window, look_back):
    """
    Documentation:
    The Change ratio is the quotient between the Rate of change(ROC) of the current candle and the ROC from a previous
    candle, this indicator is built to show momentum in a price series.

    Calculation:
    First, we get the ROC in the whole series by using the ROC indicator with his respective window
    Then, we calculate the quotient between two ROC, one from the current candle an the other from a N-previous candle

    CHANGE RATIO = ROC[t] / ROC[t - n]

    Interpretation:
    If the ratio shows a value superior to the unit (1) this means that the current ROC has a higher magnitude than the
    N-previous ROC in the series, so this means that the series in the moment [t] has higher momentum than the series in
    the moment [t - n], in the other direction if the change ratio value is below 1 this means the opposite, that the
    current ROC has a minor magnitude compared to the N-previous ROC.

    :param arr: np.array, dtype=np.float32, the time series input
    :param roc_window: integer, the window for the ROC calculations
    :param look_back: integer, the periods to compare the current ROC and the N-previous ROC
    :return: cr_val: np.array, the calculation of the change ratio
    """
    # rate of change
    rate = roc(arr.astype(np.float32), np.int32(roc_window))
    rate_val = np.where(rate == 0, 1, rate)

    # The ratio
    cr_val = np.zeros(len(arr))
    for i in range(len(arr)):
        cr_val[i] = round(np.fabs(rate_val[i]) / np.fabs(rate_val[i - look_back]), 0)
    return cr_val.astype(np.float32)


# @njit('float32[:](float32[:], float32[:], int32, int32)', cache=True, nogil=True)
@njit
def ac(high, low, fast, slow):
    """
    Documentation:
    The Accelerator Oscillator is an indicator that measures the acceleration or deceleration of the
    current market driving force. The principle of operation of the Accelerator Oscillator is based on the assumption
    of its creator Bill Williams that before the change of the direction of the price movement, the momentum of its
    change should fall.

    Calculation:
    The Accelerator Oscillator indicator is calculated as a difference between the Awesome Oscillator (AO)
    and the 5-period moving average of the AO.

    AO = SMA(median price, 5)-SMA(median price, 34)
    AC = AO-SMA(AO, 5)

    Interpretation:
    The Accelerator Oscillator is an indicator which fluctuates around a median 0.00 (zero) level which corresponds to
    a relative balance of the market driving force with the acceleration. Positive values signal a growing bullish
    trend, while negative values may be qualified as a bearish trend development. The AC indicator changes its direction
    before any actual trend reversals take place in the market therefore it serves as an early warning sign of probable
    trend direction changes.

    :param high: np.array, dtype=np.float32, the high time series
    :param low: np.array, dtype=np.float32, the low time series
    :param fast: integer, the fast period for the AO calculation and the MA of the AO
    :param slow: integer, the slow period for the AO calculation
    :return: ac_calue, np.array, the AC oscillator calculation
    """
    ao_value = ao(high.astype(np.float32), low.astype(np.float32), np.int32(fast), np.int32(slow))
    ma_ao = ma(ao_value.astype(np.float32), np.int32(fast))
    ac_value = ao_value - ma_ao
    return ac_value.astype(np.float32)


# @njit('(float32[:], float32[:], float32[:], int32, int32)', cache=True, nogil=True)
@njit
def fast_stoch(high, low, close, k_period, d_period):
    """
    Documentation:
    The Stochastic Oscillator is a momentum indicator that shows the location of the close relative to the high-low
    range over a set number of periods.
    Interpretation:
    The Stochastic Oscillator is above 50 when the close is in the upper half of the range and below 50 when
    the close is in the lower half. Low readings (below 20) indicate that price is near its low for the given
    time period. High readings (above 80) indicate that price is near its high for the given time period.
    :param high: numpy array, dtype=np.float32, this is the high time series you want to use
    :param low: numpy array, dtype=np.float32, this is the low time series you want to use
    :param close: numpy array, dtype=np.float32, this is the close time series you want to use
    :param k_period: integer value, the look back period for highest high and lowest low.
    :param d_period: integer value, the smoothing period for the ma.
    :return: numpy array, dtype=np.float32, those are the values of the indicators at every candle
    """
    h_high = rolling_max(high.astype(np.float32), np.int32(k_period))
    l_low = rolling_min(low.astype(np.float32), np.int32(k_period))
    k_line = np.zeros(len(high))
    cml = close - l_low
    hml = h_high - l_low
    hml = np.where(hml == 0, 1, hml)
    for i in range(0, len(high)):
        k_line[i] = (cml[i] / hml[i]) * 100
    d_line = ma(k_line.astype(np.float32), np.int32(d_period))
    return k_line.astype(np.float32), d_line.astype(np.float32)


# @njit('(float32[:], int32, int32, int32)', cache=True, nogil=True)
@njit
def macd(arr, fast, slow, sig):
    """
    the Moving Average Convergence/Divergence oscillator (MACD) is one of the simplest and most effective momentum
    indicators available. The MACD turns two trend-following indicators, moving averages, into a momentum oscillator
    by subtracting the longer moving average from the shorter one. As a result, the MACD offers the best of both
    worlds: trend following and momentum.

    The MACD calculation follows this structure:
    - MACD Line: (12-day EMA - 26-day EMA)

    - Signal Line: 9-day EMA of MACD Line

    - MACD Histogram: MACD Line - Signal Line

    :param arr: The price series, generally we take the close price
    :param fast: The fast EMA period
    :param slow: The slow EMA period
    :param sig:  The signal line period
    :return: two np.arrays showing the macd and signal line calculations
    """
    macd_res = ema(arr.astype(np.float32), np.int32(fast)) - ema(arr.astype(np.float32), np.int32(slow))
    sig_line = ema((macd_res[slow - 1:]).astype(np.float32), np.int32(sig))
    sig_line = np.concatenate((np.array([np.nan] * (slow - 1)), sig_line))
    macd_hist = macd_res - sig_line 
    return macd_res.astype(np.float32), sig_line.astype(np.float32), macd_hist.astype(np.float32)


# @njit('float32(float32, float32[:])', cache=True, nogil=True)
@njit
def polevl(x, coef):
    """
        Documentation:
         Polinomial evaluation for the function qnorm.
    """

    accum = 0
    for c in coef:
        accum = x * accum + c
    return accum


# @njit('float32(float32, float32, float32)', cache=True, nogil=True)
@njit
def qnorm(p, mean, std):
    """
        Documentation:
         Returns the argument, x, for which the area under the Gaussian probability density function (integrated from
         minus infinity to x) is equal to y.

         For small arguments 0 < y < exp(-2), the program computes z = sqrt( -2.0 * log(y) );  then the approximation is
         x = z - log(z)/z  - (1/z) P(1/z) / Q(1/z). There are two rational functions P/Q, one for 0 < y < exp(-32)
         and the other for y up to exp(-2).  For larger arguments, w = y - 0.5, and  x/sqrt(2pi) = w + w**3 R(w**2)/S(w**2)).

          ACCURACY:

                               Relative error:
          arithmetic   domain        # trials      peak         rms
             IEEE     0.125, 1        20000       7.2e-16     1.3e-16
             IEEE     3e-308, 0.135   50000       4.6e-16     9.8e-17


          ERROR MESSAGES:

          condition    value returned
           p <= 0     "qnorm p receive values 0 < x < 1"
           p >= 1     "qnorm p receive values 0 < x <  1"
           std <= 0   "std values most be positive"

        :param p: percentile to find, most be between 0 and 1.
        :param mean: Mean  parameter for the normal distribution.
        :param std: Standard deviation parameter for the normal distribution.

        :return: float, dtype=np.float32, this is the x value for .
        """

    s2pi = 2.50662827463100050242E0

    P0 = [
        -5.99633501014107895267E1,
        9.80010754185999661536E1,
        -5.66762857469070293439E1,
        1.39312609387279679503E1,
        -1.23916583867381258016E0,
    ]

    Q0 = [
        1,
        1.95448858338141759834E0,
        4.67627912898881538453E0,
        8.63602421390890590575E1,
        -2.25462687854119370527E2,
        2.00260212380060660359E2,
        -8.20372256168333339912E1,
        1.59056225126211695515E1,
        -1.18331621121330003142E0,
    ]

    P1 = [
        4.05544892305962419923E0,
        3.15251094599893866154E1,
        5.71628192246421288162E1,
        4.40805073893200834700E1,
        1.46849561928858024014E1,
        2.18663306850790267539E0,
        -1.40256079171354495875E-1,
        -3.50424626827848203418E-2,
        -8.57456785154685413611E-4,
    ]

    Q1 = [
        1,
        1.57799883256466749731E1,
        4.53907635128879210584E1,
        4.13172038254672030440E1,
        1.50425385692907503408E1,
        2.50464946208309415979E0,
        -1.42182922854787788574E-1,
        -3.80806407691578277194E-2,
        -9.33259480895457427372E-4,
    ]

    P2 = [
        3.23774891776946035970E0,
        6.91522889068984211695E0,
        3.93881025292474443415E0,
        1.33303460815807542389E0,
        2.01485389549179081538E-1,
        1.23716634817820021358E-2,
        3.01581553508235416007E-4,
        2.65806974686737550832E-6,
        6.23974539184983293730E-9,
    ]

    Q2 = [
        1,
        6.02427039364742014255E0,
        3.67983563856160859403E0,
        1.37702099489081330271E0,
        2.16236993594496635890E-1,
        1.34204006088543189037E-2,
        3.28014464682127739104E-4,
        2.89247864745380683936E-6,
        6.79019408009981274425E-9,
    ]

    P0 = np.array(P0)
    P1 = np.array(P1)
    P2 = np.array(P2)

    Q0 = np.array(Q0)
    Q1 = np.array(Q1)
    Q2 = np.array(Q2)

    if std <= 0:
        raise ValueError("std values most be positive")
    if p <= 0 or p >= 1:
        raise ValueError("qnorm p receive values 0 < p < 1")
    negate = True
    y = p
    if y > 1.0 - 0.13533528323661269189:
        y = 1.0 - y
        negate = False

    if y > 0.13533528323661269189:
        y = y - 0.5
        y2 = y * y
        x = y + y * (y2 * polevl(float(y2), P0.astype(np.float32)) / polevl(float(y2), Q0.astype(np.float32)))
        x = x * s2pi
        x = (x * std) + mean
        return x

    x = np.sqrt(-2.0 * np.log(y))
    x0 = x - np.log(x) / x

    z = 1.0 / x
    if x < 8.0:
        x1 = z * polevl(float(z), P1.astype(np.float32)) / polevl(float(z), Q1.astype(np.float32))
    else:
        x1 = z * polevl(float(z), P2.astype(np.float32)) / polevl(float(z), Q2.astype(np.float32))
    x = x0 - x1
    if negate:
        x = -x

    x = (x * std) + mean
    return x


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def lag(arr, lag):
    """
        Documentation:
         Compute a lagged version of a time series, shifting the time base back by a given number of observations.

        :param arr: An array time series
        :param lag: The number of lags (in units of observations).

        :return: lag array  of the time series.
        """
    lagged = np.roll(arr, lag)
    for i in range(lag):
        lagged[i] = np.NaN
    return lagged


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def bandwidth(np_bandwidth: np.array, periods: int):
    """
    The Z-Score Bandwidth indicator is built taking the Bollinger Bandwidth indicator (band sizes)
     and standardized in N Periods
    :param arr: np.array with dimension 1, should be np.float32 numbers.
    :param window: int value, its the length of the moving window, minimum value for the window may be 2.
    :return: return: numpy array, dtype=np.float32, this is the value for every moment of the calculation .
    """
    z_bandwidth = standarize(np_bandwidth, periods)

    return z_bandwidth.astype(np.float32)


# @njit('float32[:](float32[:], float32[:], int32)', cache=True, nogil=True)
@njit
def va(high, low, window):
    """
    Documentation:
    Moving average of the maximum 2-day high-low combined range over a window (highest high minus lowest low).

    Interpretation:
    The volatility average measures market volatility from a combined range of an asset price for that period.
    It can help to measure the market volatility.

    Example:
    Assume that the high equals 110, the low equals 100 and the high to the previous session was 108 and the low was
    105. Then, in a window of 10 days, the volatility average will be:
    the sum of [∑ i= t -9 to t (max(H(t), H(t-1)) - min(L(t), L(t-1)))] / 10.

    :param high: numpy array, dtype=np.float32, this is the high time series you want to use
    :param low: numpy array, dtype=np.float32, this is the low time series you want to use
    :param window: integer value, the bigger the value the slower the calculation will but also less impacted by
    outlayers, picking a good value according to the temporality is a most.

    :return: numpy array, dtype=np.float32, this is the value of the moving average for every moment of the
    calculation.
    """
    high_shifted = np.hstack((np.zeros(1) * np.nan, high[:-1]))
    low_shifted = np.hstack((np.zeros(1) * np.nan, low[:-1]))
    max_high = np.zeros(len(high))
    min_low = np.zeros((len(low)))

    for i in range(len(high)):
        if i == 0:
            continue
        if high[i] > high_shifted[i]:
            max_high[i] = high[i]
        else:
            max_high[i] = high_shifted[i]

        if low[i] < low_shifted[i]:
            min_low[i] = low[i]
        else:
            min_low[i] = low_shifted[i]

    vol_dif = max_high - min_low
    vol_ave = ma(vol_dif.astype(np.float32), np.int32(window))

    return vol_ave.astype(np.float32)


# @njit('float32[:](float32[:], float32[:], float32[:], float32[:], float32[:], int32)', cache=True, nogil=True)
@njit
def demand_index(open, high, low, close, volume, window):
    """
    Documentation:
    The Demand Index developed by James Sibbett and its counterpart, are quite useful in identifying accumulation or
     distribution. The Demand Index is a combination of price and volume, calculated as buying pressure (BP) and
     selling pressure (SP).

    Interpretation:
    1) Identify bullish and bearish divergences, i.e., determine whether the DI is moving with the prices or
    opposite to the prices.
    2) Extensively use trendlines and support/resistance levels on the DI, to determine important turning points.
    3) Separate the DI into BP and SP then determine whether the BP is above the SP (positive) or below it
    (negative).

    Example:
    If the prices rise:
        BP = V or Volume
        SP = V/P where P is the percent change in price
    If the price declines:
        BP = V/P where P is the percent change in price
        SP = V
    Because P is a decimal (i.e., less than 1), P is modified to make it greater than one by multiplying it by
    the constant K.
    For the No Limit version K=(3×C)/Va
    where C is the closing price and Va is the Volatility average which is the ten-day average of a two-day
    price range (highest high minus lowest low).
    DI = BP/SP

    :param open: numpy array, dtype=np.float32, this is the open time series you want to use
    :param high: numpy array, dtype=np.float32, this is the high time series you want to use
    :param low: numpy array, dtype=np.float32, this is the low time series you want to use
    :param close: numpy array, dtype=np.float32, this is the close time series you want to use
    :param volume: numpy array, dtype=np.float32, this is the volume time series you want to use
    :param window: integer value, the bigger the value the slower the calculation will but also less impacted by
    outlayers, picking a good value according to the temporality is a most.

    :return: numpy array, dtype=np.float32, whit the value for the demand index.
    """

    vol_ave = va(high, low, window)
    vol_ave = np.where(vol_ave == 0, 1, vol_ave)
    k = (3 * close) / vol_ave
    price_difference = (close - open) / open
    pressure = k * price_difference
    pressure = np.where(pressure == 0, 1, pressure)

    buying_pressure = np.zeros(len(high))
    selling_pressure = np.zeros(len(high))

    for i in range(len(close)):
        if close[i] > open[i]:
            buying_pressure[i] = volume[i]
            selling_pressure[i] = volume[i] / pressure[i]
        else:
            buying_pressure[i] = volume[i] / pressure[i]
            selling_pressure[i] = volume[i]

    di = np.where(np.fabs(buying_pressure) > np.fabs(selling_pressure), selling_pressure / buying_pressure,
                  buying_pressure / selling_pressure)

    ma_di = ma(di.astype(np.float32), np.int32(window))

    return ma_di.astype(np.float32)


# @njit('float32[:](float32[:], float32[:], int32)', cache=True, nogil=True)
@njit
def disparity(ma_spread_x: np.array, ma_spread_y: np.array, window: int):
    """
        Documentation:
        The strategy measure the difference between the spreads of two assets and their moving averages.

        :param ma_spread_x: numpy array, dtype=np.float32, this is the spread between the first asset and his moving
        average.
        :param ma_spread_y: numpy array, dtype=np.float32, this is the spread between the second asset and his moving
        average.
        :param window: integer value, the bigger the value the slower the calculation will but also less impacted by
        outlayers, picking a good value according to the temporality is a most.

        :return: numpy array, dtype=np.float32, this is the value for every moment of the calculation.
        """

    spread_indicator = ma_spread_x - ma_spread_y
    z_score = intermarket_standarize(spread_indicator.astype(np.float32), np.int32(window))

    return z_score.astype(np.float32)


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def bull_momentum(arr, window):
    """
    Documentation:
    The bull momentum indicator shows if the time series in a calculation window has a bullish price pattern.

    Calculation:
    For the entire series:
    If the PRICE[i] is > PRICE[i - 1], then momentum equals 1

    Then in a predefined time window we apply a sum to the momentum to see how much times this pattern is present in
    the series.

    Interpretation:
    The indicator returns a number, that number shows how much that bullish price pattern repeats in a particular
    window of time.

    :param arr: np.array, np.float32, the time series for input
    :param window: int32, the calculation window
    :return: np.array, np.float32, bull_calc, the results for the indicator calculation
    """
    momentum = np.zeros(len(arr))
    for i in range(0, len(arr)):
        if arr[i] > arr[i - 1]:
            momentum[i] = 1
        else:
            momentum[i] = 0

    bull_calc = cumsum(momentum.astype(np.float32), window)
    return bull_calc.astype(np.float32)


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def bear_momentum(arr, window):
    """
    Documentation:
    The bear momentum indicator shows if the time series in a calculation window has a bearish price pattern.

    Calculation:
    For the entire series:
    If the PRICE[i] is < PRICE[i - 1], then momentum equals 1

    Then in a predefined time window we apply a sum to the momentum to see how much times this pattern is present in
    the series.

    Interpretation:
    The indicator returns a number, that number shows how much that bearish price pattern repeats in a particular
    window of time.

    :param arr: np.array, np.float32, the time series for input
    :param window: int32, the calculation window
    :return: np.array, np.float32, bear_calc, the results for the indicator calculation
    """
    momentum = np.zeros(len(arr))
    for i in range(0, len(arr)):
        if arr[i] < arr[i - 1]:
            momentum[i] = 1
        else:
            momentum[i] = 0

    bear_calc = cumsum(momentum.astype(np.float32), window)
    return bear_calc.astype(np.float32)


# @njit('float32[:](float32[:], float32[:])', cache=True, nogil=True)
@njit
def candle_body_comparator(open, close):
    """
        Documentation:
        The candle body comparator is the quotient between the difference between the open and the close of the current
        candle and the difference between the open and the close of the previous candle. The purpose of this indicator
        is to measure the percentage that the body of the current candle represents with respect to the body of the
        previous candle.

        Calculation:
        First we get the absolute differences between the close and the open of the actual candle:
        actual_body = close[t] - open[t]
        Then we get the same difference but between the close and the open of the previous candle:
        previous_body = close[t-1] - open[t-1]
        After that we proceed to calculate the actual ratio between these differences:
        cbc ratio = actual_body[t] / previous_body[t]

        Interpretation:
        If the ratio is higher than 1, then we can say that the body of actual candle is bigger than the body of the
        previous candle.

        :param open: np.array, the High prices from the series
        :param close: np.array, the Low prices from the series
        :return: np.array, the e_ratio calculation
        """

    actual_body = np.fabs(close - open)

    previous_body = np.hstack((np.zeros(1) * np.nan, np.fabs(close[:-1] - open[:-1])))

    previous_body = np.where(previous_body == 0, 1, previous_body)  # if previous_body=0 -> previous_body=1

    cbc_ratio = np.divide(actual_body, previous_body) * 100

    return cbc_ratio.astype(np.float32)


# @njit('float32[:](float32[:], float32[:], int32)', cache=True, nogil=True)
@njit
def demarker(high, low, window):
    """
    Documentation:
    The Demarker Indicator is a tool which was introduced by Tom DeMark to help identify emerging buying and selling
    opportunities. Demarker Indicator demonstrates the price depletion phases which usually correspond with the price
    highs and bottoms. The DeMarker indicator proved to be efficient at identifying trend break-downs as well as
    spotting intra-day entry and exit points.

    Calculation:
    The DeMarker indicator is the sum of all price increment values recorded during the "i" period divided by the price
    minima. The Demarker indicator formula is:

    The DeMax(i) is calculated:
    If high(i) > high(i-1) , then DeMax(i) = high(i)-high(i-1), otherwise DeMax(i) = 0

    The DeMin(i) is calculated:
    If low(i) < low(i-1), then DeMin(i) = low(i-1)-low(i), otherwise DeMin(i) = 0

    The value of the DeMarker is calculated as:
    DMark(i) = SMA(DeMax, N)/(SMA(DeMax, N)+SMA(DeMin, N))

    Where:
    SMA - Simple Moving Average;
    N - the number of periods used in the calculation.

    Interpretation:
    DeMarker indicator fluctuates with a range between 0 to 1.
    DeMarker indicator is indicative of lower volatility and a possible price drop when reading 0.7 and higher.
    DeMarker indicator signals a possible price increase when reading below 0.3.

    :param high: np.array, dtype=np.float32, the high time series
    :param low: np.array, dtype=np.float32, the low time series
    :param window: integer, the period for the calculations
    :return: np.array, the calculation for demarker indicator
    """
    demax = np.zeros(len(high))
    for i in range(0, len(high)):
        if high[i] > high[i - 1]:
            demax[i] = high[i] - high[i - 1]
        else:
            demax[i] = 0

    demin = np.zeros(len(low))
    for i in range(0, len(low)):
        if low[i] < low[i - 1]:
            demin[i] = low[i - 1] - low[i]
        else:
            demin[i] = 0

    ma_demax = ma(demax.astype(np.float32), np.int(window))
    ma_demin = ma(demin.astype(np.float32), np.int(window))

    demark = ma_demax / (ma_demax + ma_demin)
    return demark.astype(np.float32)


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def euc_distance(y_arr, window):
    """
    Documentation:
    In mathematics, the Euclidean distance between two points in Euclidean space is a number, the length of a line
    segment between the two points. It can be calculated from the Cartesian coordinates of the points using the
    Pythagorean theorem, and is occasionally called the Pythagorean distance.

    Calculation:
    D =  √( X2-X1)^2 + (Y2-Y1)^2)

    Interpretation:
    The euclidean distance is a momentum indicator that has to be used with the standarize function to normalize the
    calculations to interpretable values, if the distance is standarized then we can read the results as if the value
    is above N number then we have an spike in momentum.

    :param y_arr: np.array, dtype=np.float32, the time series
    :param window: integer, the window for the distance calculation
    :return: np.array, the distance calculation
    """
    position = np.arange(start=1, stop=len(y_arr) + 1)
    # position = np.zeros(len(y_arr))
    # position[0] = 1
    # for i in range(1, len(y_arr)):
    #     position[i] = position[i - 1] + 1

    euc_calc = np.zeros(len(y_arr))
    euc_calc[:window] = np.NaN
    for i in range(window, len(y_arr)):
        euc_calc[i] = np.sqrt(((position[i] - position[i - window]) ** 2) + ((y_arr[i] - y_arr[i - window]) ** 2))
    return euc_calc.astype(np.float32)


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def std_fast(arr, window):
    np_std_values_close = np.array([arr[i:i + window].std() for i in range(len(arr) - window + 1)])
    std_close = (np.append(np.zeros(window - 1) + np.nan, np_std_values_close)).astype(np.float32)

    return std_close


# @njit('(float32[:], int32, int32)', cache=True, nogil=True)
@njit
def forecast_oxilator(arr, window, forecast_periods):
    """
    Documentation:
    Indicator mesure dispersions between a market value and its forecast value expressed in a linear regression
    for a defined time window. The Z - Score Regression is constructed to measure the difference between both values
    between both values (real-theoretical).

    :param arr: numpy array, dtype=np.float32
    :param window: integer value.
    :param forecast_periods: integer value.
    :return: two numpy array, dtype=np.float32, first the regreassion spread standarized, and seccond the r square of
    the regressions in the window  time period.
    """

    n = len(arr)

    np_index = np.arange(0, n, 1)

    np_forecast_oxilator = np.zeros(n) + np.nan
    np_prediction = np.zeros(n) + np.nan

    for i in range(window + forecast_periods, n):
        x = np_index[i - forecast_periods - window:i - forecast_periods]

        y = arr[i - forecast_periods - window:i - forecast_periods]

        mean_x = np.mean(x)

        mean_y = np.mean(y)

        slope = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x) ** 2)
        intercept = mean_y - slope * mean_x

        predict = intercept + slope * i
        np_prediction[i] = predict

        forecast_oxilator = (arr[i] - predict) / arr[i] * 100
        np_forecast_oxilator[i] = forecast_oxilator

    return np_forecast_oxilator.astype(np.float32), np_prediction.astype(np.float32)


@njit
def range_vec(high, low):
    range_calc = high - low
    return range_calc


# @njit('float32[:](float32[:], float32[:], int32)', cache=True, nogil=True)
@njit
def rolling_correlation(arr_x: np.array, arr_y: np.array, window: int):
    """
    Documentation:
    Measure the strength and direction of correlation between two variables within a moving window.

    :param arr_x: numpy array, dtype=np.float32, this is the close price of the time series x
    :param arr_y: numpy array, dtype=np.float32, this is the close price of the time series y
    :param window: integer value, the bigger the value the slower the calculation will but also less impacted by
    outlayers, picking a good value according to the temporality is a most.
    :return: numpy array, dtype=np.float32, this is the value of the correlation coefficient for every moment of the
    calculation.
    """

    long = len(arr_x)
    correlation = np.zeros(long) + np.nan
    for i in range(window, long + 1):
        local_correlation = np.corrcoef(arr_x[i - window:i], arr_y[i - window:i])[0][1]
        correlation[i - 1] = local_correlation

    return correlation.astype(np.float32)


# @njit('float32[:](float32[:], float32[:], int32)', cache=True, nogil=True)
@njit
def atypical_candle(high, low, periods):
    """
    Documentation:
    Indicator that measure atypical candle size with standardization of values.

    :param high: numpy array, dtype=np.float32, this is the high price of the time series x
    :param low: numpy array, dtype=np.float32, this is the low price of the time series y
    :param periods: integer value, the bigger the value the slower the calculation will but also less impacted by
    outlayers, picking a good value according to the temporality is a most.
    :return: numpy array, dtype=np.float32, this is the standardize values of the difference for every moment of the
    calculation.
    """

    length_candle = high - low
    z_candle = standarize(length_candle.astype(np.float32), np.int32(periods))

    return z_candle.astype(np.float32)


@njit
def nancorr(x_array, y_array):
    """
    calculates correlation between 2 vectors omiting all nans in the datas.
    friendly formula from wikipedia:
    https://wikimedia.org/api/rest_v1/media/math/render/svg/9363d4a765bda05563bf32c9216e3cf250ac387d

    :param x_array: first array
    :param y_array: second array
    :return: correlation omitting all the nan values
    """

    # this mask ensures we only calculate correlation using common values (non-nan values)
    mask = np.where(np.isnan(x_array) | np.isnan(y_array), np.nan, 0)
    x_array += mask
    y_array += mask

    # number of observations after omitting the nan
    n = len(x_array[~np.isnan(x_array)])
    if n == 0:
        return np.nan

    x_mean = np.nanmean(x_array)
    y_mean = np.nanmean(y_array)

    x_std = np.nanstd(x_array)
    x_std = np.where(x_std == 0, np.nan, x_std)
    y_std = np.nanstd(y_array)
    y_std = np.where(y_std == 0, np.nan, y_std)

    standard_score_x = (x_array - x_mean) / x_std
    standard_score_y = (y_array - y_mean) / y_std

    score_product = standard_score_x * standard_score_y

    corr = (1 / n) * np.nansum(score_product)

    return corr


@njit
def rolling_nancorr(x_array, y_array, window):
    """
    calculates the moving correlation between 2 vectors for a given period, omiting all nans in the datas.
    uses the nancorr function.

    :param x_array: first array
    :param y_array: second array
    :param window:  The number of periods you want to calculate the rolling correlation.
    :return: correlation omitting all the nan values
    """

    corr_values = np.array(
        [nancorr(x_array[i: i + window], y_array[i: i + window]) for i in range(len(x_array) - window + 1)])
    moving_corr = np.append(np.zeros(window - 1) + np.nan, corr_values)

    return moving_corr


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def cummax(arr, window):
    """
               Documentation:
               Returns a vector whose elements are the cumulative max, products of the elements of
               the argument.
               :param arr: numpy array, dtype=np.float32, this is the close time series you want to use
               :param window: integer value.
               :return: numpy array, dtype=np.float32, this is the value for every moment of the calculation.
            """

    values = np.array([arr[i:i + window].max() for i in range(len(arr) - window + 1)])
    cummax_result = np.append(np.zeros(window - 1) + np.nan, values)
    return cummax_result.astype(np.float32)


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def cummin(arr, window):
    """
               Documentation:
               Returns a vector whose elements are the cumulative min, products of the elements of
               the argument.
               :param arr: numpy array, dtype=np.float32, this is the close time series you want to use
               :param window: integer value.
               :return: numpy array, dtype=np.float32, this is the value for every moment of the calculation.
            """

    values = np.array([arr[i:i + window].min() for i in range(len(arr) - window + 1)])
    cummin_result = np.append(np.zeros(window - 1) + np.nan, values)
    return cummin_result.astype(np.float32)


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def slope(x1, y1, x2, y2):
    """
               Documentation:
                The slope or gradient of a line is a number that describes both the direction and the steepness of the
                line. Slope is often denoted by the letter m.\
                Slope is calculated by finding the ratio of the "vertical change" to the "horizontal change" between
                (any) two distinct points on a line. Sometimes the ratio is expressed as a quotient ("rise over run"),
                giving the same number for every two distinct points on the same line. A line that is decreasing has a
                negative "rise". The line may be practical - as set by a road surveyor, or in a diagram that models a
                road or a roof either as a description or as a plan.

                The steepness, incline, or grade of a line is measured by the absolute value of the slope. A slope with
                a greater absolute value indicates a steeper line. The direction of a line is either increasing,
                decreasing, horizontal or vertical.

                A line is increasing if it goes up from left to right. The slope is positive.
                A line is decreasing if it goes down from left to right. The slope is negative.
                If a line is horizontal the slope is zero. This is a constant function.
                If a line is vertical the slope is undefined (see below).

                In mathematical language, the slope m of the line is

                m = (y2 - y1) / (x2 - x1)

                The angle of inclination its found by

                arctan(m)


               :param y1: float value in the vertical axis, dtype=np.float32.
               :param y2: float value in the vertical axis, dtype=np.float32.
               :param x1: float value in the horizontal axis, dtype=np.float32.
               :param x2: float value in the horizontal axis, dtype=np.float32.
               :return: float value, dtype=np.float32, the angle of inclination of the slope.
    """

    slope = (y2 - y1) / (x2 - x1)
    slope = np.rad2deg(np.arctan(slope))
    return slope


# @njit('float32[:](float32[:], int32)', cache=True, nogil=True)
@njit
def slope_oxilator(arr, slope_periods, periods):
    """
               Documentation:
               Oxilator that seeks to mesure the streng of the slope of a line.
               :param arr: numpy array, dtype=np.float32, this is the close time series you want to use
               :param periods: integer value.
               :return: numpy array, dtype=np.float32, this is the value for every moment of the calculation.
            """

    n = len(arr)

    index = np.arange(0, n, 1)

    slope_array = slope(lag(index, slope_periods), lag(arr, slope_periods), index, arr)

    slope_array = slope_array + 90
    low_slope = cummin(slope_array, periods)
    high_slope = cummax(slope_array, periods)

    slope_array = slope_array.astype(np.int32)
    low_slope = low_slope.astype(np.int32)
    high_slope = high_slope.astype(np.int32)

    numerator = (slope_array - low_slope)
    denominator = (high_slope - low_slope)
    numerator = numerator.astype(np.float32)
    denominator = denominator.astype(np.float32)
    denominator[denominator == 0] = np.nan
    oxilator = (numerator / denominator) * 100
    oxilator = oxilator
    oxilator[oxilator == np.inf] = np.nan
    oxilator[oxilator == -np.inf] = np.nan

    return oxilator


@njit()
def multiple_lineal_regression(x, y):
    """
    Multiple linear regression by least squaeres method.
    :param x: np.array with dimension n, should be np.float32 numbers, independent variables.
    :param y: np.array with dimension 1, should be np.float32 numbers, dependent variable.
    :return: np.array with the coefficients of a mutiple linear model.
    """

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    data = x.copy()
    ones = np.ones(shape=x.shape[0]).reshape(-1, 1)

    if len(x.shape) == 1:
        x.reshape(-1, 1)

    data = np.concatenate((ones, data), 1)
    coefficients = np.linalg.inv(data.transpose().astype(np.float32).dot(data.astype(np.float32))).dot(
        data.transpose().astype(np.float32)).dot(y.astype(np.float32))

    return coefficients


@njit()
def single_undiff(arr, diff_arr):
    """
    Inverse operation for np.diff, for one order difference.
    :param arr: np.array with dimension 1, should be np.float32 numbers, the original arr to undifferetiated.
    :param diff_arr: np.array with dimension 1, should be np.float32 numbers, the order 1 difference.
    :return: np.array with the undifference array.
    """
    arr0 = [arr[0]]
    undiff_arr = np.append(arr0, diff_arr).cumsum()
    return undiff_arr


@njit()
def undiff(arr, diff_arr, n_diff):
    # """
    # Inverse operation for np.diff using recursive method.
    # :param arr: np.array with dimension 1, should be np.float32 numbers, the original arr to undifferetiated.
    # :param diff_arr: np.array with dimension 1, should be np.float32 numbers, the differentiated array of n order.
    # :param n_diff: np.int indicating the order of the difference.
    # :return: np.array with the undifference array.
    # """
    undiff_d = diff_arr.copy()
    for i in range(0, n_diff):
        d = n_diff - i - 1
        diff_d = np.diff(arr, d)
        if d > 0:
            undiff_d = single_undiff(diff_d, undiff_d)
        if d == 0:
            undiff_result = single_undiff(arr, undiff_d)
            return undiff_result


@njit()
def arima(arr, d, p, q, forecast_periods):
    """
    Arima model for d time series differences, from p lags and q residual lags, return forecast_periods ahead.
    :param arr: np.array with dimension 1, should be np.float32 numbers.
    :param d: np.int, time series differences, max 2.
    :param p: np.int, lags for the autoregresive model.
    :param q: np.int, lags for the ma model.
    :param forecast_periods: np.int, periods ahead to predict.
    :return: np.array with the predictions of n periods ahead.
    """
    # arr = arr.astype(np.float32)
    # d = np.int32(d)
    # p = np.int32(p)
    # q = np.int32(q)
    # forecast_periods = np.int32(forecast_periods)

    # time series differentiation order d
    real = arr.copy()
    arr = np.diff(arr, n=d)
    longitude = len(arr)
    lagged_arr = np.zeros(p * longitude) * np.nan
    for i in range(1, p + 1):
        lag_i = lag(arr, i)
        lagged_arr[(i - 1) * longitude: i * longitude] = lag_i
    final_lagged_arr = np.reshape(lagged_arr, (-1, longitude))
    transpose_arr = np.transpose(final_lagged_arr)

    # calculating coeffitiens for ar model

    ar_coeffitiens = multiple_lineal_regression(transpose_arr[p + 1:, ], arr[p + 1:, ])

    # calculating preditions for ar model
    data = transpose_arr[p + 1:, ]
    b0 = ar_coeffitiens[0]
    other_bethas = ar_coeffitiens[1:]

    predicted = np.zeros(len(data)) * np.nan
    for i in range(len(data)):
        predicted[i] = b0 + np.sum((data[i] * other_bethas))

    predicted = np.append(np.zeros(p + 1) + np.nan, predicted)

    residuals = arr - predicted

    longitude_res = len(residuals[p + 1:, ])

    lagged_res = np.zeros(q * longitude_res) * np.nan
    for i in range(1, q + 1):
        lag_res = lag(residuals[p + 1:, ], i)
        lagged_res[(i - 1) * longitude_res: i * longitude_res] = lag_res
    final_lagged_res = np.reshape(lagged_res, (-1, longitude_res))
    transpose_res = np.transpose(final_lagged_res)

    ma_coeffitiens = multiple_lineal_regression(transpose_res[q + 1:, ], residuals[(q + 1) + (p + 1):, ])

    data_ = residuals[(q + 1) + (p + 1):, ]
    b0_ = ma_coeffitiens[0]
    other_bethas_ = ma_coeffitiens[1:]

    predicted_residuals = np.zeros(len(data_)) * np.nan
    for i in range(len(data_)):
        predicted_residuals[i] = b0_ + np.sum((data_[i] * other_bethas_))

    predicted_residuals = np.append(np.zeros((q + 1) + (p + 1)) + np.nan, predicted_residuals)

    predicted_values = predicted + predicted_residuals

    # df = pd.DataFrame({'real': arr, 'predicted': predicted, 'residuals': residuals,
    #                    'predicted_residuals': predicted_residuals, 'predicted_values': predicted_values})

    forecasted_array = arr.copy()
    for j in range(0, forecast_periods):
        pred_arr = np.flip(forecasted_array[len(forecasted_array) - p:len(forecasted_array)])
        prediction = b0 + np.sum((pred_arr * other_bethas))
        forecasted_array = np.append(forecasted_array, prediction)

    forecasted_res = residuals.copy()
    for j in range(0, forecast_periods):
        pred_res = np.flip(forecasted_res[len(forecasted_res) - q:len(forecasted_res)])
        prediction_res = b0_ + np.sum((pred_res * other_bethas_))
        forecasted_res = np.append(forecasted_res, prediction_res)

    forecast_diff = forecasted_array + forecasted_res
    forecast_diff = np.append(arr, forecast_diff[len(forecast_diff) - forecast_periods:])

    forecast_array = undiff(real, forecast_diff, d)[-forecast_periods:]

    return forecast_array


@njit()
def rolling_arima(arr, d, p, q, forecast_periods, window):
    """
    Arima model for d time series differences in a rolling window, from p lags and q residual lags, return forecast_periods ahead.
    :param arr: np.array with dimension 1, should be np.float32 numbers.
    :param d: np.int, time series differences, max 2.
    :param p: np.int, lags for the autoregresive model.
    :param q: np.int, lags for the ma model.
    :param forecast_periods: np.int, periods ahead to predict.
    :param window: np.int, ts window.
    :return: np.array an array of the same lenth of the original array, each elemnt of the array predicts n periods ahead.
    """

    n = len(arr)

    values = np.array([arima(arr[i:i + window], d=d, p=p, q=q, forecast_periods=forecast_periods)[-1]
                       for i in range(n - window + 1)])

    arima_arr = np.append(np.zeros(window - 1) + np.nan, values)

    return arima_arr


def pullback(arr: np.array):
    """
    ma stands for the moving average of a price series.
    :param arr: np.array with dimension 1, should be np.float32 numbers.
    :param window: int value, its the length of the moving window, minimum value for the window may be 2.
    :return: np.array vector where the first #window-elements are np.nan, then you have the moving average of the arr.
    """
    n = len(arr)
    result = np.zeros(n) + np.NaN
    value = np.NaN
    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            value = arr[i]
            result[i] = value
        else:
            result[i] = value
    return result


def divergence(high, low, indicator, window):
    cummin_low = cummin(low, window)
    cummax_high = cummax(high, window)
    cummax_indicator = cummax(indicator, window)
    cummin_indicator = cummin(indicator, window)

    trend = np.zeros(len(high))

    for i in range(1, len(high)):
        if cummin_low[i] == low[i] and cummax_indicator[i] == indicator[i]:
            trend[i] = 1
        if cummax_high[i] == high[i] and cummin_indicator[i] == indicator[i]:
            trend[i] = -1
    return trend


@njit
def roc_spread(arr, fast_period, slow_period, slope_period):
    fast_roc = roc(arr, fast_period)
    slow_roc = roc(arr, slow_period)

    ratio = fast_roc / slow_roc

    position = np.arange(start=1, stop=len(arr) + 1)

    slope = np.zeros(len(arr))
    for i in range(0, len(arr)):
        slope[i] = (ratio[i] - ratio[i - slope_period]) / (position[i] - position[i - slope_period])

    return slope


@njit
def lin_reg(y_array, x_array, window):
    # this mask ensures we only calculate using common values (non-nan values)
    mask = np.where(np.isnan(x_array) | np.isnan(y_array), np.nan, 0)
    x_array += mask
    y_array += mask

    n = len(y_array)

    # We compute the rolling means
    x_mean_calc = np.array([np.nanmean(x_array[i:i + window]) for i in range(len(x_array) - window + 1)])
    x_mean = np.append(np.zeros(window - 1) + np.nan, x_mean_calc)
    y_mean_calc = np.array([np.nanmean(y_array[i:i + window]) for i in range(len(y_array) - window + 1)])
    y_mean = np.append(np.zeros(window - 1) + np.nan, y_mean_calc)

    # We get the mean deviations for each variable
    x = x_array - x_mean
    y = y_array - y_mean

    # We compute the square of the deviations of X
    sqr_x = x ** 2

    # The numerator of the slope
    num = x * y

    # We compute the rolling sum for the numerator
    roll_num_calc = np.array([np.nansum(num[i:i + window]) for i in range(len(x_array) - window + 1)])
    roll_num = np.append(np.zeros(window - 1) + np.nan, roll_num_calc)

    # The rolling sum for the denominator
    roll_den_calc = np.array([np.nansum(sqr_x[i:i + window]) for i in range(len(x_array) - window + 1)])
    roll_den = np.append(np.zeros(window - 1) + np.nan, roll_den_calc)

    # We ensure the denominator values differ from zero
    roll_den = np.where(roll_den == 0, 1, roll_den)

    slope = np.zeros(n)
    intercept = np.zeros(n)
    y_predict = np.zeros(n)

    for i in range(window, n):
        slope[i] = (roll_num[i] / roll_den[i]) * x[i]
        # slope[i] = np.nansum(x[i:i + window] * y[i:i + window]) / np.nansum(sqr_x[i:i + window])
        intercept[i] = y_mean[i] - slope[i] * x_mean[i]
        y_predict[i] = intercept[i] + slope[i] * x_mean[i]

    y_predict[:window] = np.nan

    return y_predict


@njit
def modified_atr(high: np.array, low: np.array, close: np.array, window):
    true_range = np.zeros(len(high))
    true_range[0] = np.nan

    for i in range(1, len(true_range)):
        high_minus_low = (high[i] / low[i]) - 1
        abs_high_minus_prev_close = (np.fabs(high[i] / close[i - 1])) - 1
        abs_low_minus_prev_close = (np.fabs(low[i] / close[i - 1])) - 1
        true_range[i] = np.amax(np.array([high_minus_low, abs_high_minus_prev_close, abs_low_minus_prev_close]))

    np_atr_values = np.array([true_range[i:i + window].mean() for i in range(len(true_range) - window + 1)])
    average_true_range = np.append(np.zeros(window - 1) + np.nan, np_atr_values)

    return average_true_range.astype(np.float32)


@njit
def pos_streak(arr):
    """Positive Streaks in a price series"""
    curr_streak = np.zeros(len(arr))
    streaks = int(0)
    for i in range(len(arr)):
        if i == 0:
            continue
        if arr[i] > arr[i - 1]:
            streaks = np.maximum(0, streaks)
            streaks += 1
            curr_streak[i] = streaks
        # elif arr[i] < arr[i - 1]:
        #     streaks = np.maximum(0, streaks)
        #     streaks += 1
        #     curr_streak[i] = streaks
        else:
            streaks = 0
            curr_streak[i] = streaks
    return curr_streak.astype(np.float32)


@njit
def neg_streak(close):
    curr_streak = np.zeros(len(close))
    streaks = int(0)
    for i in range(len(close)):
        if i == 0:
            continue
        if close[i] < close[i - 1]:
            streaks = np.maximum(0, streaks)
            streaks += 1
            curr_streak[i] = streaks
        # elif close[i] > close[i - 1]:
        #     streaks = np.minimum(0, streaks)
        #     streaks += -1
        #     curr_streak[i] = streaks
        else:
            streaks = 0
            curr_streak[i] = streaks

    return curr_streak.astype(np.float32)


@njit
def linear_regression_squares_method(arr, window):
    """
            Documentation:
            Linear regression is used to fit linear models for a window of limited periods,
            by the least squares method.

            Method
            For each (x,y) point calculate x2 and xy

            Sum all x, y, x2 and xy, which gives us Σx, Σy, Σx2 and Σxy

            Calculate Slope m:

            m =  N Σ(xy) − Σx ΣyN Σ(x2) − (Σx)2

            (N is the number of points.)

            Calculate Intercept b:

            b =  Σy − m ΣxN


            :param arr: numpy array, dtype=np.float32, this is the close time series you want to use
            :param window: integer value, the bigger the value the slower the calculation will but also less impacted by
            outlayers, picking a good value according to the temporality is a most.

            :return: numpy array, dtype=np.float32, the slope and r squared to mesure the goodness of fit.
            """
    window = window
    n = len(arr)

    prediction = np.zeros(n) + np.nan

    for i in range(window, n):
        x = np.arange(0, window, 1)
        y = arr[i - window:i]
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        if np.sum((x - mean_x) ** 2) == 0 or \
                np.sum((y - mean_y) ** 2) == 0:
            prediction[i] = np.nan
        else:
            slope = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x) ** 2)
            intercept = mean_y - slope * mean_x
            y_predict = intercept + slope * (window - 1)

            prediction[i] = y_predict

    return prediction


def squeeze_momentum(high, low, close, window_bb, mult_bb, window_kc, mult_kc, window_atr_kc,
                     window_momentum, window_sma):
    # Bollinger Bands
    bbands = bband(close, window_bb, mult_bb)
    upper_bband = bbands[0]
    lower_bband = bbands[1]
    midline_bband = bbands[2]

    # Keltner Channels
    kc = keltner_channel(high, low, close, window_kc, mult_kc, window_atr_kc)
    midline_kc = bbands[0]
    upper_kc = bbands[1]
    lower_kc = bbands[2]

    # Momentum Histogram
    highest_high = rolling_max(high, window_momentum)
    lowest_low = rolling_min(low, window_momentum)
    donchian_midline = (highest_high + lowest_low) / 2
    close_sma = ma(close, window_sma)
    squeeze_momentum_values = close - ((donchian_midline + close_sma) / 2)
    lin_squeeze_momentum_values = linear_regression_squares_method(squeeze_momentum_values, window_kc)

    # Squeeze On/Off Dots
    squeeze_on = np.zeros(len(close), dtype='bool')
    squeeze_off = np.zeros(len(close), dtype='bool')
    no_squeeze = np.zeros(len(close), dtype='bool')

    for i in range(1, len(close)):
        squeeze_on[i] = ((lower_bband[i] > lower_kc[i]) & (upper_bband[i] < upper_kc[i]))
        squeeze_off[i] = ((lower_bband[i] < lower_kc[i]) & (upper_bband[i] > upper_kc[i]))
        no_squeeze[i] = ((squeeze_on[i] == False) & (squeeze_off[i] == False))

    return lin_squeeze_momentum_values, squeeze_on, squeeze_off, no_squeeze


@njit
def candle_body(open, high, low, close):
    actual_body = np.fabs(close - open)
    actual_size = np.fabs(high - low)
    body_size = actual_body / actual_size

    return body_size.astype(np.float32)


@njit()
def highs_counter(column: np.array, window: int):
    init_high = 0
    lot_high = []

    for i in range(window - 1, len(column)):

        for n in range(window - 1):

            if column[i - n] > column[i - n - 1]:
                init_high += 1

        lot_high.append(init_high)
        init_high = 0

    lot_high = np.array(lot_high)
    np_total_high = np.append(np.zeros(window - 1) + np.nan, lot_high)

    return np_total_high.astype(np.float32)


@njit()
def lows_counter(column: np.array, window: int):
    init_low = 0
    lot_low = []

    for i in range(window - 1, len(column)):

        for n in range(window - 1):

            if column[i - n] < column[i - n - 1]:
                init_low += 1

        lot_low.append(init_low)
        init_low = 0

    lot_low = np.array(lot_low)
    np_total_low = np.append(np.zeros(window - 1) + np.nan, lot_low)

    return np_total_low.astype(np.float32)


@njit()
def minor_close(column: np.array, window: int):
    minor = []
    n = window - 1

    for i in range(window - 1, len(column)):

        if column[i] < column[i - n]:
            minor.append(True)

        else:
            minor.append(False)

    minor = np.array(minor)
    np_total_low = np.append(np.zeros(window - 1) + np.nan, minor)

    return np_total_low.astype(np.float32)


@njit()
def major_close(column: np.array, window: int):
    higher = []
    n = window - 1

    for i in range(window - 1, len(column)):

        if column[i] > column[i - n]:
            higher.append(True)

        else:
            higher.append(False)

    higher = np.array(higher)
    np_total_high = np.append(np.zeros(window - 1) + np.nan, higher)

    return np_total_high.astype(np.float32)


@njit
def returns(arr):

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
def returns(arr):
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


def sigma_returns(df, window: int, sigma_param: float):
    df['date'] = pd.to_datetime(df['date'].dt.date)
    close = df['close'].values.astype(np.float32)
    np_returns = returns(arr=close)
    df['close_returns'] = np_returns

    if window == 0:
        df['returns_mean'] = df['close_returns'].expanding().mean()
        df['returns_std'] = df['close_returns'].expanding().std()
        df['sigma_returns'] = sigma_param * df.returns_std + df.returns_mean

    if window > 0:
        dates = df['date'].values.astype('datetime64[D]')
        close_returns = df['close_returns'].values.astype(np.float32)

        sigma = np.zeros(len(close_returns)) + np.NaN

        for i in range(1,len(close_returns)):

            end_date = dates[i]

            year = dates[i].astype('datetime64[Y]').astype(int) + 1970 - window
            start_date = pd.to_datetime(str(year) + '-01-01')

            mask = ((dates >= start_date) & (dates <= end_date))

            filtered_return = close_returns[mask]

            sigma[i] = sigma_param * np.nanstd(filtered_return) + np.nanmean(filtered_return)

        df['sigma_returns'] = sigma

    return df


@njit
def congestion_index(high, low, close, h_l_window):
    ci = np.zeros(len(close))
    highest = rolling_max(high, h_l_window)
    lowest = rolling_min(low, h_l_window)

    for i in range(len(close)):
        ci[i] = ((close[i] - close[i - h_l_window]) / close[i - h_l_window]) * 100 / (
                    (highest[i] - lowest[i]) / lowest[i])

    return ci.astype(np.float32)


@njit
def klinger_oscilator(high, low, close, volume):
    np_hlc = high + low + close

    value = np.append(
        np.zeros(1) + np.nan, np.array([1 if np_hlc[i] > np_hlc[i - 1] else -1 for i in range(1, len(np_hlc))]))

    np_hl = high - low

    accumulated = np.zeros(len(np_hl)) * np.NaN
    accumulated[0] = np.nan
    accumulated[1] = np_hl[0] + np_hl[1]

    for i in range(2, len(np_hl)):

        if value[i] == 1 and value[i - 1] == 1:
            accumulated[i] = np_hl[i] + accumulated[i - 1]

        elif (value[i] == 1 and value[i - 1] == -1) or (value[i] == -1 and value[i - 1] == 1):
            accumulated[i] = np_hl[i] + np_hl[i - 1]

        elif value[i] == -1 and value[i - 1] == -1:

            if np_hl[i] + accumulated[i - 1] > np_hl[i] + np_hl[i - 1]:
                accumulated[i] = np_hl[i] + accumulated[i - 1]

            else:
                accumulated[i] = np_hl[i] + np_hl[i - 1]

    accumulated[accumulated == 0] = np.nan

    fv = volume * np.absolute(2 * (np_hl / accumulated) * (-1)) * value * 100

    sig = ma(fv, 13)

    fv_ma34 = ma(fv, 34)

    fv_ma55 = ma(fv, 55)

    ko = fv_ma34 - fv_ma55

    return ko, sig


def to_military(hour: int, minute: int):
    return hour * 10 ** 2 + minute


@njit
def sum_range_to_military_time(military_time, hours, minutes):
    military_sum = military_time + hours * 100 + minutes
    tens = (military_sum % 100) // 10

    if tens > 60:
        military_sum = military_sum + 100 - 60
    if military_sum > 2359:
        military_sum = military_sum - 2360
    return military_sum



def dates_as_integer(date, min_date):
    days = (date - min_date).days + 1
    return days

def datetime_int(date, hour):
    return date * 10 ** 4 + hour

def time_bias(df, entry_time):

    df['hour'] = pd.to_datetime(df.date).dt.hour
    df['minute'] = pd.to_datetime(df.date).dt.minute

    df['military_time'] = to_military(df['hour'], df['minute'])

    df['date_'] = pd.to_datetime(df["date"]).dt.date
    min_date = min(df["date_"])

    df['date_aux'] = min_date

    df['day'] = df['date_'] - df['date_aux']

    df['day'] = df['day'] / np.timedelta64(1, 'D') + 1

    df['datetime'] = datetime_int(df['day'] , df['military_time'])

    entry_time_low = entry_time

    entry_time_high = sum_range_to_military_time(entry_time_low, hours=0, minutes=20)

    df['entry_time_low'] = datetime_int(df['day'] , entry_time_low)

    df['entry_time_high'] = datetime_int(df['day'] , entry_time_high)
    df['entry_time_high'] = np.where(df['entry_time_high'] < df['entry_time_low'], datetime_int(df['day'] + 1 , entry_time_high), df['entry_time_high'])

    def entry_aux(datetime, entry_time_low, entry_time_high, lag_entry_time_low,  lag_entry_time_high):

        if datetime <= lag_entry_time_high:
            entry_low = lag_entry_time_low
            entry_high = lag_entry_time_high
        else:
            entry_low = entry_time_low
            entry_high = entry_time_high

        return entry_low, entry_high



    datetime = df['datetime'].values
    entry_time_high = df['entry_time_high'].values
    entry_time_low = df['entry_time_low'].values

    for i in range(1, len(datetime)):
        entry_time_low[i], entry_time_high[i] = \
            entry_aux(datetime=datetime[i], entry_time_low=entry_time_low[i],  entry_time_high=entry_time_high[i],
                      lag_entry_time_low=entry_time_low[i - 1], lag_entry_time_high=entry_time_high[i - 1])


    df['entry_time_high'] = entry_time_high
    df['entry_time_low'] = entry_time_low
    df['entry_time_high'] = entry_time_high
    df['trade_index'] = entry_time_high

    return df

@njit()
def exit_datetime(entry_day, entry_time, exit_time):
    exit_dt = entry_day * 10 ** 4 + exit_time
    if exit_dt <= entry_time:
        exit_dt = (entry_day + 1) * 10 ** 4 + exit_time
    return exit_dt
