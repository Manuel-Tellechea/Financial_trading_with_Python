import functools
import time
import pandas as pd
import numpy as np
from numba import njit, objmode
import numba
import logging
from logging.handlers import RotatingFileHandler
import platform
from datetime import datetime
import itertools
import pymongo
from quantstats.stats import *
from colorama import *
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, CrosshairTool
from decouple import config
import json


def timer(func):
    """Print the runtime of the decorated function"""
    print('/// @TIMER DECORATOR ACTIVATED')
    print()

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print()
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def debug(func):
    """Print the function signature and return value"""
    print('/// @DEBUGER DECORATOR ACTIVATED')
    print()

    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"DEBUGER --->>>, Calling {func.__name__}({signature})")
        print()
        value = func(*args, **kwargs)
        print()
        print(f"DEBUGER --->>> {func.__name__!r} returned {value!r}")

        return value

    return wrapper_debug


def standalone_testing(func):
    """Helps test Numba do not fail before going to quant optimization script"""
    print(f'/// @STANDALONE_TESTING DECORATOR ACTIVATED ON FUNCTION: {func}')
    print()

    @functools.wraps(func)
    def wraper_standalone(*args):
        df = pd.read_csv('is_data_CL.csv', nrows=200_000)

        dft = df[['open', 'high', 'low', 'close']].values.T
        dft = dft[:4].astype(np.float32)
        # data_timetamps = df['date'].values.T

        details = {'market_type': 3, 'margin': 5000, 'tick_size': 0.001, 'contract_size': 1000, 'fee': 7}
        instrument_detail = np.array(list(details.values()), dtype=np.float32)

        print()
        print('STANDALONE --->>> Loading CL data and details: SUCCESS')

        parameters = np.array(args[0], dtype=np.float32)
        direction = args[1]

        true_args = dft, parameters, instrument_detail, 500_000, direction

        print()
        print('STANDALONE --->>> Loading parameters you pass', parameters)
        print('STANDALONE --->>> Loading direction you pass', direction)

        print()
        print('STANDALONE --->>> Runing Function with', true_args)
        np_trades, capital_line = func(*true_args)

        print()
        print('STANDALONE --->>> Running function with your parameters')

        print()
        print('STANDALONE --->>> Summary: ')
        print('they are ', np_trades.shape[0], ' trades with ')
        print('Capital line ended in: ', capital_line[-1])

        df_trades = pd.DataFrame(np_trades,
                                 columns=['entry_date', 'exit_date', 'entry_price', 'exit_price', 'quantity', 'fee',
                                          'capital', 'mae', 'quantity_active', 'profit_loss', 'exit_type',
                                          'trade_direction'])
        print()
        print(df_trades.head(20))

        print()
        print('STANDALONE --->>> Check the table above, if no Numba error appears... CONGRATS!')

        return np_trades, capital_line

    return wraper_standalone


@njit
def please_set_me_all(capital, ins_det, my_data, target_number_columns=12, max_trades=10000, look_after=0):
    capital_line = np.zeros(max_trades) * np.NaN
    capital_line[0] = capital

    condition = False

    # trades_line can support up to 12 * 10.000 values (12 columns * 10.000 rows)
    trades_line = np.zeros(target_number_columns * max_trades) * np.NaN
    # since 12 is the number of features our output vector should have
    max_periods = int(look_after)

    trade_number = 0

    type = ins_det[0]
    margin = ins_det[1]
    tick_size = ins_det[2]
    contract_size = ins_det[3]
    fee = ins_det[4]

    open = my_data[0]
    high = my_data[1]
    low = my_data[2]
    close = my_data[3]
    volume = my_data[4]  # This line can potentially cause trouble

    len_data = len(open)

    return capital_line, condition, trades_line, max_periods, trade_number, \
           type, margin, tick_size, contract_size, fee, open, high, low, close, volume, len_data


@njit
def please_set_me_all_intermarket(capital, ins_det, my_data, target_number_columns=12, max_trades=10000, look_after=0,
                                  intermarket=False):
    # TODO: Creo que el problema es que no puede tener diferente numero de retornos,
    #  tanto el if como el else deberan traer igual cantidad de elementos

    capital_line = np.zeros(max_trades) * np.NaN
    capital_line[0] = capital

    condition = False

    # trades_line can support up to 12 * 10.000 values (12 columns * 10.000 rows)
    trades_line = np.zeros(target_number_columns * max_trades) * np.NaN
    # since 12 is the number of features our output vector should have
    max_periods = int(look_after)

    trade_number = 0

    # print('my ins det:')
    # print(ins_det)

    type = ins_det[0]
    margin = ins_det[1]
    tick_size = ins_det[2]
    contract_size = ins_det[3]
    fee = ins_det[4]

    # print(type, margin, tick_size, contract_size, fee)
    # print('-------------------')

    if not intermarket:
        open = my_data[0]
        high = my_data[1]
        low = my_data[2]
        close = my_data[3]
        volume = my_data[4]  # This line can potentially cause trouble

        len_data = len(open)

        nan_vector = (np.zeros(len_data) + np.nan).astype(np.float32)

        return capital_line, condition, trades_line, max_periods, trade_number, \
               type, margin, tick_size, contract_size, fee, \
               open, high, low, close, volume, len_data, \
               nan_vector, nan_vector, nan_vector, nan_vector, nan_vector

    else:
        open_x, open_y = my_data[0], my_data[5]
        high_x, high_y = my_data[1], my_data[6]
        low_x, low_y = my_data[2], my_data[7]
        close_x, close_y = my_data[3], my_data[8]
        volume_x, volume_y = my_data[4], my_data[9]

        len_data = len(open_x)

        return capital_line, condition, trades_line, max_periods, trade_number, \
               type, margin, tick_size, contract_size, fee, \
               open_x, high_x, low_x, close_x, volume_x, len_data, \
               open_y, high_y, low_y, close_y, volume_y


@njit
def please_set_me_all_inter(capital, ins_det, my_data, target_number_columns=12, max_trades=10000, look_after=0):
    # TODO: Creo que el problema es que no puede tener diferente numero de retornos,
    #  tanto el if como el else deberan traer igual cantidad de elementos

    capital_line = np.zeros(max_trades) * np.NaN
    capital_line[0] = capital

    condition = False

    # trades_line can support up to 12 * 10.000 values (12 columns * 10.000 rows)
    trades_line = np.zeros(target_number_columns * max_trades) * np.NaN
    # since 12 is the number of features our output vector should have
    max_periods = int(look_after)

    trade_number = 0

    # print('my ins det:')
    # print(ins_det)

    type = ins_det[0]
    margin = ins_det[1]
    tick_size = ins_det[2]
    contract_size = ins_det[3]
    fee = ins_det[4]

    # print(type, margin, tick_size, contract_size, fee)
    # print('-------------------')

    open_x, open_y, open_z = my_data[0], my_data[5], my_data[10]
    high_x, high_y, high_z = my_data[1], my_data[6], my_data[11]
    low_x, low_y, low_z = my_data[2], my_data[7], my_data[12]
    close_x, close_y, close_z = my_data[3], my_data[8], my_data[13]
    volume_x, volume_y, volume_z = my_data[4], my_data[9], my_data[14]
    trade_index = my_data[15]
    exit_index = 0
    capital_0 = capital
    entry_index = np.NaN
    exit_condition = False
    size = 14
    len_data = len(open_x)

    return capital_line, condition, trades_line, max_periods, trade_number, \
           type, margin, tick_size, contract_size, fee, \
           open_x, high_x, low_x, close_x, volume_x, len_data, \
           open_y, high_y, low_y, close_y, volume_y, \
           open_z, high_z, low_z, close_z, volume_z, \
           trade_index, exit_index, capital_0, entry_index, exit_condition, size, len_data


def validation(func):
    @functools.wraps(func)
    def wraper_validation(*args):

        arr = args[0]
        window = args[1]

        if type(window) != int:
            raise EwmErrors(f'Indicators.py function {func} error: '
                            f'window parameter should be an integer not {type(window)}!')

        if type(arr) != np.ndarray:
            raise EwmErrors(f'Indicators.py function {func} error: '
                            f'arr parameter should be an np.ndarray not {type(arr)}')

        if len(arr) < window:
            raise EwmErrors(f'Indicators.py function {func} error: '
                            f'len() of arr: {len(arr)} should be bigger than window param: {window}')

        if window < 1:
            raise EwmErrors(f'Indicators.py function {func} error: '
                            f'window parameter should be more than 1 not {window}')

        function = func(arr, window)

        return function

    return wraper_validation


class EwmErrors(Exception):
    """Custom error you want to rise in order to debug"""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def old_log_me(my_logger, message, object, pikachu=False):
    """
    DEPRECATED, only for documentation purposes
    """

    if pikachu is True:
        my_logger.info('''
                ⣿    ⣿ ⣿⣿⣿⣿⣿⣿⣿⣿⣿ ⣿⣿⣿ ⣿⣿⣿ ⣿⣿⣿⣿
                ⣿ ⣿⣿⣿⣿ ⣿⣿⣿ ⣿⣿⣿⣿ ⣿⣿⣿ ⣿ ⣿ ⣿ ⣿⣿⣿
                ⣿    ⣿⣿ ⣿⣿ ⣿ ⣿⣿ ⣿⣿⣿ ⣿⣿⣿ ⣿⣿ ⣿⣿⣿
                ⣿ ⣿⣿⣿⣿⣿ ⣿ ⣿⣿ ⣿ ⣿⣿⣿ ⣿⣿⣿⣿⣿⣿⣿ ⣿⣿
                ⣿    ⣿⣿⣿ ⣿⣿⣿⣿ ⣿⣿⣿ ⣿⣿⣿⣿⣿⣿⣿⣿⣿ ⣿
                ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
                ⣿⣿⣿⡏⠉⠛⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿
                ⣿⣿⣿⣿⠀⠀⠀⠈⠛⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⠉⠁⠀⣿
                ⣿⣿⣿⣿⣧⡀⠀⠀⠀⠀⠙⠿⠿⠿⠻⠿⠿⠟⠿⠛⠉⠀⠀⠀⠀⠀⣸⣿
                ⣿⣿⣿⣿⣿⣷⣄⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⣿
                ⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⣴⣿⣿⣿⣿
                ⣿⣿⣿⣿⣿⣿⡟⠀⠀⢰⣹⡆⠀⠀⠀⠀⠀⠀⣭⣷⠀⠀⠀⠸⣿⣿⣿⣿
                ⣿⣿⣿⣿⣿⣿⠃⠀⠀⠈⠉⠀⠀⠤⠄⠀⠀⠀⠉⠁⠀⠀⠀⠀⢿⣿⣿⣿
                ⣿⣿⣿⣿⣿⣿⢾⣿⣷⠀⠀⠀⠀⡠⠤⢄⠀⠀⠀⠠⣿⣿⣷⠀⢸⣿⣿⣿
                ⣿⣿⣿⣿⣿⣿⡀⠉⠀⠀⠀⠀⠀⢄⠀⢀⠀⠀⠀⠀⠉⠉⠁⠀⠀⣿⣿⣿
                ⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣿⣿
                ⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿
                ''')

    my_logger.info(message)
    my_logger.info(object)


@njit
def please_buy_me(i, future_open, instrument_tick_size, profit_target_param, stop_loss_param, capital,
                  instrument_type, instrument_margin, instrument_contract_size, instrument_fee):
    # BUG-> Serious issue with this -> I can't break the code without the for loop... I'm stuck

    entry_date = i

    entry_price = future_open + instrument_tick_size
    profit_target_price = numba.float32(entry_price * (1 + profit_target_param))
    stop_loss_price = entry_price - (entry_price * stop_loss_param)

    # position size logic:
    position_size = np.round((capital * 0.01) / (entry_price * stop_loss_param *
                                                 instrument_contract_size))

    entry_position_size_usd = position_size * instrument_contract_size * entry_price

    # 1 Means Futuros
    if instrument_type == 1:
        if capital < instrument_margin:
            return None  # Break
            # with objmode():
            #     log_me(logger, 'EWM breaking execution, capital is < instrument margin!', None, pikachu=True)

        if position_size == 0:
            position_size = 1
            entry_position_size_usd = position_size * instrument_contract_size * entry_price

        total_fee = instrument_fee * 2 * position_size
        margin_total = instrument_margin * position_size

    # 2 Means Acciones
    elif instrument_type == 2:

        if entry_position_size_usd > capital:
            position_size = np.floor(capital / (instrument_contract_size * entry_price))
            entry_position_size_usd = position_size * instrument_contract_size * entry_price

        if position_size == 0:
            # with objmode():
            #     log_me(logger, 'EWM breaking execution, position size is 0!', None, pikachu=True)
            return None  # break

        # We assume the max amount to pay in fees is 1% of the total. According to current rules.
        # This may change in the future without notice.
        total_fee = min(max(instrument_fee * position_size, 1), entry_price * position_size * 0.01)
        margin_total = np.NaN

        return entry_date, entry_price, stop_loss_price, position_size, entry_position_size_usd, total_fee, margin_total


def get_logging(stage: int, message: str, message_variable=''):
    """
    as it is on Skynet repository, this is the logging function we use for the strategies.
    :param stage: Genera el archivo .log o guarda el mensaje dentro del archivo.
    :param message: Mensaje a guardar en el .log
    :param message_variable: Mensaje a guardar en el .log
    :return:
    """

    strategies_logger = logging.getLogger('skynet_strategies_logger')
    server = platform.node()

    if stage == 0:
        strategies_logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(f'logs/{server}_strategies_logger.log', maxBytes=1000000000)
        formatter = logging.Formatter('%(levelname)s | %(asctime)s | %(message)s')
        handler.setFormatter(formatter)
        strategies_logger.addHandler(handler)
        strategies_logger.info(message)
    else:
        # Formating message (this is to workaround Numba liminations)
        message += f' {message_variable}'
        strategies_logger.info(message)


def add_weekday(dataframe: pd.DataFrame):
    """
    add_weekday takes a dataframe and create a new column with the day of the week,
    0 means Monday and 6 means Sunday/
    :param dataframe: any pd.Dataframe you want to use, be sure a column called date in inside.
    :return: None, this function works inplace.
    """
    dataframe['year'] = dataframe.date.apply(lambda row: int(row.split()[0].split('-')[0]))
    dataframe['month'] = dataframe.date.apply(lambda row: int(row.split()[0].split('-')[1]))
    dataframe['day'] = dataframe.date.apply(lambda row: int(row.split()[0].split('-')[2]))

    dataframe['weekday'] = dataframe.apply(lambda row:
                                           datetime.date(row['year'], row['month'], row['day']).weekday(), axis=1)

    del dataframe['year'], dataframe['month'], dataframe['day']


def how_many_nans_in_intermarket(dataframe1, dataframe2):
    """
    This function tells you some statistics about the dataframes you want to use for your intermarket strategy.
    :param dataframe1: pd.DataFrame you want to use as the leading dataframe.
    :param dataframe2: pd.DataFrame you want to use as the auxiliary dataframe.
    :return: None, it only prints a couple of messages.
    """
    d1, d2 = set(dataframe1.date), set(dataframe2.date)

    long = len(d1)

    interception = d1 & d2
    all_rows = d1 | d2

    d1_minus_d2 = d1 - d2
    d2_minus_d1 = d2 - d1

    print('there will be', len(d1_minus_d2), 'NAN values in data1 this is equal to',
          round(100 * len(d1_minus_d2) / len(all_rows), 2), '% of all data')
    print('there will be', len(d2_minus_d1), 'NAN values in data1 this is equal to',
          round(100 * len(d2_minus_d1) / len(all_rows), 2), '% of all data')

    simetric_diff = d1.symmetric_difference(d2)
    print(len(simetric_diff), 'rows with at least 1 NAN after join this is equal to',
          100 * round(len(simetric_diff) / len(all_rows), 2), '% of all data')

    print()
    print(100 * round(len(interception) / long, 2),
          '% of the records are in common between data1 and data2 should be 100%')
    print(100 * round(len(all_rows) / long, 2) - 100,
          '% more records than expected when matching date1 and date2 should be 0%')

    print()


@njit
def stop_loss_exit(index, open_price, low_price, high_price, stop_loss_price, position_size, instrument_contract_size,
                   entry_position_size_usd, total_fee, direction):
    if direction == 1:

        final_price_to_exit = np.nan  # this variables is always set to nan unless you actually are going to sell.
        if open_price > stop_loss_price:
            # since this is true that means you are not going to sell at the open.
            if low_price <= stop_loss_price:
                # if the low < stop loss < open that means we should sell at the stop loss.
                final_price_to_exit = stop_loss_price
            else:
                # the low is bigger than the stop loss, so go to the selling condition.
                pass
        else:
            # the open price is < than the stop loss price, so sell at the open price!
            final_price_to_exit = open_price

        if not np.isnan(final_price_to_exit):
            # Since the selling price is different than np.nan then sell at the final_price_to_exit price.

            exit_price = final_price_to_exit * 0.99975
            exit_position_size_usd = position_size * instrument_contract_size * exit_price
            profit_loss = exit_position_size_usd - entry_position_size_usd - total_fee
            exit_date = index

        if np.isnan(final_price_to_exit):
            profit_loss = np.nan
            exit_date = np.nan
            exit_price = np.nan

    if direction == -1:

        final_price_to_exit = np.nan  # this variables is always set to nan unless you actually are going to sell.
        if open_price < stop_loss_price:
            # since this is true that means you are not going to sell at the open.
            if high_price >= stop_loss_price:
                # if the low < stop loss < open that means we should sell at the stop loss.
                final_price_to_exit = stop_loss_price
            else:
                # the low is bigger than the stop loss, so go to the selling condition.
                pass
        else:
            # the open price is < than the stop loss price, so sell at the open price!
            final_price_to_exit = open_price

        if not np.isnan(final_price_to_exit):
            # Since the selling price is different than np.nan then sell at the final_price_to_exit price.

            exit_price = final_price_to_exit * 1.00025
            exit_position_size_usd = position_size * instrument_contract_size * exit_price
            profit_loss = ((exit_position_size_usd - entry_position_size_usd) * -1) - total_fee
            exit_date = index

        if np.isnan(final_price_to_exit):
            profit_loss = np.nan
            exit_date = np.nan
            exit_price = np.nan

    return profit_loss, exit_date, exit_price


@njit
def profit_target_exit(index, open_price, low_price, high_price, profit_target_price, position_size,
                       instrument_contract_size, entry_position_size_usd, total_fee, direction):
    if direction == 1:

        final_price_to_exit = np.nan  # this variables is always set to nan unless you actually are going to sell.
        if open_price < profit_target_price:
            # since this is true that means you are not going to sell at the open.
            if high_price >= profit_target_price:
                # if the low < stop loss < open that means we should sell at the stop loss.
                final_price_to_exit = profit_target_price
            else:
                # the low is bigger than the stop loss, so go to the selling condition.
                pass
        else:
            # the open price is < than the stop loss price, so sell at the open price!
            final_price_to_exit = open_price

        if not np.isnan(final_price_to_exit):
            # Since the selling price is different than np.nan then sell at the final_price_to_exit price.

            exit_price = final_price_to_exit * 0.99975
            exit_position_size_usd = position_size * instrument_contract_size * exit_price
            profit_loss = exit_position_size_usd - entry_position_size_usd - total_fee
            exit_date = index

        if np.isnan(final_price_to_exit):
            profit_loss = np.nan
            exit_date = np.nan
            exit_price = np.nan

    if direction == -1:

        final_price_to_exit = np.nan  # this variables is always set to nan unless you actually are going to sell.
        if open_price > profit_target_price:
            # since this is true that means you are not going to sell at the open.
            if low_price <= profit_target_price:
                # if the low < stop loss < open that means we should sell at the stop loss.
                final_price_to_exit = profit_target_price
            else:
                # the low is bigger than the stop loss, so go to the selling condition.
                pass
        else:
            # the open price is < than the stop loss price, so sell at the open price!
            final_price_to_exit = open_price

        if not np.isnan(final_price_to_exit):
            # Since the selling price is different than np.nan then sell at the final_price_to_exit price.

            exit_price = final_price_to_exit * 1.00025
            exit_position_size_usd = position_size * instrument_contract_size * exit_price
            profit_loss = ((exit_position_size_usd - entry_position_size_usd) * -1) - total_fee
            exit_date = index

        if np.isnan(final_price_to_exit):
            profit_loss = np.nan
            exit_date = np.nan
            exit_price = np.nan

    return profit_loss, exit_date, exit_price


@njit
def trailing_stop_exit(index, open_price, low_price, high_price, trailing_stop, position_size,
                       instrument_contract_size, entry_position_size_usd, total_fee, direction):
    if direction == 1:

        final_price_to_exit = np.nan  # this variables is always set to nan unless you actually are going to sell.
        if open_price > trailing_stop:
            # since this is true that means you are not going to sell at the open.
            if low_price <= trailing_stop:
                # if the low < stop loss < open that means we should sell at the stop loss.
                final_price_to_exit = trailing_stop
            else:
                # the low is bigger than the stop loss, so go to the selling condition.
                pass
        else:
            # the open price is < than the stop loss price, so sell at the open price!
            final_price_to_exit = open_price

        if not np.isnan(final_price_to_exit):
            # Since the selling price is different than np.nan then sell at the final_price_to_exit price.

            exit_price = final_price_to_exit * 0.99975
            exit_position_size_usd = position_size * instrument_contract_size * exit_price
            profit_loss = exit_position_size_usd - entry_position_size_usd - total_fee
            exit_date = index

        if np.isnan(final_price_to_exit):
            profit_loss = np.nan
            exit_date = np.nan
            exit_price = np.nan

    if direction == -1:

        final_price_to_exit = np.nan  # this variables is always set to nan unless you actually are going to sell.
        if open_price < trailing_stop:
            # since this is true that means you are not going to sell at the open.
            if high_price >= trailing_stop:
                # if the low < stop loss < open that means we should sell at the stop loss.
                final_price_to_exit = trailing_stop
            else:
                # the low is bigger than the stop loss, so go to the selling condition.
                pass
        else:
            # the open price is < than the stop loss price, so sell at the open price!
            final_price_to_exit = open_price

        if not np.isnan(final_price_to_exit):
            # Since the selling price is different than np.nan then sell at the final_price_to_exit price.

            exit_price = final_price_to_exit * 1.00025
            exit_position_size_usd = position_size * instrument_contract_size * exit_price
            profit_loss = ((exit_position_size_usd - entry_position_size_usd) * -1) - total_fee
            exit_date = index

        if np.isnan(final_price_to_exit):
            profit_loss = np.nan
            exit_date = np.nan
            exit_price = np.nan

    return profit_loss, exit_date, exit_price


@njit
def trailing_stop_minute_exit(index, next_open_price, close_price, trailing_stop, position_size,
                              instrument_contract_size, entry_position_size_usd, total_fee, direction,
                              instrument_type, instrument_tick_size):
    """
    :param index: actual position in the array.
    :param next_open_price: the trade is established in the next open price (open[i + 1]).
    :param close_price: np.array, the Close prices from the series.
    :param trailing_stop: it's necessary use the stop_loss_price value because the trailing_stop[i] has a nan value.
    :param position_size: position size value.
    :param instrument_contract_size: instrument contract size value.
    :param entry_position_size_usd: entry position size amount in dollars.
    :param total_fee: total fee value.
    :param direction: 1 for long direction or 2 for short direction
    :param instrument_type: 1 for Futures or 2 for Stocks or ETF
    :param instrument_tick_size: instrument tick size
    :return: profit_loss, exit_date, exit_price values
    """

    if direction == 1:

        final_price_to_exit = np.nan  # this variables is always set to nan unless you actually are going to sell.
        if close_price <= trailing_stop:
            final_price_to_exit = next_open_price

        if not np.isnan(final_price_to_exit):
            # Since the selling price is different than np.nan then sell at the final_price_to_exit price.

            # exit_price = final_price_to_exit * 0.99975
            exit_price = price_with_slippage(final_price_to_exit, 1, instrument_type, instrument_tick_size, "exit")
            exit_position_size_usd = position_size * instrument_contract_size * exit_price
            profit_loss = exit_position_size_usd - entry_position_size_usd - total_fee
            exit_date = index + 1

        if np.isnan(final_price_to_exit):
            profit_loss = np.nan
            exit_date = np.nan
            exit_price = np.nan

    if direction == -1:

        final_price_to_exit = np.nan  # this variables is always set to nan unless you actually are going to sell.
        if close_price >= trailing_stop:
            final_price_to_exit = next_open_price

        if not np.isnan(final_price_to_exit):
            # Since the selling price is different than np.nan then sell at the final_price_to_exit price.

            exit_price = price_with_slippage(final_price_to_exit, -1, instrument_type, instrument_tick_size, "exit")
            exit_position_size_usd = position_size * instrument_contract_size * exit_price
            profit_loss = ((exit_position_size_usd - entry_position_size_usd) * -1) - total_fee
            exit_date = index + 1

        if np.isnan(final_price_to_exit):
            profit_loss = np.nan
            exit_date = np.nan
            exit_price = np.nan

    return profit_loss, exit_date, exit_price


@njit
def stop_loss_minute_exit(index, next_open_price, close_price, stop_loss_price, position_size, instrument_contract_size,
                          entry_position_size_usd, total_fee, direction, instrument_type, instrument_tick_size):
    """
    :param index: actual position in the array.
    :param next_open_price: np.array, the Open prices from the series. The trade is established in the next open
    price (open[i + 1]).
    :param close_price: np.array, the Close prices from the series.
    :param stop_loss_price: stop loss value.
    :param position_size: position size value.
    :param instrument_contract_size: instrument contract size value.
    :param entry_position_size_usd: entry position size amount in dollars.
    :param total_fee: total fee value.
    :param direction: 1 for long direction or 2 for short direction
    :param instrument_type: 1 for Futures or 2 for Stocks or ETF
    :param instrument_tick_size: instrument tick size
    :return: profit_loss, exit_date, exit_price values
    """

    if direction == 1:

        final_price_to_exit = np.nan  # this variables is always set to nan unless you actually are going to sell.
        if close_price <= stop_loss_price:
            final_price_to_exit = next_open_price

        if not np.isnan(final_price_to_exit):
            # Since the selling price is different than np.nan then sell at the final_price_to_exit price.

            exit_price = price_with_slippage(final_price_to_exit, 1, instrument_type, instrument_tick_size, "exit")
            exit_position_size_usd = position_size * instrument_contract_size * exit_price
            profit_loss = exit_position_size_usd - entry_position_size_usd - total_fee
            exit_date = index + 1

        if np.isnan(final_price_to_exit):
            profit_loss = np.nan
            exit_date = np.nan
            exit_price = np.nan

    if direction == -1:

        final_price_to_exit = np.nan  # this variables is always set to nan unless you actually are going to sell.
        if close_price >= stop_loss_price:
            final_price_to_exit = next_open_price

        if not np.isnan(final_price_to_exit):
            # Since the selling price is different than np.nan then sell at the final_price_to_exit price.

            exit_price = price_with_slippage(final_price_to_exit, -1, instrument_type, instrument_tick_size, "exit")
            exit_position_size_usd = position_size * instrument_contract_size * exit_price
            profit_loss = ((exit_position_size_usd - entry_position_size_usd) * -1) - total_fee
            exit_date = index + 1

        if np.isnan(final_price_to_exit):
            profit_loss = np.nan
            exit_date = np.nan
            exit_price = np.nan

    return profit_loss, exit_date, exit_price


@njit
def profit_target_minute_exit(index, next_open_price, close_price, profit_target_price, position_size,
                              instrument_contract_size, entry_position_size_usd, total_fee, direction,
                              instrument_type, instrument_tick_size):
    """
        :param index: actual position in the array.
        :param next_open_price: np.array, the Open prices from the series. The trade is established in the next open
        price (open[i + 1]).
        :param close_price: np.array, the Close prices from the series.
        :param profit_target_price: profit target value.
        :param position_size: position size value.
        :param instrument_contract_size: instrument contract size value.
        :param entry_position_size_usd: entry position size amount in dollars.
        :param total_fee: total fee value.
        :param direction: 1 for long direction or 2 for short direction
        :param instrument_type: 1 for Futures or 2 for Stocks or ETF
        :param instrument_tick_size: instrument tick size
        :return: profit_loss, exit_date, exit_price values
        """

    if direction == 1:

        final_price_to_exit = np.nan  # this variables is always set to nan unless you actually are going to sell.
        if close_price >= profit_target_price:
            final_price_to_exit = next_open_price

        if not np.isnan(final_price_to_exit):
            # Since the selling price is different than np.nan then sell at the final_price_to_exit price.

            exit_price = price_with_slippage(final_price_to_exit, 1, instrument_type, instrument_tick_size, "exit")
            exit_position_size_usd = position_size * instrument_contract_size * exit_price
            profit_loss = exit_position_size_usd - entry_position_size_usd - total_fee
            exit_date = index + 1

        if np.isnan(final_price_to_exit):
            profit_loss = np.nan
            exit_date = np.nan
            exit_price = np.nan

    if direction == -1:

        final_price_to_exit = np.nan  # this variables is always set to nan unless you actually are going to sell.
        if close_price <= profit_target_price:
            final_price_to_exit = next_open_price

        if not np.isnan(final_price_to_exit):
            # Since the selling price is different than np.nan then sell at the final_price_to_exit price.

            exit_price = price_with_slippage(final_price_to_exit, -1, instrument_type, instrument_tick_size, "exit")
            exit_position_size_usd = position_size * instrument_contract_size * exit_price
            profit_loss = ((exit_position_size_usd - entry_position_size_usd) * -1) - total_fee
            exit_date = index + 1

        if np.isnan(final_price_to_exit):
            profit_loss = np.nan
            exit_date = np.nan
            exit_price = np.nan

    return profit_loss, exit_date, exit_price


def mongodb_authentication(db: str):
    """
    :param db: Nombre de la base de datos.
    :return: Cliente
    """

    uri = config('MONGO_URI')
    try:
        MONGO_CLIENT = pymongo.MongoClient(uri)
        client_server_skynet = MONGO_CLIENT[db]
        return client_server_skynet
    except Exception as e:
        print(e)
        exit()


def get_data(symbol: str):
    try:
        df_data = pd.read_csv(f'is_data_{symbol}.csv')  # Download your data from skynet-strategies/is_data_download.py
        df_data["date"] = pd.to_datetime((df_data["date"]))
        print(Fore.GREEN + f"[{symbol}]" + Fore.WHITE + " - Get_data done.")

    except Exception as e:
        print(Fore.RED + f"[{symbol}] - {e}")
    return df_data


def get_dayly_data(client, symbol: str, group: str):
    try:
        historical_data_collection = client["HistoricalDataDaily"]
        data = historical_data_collection.find({'symbol': symbol, 'group': group},
                                               {'_id': 0, 'date': 1, 'date': 1, 'open': 1, 'high': 1, 'low': 1,
                                                'close': 1, 'volume': 1}).sort('date', 1)
        df_data = pd.DataFrame(list(data))
        df_data = df_data[['date', 'open', 'high', 'low', 'close', 'volume']]

        if not df_data.empty:
            return df_data
        else:
            print(f'{symbol} data not found')

    except Exception as e:
        print(e)


def columns_apply(len_columns: int):
    """
    :param len_columns: Longitud de columnas de la data histórica del instrumento
    :return: Diccionario con las columnas necesarias para el resample.
    """
    if len_columns > 6:
        apply = {'open_x': 'first', 'high_x': 'max', 'low_x': 'min', 'close_x': 'last',
                 'volume_x': 'sum',
                 'open_y': 'first', 'high_y': 'max', 'low_y': 'min', 'close_y': 'last',
                 'volume_y': 'sum'}
    else:
        apply = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    return apply


def get_data_resampled(data: pd.DataFrame, time: str, instrument_details: dict, client):
    """
    :param data: Data histórica del instrumento.
    :param time: Temporalidad en la que se agruparà la data.
    :param instrument_details: Detalles del instrumento, margin, tick_size, fee, etc.
    :param client: detalles del usuario.
    :return: Data histórica agrupada en la temporalidad ingresada.
    """
    # if portfolio_type == 'swing':
    #     data['date'] = pd.to_datetime(data['date'].dt.strftime("%Y-%m-%d"))
    data_aux = data.copy()
    if len(data_aux.columns) > 6:
        data_aux.reset_index(inplace=True)
    trading_hours_collection = client["TradingHours"]
    calendar_collection = client["TradingHoursCalendar"]
    df_historical_data_fixed = pd.DataFrame()
    grouping_data_no_tradinghours = False
    if not instrument_details['trading_hours'] == "":
        trading_hours_info = trading_hours_collection.find_one({"name": instrument_details['trading_hours']},
                                                               {"_id": 0, "time_zone": 1, "start_time": 1})
        if trading_hours_info is not None:
            time_zone = trading_hours_info["time_zone"]
            start_time = trading_hours_info["start_time"]
            calendar = calendar_collection.find_one({"time_zone": time_zone})
            if calendar is not None:
                calendar_years_df = pd.DataFrame(list(calendar["years"]))
                calendar_hour_change = calendar["change"]
                data_date_init = data_aux.date.iloc[0]
                data_date_end = data_aux.date.iloc[-1]
                data_year_init = data_date_init.year
                data_year_end = data_date_end.year
                calendar_aux_df = calendar_years_df[(calendar_years_df['year'] >= data_year_init - 28) &
                                                    (calendar_years_df['year'] <= data_year_end - 28)]
                calendar_aux_df.reset_index(inplace=True)
                for i in range(0, len(calendar_aux_df)):
                    calendar_aux_df["year"][i] += 28
                    calendar_aux_df["start_date"][i] = calendar_aux_df["start_date"][i].replace(
                        year=calendar_aux_df["year"][i])
                    calendar_aux_df["end_date"][i] = calendar_aux_df["end_date"][i].replace(
                        year=calendar_aux_df["year"][i])
                    # Antes del Daylight Saving
                    df_aux = data_aux[(data_aux['date'] >=
                                       f'{calendar_aux_df["year"][i]}-01-01') &
                                      (data_aux['date'] < calendar_aux_df["start_date"][i])]
                    if not df_aux.empty:
                        origin_date = df_aux.date.iloc[0].replace(hour=int(start_time[:2]), minute=int(start_time[3:]))
                        df_aux.set_index('date', inplace=True)
                        apply = columns_apply(len(data_aux.columns))
                        df_aux = df_aux.resample(time, origin=origin_date, label='right', closed='right').apply(apply)
                        df_historical_data_fixed = df_historical_data_fixed.append(df_aux)
                    # Dentro del Daylight Saving
                    df_aux = data_aux[(data_aux['date'] >=
                                       f'{calendar_aux_df["start_date"][i]}') &
                                      (data_aux['date'] <= calendar_aux_df["end_date"][i])]
                    if not df_aux.empty:
                        origin_date = df_aux.date.iloc[0].replace(
                            hour=int(start_time[:2]) - int(calendar_hour_change[:2]),
                            minute=int(start_time[3:]) - int(
                                calendar_hour_change[3:]))
                        df_aux.set_index('date', inplace=True)
                        apply = columns_apply(len(data_aux.columns))
                        df_aux = df_aux.resample(time, origin=origin_date, label='right', closed='right').apply(apply)
                        df_historical_data_fixed = df_historical_data_fixed.append(df_aux)
                    # Despues del Daylight Saving
                    df_aux = data_aux[(data_aux['date'] >
                                       f'{calendar_aux_df["end_date"][i]}') &
                                      (data_aux['date'] <= f'{calendar_aux_df["year"][i]}-12-31 23:59:59')]
                    if not df_aux.empty:
                        origin_date = df_aux.date.iloc[0].replace(hour=int(start_time[:2]), minute=int(start_time[3:]))
                        df_aux.set_index('date', inplace=True)
                        apply = columns_apply(len(data_aux.columns))
                        df_aux = df_aux.resample(time, origin=origin_date, label='right', closed='right').apply(apply)
                        df_historical_data_fixed = df_historical_data_fixed.append(df_aux)
            else:
                # Tradinghours registrado pero sin Time zone
                origin_date = data_aux.date.iloc[0].replace(hour=int(start_time[:2]), minute=int(start_time[3:]))
                data_aux.set_index('date', inplace=True)
                apply = columns_apply(len(data_aux.columns))
                df_historical_data_fixed = data_aux.resample(time, origin=origin_date,
                                                             label='right', closed='right').apply(apply)
        else:
            grouping_data_no_tradinghours = True
    else:
        grouping_data_no_tradinghours = True
    if grouping_data_no_tradinghours is True:
        # TradingHours vacio o no registrado
        data_aux.set_index('date', inplace=True)
        apply = columns_apply(len(data_aux.columns))
        df_historical_data_fixed = data_aux.resample(time, label='right', closed='right').apply(apply)
    df_historical_data_fixed.dropna(inplace=True)
    df_historical_data_fixed.sort_values('date', ascending=True, inplace=True)
    df_historical_data_fixed.reset_index(inplace=True)
    data_aux.reset_index(inplace=True)
    print(Fore.GREEN + f"[{instrument_details['symbol']}]" + Fore.WHITE + " - get_data_resampled done.\n")
    return df_historical_data_fixed


def get_instrument_details(client, symbol: str, group: str):
    # get instrument details. fee, margin, contract_size, tick_size. etc..
    instrument_collection = client["Instrument"]
    instrument_details = instrument_collection.find_one({'symbol': symbol, 'group': group}, {'_id': 0, 'symbol': 1,
                                                                                             'margin': 1,
                                                                                             'tick_size': 1,
                                                                                             'fee': 1,
                                                                                             'contract_size': 1,
                                                                                             'group': 1,
                                                                                             "trading_hours": 1})
    if instrument_details is None:
        print(Fore.RED + f"[{symbol}] - This symbol is not registered in the database.\n")
        exit()
    else:
        print(Fore.GREEN + f"[{symbol}]" + Fore.WHITE + " - Get_instrument_details done.\n")
        return instrument_details


def get_stats(trades: pd.DataFrame, capital_line: list):
    try:
        diff_capital = abs(trades["capital"].diff())
        del diff_capital[0]
        sum_diff_capital = sum(diff_capital)
        if sum_diff_capital != 0:
            noise = abs(trades["capital"].iloc[-1] - trades["capital"].iloc[0]) / sum_diff_capital
        else:
            noise = 0
    except:
        noise = 0

    df_aux = pd.DataFrame()
    df_aux["capital"] = capital_line
    df_aux["exit_date"] = pd.to_datetime(trades["exit_date"].shift(1))
    df_aux["exit_date"].iloc[0] = df_aux["exit_date"].iloc[1] - pd.DateOffset(1)
    df_aux.dropna(inplace=True)
    df_aux.set_index('exit_date', inplace=True)

    drawdown_series = to_drawdown_series(df_aux)
    dd_details = drawdown_details(drawdown_series)
    dd_details.reset_index(inplace=True, drop=True)
    dd_details.sort_values(by=[('capital', 'max drawdown')], ascending=True, inplace=True)

    try:
        max_dd = (dd_details["capital"]["max drawdown"].iloc[0] / 100).astype(float) * -1
        drawdown_recovery = (dd_details["capital"]["days"].iloc[0]).astype(float)
    except:
        max_dd = 0
        drawdown_recovery = 0

    cumulative_return = (df_aux["capital"].iloc[-1] / df_aux["capital"].iloc[0]) - 1

    risk_return = cumulative_return / max_dd if max_dd != 0 else 0

    num_trades = len(trades["capital"])

    result_trade = trades['profit_loss'].apply(lambda x: 1 if x > 0 else 0).sum()
    percent_profitable = result_trade / num_trades
    percent_loss = 1 - percent_profitable

    avg_profit = trades['profit_loss'][trades['profit_loss'] > 0].mean()
    avg_loss = trades['profit_loss'][trades['profit_loss'] < 0].mean()
    profit_factor = avg_profit * percent_profitable / (-1 * avg_loss * percent_loss)

    percent_strategy_exit = 0
    percent_stoploss_exit = 0

    for i in range(0, len(trades)):
        if trades.exit_type[i] == 'ExitLong' or trades.exit_type[i] == 'ExitShort':
            percent_strategy_exit += 1
        else:
            percent_stoploss_exit += 1
    if percent_stoploss_exit == 0:
        percent_strategy_exit = percent_strategy_exit / num_trades

    else:
        percent_strategy_exit = percent_strategy_exit / num_trades
        percent_stoploss_exit = 1 - percent_strategy_exit

    mae_mean = trades.mae.mean()
    mae_max = trades.mae.max()
    mae_min = trades.mae.min()

    trades_consectv = (trades['profit_loss'].apply(lambda x: 1 if x > 0 else -1))
    trades_consectv = [(k, sum(1 for k in g)) for k, g in itertools.groupby(trades_consectv)]
    r_trades_aux = []

    for i in trades_consectv:
        r_trades_aux.append(i[0] * i[1])

    max_consectv_wins = max(r_trades_aux)
    max_consectv_loss = min(r_trades_aux) * -1

    long_in_days = (trades.exit_date.iloc[-1] - trades.entry_date[0]) / np.timedelta64(1, 'D')

    cumulative_return = -0.99 if cumulative_return < -1 else cumulative_return

    annual_return = -1 + (cumulative_return + 1) ** (365 / long_in_days)

    results_isi_par_row = pd.DataFrame({
        'cumulative_return': [cumulative_return],
        'max_drawdown': [max_dd],
        'num_trades': [num_trades],
        'percent_profitable': percent_profitable,
        'percent_loss': percent_loss,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'percent_strategy_exit': percent_strategy_exit,
        'percent_stoploss_exit': percent_stoploss_exit,
        'mae_mean': mae_mean,
        'mae_max': mae_max,
        'mae_min': mae_min,
        'risk_return': risk_return,
        'max_consectv_wins': max_consectv_wins,
        'max_consectv_loss': max_consectv_loss,
        'annual_return': annual_return,
        'noise': noise,
        "drawdown_recovery": drawdown_recovery
    })

    print(Fore.GREEN + "\n[SUCCESS]" + Fore.WHITE + " Function get_stats done.\n")

    print(results_isi_par_row)  # borrar esto
    return results_isi_par_row


def get_scores(stats: pd.DataFrame):
    try:
        min_risk_return = stats.risk_return.min()
        diff_risk_return = stats.risk_return.max() - min_risk_return

        min_cumulative_return = stats.cumulative_return.min()
        diff_cumulative_return = stats.cumulative_return.max() - min_cumulative_return

        min_profit_factor = stats.profit_factor.min()
        diff_profit_factor = stats.profit_factor.max() - min_profit_factor

        max_max_drawdown = stats.max_drawdown.max()
        diff_max_drawdown = max_max_drawdown - stats.max_drawdown.min()

        scores = [((0.4 * (r.risk_return - min_risk_return) / diff_risk_return) +
                   (0.2 * (r.cumulative_return - min_cumulative_return) / diff_cumulative_return) +
                   (0.2 * (r.profit_factor - min_profit_factor) / diff_profit_factor) +
                   (0.2 * (max_max_drawdown - r.max_drawdown) / diff_max_drawdown)) for r in stats.itertuples()]

        print(Fore.GREEN + "[SUCCESS]" + Fore.WHITE + " Function get_scores done.\n")
        return scores

    except Exception as e:
        print(e)
        exit()


def pre_vectorization(data: pd.DataFrame, direction: str, instrument_details: dict, parameters: tuple, capital: int,
                      intermarket: bool):
    """
    :param data: Data histórica agrupada
    :param direction: Dirección de la estrategia
    :param instrument_details: Detalles del instrumento, margin, tick_size, fee, etc.
    :param parameters: Combinación de parámetros a optimizar
    :param capital: Capital a usar en la optimización
    :param intermarket: indica si el la funcion prevec se usa para una estrategia de intermarket o no.
    :return: Variables ingresadas transformadas en arreglos y valores aceptados por Numba.
    """
    if direction == 'Long':
        value = 1
    elif direction == 'Short':
        value = -1
    else:
        value = 0

    if instrument_details["group"] == 'Futuros':
        instrument_details["group"] = 1
    elif instrument_details["group"] in ['Acciones', 'ETF', 'ETFx2']:
        instrument_details["group"] = 2
    elif instrument_details["group"] in ['FOREX']:
        if instrument_details["symbol"].endswith("USD"):
            instrument_details["group"] = 3
        elif instrument_details["symbol"].startwith("USD"):
            instrument_details["group"] = 4

    del instrument_details["symbol"]
    del instrument_details["trading_hours"]

    instrument_details = np.array(list(instrument_details.values()), dtype=np.float32)
    np_parameters = np.array(parameters, dtype=np.float32)
    capital = np.float32(capital)

    if not intermarket:
        data = data[['date', 'open', 'high', 'low', 'close', 'volume']].copy()

    dft = data.values.T
    data_timetamps = dft[0]
    data = dft[1:].astype(np.float32)

    return data, data_timetamps, value, instrument_details, np_parameters, capital


# def get_orderbook_data_resampled(symbol, group, time, client):
#     # client = check_production_mode(production_mode=True)
#     if 'T' in time:
#         time = int(time.replace('T', ''))
#     elif 'H' in time:
#         time = int(time.replace('H', '')) * 60
#     start_date = datetime(2020, 1, 1, 0, 0, 0)
#     minute = 60 * 1000
#     label = {'$add': [(time - 1) * minute,
#                       {'$subtract': ["$date",
#                                      {'$mod': [{'$subtract': [{'$add': [minute * -1, "$date"]}, start_date]},
#                                                time * minute]}]}]}
#     resampled_data = client.HistoricalOrderBook.aggregate([
#         {'$match': {'symbol': symbol, 'group': group, 'date': {'$gt': start_date}}},
#         {'$project': {'_id': 0, 'label': label, 'date': 1, 'buys': 1, 'sells': 1, 'open': "$ohlc.open",
#                       'high': "$ohlc.high", 'low': "$ohlc.low", 'close': "$ohlc.close", 'volume': "$ohlc.volume",
#                       'ask_total_volume': 1, 'bid_total_volume': 1, 'buys_volume': 1, 'sells_volume': 1}},
#         {'$group': {'_id': {'new_date': '$label'}, 'buys': {'$sum': "$buys"}, 'sells': {'$sum': "$sells"},
#                     'open': {'$first': "$open"}, 'high': {'$max': "$high"}, 'low': {'$min': "$low"},
#                     'close': {'$last': "$close"}, 'market_volume': {'$sum': "$volume"},
#                     'ask_total_volume': {'$sum': "$ask_total_volume"},
#                     'bid_total_volume': {'$sum': "$bid_total_volume"},
#                     'buys_volume': {'$sum': '$buys_volume'}, 'sells_volume': {'$sum': '$sells_volume'}}},
#         {'$sort': {'_id.new_date': -1}},
#         {'$sort': {'_id.new_date': 1}},
#         {'$project': {'_id': 0, 'date': "$_id.new_date", 'open': 1, 'high': 1, 'low': 1, 'close': 1, 'sells': 1,
#                       'buys': 1, 'ask_total_volume': 1, 'bid_total_volume': 1, 'buys_volume': 1, 'sells_volume': 1,
#                       'buys_sells': {'$add': ['$buys', '$sells']},
#                       'buys_sells_volume': {'$add': ['$buys_volume', '$sells_volume']},
#                       'transaction_volume': {
#                           '$cond': [{'$eq': ['$sells_volume', 0]}, 0, {'$divide': ['$buys_volume', '$sells_volume']}]},
#                       'bid_ask_ratio': {'$divide': ['$bid_total_volume', '$ask_total_volume']},
#                       'transaction_ratio': {'$divide': ['$buys', '$sells']},
#                       'market_volume': {'$toInt': '$market_volume'},
#                       'whale': {'$divide': ['$market_volume', {'$add': ['$ask_total_volume', '$bid_total_volume']}]}}}
#     ], allowDiskUse=True)
#     df_resampled_data = pd.concat([pd.DataFrame([row]) for row in resampled_data], ignore_index=True)
#
#     def get_levels_data(book: str):
#         if book.lower() not in ['ask', 'bid']:
#             raise Exception("book value not recognized. book should be either ask or bid")
#         sort = 1 if book == 'ask' else -1
#         level = {'$add': [{'$multiply': ['$book.index', sort]}, sort]}
#         levels_data = client.HistoricalOrderBook.aggregate([
#             {'$match': {'symbol': symbol, 'group': group, 'date': {'$gt': start_date}}},
#             {'$project': {'_id': 0, 'book': f'${book}', 'date': 1, 'label': label}},
#             {'$sort': {'date': 1}},
#             {'$unwind': '$book'},
#             {'$group': {'_id': {'new_date': "$label", 'price': "$book.price"}, 'volume': {'$sum': "$book.volume"}}},
#             {'$sort': {'_id.new_date': -1, '_id.price': sort}},
#             {'$group': {'_id': {'date': '$_id.new_date'}, f'{book}_total_volume': {'$sum': '$volume'},
#                         'book': {'$push': {'level': '$_id.level', 'price': '$_id.price', 'volume': '$volume'}}}},
#             {'$unwind': {'path': '$book', 'includeArrayIndex': 'book.index'}},
#             {'$group': {'_id': {'date': '$_id.date'},
#                         'book': {'$push': {'price': '$book.price', 'volume': '$book.volume',
#                                            'level': {'$toInt': level}}},
#                         f'{book}_ponderado': {'$sum': {
#                             '$multiply': [{'$divide': ['$book.volume', f'${book}_total_volume']}, '$book.price']}}}},
#             {'$sort': {'_id.date': -1}},
#             {'$sort': {'_id.date': 1}},
#             {'$project': {'_id': 0, 'date': '$_id.date', 'book': 1, f'{book}_ponderado': 1}}
#         ], allowDiskUse=True)
#         df_levels_data = pd.concat([pd.DataFrame([row]) for row in levels_data], ignore_index=True)
#         df_levels_data.rename(columns={'book': book}, inplace=True)
#         return df_levels_data
#
#     df_ask_data = get_levels_data('ask')
#     df_bid_data = get_levels_data('bid')
#     df_resampled_data = df_resampled_data.merge(df_ask_data, on='date')
#     df_resampled_data = df_resampled_data.merge(df_bid_data, on='date')
#     df_resampled_data['order_spread_ask'] = df_resampled_data.ask_ponderado - df_resampled_data.close
#     df_resampled_data['order_spread_bid'] = df_resampled_data.close - df_resampled_data.bid_ponderado
#     df_resampled_data['order_spread_book'] = df_resampled_data.order_spread_bid - df_resampled_data.order_spread_ask
#
#     def calculate_market_level(row: pd.Series):
#         volume = row['volume']
#         book = row['book']
#         curr_ml = book[-1]['level']
#         book_length = len(book)
#         i = 0
#         while volume > 0 & i < book_length:
#             volume -= book[i]['volume']
#             i += 1
#         if volume <= 0:
#             curr_ml = book[i - 1]['level']
#         return curr_ml
#
#     df_subset = df_resampled_data[['bid_ask_ratio', 'ask', 'bid_total_volume']].copy()
#     df_subset.rename(columns={'ask': 'book', 'bid_total_volume': 'volume'}, inplace=True)
#     for idx, row in df_resampled_data[df_resampled_data.bid_ask_ratio >= 1].iterrows():
#         df_subset['book'].iloc[idx] = row['bid']
#         df_subset['volume'].iloc[idx] = row['ask_total_volume']
#     df_resampled_data['market_level'] = df_subset.apply(calculate_market_level, axis=1)
#
#     print(Fore.GREEN + f"[{symbol}]" + Fore.WHITE + " - get_orderbook_data_resampled done.\n")
#     return df_resampled_data

# Convierte temporalidad en minutes: 1H = 60
def convert_time_to_minutes(time):
    if 'T' in time:
        time = int(time.replace('T', ''))
    elif 'H' in time:
        time = int(time.replace('H', '')) * 60

    return time


def get_orderbook_data_resampled(symbol, group, time, client):

    time = convert_time_to_minutes(time=time)

    start_date = datetime(2020, 1, 1, 22, 0, 0)
    minute = 60 * 1000

    label = {'$add': [(time - 1) * minute,
                      {'$subtract': ["$date",
                                     {'$mod': [{'$subtract': [{'$add': [minute * -1, "$date"]}, start_date]},
                                               time * minute]}]}]}

    resampled_data = client.HistoricalOrderBook.aggregate([
        {'$match': {'symbol': symbol, 'group': group, 'date': {'$gt': start_date}}},
        {'$project': {'_id': 0, 'label': label, 'date': 1, 'buys': 1, 'sells': 1, 'open': "$ohlc.open",
                      'high': "$ohlc.high", 'low': "$ohlc.low", 'close': "$ohlc.close", 'volume': "$ohlc.volume",
                      'ask_total_volume': 1, 'bid_total_volume': 1, 'buys_volume': 1, 'sells_volume': 1, 'ofi': 1}},

        {'$group': {'_id': {'new_date': '$label'}, 'buys': {'$sum': "$buys"}, 'sells': {'$sum': "$sells"},
                    'open': {'$first': "$open"}, 'high': {'$max': "$high"}, 'low': {'$min': "$low"},
                    'close': {'$last': "$close"}, 'market_volume': {'$sum': "$volume"}, 'ofi': {'$sum': '$ofi'},
                    'ask_total_volume': {'$sum': "$ask_total_volume"},
                    'bid_total_volume': {'$sum': "$bid_total_volume"},
                    'buys_volume': {'$sum': '$buys_volume'}, 'sells_volume': {'$sum': '$sells_volume'}}},

        {'$sort': {'_id.new_date': -1}},
        {'$sort': {'_id.new_date': 1}},

        {'$project': {'_id': 0, 'date': "$_id.new_date", 'open': 1, 'high': 1, 'low': 1, 'close': 1, 'sells': 1,
                      'buys': 1, 'ask_total_volume': 1, 'bid_total_volume': 1,
                      'buys_volume': 1, 'sells_volume': 1, 'ofi': 1,

                      'buys_sells': {'$add': ['$buys', '$sells']},
                      'buys_sells_volume': {'$add': ['$buys_volume', '$sells_volume']},
                      'transaction_volume': {
                          '$cond': [{'$eq': ['$sells_volume', 0]}, 0, {'$divide': ['$buys_volume', '$sells_volume']}]},
                      'bid_ask_ratio': {'$divide': ['$bid_total_volume', '$ask_total_volume']},
                      'transaction_ratio': {'$divide': ['$buys', '$sells']},
                      'market_volume': {'$toInt': '$market_volume'},
                      'whale': {'$divide': ['$market_volume', {'$add': ['$ask_total_volume', '$bid_total_volume']}]}}}
    ], allowDiskUse=True)

    df_resampled_data = pd.concat([pd.DataFrame([row])
                                   for row in resampled_data], ignore_index=True)

    def get_levels_data(book: str):
        if book.lower() not in ['ask', 'bid']:
            raise Exception(
                "book value not recognized. book should be either ask or bid")

        sort = 1 if book == 'ask' else -1
        level = {'$add': [{'$multiply': ['$book.index', sort]}, sort]}

        levels_data = client.HistoricalOrderBook.aggregate([
            {'$match': {'symbol': symbol, 'group': group, 'date': {'$gt': start_date}}},
            {'$project': {'_id': 0, 'book': f'${book}', 'date': 1, 'label': label}},
            {'$sort': {'date': 1}},
            {'$unwind': '$book'},
            {'$group': {'_id': {'new_date': "$label", 'price': "$book.price"},
                        'volume': {'$sum': "$book.volume"}}},
            {'$sort': {'_id.new_date': -1, '_id.price': sort}},
            {'$group': {'_id': {'date': '$_id.new_date'}, f'{book}_total_volume': {'$sum': '$volume'},
                        'book': {'$push': {'level': '$_id.level', 'price': '$_id.price', 'volume': '$volume'}}}},
            {'$unwind': {'path': '$book', 'includeArrayIndex': 'book.index'}},
            {'$group': {'_id': {'date': '$_id.date'},
                        'book': {'$push': {'price': '$book.price', 'volume': '$book.volume',
                                           'level': {'$toInt': level}}},
                        f'{book}_ponderado': {'$sum': {
                            '$multiply': [{'$divide': ['$book.volume', f'${book}_total_volume']}, '$book.price']}}}},
            {'$sort': {'_id.date': -1}},
            {'$sort': {'_id.date': 1}},
            {'$project': {'_id': 0, 'date': '$_id.date',
                          'book': 1, f'{book}_ponderado': 1}}
        ], allowDiskUse=True)

        df_levels_data = pd.concat([pd.DataFrame([row])
                                    for row in levels_data], ignore_index=True)
        df_levels_data.rename(columns={'book': book}, inplace=True)
        return df_levels_data

    df_ask_data = get_levels_data('ask')
    df_bid_data = get_levels_data('bid')

    df_resampled_data = df_resampled_data.merge(df_ask_data, on='date')
    df_resampled_data = df_resampled_data.merge(df_bid_data, on='date')

    df_resampled_data['order_spread_ask'] = df_resampled_data.ask_ponderado - df_resampled_data.close
    df_resampled_data['order_spread_bid'] = df_resampled_data.close - df_resampled_data.bid_ponderado

    df_resampled_data['order_spread_book'] = df_resampled_data.order_spread_bid - df_resampled_data.order_spread_ask

    def calculate_market_level(row: pd.Series):
        volume = row['volume']
        book = row['book']
        curr_ml = book[-1]['level']
        book_length = len(book)
        i = 0

        while volume > 0 & i < book_length:
            volume -= book[i]['volume']
            i += 1

        if volume <= 0:
            curr_ml = book[i - 1]['level']

        return curr_ml

    df_subset = df_resampled_data[[
        'bid_ask_ratio', 'ask', 'bid_total_volume']].copy()
    df_subset.rename(
        columns={'ask': 'book', 'bid_total_volume': 'volume'}, inplace=True)

    for idx, row in df_resampled_data[df_resampled_data.bid_ask_ratio >= 1].iterrows():
        df_subset['book'].iloc[idx] = row['bid']
        df_subset['volume'].iloc[idx] = row['ask_total_volume']

    df_resampled_data['market_level'] = df_subset.apply(
        calculate_market_level, axis=1)
    return df_resampled_data


# Strategy Specifications
def get_object_variables(instrument_details1: dict, instrument_details2: dict, temporality1: str, temporality2: str, temporality3: str,
                         direction: str, strategy_name: str, parameters: dict):
    # Strategy Object
    object_variables = {'symbols_1': instrument_details1["symbol"],
                        'symbols_2': instrument_details2["symbol"],
                        'temporality1': temporality1,
                        'temporality2': temporality2,
                        'temporality3': temporality3,
                        'fee': instrument_details1["fee"],
                        'tick_size': instrument_details1["tick_size"],
                        'contract_size': instrument_details1["contract_size"],
                        'direction': direction,
                        'capital': 500000,
                        'margin': instrument_details1["margin"],
                        'strategy_name': strategy_name,
                        'parameters': parameters
                        }

    parameter_combinations = list()

    # Uncomment this section if you want to run only a combination of parameters
    aux = object_variables["parameters"].values()

    parameter_combinations.extend(list(itertools.product(*aux)))

    return object_variables, parameter_combinations


def dataframe_branding(numpy_trades, np_data_times):
    df_trades = pd.DataFrame(numpy_trades,
                             columns=['entry_date', 'exit_date', 'entry_price', 'exit_price', 'quantity', 'fee',
                                      'capital', 'mae', 'quantity_active', 'profit_loss', 'exit_type',
                                      'trade_direction', 'stop_loss', 'target_profit'])
    # exit()
    # df_trades.to_csv(f'{"trades_vec"}.csv')
    def data_index(index):
        if np.isnan(index):
            return np.NaN
        else:
            return np_data_times[int(index)]

    if len(df_trades) > 1:
        # Modifiying numba numbers into our target strings

        # 1 means Long, 0 LongSHort and -1 Short
        df_trades['trade_direction'] = df_trades['trade_direction'].replace(
            {1.0: 'Long', 0.: 'LongSHort', -1.: 'Short'})
        # -1 means stoploss, 1 strategy exit
        df_trades['exit_type'] = df_trades['exit_type'].replace({1.0: 'Exit', -1.: 'StopLoss'})
        df_trades['exit_type'] = df_trades['exit_type'] + df_trades['trade_direction']
        # now with entry_date and exit_date
        df_trades['entry_date'] = df_trades['entry_date'].apply(lambda index: data_index(index))
        df_trades['exit_date'] = df_trades['exit_date'].apply(lambda index: data_index(index))

        df_trades['type_trade'] = 'Backtesting'
        # df_trades = df_trades[df_trades['exit_price'] != np.NaN]
        df_trades.reset_index(inplace=True, drop=True)
        print(df_trades)  # borrar esto
    else:
        df_trades = pd.DataFrame()

    # df_trades.to_csv(f'{"trades_vec"}.csv')
    return df_trades


# Run trades, statistics and graphic validation
def optimization(parameter_combinations: list, is_data: pd.DataFrame, instrument_details,
                 capital: str, direction: str, data_graphic: pd.DataFrame, dataframe_branding, strategy,
                 graphic_validation, object_variables, generate_graphic_validation=False):
    if not generate_graphic_validation:

        is_stats = pd.DataFrame()

        for parameters in parameter_combinations:

            print(Fore.CYAN + "[INFO]" + Fore.WHITE + f" Testing parameters: {parameters}")

            np_parameters = parameters
            np_parameters = np.array(np_parameters, dtype=np.float32)

            np_data, data_timetamps, direction, instrument_details, np_parameters, capital = \
                pre_vectorization(data=is_data, direction=direction, instrument_details=instrument_details,
                                  parameters=np_parameters, capital=int(capital), intermarket=True)

            # this executes the strategy as usual
            trades, capital_line = strategy(np_data, np_parameters, instrument_details, capital, direction)

            trades = dataframe_branding(trades, data_timetamps)

            if not trades.empty:
                statistics = get_stats(trades, capital_line)
                statistics["parameters"] = str(parameters)

                is_stats = pd.concat([is_stats, statistics])
            else:
                print(f"Trades empty for the combination {parameters} please check your strategy")

        if len(is_stats) > 1:
            is_stats['score'] = get_scores(is_stats)
            is_stats.sort_values(by='score', ascending=False, inplace=True)
            is_stats.reset_index(inplace=True, drop=True)

            is_stats.to_csv(f'{object_variables["strategy_name"]}_{object_variables["direction"]}_'
                            f'{object_variables["temporality1"]}.csv')

        else:
            is_stats['score'] = 1
            is_stats.to_csv(f'{object_variables["strategy_name"]}_{object_variables["direction"]}_'
                            f'{object_variables["temporality1"]}.csv')

    elif generate_graphic_validation:

        is_stats = pd.DataFrame()

        for parameters in parameter_combinations:

            print(Fore.CYAN + "[INFO]" + Fore.WHITE + f" Testing parameters: {parameters}")

            np_parameters = parameters
            np_parameters = np.array(np_parameters, dtype=np.float32)

            np_data, data_timetamps, direction, instrument_details, np_parameters, capital = \
                pre_vectorization(data=is_data, direction=direction, instrument_details=instrument_details,
                                  parameters=np_parameters, capital=int(capital), intermarket=True)

            # this executes the strategy as usual
            trades, capital_line, np_data_with_indicators = strategy(np_data, np_parameters, instrument_details,
                                                                     capital, direction)

            trades = dataframe_branding(trades, data_timetamps)

            if not trades.empty:
                statistics = get_stats(trades, capital_line)
                statistics["parameters"] = str(parameters)

                is_stats = pd.concat([is_stats, statistics])
            else:
                print(f"Trades empty for the combination {parameters} please check your strategy")

        if len(is_stats) > 1:
            is_stats['score'] = get_scores(is_stats)
            is_stats.sort_values(by='score', ascending=False, inplace=True)
            is_stats.reset_index(inplace=True, drop=True)

            is_stats.to_csv(f'{object_variables["strategy_name"]}_{object_variables["direction"]}_'
                            f'{object_variables["temporality1"]}.csv')

        else:
            is_stats['score'] = 1
            is_stats.to_csv(f'{object_variables["strategy_name"]}_{object_variables["direction"]}_'
                            f'{object_variables["temporality1"]}.csv')

            np_dates = pd.DataFrame(is_data['date'])

        # is_data.to_csv(f'{"data"}.csv')

        graphic_validation(trades, np_dates, np_data_with_indicators, data_graphic, parameter_combinations,
                           object_variables)


def data_strategy(symbol1, symbol2, instrument_details1, instrument_details2, client, object_variables, type_of_strategy: str):
    if type_of_strategy == 'Normal':
        # Get historical data
        data1 = get_data(symbol=symbol1)
        df1 = data1.copy()
        df2 = data1.copy()

        df1.date = pd.to_datetime(df1.date)
        df2.date = pd.to_datetime(df2.date)
        df1 = df1.drop(df1.columns[[0]], axis=1)
        df2 = df2.drop(df2.columns[[0]], axis=1)

        data = []
        resampled_df1 = get_data_resampled(df1, object_variables['temporality1'], instrument_details1, client=client)
        resampled_df2 = get_data_resampled(df2, object_variables['temporality2'], instrument_details1, client=client)
        is_percent = int(len(resampled_df1) * .4)
        data.append(resampled_df1[:is_percent])
        data.append(resampled_df2[(resampled_df2['date'] >= data[0]['date'].iloc[0]) &
                                  (resampled_df2['date'] <= data[0]['date'].iloc[-1])])

        return data

    elif type_of_strategy == 'Intermarket':
        # Get historical data
        data1 = get_data(symbol=symbol1)
        data2 = get_data(symbol=symbol2)

        df1 = data1.copy()
        df2 = data1.copy()
        df3 = data2.copy()

        is_percent = int(len(df1) * 0.4)
        df1 = df1[:is_percent]

        df1.date = pd.to_datetime(df1.date)
        df2.date = pd.to_datetime(df2.date)
        df3.date = pd.to_datetime(df3.date)

        # Filter dates
        min_date_df1 = min(df1['date'])
        min_date_df3 = min(df3['date'])
        min_date = max(min_date_df1, min_date_df3)

        max_date_df1 = max(df1['date'])
        max_date_df3 = max(df3['date'])
        max_date = min(max_date_df1, max_date_df3)

        df1 = df1[(df1['date'] >= min_date) & (df1['date'] <= max_date)]
        df2 = df2[(df2['date'] >= min_date) & (df2['date'] <= max_date)]
        df3 = df3[(df3['date'] >= min_date) & (df3['date'] <= max_date)]

        df1 = df1.drop(df1.columns[[0]], axis=1)
        df2 = df2.drop(df2.columns[[0]], axis=1)
        df3 = df3.drop(df3.columns[[0]], axis=1)

        resampled_df1 = get_data_resampled(df1, object_variables['temporality1'], instrument_details1, client=client)
        resampled_df2 = get_data_resampled(df2, object_variables['temporality2'], instrument_details1, client=client)
        resampled_df3 = get_data_resampled(df3, object_variables['temporality3'], instrument_details2, client=client)

        dates = [resampled_df1['date'], resampled_df3['date']]
        dates = pd.DataFrame(pd.concat(dates))
        dates = dates.drop_duplicates()

        resampled_df1 = pd.merge(resampled_df1, dates, on='date', how='outer').sort_values(by='date')
        resampled_df3 = pd.merge(resampled_df3, dates, on='date', how='outer').sort_values(by='date')

        resampled_df2.sort_values(["date"], ascending=True)

        data = [resampled_df1, resampled_df2, resampled_df3]
        return data

    elif type_of_strategy == 'Intertemp':
        # Get historical data
        data1 = get_data(symbol=symbol1)

        df1 = data1.copy()
        df2 = data1.copy()
        df3 = data1.copy()

        df1.date = pd.to_datetime(df1.date)
        df2.date = pd.to_datetime(df2.date)
        df3.date = pd.to_datetime(df3.date)

        df1 = df1.drop(df1.columns[[0]], axis=1)
        df2 = df2.drop(df2.columns[[0]], axis=1)
        df3 = df3.drop(df3.columns[[0]], axis=1)

        resampled_df1 = get_data_resampled(df1, object_variables['temporality1'], instrument_details1, client=client)
        resampled_df2 = get_data_resampled(df2, object_variables['temporality2'], instrument_details1, client=client)
        resampled_df3 = get_data_resampled(df3, object_variables['temporality3'], instrument_details1, client=client)

        data = [resampled_df1, resampled_df2, resampled_df3]
        return data

    elif type_of_strategy == 'OrderBook':
        # Get historical data
        data1 = get_data(symbol=symbol1)
        # Get Order Book Data
        data_order_book = get_orderbook_data_resampled(symbol=symbol1, group=instrument_details1['group'],
                                                       time=object_variables['temporality3'], client=client)

        # Filter dates
        min_order_book_date = min(data_order_book['date'])
        min_data1_date = min(data1['date'])
        min_date = max(min_order_book_date, min_data1_date)

        max_order_book_date = max(data_order_book['date'])
        max_data1_date = max(data1['date'])
        max_date = min(max_order_book_date, max_data1_date)

        data1 = data1[(data1['date'] >= min_date) & (data1['date'] <= max_date)]
        data_order_book = data_order_book[
            (data_order_book['date'] >= min_date) & (data_order_book['date'] <= max_date)]

        df1 = data1.copy()
        df2 = data1.copy()

        df1.date = pd.to_datetime(df1.date)
        df2.date = pd.to_datetime(df2.date)
        df1 = df1.drop(df1.columns[[0]], axis=1)
        df2 = df2.drop(df2.columns[[0]], axis=1)

        resampled_df1 = get_data_resampled(df1, object_variables['temporality1'], instrument_details1, client=client)
        resampled_df2 = get_data_resampled(df2, object_variables['temporality2'], instrument_details1, client=client)

        data = [resampled_df1, resampled_df2, data_order_book]
        return data

    elif type_of_strategy == 'Sigma':
        # Get historical data
        data1 = get_data(symbol=symbol1)
        df1 = get_dayly_data(client=client, symbol=symbol1, group=instrument_details1['group'])

        df2 = data1.copy()

        df2.date = pd.to_datetime(df2.date)
        df2 = df2.drop(df2.columns[[0]], axis=1)

        resampled_df1 = df1
        resampled_df2 = get_data_resampled(df2, object_variables['temporality1'], instrument_details1, client=client)

        data = [resampled_df1, resampled_df2]
        return data


def strategy_execute(client, symbol1, symbol2, group1, group2, temporality1, temporality2, temporality3, direction,
                     strategy_name, parameters, strategy_type, strategy, dataframe_branding, graphic_validation,
                     backstage, generate_graphic_validation):
    # Get instrument details
    instrument_details1 = get_instrument_details(client=client, symbol=symbol1, group=group1)
    instrument_details2 = get_instrument_details(client=client, symbol=symbol2, group=group2)

    # Get object of the strategy and parameters combination
    # Timeframes are order from major to minor
    object_variables, parameter_combinations = get_object_variables(instrument_details1=instrument_details1,
                                                                    instrument_details2=instrument_details2,
                                                                    temporality1=temporality1,
                                                                    temporality2=temporality2,
                                                                    temporality3=temporality3, direction=direction,
                                                                    strategy_name=strategy_name, parameters=parameters)
    parameter_combinations = list()
    aux = object_variables["parameters"].values()
    parameter_combinations.extend(list(itertools.product(*aux)))

    data = data_strategy(symbol1=symbol1, symbol2=symbol2,
                         instrument_details1=instrument_details1, instrument_details2=instrument_details2,
                         client=client, object_variables=object_variables, type_of_strategy=strategy_type)
    if generate_graphic_validation:
        is_data, data_graphic = backstage(data, parameters=parameter_combinations[0])
    else:
        is_data = backstage(data, parameters=parameter_combinations[0])
        data_graphic = None

    # Optimization between parameters combinations
    # If you want to run the BT on all the signals then change all_signals to True
    optimization(parameter_combinations=parameter_combinations, is_data=is_data, instrument_details=instrument_details1,
                 capital=object_variables["capital"], direction=object_variables["direction"],
                 data_graphic=data_graphic, dataframe_branding=dataframe_branding, strategy=strategy,
                 graphic_validation=graphic_validation,
                 object_variables=object_variables, generate_graphic_validation=generate_graphic_validation)


@njit
def price_with_slippage(price, direction, group, slippage, type_order):
    if group == 1:
        slippage = slippage
        if direction == 1:
            if type_order == "entry":
                slippage_price = price + (slippage * 2)
            elif type_order == "exit":
                slippage_price = price - (slippage * 2)
        elif direction == -1:
            if type_order == "entry":
                slippage_price = price - (slippage * 2)
            elif type_order == "exit":
                slippage_price = price + (slippage * 2)
    elif group == 2:
        slippage = 0.00025
        if direction == 1:
            if type_order == "entry":
                slippage_price = price * (1 + slippage)
            elif type_order == "exit":
                slippage_price = price * (1 - slippage)
        elif direction == -1:
            if type_order == "entry":
                slippage_price = price * (1 - slippage)
            elif type_order == "exit":
                slippage_price = price * (1 + slippage)
    elif group == 3:
        slippage = slippage
        if direction == 1:
            if type_order == "entry":
                slippage_price = price + (slippage * 2)
            elif type_order == "exit":
                slippage_price = price - (slippage * 2)
        elif direction == -1:
            if type_order == "entry":
                slippage_price = price - (slippage * 2)
            elif type_order == "exit":
                slippage_price = price + (slippage * 2)

    return slippage_price


def candle_stick_chart(is_data, trades, object_variables, candle_size=0.15):
    # Candlestick
    inc = is_data.close > is_data.open
    dec = is_data.open > is_data.close
    w = (17 * 30 * 30 * 25) * candle_size

    if trades.empty:
        trades = pd.DataFrame(columns=['entry_date', 'exit_date', 'entry_price', 'exit_price', 'quantity', 'fee',
                                          'capital', 'mae', 'quantity_active', 'profit_loss', 'exit_type',
                                          'trade_direction'])

    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    source = ColumnDataSource(trades)
    source_is_data = ColumnDataSource(is_data)

    crosshair = CrosshairTool(dimensions='both')

    def clean_text(text, chars='{}[]"'):
        for char in chars:
            text = text.replace(char, "")
        return text

    parameters = clean_text(json.dumps(object_variables["parameters"]))

    # Figures
    p = figure(x_axis_type="datetime", plot_height=500, plot_width=1000,
               title=f'Symbol: {object_variables["symbols_1"]}, '
                     f'Timeframe: {object_variables["temporality1"]}, '
                     f'Parameters: {parameters}, '
                     f'Direction: {object_variables["direction"]}')

    bar_high_low = p.segment("date", "high", "date", "low", color="black", source=source_is_data)
    bar_open = p.vbar(is_data.date[inc], w, is_data.open[inc], is_data.close[inc], fill_color="green",
                      line_color="green")
    bar_close = p.vbar(is_data.date[dec], w, is_data.open[dec], is_data.close[dec], fill_color="red",
                       line_color="red")
    # OHLC Hoover
    ohlc_hover = HoverTool(
        renderers=[bar_high_low],
        tooltips=[
            ('date', '@date{%Y-%m-%d %H:%M:%S}'),
            ('open', '$@open{0,0.00}'),
            ('high', '$@high{0,0.00}'),
            ('low', '$@low{0,0.00}'),
            ('close', '$@close{0,0.00}'),
            ('volume', '@volume{0,0}'),
        ],

        formatters={
            '@date': 'datetime',  # use 'datetime' formatter for '@date' field
            # use default 'numeral' formatter for other fields
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='mouse'
    )

    entry_circle = p.circle('entry_date', 'entry_price', fill_color="lime", line_color="darkgreen",
                            legend_label='Entry condition',
                            size=12, fill_alpha=0.8, source=source, name='entry_condition')

    entry_tooltip = """
    <div class = "entry_tooltip">
        <div>
            <span style="font-size: 12px; font-weight: bold; color: #696"> Entry condition</span>
        </div>                
        <div>
            <span style="font-size: 10px; font-weight: bold;">Entry date: </span>
            <span style="font-size: 10px;">@entry_date{%Y-%m-%d %H:%M:%S}</span>
        </div>
        <div>
            <span style="font-size: 10px; font-weight: bold;">Entry price: </span>
            <span style="font-size: 10px;">$@entry_price{0,0.00}</span>
        </div>
    </div> 
    """
    p_hover_entry = HoverTool(
        name="entry_tooltip",
        renderers=[entry_circle],
        tooltips=entry_tooltip,
        formatters={'entry_date': 'datetime'},
        mode='mouse'
    )

    exit_circle = p.circle('exit_date', 'exit_price', fill_color="red", line_color="darkred",
                           legend_label='Exit condition',
                           size=12, fill_alpha=0.8, source=source, name='exit_condition')

    exit_tooltip = """
    <div class = "exit_tooltip">
        <div>
            <span style="font-size: 12px; font-weight: bold; color: #FF0000"> Exit condition</span>
        </div>                
        <div>
            <span style="font-size: 10px; font-weight: bold;">Exit date: </span>
            <span style="font-size: 10px;">@exit_date{%Y-%m-%d %H:%M:%S}</span>
        </div>
        <div>
            <span style="font-size: 10px; font-weight: bold;">Exit price: </span>
            <span style="font-size: 10px;">$@exit_price{0,0.00}</span>
        </div>
            <div>
            <span style="font-size: 10px; font-weight: bold;">Profit/Loss: </span>
            <span style="font-size: 10px;">$@profit_loss{0,0.00}</span>
        </div>
            <div>
            <span style="font-size: 10px; font-weight: bold;">Exit type: </span>
            <span style="font-size: 10px;">@exit_type</span>
        </div>
    </div> 
    """
    p_hover_exit = HoverTool(
        renderers=[exit_circle],
        tooltips=exit_tooltip,
        formatters={'exit_date': 'datetime'},
        mode='mouse'
    )
    # Axis
    p.xaxis.axis_label = 'TIME'
    p.yaxis.axis_label = 'PRICE'
    len_is_data = len(is_data)
    nans_profit_target = np.count_nonzero(np.isnan(is_data['profit_target_vector']))
    nans_stop_loss = np.count_nonzero(np.isnan(is_data['stop_loss_vector']))
    nans_trailing_stop = np.count_nonzero(np.isnan(is_data['trailing_stop']))

    plot_profit_target = len_is_data != nans_profit_target
    plot_stop_loss = len_is_data != nans_stop_loss
    plot_trailing_stop = len_is_data != nans_trailing_stop

    if plot_stop_loss:
        # lines
        stop_loss = p.line("date", "stop_loss_vector", line_color="red", legend_label='Stop Loss', line_width=2,
                           source=source_is_data)

        stop_loss_hover = HoverTool(
            renderers=[stop_loss],
            tooltips=[
                ('date', '@date{%Y-%m-%d %H:%M:%S}'),
                ('Stop Loss', '$@stop_loss_vector{0,0.00}'),
            ],

            formatters={
                '@date': 'datetime',  # use 'datetime' formatter for '@date' field
                # use default 'numeral' formatter for other fields
            },

            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode='vline'
        )
        p.add_tools(stop_loss_hover)

    if plot_profit_target:

        target_profit = p.line("date", "profit_target_vector", line_color="green", legend_label='Profit Target',
                               line_width=2,
                               source=source_is_data)

        target_profit_hover = HoverTool(
            renderers=[target_profit],
            tooltips=[
                ('date', '@date{%Y-%m-%d %H:%M:%S}'),
                ('Profit Target', '$@profit_target_vector{0,0.00}'),
            ],

            formatters={
                '@date': 'datetime',  # use 'datetime' formatter for '@date' field
                # use default 'numeral' formatter for other fields
            },

            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode='vline'
        )
        p.add_tools(target_profit_hover)

    if plot_trailing_stop:
        # lines
        trailing_stop = p.line("date", "trailing_stop", line_color="orange", legend_label='Trailing Stop', line_width=2,
                               source=source_is_data)

        stop_loss_hover = HoverTool(
            renderers=[trailing_stop],
            tooltips=[
                ('date', '@date{%Y-%m-%d %H:%M:%S}'),
                ('Trailing Stop', '$@trailing_stop{0,0.00}'),
            ],

            formatters={
                '@date': 'datetime',  # use 'datetime' formatter for '@date' field
                # use default 'numeral' formatter for other fields
            },

            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode='vline'
        )
        p.add_tools(stop_loss_hover)

    # Tools
    p.add_tools(p_hover_entry, p_hover_exit, ohlc_hover, crosshair)
    return p


@njit
def minute_exit(index, next_open_price, position_size, instrument_contract_size, entry_position_size_usd, total_fee,
                direction, instrument_type, instrument_tick_size, trade_index, capital, capital_line, trades_line,
                trade_number, size, exit_strategy: bool):
    """
    :param index: actual position in the array.
    :param next_open_price: np.array, the Open prices from the series. The trade is established in the next open
    price (open[i + 1]).
    :param position_size: position size value.
    :param instrument_contract_size: instrument contract size value.
    :param entry_position_size_usd: entry position size amount in dollars.
    :param total_fee: total fee value.
    :param direction: 1 for long direction or 2 for short direction
    :param instrument_type: 1 for Futures or 2 for Stocks or ETF
    :param instrument_tick_size: instrument tick size
    :param trade_index: principal temporality index
    :param capital: capital
    :param capital_line: capital line
    :param trades_line: register of trades
    :param trade_number: trade number
    :param size: size of trades_line
    :param exit_strategy: type of exit
    :return: trades_line, capital, capital_line, exit_index arrays
    """

    if direction == 1:

        final_price_to_exit = next_open_price

        exit_price = price_with_slippage(final_price_to_exit, 1, instrument_type, instrument_tick_size, "exit")
        exit_position_size_usd = position_size * instrument_contract_size * exit_price
        profit_loss = exit_position_size_usd - entry_position_size_usd - total_fee
        exit_date = index + 1

        exit_index = trade_index[index]

        capital += profit_loss

        if capital < 0:
            capital_line[trade_number + 1] = np.NaN

            trades_line[trade_number * size + 1] = np.NaN
            trades_line[trade_number * size + 3] = np.NaN
            trades_line[trade_number * size + 6] = np.NaN
            trades_line[trade_number * size + 7] = np.NaN
            trades_line[trade_number * size + 9] = np.NaN
            if profit_loss > 0 or exit_strategy:
                trades_line[trade_number * size + 10] = np.NaN  # -1 means stop_loss_price, 1 strategy exit
            else:
                trades_line[trade_number * size + 10] = np.NaN  # -1 means stop_loss_price, 1 strategy exit
            trades_line[trade_number * size + 11] = np.NaN  # 1 means Long, 0 LongSHort and -1 Short
        else:
            capital_line[trade_number + 1] = capital

            trades_line[trade_number * size + 1] = exit_date
            trades_line[trade_number * size + 3] = exit_price
            trades_line[trade_number * size + 6] = capital
            trades_line[trade_number * size + 7] = 0
            trades_line[trade_number * size + 9] = profit_loss
            if profit_loss > 0 or exit_strategy:
                trades_line[trade_number * size + 10] = 1  # -1 means stop_loss_price, 1 strategy exit
            else:
                trades_line[trade_number * size + 10] = -1  # -1 means stop_loss_price, 1 strategy exit
            trades_line[trade_number * size + 11] = 1  # 1 means Long, 0 LongSHort and -1 Short

        return trades_line, capital, capital_line, exit_index

    if direction == -1:

        final_price_to_exit = next_open_price

        exit_price = price_with_slippage(final_price_to_exit, -1, instrument_type, instrument_tick_size, "exit")
        exit_position_size_usd = position_size * instrument_contract_size * exit_price
        profit_loss = ((exit_position_size_usd - entry_position_size_usd) * -1) - total_fee
        exit_date = index + 1

        exit_index = trade_index[index]

        capital += profit_loss

        if capital < 0:

            capital_line[trade_number + 1] = np.NaN

            trades_line[trade_number * size + 1] = np.NaN
            trades_line[trade_number * size + 3] = np.NaN
            trades_line[trade_number * size + 6] = np.NaN
            trades_line[trade_number * size + 7] = np.NaN
            trades_line[trade_number * size + 9] = np.NaN
            if profit_loss > 0 or exit_strategy:
                trades_line[trade_number * size + 10] = np.NaN  # -1 means stop_loss_price, 1 strategy exit
            else:
                trades_line[trade_number * size + 10] = np.NaN  # -1 means stop_loss_price, 1 strategy exit
            trades_line[trade_number * size + 11] = np.NaN  # 1 means Long, 0 LongSHort and -1 Short

        else:
            capital_line[trade_number + 1] = capital

            trades_line[trade_number * size + 1] = exit_date
            trades_line[trade_number * size + 3] = exit_price
            trades_line[trade_number * size + 6] = capital
            trades_line[trade_number * size + 7] = 0
            trades_line[trade_number * size + 9] = profit_loss
            if profit_loss > 0 or exit_strategy:
                trades_line[trade_number * size + 10] = 1  # -1 means stop_loss_price, 1 strategy exit
            else:
                trades_line[trade_number * size + 10] = -1  # -1 means stop_loss_price, 1 strategy exit
            trades_line[trade_number * size + 11] = -1  # 1 means Long, 0 LongSHort and -1 Short

        return trades_line, capital, capital_line, exit_index


@njit
def profit_target_value(entry_price, profit_target_param, direction):
    if direction == 1:
        profit_target_price = entry_price * (1 + profit_target_param)
    elif direction == -1:
        profit_target_price = entry_price * (1 - profit_target_param)
    return profit_target_price


@njit
def stop_loss_value(entry_price, stop_loss_param, direction):
    if direction == 1:
        stop_loss_price = entry_price * (1 - stop_loss_param)
    elif direction == -1:
        stop_loss_price = entry_price * (1 + stop_loss_param)
    return stop_loss_price


@njit
def minute_entry(index, next_open_price, profit_target_param, stop_loss_param, instrument_contract_size,
                 instrument_margin, direction, instrument_type, instrument_tick_size, trade_index, capital, capital_0,
                 instrument_fee, trades_line, trade_number, size, logger=True):
    entry_date = index + 1
    entry_price = price_with_slippage(price=next_open_price, direction=direction, group=instrument_type,
                                      slippage=instrument_tick_size, type_order="entry")
    profit_target_price = profit_target_value(entry_price=entry_price, profit_target_param=profit_target_param,
                                              direction=direction)
    stop_loss_price = stop_loss_value(entry_price=entry_price, stop_loss_param=stop_loss_param, direction=direction)
    entry_index = trade_index[index]

    # position size logic:
    if instrument_type == 4:
        position_size = np.round((capital_0 * 0.01) / ((1/entry_price) * stop_loss_param * instrument_contract_size))
    else:
        position_size = np.round((capital_0 * 0.01) / (entry_price * stop_loss_param * instrument_contract_size))

    # # For graphic validation
    # trailing_stop[i] = stop_loss_price

    entry_position_size_usd = position_size * instrument_contract_size * entry_price

    # 1 Means Futures
    if instrument_type == 1:
        if capital < instrument_margin:
            if logger:
                with objmode():
                    get_logging(stage=1, message='EWM breaking execution, capital is < instrument margin!')

            raise EwmErrors('EWM breaking execution, capital is < instrument margin!')

        if position_size == 0:
            position_size = 1
            entry_position_size_usd = position_size * instrument_contract_size * entry_price

        total_fee = instrument_fee * 2 * position_size
        margin_total = instrument_margin * position_size

    # 2 Means Stocks, ETF
    elif instrument_type == 2:

        if entry_position_size_usd > capital:
            position_size = np.floor(capital_0 / (instrument_contract_size * entry_price))
            entry_position_size_usd = position_size * instrument_contract_size * entry_price

        if position_size == 0:
            if logger:
                with objmode():
                    get_logging(stage=1, message='EWM breaking execution, position size is 0!')

            raise EwmErrors('EWM breaking execution, position size is 0!')

        # We assume the max amount to pay in fees is 1% of the total. According to current rules.
        # This may change in the future without notice.
        total_fee = min(max(instrument_fee * position_size, 1), entry_price * position_size * 0.01)
        margin_total = np.NaN

        # 1 Means Forex
    elif instrument_type == 3 or instrument_type == 4:
        if capital < instrument_margin:
            if logger:
                with objmode():
                    get_logging(stage=1, message='EWM breaking execution, capital is < instrument margin!')

            raise EwmErrors('EWM breaking execution, capital is < instrument margin!')

        if position_size == 0:
            position_size = 1
            if instrument_type == 4:
                entry_position_size_usd = position_size * instrument_contract_size * (1/entry_price)
            else:
                entry_position_size_usd = position_size * instrument_contract_size * entry_price

        total_fee = instrument_fee * 2 * position_size
        margin_total = instrument_margin * position_size

    # np_aux NEEDS to be in this order:
    # 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'quantity',
    # 'fee', 'capital','mae', 'quantity_active', 'profit_loss', 'exit_type', 'trade_direction'

    trades_line[trade_number * size: (trade_number + 1) * size] = np.array(
        (entry_date, np.NaN, entry_price, np.NaN, position_size, total_fee,
         np.NaN, np.NaN, margin_total, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN))

    return trades_line, entry_position_size_usd, entry_index, profit_target_price, stop_loss_price, total_fee, \
           position_size


@njit
def exit_type(exit_type_str: str):
    if exit_type_str == "sl_strategy":
        exit_type_value = 1
    elif exit_type_str == "sl_tp":
        exit_type_value = 2
    elif exit_type_str == "trailing":
        exit_type_value = 3
    elif exit_type_str == "sl_tp_time":
        exit_type_value = 4
    elif exit_type_str == "signals":
        exit_type_value = 5


    return exit_type_value


@njit
def graphic_validation_vectors(len_np_data):

    # VECTORS FOR GRAPHIC VALIDATION
    # Stop Loss vector
    stop_loss_vector = np.zeros(len_np_data) * np.NaN
    stop_loss_price = np.NaN
    # Profit Target vector
    profit_target_vector = np.zeros(len_np_data) * np.NaN
    profit_target_price = np.NaN
    # Trailing Stop vector
    trailing_stop = np.zeros(len_np_data) * np.NaN
    return stop_loss_vector, stop_loss_price, profit_target_vector, profit_target_price, trailing_stop


@njit
def stop_loss_condition(close, stop_loss_price, direction):

    if close <= stop_loss_price and direction == 1:
        condition = True
    elif close >= stop_loss_price and direction == -1:
        condition = True
    else:
        condition = False

    return condition


@njit
def profit_target_condition(close, profit_target_price, direction):

    if close >= profit_target_price and direction == 1:
        condition = True
    elif close <= profit_target_price and direction == -1:
        condition = True
    else:
        condition = False

    return condition


@njit
def trailing_stop_update(direction, close, stop_loss_param, stop_loss_price):
    if direction == 1:
        trailing = close - (close * stop_loss_param)
        stop_loss_price = max(trailing, stop_loss_price)
    elif direction == -1:
        trailing = close + (close * stop_loss_param)
        stop_loss_price = min(trailing, stop_loss_price)

    return stop_loss_price


def backstage_data_copies(data, strategy_type):

    if strategy_type == "order_book_strategy":
        # Creating copies:
        order_book_df = data[2].copy()
        order_book_df = order_book_df[['date', 'open', 'high', 'low', 'close', 'ofi',
                                       'buys', 'buys_volume',
                                       'sells', 'sells_volume',
                                       'buys_sells', 'buys_sells_volume',
                                       'ask', 'ask_total_volume',
                                       'bid', 'bid_total_volume',
                                       'ask_ponderado', 'bid_ponderado',
                                       'bid_ask_ratio', 'transaction_ratio',
                                       'transaction_volume', 'market_volume',
                                       'whale', 'market_level',
                                       'order_spread_ask', 'order_spread_bid', 'order_spread_book']]

        df1 = data[0].copy()
        df2 = data[1].copy()
        order_book_df.date = pd.to_datetime(order_book_df.date)
        df1.date = pd.to_datetime(df1.date)
        df2.date = pd.to_datetime(df2.date)
        return df1, df2, None, order_book_df

    elif strategy_type == "normal" or strategy_type == "sigma":
        df1 = data[0].copy()
        df2 = data[1].copy()
        df1.date = pd.to_datetime(df1.date)
        df2.date = pd.to_datetime(df2.date)
        return df1, df2, None, None

    elif strategy_type == "intermarket" or  strategy_type == "intertemp":
        df1 = data[0].copy()
        df2 = data[1].copy()
        df3 = data[2].copy()
        df1.date = pd.to_datetime(df1.date)
        df2.date = pd.to_datetime(df2.date)
        df3.date = pd.to_datetime(df3.date)
        return df1, df2, df3, None


def backstage_data_frame_joining(df1, df2, df3, order_book_df, strategy_type):
    # Joining the dataframes:
    if strategy_type == "order_book_strategy":
        order_book_df.set_index('date', inplace=True)
        df1.set_index('date', inplace=True)
        df2.set_index('date', inplace=True)
        df1.columns = df1.columns.map(lambda x: str(x) + '_x')
        df2.columns = df2.columns.map(lambda x: str(x) + '_y')
        order_book_df.columns = order_book_df.columns.map(lambda x: str(x) + '_z')
        df = order_book_df.join(df1, how='outer')
        final_df = df.join(df2, how='outer')
        final_df['trade_index'] = final_df['trade_index_x']
        final_df.reset_index(inplace=True)
        final_df['volume_z'] = np.zeros(len(final_df))
        # final_df = final_df.rename(columns={"returns_y": "returns"})

        return final_df
    elif strategy_type == 'normal' or strategy_type == 'sigma':

        df1.set_index('date', inplace=True)
        df2.set_index('date', inplace=True)
        df3 = pd.DataFrame().reindex_like(df2)
        df1.columns = df1.columns.map(lambda x: str(x) + '_x')
        df2.columns = df2.columns.map(lambda x: str(x) + '_y')
        df3.columns = df3.columns.map(lambda x: str(x) + '_z')
        df = df1.join(df2, how='outer')
        final_df = df.join(df3, how='outer')
        final_df['trade_index'] = final_df['trade_index_x']
        final_df.reset_index(inplace=True)
        # final_df = final_df.rename(columns={"returns_y": "returns"})

        return final_df
    elif strategy_type == "intermarket" or strategy_type == 'intertemp':
        df1.set_index('date', inplace=True)
        df2.set_index('date', inplace=True)
        df3.set_index('date', inplace=True)
        df1.columns = df1.columns.map(lambda x: str(x) + '_x')
        df2.columns = df2.columns.map(lambda x: str(x) + '_y')
        df3.columns = df3.columns.map(lambda x: str(x) + '_z')
        df = df1.join(df2, how='outer')
        final_df = df.join(df3, how='outer')
        final_df['trade_index'] = final_df['trade_index_x']
        final_df.reset_index(inplace=True)
        # final_df = final_df.rename(columns={"returns_y": "returns"})
        return final_df


def inter_graphic_validation(data_graphic, np_dates, indicators, strategy_type, trades, object_variables, candle_size):

    if strategy_type == "intermarket" or strategy_type == "intertemp":
        data_graphic_1 = data_graphic.copy()
        data_graphic_2 = data_graphic.copy()

        data_graphic_1 = data_graphic_1[['date', 'open_x', 'high_x', 'low_x', 'close_x', 'volume_x']]
        data_graphic_1.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

        data_graphic_2 = data_graphic_2[['date', 'open_z', 'high_z', 'low_z', 'close_z', 'volume_z']]
        data_graphic_2.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

        data_graphic_1.date = pd.to_datetime(data_graphic_1.date)
        data_graphic_2.date = pd.to_datetime(data_graphic_2.date)

        is_data_indicators = pd.concat([np_dates, indicators], axis=1)

        is_data_indicators.date = pd.to_datetime(is_data_indicators.date)

        is_data_indicators.set_index('date', inplace=True)

        data_graphic_1.set_index('date', inplace=True)

        is_data = data_graphic_1.join(is_data_indicators, how='left')

        is_data.reset_index(inplace=True)

        indicators2 = pd.DataFrame().reindex_like(indicators)
        is_data_2 = data_graphic_2.join(indicators2, how='left')

        is_data_2.reset_index(inplace=True)

        p_1 = candle_stick_chart(is_data=is_data, trades=trades, object_variables=object_variables,
                                 candle_size=candle_size)

        trades_empty = pd.DataFrame().reindex_like(trades)

        p_2 = candle_stick_chart_2(is_data=is_data_2, trades=trades_empty, object_variables=object_variables,
                                   candle_size=candle_size)
        return p_1, p_2, is_data

    elif strategy_type == "order_book_strategy" or strategy_type == "normal" or strategy_type == "sigma":

        data_graphic = data_graphic[['date', 'open_x', 'high_x', 'low_x', 'close_x', 'volume_x']]
        data_graphic.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

        data_graphic.date = pd.to_datetime(data_graphic.date)

        is_data_indicators = pd.concat([np_dates, indicators], axis=1)

        is_data_indicators.date = pd.to_datetime(is_data_indicators.date)

        is_data_indicators.set_index('date', inplace=True)

        data_graphic.set_index('date', inplace=True)

        is_data = data_graphic.join(is_data_indicators, how='left')

        is_data.reset_index(inplace=True)

        p = candle_stick_chart(is_data=is_data, trades=trades, object_variables=object_variables,
                               candle_size=candle_size)

        return p, None, is_data


def candle_stick_chart_2(is_data, trades, object_variables, candle_size):
    # Candlestick
    inc = is_data.close > is_data.open
    dec = is_data.open > is_data.close
    w = (17 * 30 * 30 * 25) * candle_size

    if trades.empty:
        trades = pd.DataFrame(columns=['entry_date', 'exit_date', 'entry_price', 'exit_price', 'quantity', 'fee',
                                       'capital', 'mae', 'quantity_active', 'profit_loss', 'exit_type',
                                       'trade_direction'])

    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    source = ColumnDataSource(trades)
    source_is_data = ColumnDataSource(is_data)

    crosshair = CrosshairTool(dimensions='both')

    def clean_text(text, chars='{}[]"'):
        for char in chars:
            text = text.replace(char, "")
        return text

    parameters = clean_text(json.dumps(object_variables["parameters"]))

    # Figures
    p = figure(x_axis_type="datetime", plot_height=500, plot_width=1000,
               title=f'Symbol: {object_variables["symbols_2"]}, '
                     f'Timeframe: {object_variables["temporality3"]}, '
                     f'Parameters: {parameters}, '
                     f'Direction: {object_variables["direction"]}')

    bar_high_low = p.segment("date", "high", "date", "low", color="black", source=source_is_data)
    bar_open = p.vbar(is_data.date[inc], w, is_data.open[inc], is_data.close[inc], fill_color="green",
                      line_color="green")
    bar_close = p.vbar(is_data.date[dec], w, is_data.open[dec], is_data.close[dec], fill_color="red",
                       line_color="red")
    # OHLC Hoover
    ohlc_hover = HoverTool(
        renderers=[bar_high_low],
        tooltips=[
            ('date', '@date{%Y-%m-%d %H:%M:%S}'),
            ('open', '$@open{0,0.00}'),
            ('high', '$@high{0,0.00}'),
            ('low', '$@low{0,0.00}'),
            ('close', '$@close{0,0.00}'),
            ('volume', '@volume{0,0}'),
        ],

        formatters={
            '@date': 'datetime',  # use 'datetime' formatter for '@date' field
            # use default 'numeral' formatter for other fields
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='mouse'
    )

    entry_circle = p.circle('entry_date', 'entry_price', fill_color="lime", line_color="darkgreen",
                            legend_label='Entry condition',
                            size=12, fill_alpha=0.8, source=source, name='entry_condition')

    entry_tooltip = """
    <div class = "entry_tooltip">
        <div>
            <span style="font-size: 12px; font-weight: bold; color: #696"> Entry condition</span>
        </div>                
        <div>
            <span style="font-size: 10px; font-weight: bold;">Entry date: </span>
            <span style="font-size: 10px;">@entry_date{%Y-%m-%d %H:%M:%S}</span>
        </div>
        <div>
            <span style="font-size: 10px; font-weight: bold;">Entry price: </span>
            <span style="font-size: 10px;">$@entry_price{0,0.00}</span>
        </div>
    </div> 
    """
    p_hover_entry = HoverTool(
        name="entry_tooltip",
        renderers=[entry_circle],
        tooltips=entry_tooltip,
        formatters={'entry_date': 'datetime'},
        mode='mouse'
    )

    exit_circle = p.circle('exit_date', 'exit_price', fill_color="red", line_color="darkred",
                           legend_label='Exit condition',
                           size=12, fill_alpha=0.8, source=source, name='exit_condition')

    exit_tooltip = """
    <div class = "exit_tooltip">
        <div>
            <span style="font-size: 12px; font-weight: bold; color: #FF0000"> Exit condition</span>
        </div>                
        <div>
            <span style="font-size: 10px; font-weight: bold;">Exit date: </span>
            <span style="font-size: 10px;">@exit_date{%Y-%m-%d %H:%M:%S}</span>
        </div>
        <div>
            <span style="font-size: 10px; font-weight: bold;">Exit price: </span>
            <span style="font-size: 10px;">$@exit_price{0,0.00}</span>
        </div>
            <div>
            <span style="font-size: 10px; font-weight: bold;">Profit/Loss: </span>
            <span style="font-size: 10px;">$@profit_loss{0,0.00}</span>
        </div>
            <div>
            <span style="font-size: 10px; font-weight: bold;">Exit type: </span>
            <span style="font-size: 10px;">@exit_type</span>
        </div>
    </div> 
    """
    p_hover_exit = HoverTool(
        renderers=[exit_circle],
        tooltips=exit_tooltip,
        formatters={'exit_date': 'datetime'},
        mode='mouse'
    )
    # Axis
    p.xaxis.axis_label = 'TIME'
    p.yaxis.axis_label = 'PRICE'
    len_is_data = len(is_data)
    nans_profit_target = np.count_nonzero(np.isnan(is_data['profit_target_vector']))
    nans_stop_loss = np.count_nonzero(np.isnan(is_data['stop_loss_vector']))
    nans_trailing_stop = np.count_nonzero(np.isnan(is_data['trailing_stop']))

    plot_profit_target = len_is_data != nans_profit_target
    plot_stop_loss = len_is_data != nans_stop_loss
    plot_trailing_stop = len_is_data != nans_trailing_stop

    if plot_stop_loss:
        # lines
        stop_loss = p.line("date", "stop_loss_vector", line_color="red", legend_label='Stop Loss', line_width=2,
                           source=source_is_data)

        stop_loss_hover = HoverTool(
            renderers=[stop_loss],
            tooltips=[
                ('date', '@date{%Y-%m-%d %H:%M:%S}'),
                ('Stop Loss', '$@stop_loss_vector{0,0.00}'),
            ],

            formatters={
                '@date': 'datetime',  # use 'datetime' formatter for '@date' field
                # use default 'numeral' formatter for other fields
            },

            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode='vline'
        )
        p.add_tools(stop_loss_hover)

    if plot_profit_target:

        target_profit = p.line("date", "profit_target_vector", line_color="green", legend_label='Profit Target',
                               line_width=2,
                               source=source_is_data)

        target_profit_hover = HoverTool(
            renderers=[target_profit],
            tooltips=[
                ('date', '@date{%Y-%m-%d %H:%M:%S}'),
                ('Profit Target', '$@profit_target_vector{0,0.00}'),
            ],

            formatters={
                '@date': 'datetime',  # use 'datetime' formatter for '@date' field
                # use default 'numeral' formatter for other fields
            },

            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode='vline'
        )
        p.add_tools(target_profit_hover)

    if plot_trailing_stop:
        # lines
        trailing_stop = p.line("date", "trailing_stop", line_color="orange", legend_label='Trailing Stop', line_width=2,
                               source=source_is_data)

        stop_loss_hover = HoverTool(
            renderers=[trailing_stop],
            tooltips=[
                ('date', '@date{%Y-%m-%d %H:%M:%S}'),
                ('Trailing Stop', '$@trailing_stop{0,0.00}'),
            ],

            formatters={
                '@date': 'datetime',  # use 'datetime' formatter for '@date' field
                # use default 'numeral' formatter for other fields
            },

            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode='vline'
        )
        p.add_tools(stop_loss_hover)

    # Tools
    p.add_tools(p_hover_entry, p_hover_exit, ohlc_hover, crosshair)
    return p


def relative_strength_candle_stick_chart(data_graphic, trades, object_variables, candle_size):
    data_graphic_rs = data_graphic.copy()

    data_graphic_rs = data_graphic_rs[['date', 'rs_open_x', 'rs_high_x', 'rs_low_x', 'rs_close_x']]
    data_graphic_rs.columns = ['date', 'rs_open', 'rs_high', 'rs_low', 'rs_close']

    # Candlestick
    inc_rs = data_graphic_rs.rs_close > data_graphic_rs.rs_open
    dec_rs = data_graphic_rs.rs_open > data_graphic_rs.rs_close

    w = (17 * 30 * 30 * 25) * candle_size

    if trades.empty:
        trades = pd.DataFrame(columns=['entry_date', 'exit_date', 'entry_price', 'exit_price', 'quantity', 'fee',
                                       'capital', 'mae', 'quantity_active', 'profit_loss', 'exit_type',
                                       'trade_direction'])

    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    source = ColumnDataSource(trades)
    source_is_data = ColumnDataSource(data_graphic_rs)

    crosshair = CrosshairTool(dimensions='both')

    def clean_text(text, chars='{}[]"'):
        for char in chars:
            text = text.replace(char, "")
        return text

    parameters = clean_text(json.dumps(object_variables["parameters"]))

    p_rs = figure(x_axis_type="datetime", plot_height=500, plot_width=1000,
                  title=f'Relative Strength: {object_variables["symbols_1"]} / {object_variables["symbols_2"]}, '
                        f'Timeframe: {object_variables["temporality1"]}, '
                        f'Direction: {object_variables["direction"]}')

    bar_high_low = p_rs.segment("date", "rs_high", "date", "rs_low", color="black", source=source_is_data)
    bar_open = p_rs.vbar(data_graphic_rs.date[inc_rs], w, data_graphic_rs.rs_open[inc_rs],
                         data_graphic_rs.rs_close[inc_rs], fill_color="green", line_color="green")
    bar_close = p_rs.vbar(data_graphic_rs.date[dec_rs], w, data_graphic_rs.rs_open[dec_rs],
                          data_graphic_rs.rs_close[dec_rs], fill_color="red", line_color="red")
    # OHLC Hoover
    ohlc_hover = HoverTool(
        renderers=[bar_high_low],
        tooltips=[
            ('date', '@date{%Y-%m-%d %H:%M:%S}'),
            ('open', '$@rs_open{0,0.00000}'),
            ('high', '$@rs_high{0,0.00000}'),
            ('low', '$@rs_low{0,0.00000}'),
            ('close', '$@rs_close{0,0.00000}'),
        ],

        formatters={
            '@date': 'datetime',  # use 'datetime' formatter for '@date' field
            # use default 'numeral' formatter for other fields
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='mouse'
    )

    # Axis
    p_rs.xaxis.axis_label = 'TIME'
    p_rs.yaxis.axis_label = 'PRICE'

    # Tools
    p_rs.add_tools(ohlc_hover, crosshair)
    return p_rs


@njit
def progress_bar(i, len_data):
    percentage = i / len_data

    p = int((percentage * 10))
    q = int((1 - percentage) * 10)
    with objmode():
      print("\r", end="")
      print("[" + "#" * p + "_" * q + "]", int(percentage * 100), "% complete", end="")