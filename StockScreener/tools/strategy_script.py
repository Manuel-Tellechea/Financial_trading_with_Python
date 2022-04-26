import warnings
from numba import njit, objmode
import numpy as np
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, CrosshairTool
import sys

# TURN ON OR OFF TO GENERATE GRAPHIC VALIDATION
generate_graphic_validation = False
skynet = True
# SET YOUR STRATEGY TYPE (order_book_strategy, normal, intermarket, intertemp)
strategy_type = "normal"

if generate_graphic_validation or not skynet:
    from utils.ewm_tools import please_set_me_all_inter, get_logging, minute_exit, minute_entry, \
        exit_type, graphic_validation_vectors, stop_loss_condition, profit_target_condition, trailing_stop_update, \
        backstage_data_copies, backstage_data_frame_joining, inter_graphic_validation, progress_bar

    from utils.Indicators import rsi, cci, cumsum

if skynet:
    from ..utils.ewm_tools import please_set_me_all_inter, get_logging, minute_exit, minute_entry, \
        exit_type, graphic_validation_vectors, stop_loss_condition, profit_target_condition, trailing_stop_update, \
        backstage_data_copies, backstage_data_frame_joining, inter_graphic_validation, progress_bar
    from ..utils.Indicators import rsi, cci, cummax

np.set_printoptions(threshold=sys.maxsize)
# Pandas output settings
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.width', 1000)
# Hide Pandas Future Warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

LOGGER = True
if LOGGER:
    get_logging(stage=0, message='oil_rsi_cci.py')


@njit
def backtesting(np_data_type_float, instrument_details, capital, direction, exit_type_value,
                profit_target_param, stop_loss_param):
    # ---------------------- Same for all strategies! ---------------------- #
    capital_line, condition, trades_line, max_periods, trade_number,\
    instrument_type, instrument_margin, instrument_tick_size, instrument_contract_size, instrument_fee,\
    np_data_open_x, np_data_high_x, np_data_low_x, np_data_close_x, np_data_volume_x, len_np_data, \
    np_data_open_y, np_data_high_y, np_data_low_y, np_data_close_y, np_data_volume_y, \
    np_data_open_z, np_data_high_z, np_data_low_z, np_data_close_z, np_data_volume_z, \
    trade_index, exit_index, capital_0, entry_index, exit_condition, size, len_data \
        = please_set_me_all_inter(capital, instrument_details, np_data_type_float, 12, 50000, 0)

    entry_date, entry_price, profit_target_price, stop_loss_price, \
    position_size, entry_position_size_usd, total_fee, margin_total \
        = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    # ---------------------- Same for all strategies! ---------------------- #

    # set your needed indicators here:
    signal = np_data_type_float[16]
    exit_condition1 = np_data_type_float[17]

    entry_number = 0
    stop_loss_vector, stop_loss_price, profit_target_vector, profit_target_price, trailing_stop = \
        graphic_validation_vectors(len_np_data=len_np_data)

    for i in range(max_periods, len_np_data - 1):

        if condition:
            trades_line[trade_number * size + 12] = stop_loss_price  # 1 means Long, 0 LongSHort and -1 Short
            trades_line[trade_number * size + 13] = profit_target_price  # 1 means Long, 0 LongSHort and -1 Short

        if generate_graphic_validation:
            progress_bar(i=i, len_data=len_np_data)

        if np.isnan(signal[i]) or np.isnan(signal[i + 1]):
            continue

        if i % 30_000 == 0 and LOGGER:
            with objmode():
                get_logging(stage=1, message='Im inside long/short, attempt number i our of len_np_data: ',
                            message_variable=(i, len_np_data))

        # For graphic validation
        if condition and exit_type_value == 1:
            stop_loss_vector[i] = stop_loss_price
        elif condition and (exit_type_value == 2 or exit_type_value == 4):
            stop_loss_vector[i] = stop_loss_price
            profit_target_vector[i] = profit_target_price
        elif condition and exit_type_value == 3:
            trailing_stop[i] = stop_loss_price

        # EXIT STOPLOSS - TRAILING STOP CONDITION
        if condition and i != entry_date and i + 1 < len_np_data:

            if stop_loss_condition(close=np_data_close_y[i], stop_loss_price=stop_loss_price, direction=direction):

                trades_line_, capital_, capital_line_, exit_index_ = minute_exit(index=i,
                                                                                 next_open_price=np_data_open_y[i + 1],
                                                                                 position_size=position_size,
                                                                                 instrument_contract_size=
                                                                                 instrument_contract_size,
                                                                                 entry_position_size_usd=
                                                                                 entry_position_size_usd,
                                                                                 total_fee=total_fee,
                                                                                 direction=direction,
                                                                                 instrument_type=instrument_type,
                                                                                 instrument_tick_size=
                                                                                 instrument_tick_size,
                                                                                 trade_index=trade_index,
                                                                                 capital=capital,
                                                                                 capital_line=capital_line,
                                                                                 trades_line=trades_line,
                                                                                 trade_number=trade_number, size=size,
                                                                                 exit_strategy=False)

                if capital_ < 0:
                    if LOGGER:
                        with objmode():
                            get_logging(stage=1, message='EWM breaking execution, the strategy ran out of money!')
                    break

                trades_line = trades_line_
                capital = capital_
                capital_line = capital_line_
                exit_index = exit_index_
                condition = False
                exit_condition_, exit_condition = False, False
                trade_number += 1

        # EXIT CONDITION
        if condition and exit_type_value != 3 and i != entry_date and i + 1 < len_np_data:

            if exit_type_value == 1 or exit_type_value == 4:

                # SETUP YOUR EXIT CONDITION
                exit_condition = False

            if exit_type_value == 2 or exit_type_value == 4:

                exit_condition_ = profit_target_condition(close=np_data_close_y[i],
                                                          profit_target_price=profit_target_price, direction=direction)

                exit_condition = (exit_condition or exit_condition_)

            if exit_condition:

                trades_line_, capital_, capital_line_, exit_index_ = minute_exit(index=i,
                                                                                 next_open_price=
                                                                                 np_data_open_y[i + 1],
                                                                                 position_size=position_size,
                                                                                 instrument_contract_size=
                                                                                 instrument_contract_size,
                                                                                 entry_position_size_usd=
                                                                                 entry_position_size_usd,
                                                                                 total_fee=total_fee,
                                                                                 direction=direction,
                                                                                 instrument_type=instrument_type,
                                                                                 instrument_tick_size=
                                                                                 instrument_tick_size,
                                                                                 trade_index=trade_index,
                                                                                 capital=capital,
                                                                                 capital_line=capital_line,
                                                                                 trades_line=trades_line,
                                                                                 trade_number=trade_number,
                                                                                 size=size, exit_strategy=True)

                if capital_ < 0:
                    if LOGGER:
                        with objmode():
                            get_logging(stage=1, message='EWM breaking execution, the strategy ran out of money!')
                    break

                trades_line = trades_line_
                capital = capital_
                capital_line = capital_line_
                exit_index = exit_index_
                condition = False
                exit_condition_, exit_condition = False, False
                trade_number += 1

        # ENTRY CONDITION
        # SETUP YOUR ENTRY CONDITION
        if not condition and i + 1 < len_np_data and trade_index[i] != exit_index \
                and signal[i]:
            try:
                trades_line_, entry_position_size_usd_, entry_index_, profit_target_price_, stop_loss_price_, total_fee_, \
                    position_size_ = minute_entry(index=i, next_open_price=np_data_open_y[i + 1],
                                                  profit_target_param=profit_target_param,
                                                  stop_loss_param=stop_loss_param,
                                                  instrument_contract_size=instrument_contract_size,
                                                  instrument_margin=instrument_margin, direction=direction,
                                                  instrument_type=instrument_type,
                                                  instrument_tick_size=instrument_tick_size,
                                                  trade_index=trade_index, capital=capital,
                                                  capital_0=capital_0,
                                                  instrument_fee=instrument_fee, trades_line=trades_line,
                                                  trade_number=trade_number, size=size, logger=True)
            except:
                break

            trades_line = trades_line_
            entry_position_size_usd = entry_position_size_usd_
            entry_index = entry_index_
            profit_target_price = profit_target_price_
            stop_loss_price = stop_loss_price_
            total_fee = total_fee_
            position_size = position_size_

            condition = True
            entry_number += 1
            continue

        # TRAILING STOP VALUE ACTUALIZATION (ORDERBOOK STRATEGIES CLOSE VALUE IS DIFFERENT FROM THE OTHERS)
        if condition and entry_index != trade_index[i] and exit_type_value == 3:
            stop_loss_price = trailing_stop_update(direction=direction, close=np_data_close_x[i],
                                                   stop_loss_param=stop_loss_param, stop_loss_price=stop_loss_price)
            trailing_stop[i] = stop_loss_price

    # Cleaning the results (erasing original np.NaN out of the strings)
    capital_line = capital_line[:trade_number + 1]  # we need to add +1, remember the first element is the init capital
    trades_line = trades_line[:entry_number * size]

    # transforming trades_line from array to matrix of shape (number of trades, size columns)
    np_trades = np.reshape(trades_line, (-1, size))

    # RETURNS
    return np_trades, capital_line, profit_target_vector, stop_loss_vector, trailing_stop


@njit
def strategy(data, parameters, instrument_details, capital, direction, all_signals=False):
    if LOGGER:
        with objmode:
            get_logging(stage=1, message='Using parameters', message_variable=parameters)
            get_logging(stage=1, message='Instrument details', message_variable=instrument_details)
            get_logging(stage=1, message='Direction', message_variable=direction)

    # SET YOUR STRATEGY TYPE (sl_strategy, sl_tp, trailing, sl_tp_time)
    exit_type_value = exit_type("trailing")

    window_observation = int(parameters[0])  # evaluation window
    rsi_overbought_level = int(parameters[1])  # rsi_overbought_level
    rsi_oversold_level = int(parameters[2])  # rsi_oversold_level
    profit_target_param = np.float32(np.NaN)  # profit_target
    stop_loss_param = np.float32(parameters[3])  # stop_loss

    np_open_x, np_open_y, np_open_z = data[0], data[5], data[10]
    np_high_x, np_high_y, np_high_z = data[1], data[6], data[11]
    np_low_x, np_low_y, np_low_z = data[2], data[7], data[12]
    np_close_x, np_close_y, np_close_z = data[3], data[8], data[13]
    np_volume_x, np_volume_y, np_volume_z = data[4], data[9], data[14]
    trade_index = data[15]

    # call your backstage indicators:
    oil_rsi_x = data[16]
    oil_cci = data[17]
    signal = data[18]
    exit_condition = data[19]

    np_data_with_indicators = np.vstack((np_open_x, np_high_x, np_low_x, np_close_x, np_volume_x,
                                         np_open_y, np_high_y, np_low_y, np_close_y, np_volume_y,
                                         np_open_z, np_high_z, np_low_z, np_close_z, np_volume_z,
                                         trade_index,
                                         signal, exit_condition))

    np_data_with_indicators = np_data_with_indicators.astype(np.float32)

    np_trades, capital_line, profit_target_vector, stop_loss_vector, trailing_stop = \
        backtesting(np_data_type_float=np_data_with_indicators, instrument_details=instrument_details, capital=capital,
                    direction=direction, exit_type_value=exit_type_value,
                    profit_target_param=profit_target_param, stop_loss_param=stop_loss_param)

    np_data_with_indicators_ = np.vstack((oil_rsi_x, oil_cci,
                                         profit_target_vector, stop_loss_vector, trailing_stop))

    if generate_graphic_validation:
        return np_trades, capital_line, np_data_with_indicators_
    else:
        return np_trades, capital_line


def backstage(data, parameters):
    """
    Documentation in progress, this function will handle the creation on indicators BEFORE the join of the datas,
    also it will handles the resample and the join itself.
    """

    # SAME FOR ALL STRATEGIES ################################################
    df1, df2, df3, order_book_df = backstage_data_copies(data=data, strategy_type=strategy_type)

    # Creating index for trades
    df1['trade_index'] = np.arange(df1.shape[0])
    ############################################################################################

    # SET YOUR INDICATORS ###################################################

    # creating indicators:
    high1 = df1['high'].values.astype(np.float32)
    low1 = df1['low'].values.astype(np.float32)
    close1 = df1['close'].values.astype(np.float32)

    # Parameters
    window_observation = parameters[0]
    rsi_overbought_level = parameters[1]
    rsi_oversold_level = parameters[2]

    oil_rsi = rsi(close1.astype(np.float32), np.int32(14))
    oil_cci = cci(high1.astype(np.float32), low1.astype(np.float32), close1.astype(np.float32), np.int32(20))

    signal1 = np.zeros(len(close1))
    count = 0
    wait = False
    wait2 = False
    for i in range(len(close1)):
        if not wait and oil_rsi[i] > rsi_overbought_level:
            count += 1
            if count == window_observation:
                count = 0
                wait = True
        elif wait and oil_rsi[i] < rsi_oversold_level:
            wait2 = True
        elif wait2 and oil_cci[i] > 0:
            signal1[i] = True
            wait = False
            wait2 = False
        else:
            count = 0

    exit_condition = oil_cci < 0

    df1['oil_rsi'] = oil_rsi
    df1['oil_cci'] = oil_cci
    df1['signal1'] = signal1
    df1['exit_condition'] = exit_condition
    ############################################################################################

    # SAME FOR ALL STRATEGIES ################################################

    final_df = backstage_data_frame_joining(df1=df1, df2=df2, df3=df3,
                                            order_book_df=order_book_df,
                                            strategy_type=strategy_type)
    #############################################################################################

    # SET YOUR INDICATORS ###################################################

    final_df = final_df[['date', 'open_x', 'high_x', 'low_x', 'close_x', 'volume_x',
                                 'open_y', 'high_y', 'low_y', 'close_y', 'volume_y',
                                 'open_z', 'high_z', 'low_z', 'close_z', 'volume_z',
                                 'trade_index',  # SAME FOR ALL STRATEGIES
                                 'oil_rsi_x', 'oil_cci_x', 'signal1_x', 'exit_condition_x']]  # ADD YOUR INDICATORS
    #############################################################################################

    # SAME FOR ALL STRATEGIES ################################################
    data_graphic = final_df.copy()
    final_df = final_df.fillna(method='ffill')

    data_graphic = data_graphic[data_graphic.columns[~data_graphic.columns.str.endswith('_y')]]
    data_graphic = data_graphic.dropna(how='all', thresh=2)
    # final_df.to_csv(f'{"data"}.csv')

    if generate_graphic_validation:
        return final_df, data_graphic
    else:
        return final_df


# Generate validation graph
def graphic_validation(trades: pd.DataFrame, np_dates, np_data_with_indicators, data_graphic: pd.DataFrame,
                       parameter_combinations: list, object_variables):

    indicators = pd.DataFrame(np.transpose(np_data_with_indicators),
                              # ADD YOUR INDICATORS
                              columns=['oil_rsi_x', 'oil_cci_x',
                                       # SAME FOR ALL STRATEGIES
                                       'profit_target_vector', 'stop_loss_vector', 'trailing_stop'])

    candle_size = 5
    # SAME FOR ALL STRATEGIES ################################################
    p_1, p_2, is_data = inter_graphic_validation(data_graphic=data_graphic, np_dates=np_dates, indicators=indicators,
                                                 strategy_type=strategy_type, trades=trades,
                                                 object_variables=object_variables, candle_size=candle_size)

    # Indicators
    source_is_data = ColumnDataSource(is_data)

    # SET THE GRAPHIC AND LINES FOR YOUR INDICATORS ###################################################

    p_aux = figure(x_axis_type="datetime", plot_height=350, plot_width=1500, title="Relative Strength Index",
                   x_range=p_1.x_range)

    p_aux_2 = figure(x_axis_type="datetime", plot_height=350, plot_width=1500, title="Commodity Channel Index",
                     x_range=p_1.x_range)

    line1 = p_aux.line("date", "oil_rsi_x", line_color="purple", legend_label='OIL RSI',
                       line_width=1, source=source_is_data)
    line2 = p_aux_2.line("date", "oil_cci_x", line_color="black", legend_label='OIL CCI',
                         line_width=1, source=source_is_data)

    line1_hover = HoverTool(
        renderers=[line1],
        tooltips=[
            ('date', '@date{%Y-%m-%d %H:%M:%S}'),
            ('oil_rsi_x', '@oil_rsi_x{0.00}'),
        ],

        formatters={
            '@date': 'datetime',  # use 'datetime' formatter for '@date' field
            # use default 'numeral' formatter for other fields
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='vline'
    )

    line2_hover = HoverTool(
        renderers=[line2],
        tooltips=[
            ('date', '@date{%Y-%m-%d %H:%M:%S}'),
            ('oil_cci_x', '@oil_cci_x{0.00}'),
        ],

        formatters={
            '@date': 'datetime',  # use 'datetime' formatter for '@date' field
            # use default 'numeral' formatter for other fields
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='vline'
    )

    crosshair = CrosshairTool(dimensions='both')

    # Tools
    p_1.add_tools(crosshair)
    p_aux.add_tools(line1_hover, crosshair)
    p_aux_2.add_tools(line2_hover, crosshair)

    g = gridplot([[p_1], [p_aux], [p_aux_2]], sizing_mode='scale_width')

    output_file(f'{object_variables["strategy_name"]}_{object_variables["direction"]}_'
                f'{object_variables["temporality1"]}.html', title="graphic_validation", mode="cdn")
    show(g)
