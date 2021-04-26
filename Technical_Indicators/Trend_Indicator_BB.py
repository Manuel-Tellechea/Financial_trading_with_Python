import talib
import matplotlib.pyplot as plt
import bt

stock_data = bt.get('goog', start='2020-6-1', end='2020-12-1')

# Define the Bollinger Bands with 1-sd
upper_1sd, mid_1sd, lower_1sd = talib.BBANDS(stock_data['Close'], nbdevup=1, nbdevdn=1, timeperiod=20)
# Plot the upper and lower Bollinger Bands
plt.plot(stock_data['Close'], color='green', label='Price')
plt.plot(upper_1sd, color='tomato', label="Upper 1sd")
plt.plot(lower_1sd, color='tomato', label='Lower 2sd')

# Customize and show the plot
plt.legend(loc='upper left')
plt.title('Bollinger Bands (1sd)')
plt.show()
