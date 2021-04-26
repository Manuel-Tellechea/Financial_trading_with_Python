import talib
import matplotlib.pyplot as plt
import bt

stock_data = bt.get('goog', start='2020-6-1', end='2020-12-1')

# Calculate RSI
stock_data['RSI'] = talib.RSI(stock_data['Close'])

# Create subplots
fig, (ax1, ax2) = plt.subplots(2)
# Plot RSI with the price
ax1.set_ylabel('Price')
ax1.plot(stock_data['Close'])
ax2.set_ylabel('RSI')
ax2.plot(stock_data['RSI'], color='orangered')

ax1.set_title('Price and RSI')
plt.show()
