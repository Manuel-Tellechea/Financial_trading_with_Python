import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
# matplotlib.use('TkAgg')

# Load the data
apple_data = pd.read_csv('AAPL.csv', index_col='Date', parse_dates=True)

# Print the top 5 rows
print(apple_data.head(5))

# Plot the daily high price
plt.plot(apple_data['High'], color='green')
# Plot the daily low price
plt.plot(apple_data['Low'], color='red')

plt.title('Daily high low prices')
plt.show()
