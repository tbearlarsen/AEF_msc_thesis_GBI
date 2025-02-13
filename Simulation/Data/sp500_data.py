import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Create a Ticker object for ^GSPC
gspc = yf.Ticker("^GSPC")

# Retrieve the maximum available historical data with weekly intervals
sp500 = gspc.history(period="max", interval="1wk")


#EXPLORE THE DATA
sp500.head()
sp500.info()
sp500.describe()
sp500.isnull().sum()

# Remove timezone information from the datetime index
sp500.index = sp500.index.tz_localize(None)

# Write the DataFrame to an Excel file
sp500.to_excel("sp500.xlsx")

#Visulaise the data
plt.figure(figsize=(10,6))
plt.plot(sp500.index, sp500['Close'],label="S&P 500")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

#Separate data
close = sp500['Close']
close = close[close.index >= '1997-01-01']

