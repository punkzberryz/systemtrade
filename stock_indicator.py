import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from stock_indicator.fetch_data import DataFetcher
from stock_indicator.forecast import ForecastSystem
start_date = "2014-01-03"

#let's firstly fetch data
aaplData = DataFetcher("AAPL")
aaplData.fetch_yf_data(start_date=start_date)

aaplForecast = ForecastSystem(ticker="AAPL",
                            #   start_date=start_date
                             )
# aaplForecast.data["EWMAC16_64"].plot()
aaplForecast.data["SIGNAL"].plot()

# aaplForecast.data["CARRY"].plot()

plt.show()