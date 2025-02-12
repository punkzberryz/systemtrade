import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from stock_indicator.fetch_data import DataFetcher
from stock_indicator.forecast import ForecastSystem, default_rules
from stock_indicator.portfolio import Portfolio
from stock_indicator.strategy.buy_strong_signal import BuyStrongSignal
start_date = "2014-01-03"

instrument_list = [
    {
        'ticker': 'AAPL',
        'rules': default_rules
    },
    {
        'ticker': 'TSLA',
        'rules': default_rules
    },
    {
        'ticker': 'NVDA',
        'rules': default_rules
    },
]
port = Portfolio(instrument_list=instrument_list)

strategy = BuyStrongSignal()
strategy.trade(port=port)
# aaplForecast = ForecastSystem(ticker="AAPL",
#                             #   start_date=start_date
#                              )
# aaplForecast.data["EWMAC16_64"].plot()
# aaplForecast.data["SIGNAL"].plot()
# aaplForecast.data["CARRY"].plot()

# plt.show()