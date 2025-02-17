import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from stock_indicator.fetch_data import DataFetcher
from stock_indicator.forecast.forecast import ForecastSystem, default_rules
from stock_indicator.portfolio import Portfolio
from stock_indicator.strategy.buy_strong_signal import BuyStrongSignal
from stock_indicator.strategy.buy_and_hold import BuyAndHold
from stock_indicator.strategy.benchmark import Benchmark
from stock_indicator.util import find_nearest_trading_date
from scipy.stats import skew


start_date = "2015-01-03"

dca_capital = 1000
instrument_list = [
    {
        'ticker': 'META',
        'rules': default_rules
    },
    {
        'ticker': 'NFLX',
        'rules': default_rules
    },
    {
        'ticker': 'AMZN',
        'rules': default_rules
    },
    {
        'ticker': 'AAPL',
        'rules': default_rules
    },
    {
        'ticker': 'NVDA',
        'rules': default_rules
    },
    {
        'ticker': 'CRWD',
        'rules': default_rules
    },
    {
        'ticker': 'TOST',
        'rules': default_rules
    },
    {
        'ticker': 'DUOL',
        'rules': default_rules
    },
    {
        'ticker': 'MNDY',
        'rules': default_rules
    },
    {
        'ticker': 'UBER',
        'rules': default_rules
    },
]
port = Portfolio(instrument_list=instrument_list)
refport = Portfolio(instrument_list=[{
    'ticker': 'SPY',
    'rules': default_rules
}])
benchmark = Benchmark(symbol="SPY",
                      dca_capital=dca_capital,
                      start_date=start_date)
buystrong = BuyStrongSignal(port=port,
                            dca_capital=dca_capital,
                            start_date=start_date)
buyandhold = BuyAndHold(port=port,                        
                        dca_capital=dca_capital,
                        start_date=start_date)

buystrong.get_latest_indicator()

curve = pd.DataFrame()
curve["BUY STRONG"] = buystrong.total_curve
curve["BUY AND HOLD"] = buyandhold.total_curve
curve["SPY"] = benchmark.total_curve

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(2, figsize=(15, 10))
ax[0].plot(curve.loc[start_date:], label=curve.columns)
ax[0].set_ylabel('Curve ($)')
ax[0].set_xlabel('Date')
ax[0].legend(loc=1)
ax[1].set_title(f'Cumulative Returns of each strategy')

ax[1].plot(buystrong.curve.loc[start_date:], label=buystrong.curve.columns)
ax[1].set_ylabel('Buy and hold Curve ($)')
ax[1].set_xlabel('Date')
ax[1].set_title(f'Cumulative Returns of each instrument in Buy Strong Strategy')
ax[1].legend(loc=2)
plt.tight_layout()
plt.show()