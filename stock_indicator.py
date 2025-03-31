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

#### let's plot the signal of last 3 months

# Get the current date with UTC timezone
current_date = pd.Timestamp.now(tz='UTC')

# Calculate the date 90 days ago (with timezone)
date_90_days_ago = current_date - pd.Timedelta(days=90)
signals = buystrong.signals[buystrong.signals.index >= date_90_days_ago]
prices = buystrong.prices[buystrong.prices.index >= date_90_days_ago]


# Normalize the prices
# normalized_prices = buystrong.prices[buystrong.prices.index >= date_90_days_ago].apply(lambda x: (x - x.mean()) / x.std())
# prices_percentage_change = normalized_prices.pct_change().dropna()

# Normalize the prices by scaling between 0 and 1 using the first value as reference
prices = buystrong.prices[buystrong.prices.index >= date_90_days_ago]
normalized_prices = prices.div(prices.iloc[0])


# Plotting the signals with different colors and line styles
# Plotting signals and prices in subplots
fig, axs = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyles = ['-', '--', '-.', ':']

# Plot signals on the first subplot
for i, column in enumerate(signals.columns):
    axs[0].plot(signals[column], linestyle=linestyles[i % len(linestyles)], color=colors[i % len(colors)], label=column)

axs[0].set_ylabel('Signal Value')
axs[0].set_title('Signals from Buy Strong Strategy (Last 3 Months)')
axs[0].legend(loc='upper left')

# Plot prices on the second subplot

for i, column in enumerate(normalized_prices.columns):
    axs[1].plot(normalized_prices[column], linestyle=linestyles[i % len(linestyles)], color=colors[i % len(colors)], label=column,
                # marker='o'
                )
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Price Change ($/$)')
axs[1].set_title('Normalized Prices of Instruments (Last 3 Months)')
axs[1].legend(loc='upper left')

# Adjust layout
plt.tight_layout()
plt.show()