import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import lib.repository.repository as repo
from datetime import timedelta
from scipy import stats

from leveraged_trading.plot import plot_returns
from leveraged_trading.starter_system import StarterSystem
from leveraged_trading.trading_system import TradingSystem
from leveraged_trading.trading_rules import TradingRules
from leveraged_trading.starter_system_multi_rule import StarterSystemMultiRule
from leveraged_trading.calculator import getStats, benchmark_data

rule_dict_no_carry = {
    'MAC' : {
        0: {'fast': 8,
            'slow': 32},
        1: {'fast': 16,
            'slow': 64},
        2: {'fast': 32,
            'slow': 128},
        3: {'fast': 64,
            'slow': 256}
    },
    'MBO': {
        0: 20,
        1: 40,
        2: 80,
        3: 160,
        4: 320
    }    
}
rule_dict_no_carry["CAR"] = {0 : True}

start_date = "2000-01-03"
# start_date = "2010-01-03"
# start_date = "2019-01-03"
# start_date = "2024-10-10"
end_date = None
ticker = "SPY"
capital = 1000

rules = TradingRules()
aapl = TradingSystem(
                    ticker=ticker,
                    # ticker="JASIF.BK",
                    risk_target=0.12,
                    capital=capital,
                    cost_per_trade=0,
                    start_date=start_date,
                    stop_loss_fraction=0.5,
                    interest_on_balance=0,
                    margin_cost=0.07,
                    short_cost=0.02,
                    # rules = ["mac16_64", "mac32_128", "breakout20", "breakout40", "carry"],
                    rules = ["mac8_32", "mac16_64", "mac32_128", "mac64_256", "breakout20", "breakout40", "breakout80", "breakout160", "breakout320", "carry"],
                    weights=[0.0625, 0.0625, 0.0625, 0.0625, 0.05, 0.05, 0.05, 0.05, 0.05, 0.5],
                    trading_rules=rules
                    )



plt.show()
aapl.portfolio.plot(figsize=(12, 8))
aapl.price.plot(figsize=(12, 8), secondary_y=True)
plt.show()


aapl.combined_signal.plot(figsize=(12, 8), label="Combined Signal")
aapl.position.plot(figsize=(12, 8), secondary_y=True, label="Position")
#add label

plt.show()

