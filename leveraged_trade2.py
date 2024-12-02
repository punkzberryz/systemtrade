import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import lib.repository.repository as repo
from datetime import timedelta
from scipy import stats
import random

from leveraged_trading.plot import plot_returns
from leveraged_trading.starter_system import StarterSystem
from leveraged_trading.trading_system import TradingSystem
from leveraged_trading.trading_rules import TradingRules
from leveraged_trading.starter_system_multi_rule import StarterSystemMultiRule
from leveraged_trading.calculator import getStats, benchmark_data
from leveraged_trading.optimization import generate_fitting_dates, optimise_over_periods
from leveraged_trading.diversification_multiplier import get_diversification_multiplier
from lib.service.vol import robust_daily_vol_given_price

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

# start_date = "2000-01-03"
# start_date = "2010-01-03"
start_date = "2016-01-03"
# start_date = "2024-10-10"
end_date = None
ticker = "USO"
capital = 1000

# rule_names = ["mac8_32", "mac16_64", "mac32_128", "mac64_256", "breakout20", "breakout40", "breakout80", "breakout160", "breakout320", "carry_fx"]
rule_names = ["mac8_32", "mac16_64","breakout20", "breakout40", "carry"]
# rule_names = ["mac8_32", "mac16_64", "mac32_128", "mac64_256", "breakout20", "breakout40", "breakout80", "breakout160", "breakout320", "carry"]
# rule_names = ["mac16_64", "mac32_128", "breakout20", "carry"]

rules = TradingRules(rule_names=rule_names)
aapl = TradingSystem(
                    ticker=ticker,
                    # ticker="JASIF.BK",
                    risk_target=0.12,
                    capital=capital,
                    cost_per_trade=1,
                    start_date=start_date,
                    stop_loss_fraction=0.5,
                    interest_on_balance=0,
                    margin_cost=0.07,
                    short_cost=0.02,                    
                    rules=rule_names,
                    trading_rules=rules,
                    # optimization_method="one_period",
                    optimization_method="bootstrap",
                    deviation_in_exposure_to_trade=0.1                    
                    )
pandl = aapl.data["curve_pre_cost"].diff()
ann_std = pandl.std() * np.sqrt(252)
total = pandl.sum()
divisor = aapl.number_of_years_trade
ann_mean = total / divisor
sharpe = ann_mean / ann_std
print(sharpe)
