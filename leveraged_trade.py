import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import lib.repository.repository as repo
from datetime import timedelta
from scipy import stats

from leveraged_trading.plot import plot_returns
from leveraged_trading.starter_system import StarterSystem
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

# start_date = "2000-01-03"
start_date = "2010-01-03"
end_date = None
ticker = "HAL"
capital = 1000
sys = StarterSystem(
                    ticker=ticker,
                    # ticker="JASIF.BK",
                    risk_target=0.12,
                    capital=capital,
                    cost_per_trade=1,
                    start_date=start_date,
                    stop_loss_fraction=0.5,
                    rule = {
                        # 'MBO': 20
                        'MAC': {'fast': 16, 'slow': 64},
                    }                    
                    )

sys_multi_rule = StarterSystemMultiRule(
                    ticker=ticker,
                    # ticker="JASIF.BK",
                    risk_target=0.12,
                    capital=capital,
                    cost_per_trade=1,
                    start_date=start_date,
                    stop_loss_fraction=0.5,
                    rules=rule_dict_no_carry
)

# sys.trade()
sys_multi_rule.trade()
df_spy = benchmark_data(ticker="SPY", start_date=start_date, capital=capital)

# plot_returns(sys.data, df_spy, ticker)
plot_returns(sys_multi_rule.data, df_spy, ticker)

# strat_stats = pd.DataFrame(
#     getStats(sys.data["curve"], capital=capital),
#     index=["Strategy"]
# )
# buy_and_hold_stats = pd.DataFrame(
#     getStats(sys.data["buy_and_hold_curve"], capital=capital),
#     index=["Buy and Hold"]
# )
# spy_stats = pd.DataFrame(
#     getStats(df_spy["curve"], capital=capital),
#     index=["SPY"]
# )
# stats = pd.concat([strat_stats, buy_and_hold_stats, spy_stats])
# print(stats)