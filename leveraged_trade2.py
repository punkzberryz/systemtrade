import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import lib.repository.repository as repo
from datetime import timedelta
from scipy import stats

from leveraged_trading.plot import plot_returns
from leveraged_trading.syarter_system import StarterSystem

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

# start_date = "2000-01-03"
start_date = "2010-01-03"
end_date = None

sys = StarterSystem(
                    ticker="AES",
                    # ticker="JASIF.BK",
                    risk_target=0.12,
                    capital=1000,
                    cost_per_trade=1,
                    start_date=start_date,
                    stop_loss_fraction=0.5,
                    rule = {
                        # 'MBO': 20
                        'MAC': {'fast': 16, 'slow': 64},
                    }
                    # rules=rule_dict_no_carry
                    )

sys.trade()

df = sys.data
ticker = sys.ticker
capital = sys.capital

# benchmark using SPY
df_spy = yf.download("SPY", start=start_date, end=end_date)
df_spy["position"] = capital / df_spy["Close"].iloc[0]
df_spy["pandl"] = df_spy["position"].shift(1) * df_spy["Close"].diff()
df_spy["curve"] = df_spy["pandl"].cumsum()

plot_returns(sys.data, df_spy, ticker)
