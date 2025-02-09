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
from leveraged_trading.calculator import calculate_instrument_risk, calculate_robust_instrument_risk, calculate_position_size, calculate_stop_loss
from leveraged_trading.optimization import _equalise_vols, generate_fitting_dates,_addem,_neg_SR, minimize
from leveraged_trading.trading_portfolio import TradingPortfolio
from leveraged_trading.rules.ewmac import ewmac_forecast
from leveraged_trading.rules.breakout import breakout_forecast
from leveraged_trading.rules.carry import carry_forecast
# start_date = "2000-01-03"
# start_date = "2010-01-03"
# start_date = "2014-01-03"
start_date = "2022-10-10"
end_date = None
ticker = "USO"
capital = 1000


rule_names = ["ewmac2_8", "ewmac4_16", "ewmac8_32", "ewmac16_64", "ewmac32_128", "ewmac64_256",
              "breakout10", "breakout20", "breakout40", "breakout80", "breakout160", "breakout320",
              "carry",
            #   "carry_fx",
              ]
rule_names2 = ["ewmac2_8", "ewmac4_16", "ewmac8_32", "ewmac16_64", "ewmac32_128", "ewmac64_256",
              "breakout10", "breakout20", "breakout40", "breakout80", "breakout160", "breakout320",
            #   "carry",
              # "carry_fx",
              ]
rule_names3 = ["ewmac2_8", "ewmac4_16", "ewmac8_32", "ewmac16_64", "ewmac32_128", "ewmac64_256",
              "breakout10", "breakout20", "breakout40", "breakout80", "breakout160", "breakout320",
            #   "carry",
              "carry_fx",
              ]
# rules = TradingRules(rule_names=rule_names)

tickers = [
        #mag7
        'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'TSLA'
        #stocks
        # "AAPL", "AES", "HAL", "LULU", "CPALL.BK", "JASIF.BK",
        # "MSFT", "GOOGL", "META", "SPY", 
        # #commodities
        # "CORN", "USO", "IAU",
        # #fx
        # "EURUSD=X", "AUDUSD=X", "USDJPY=X",
        # #crypto
        # "BTC-USD", "ETH-USD", 
        # #volatility
        # "VIXM",
        # #bond
        # # "UTHY", "SHY",
        # "IEF",
        # "JASIF.BK", "CPALL.BK", "TCAP.BK", "HMPRO.BK", "MEGA.BK", "HANA.BK", "COM7.BK"
        ]
rules = TradingRules(instrument_tickers=tickers)

port = TradingPortfolio(trading_rules=rules,
                        # optimization_method="bootstrap",
                        risk_target=0.3,
                        start_date=start_date
                        )
# port.add_instrument("AAPL",
#                     risk_target=0.3,
#                     capital=capital,
#                     rule_names=rule_names,
#                     short_cost=0.02,
#                     margin_cost=0.07,
#                     interest_on_balance=0.001,
#                     deviation_in_exposure_to_trade=1,
#                     )
# port.add_instrument("MSFT",
#                     risk_target=0.3,
#                     capital=capital,
#                     rule_names=rule_names,
#                     # rule_names=["breakout10","ewmac8_32","carry_fx",],
#                     short_cost=0.02,
#                     margin_cost=0.07,
#                     interest_on_balance=0.001,
#                     deviation_in_exposure_to_trade=1,
#                     )
# port.add_instrument("GOOGL",
#                     risk_target=0.3,
#                     capital=capital,
#                     rule_names=rule_names,
#                     # rule_names=["breakout10","ewmac8_32","carry_fx",],
#                     short_cost=0.02,
#                     margin_cost=0.07,
#                     interest_on_balance=0.001,
#                     deviation_in_exposure_to_trade=1,
#                     )
# port.add_instrument("META",
#                     risk_target=0.3,
#                     capital=capital,
#                     rule_names=rule_names,
#                     # rule_names=["breakout10","ewmac8_32","carry_fx",],
#                     short_cost=0.02,
#                     margin_cost=0.07,
#                     interest_on_balance=0.001,
#                     deviation_in_exposure_to_trade=1,
#                     )
# port.add_instrument("AMZN",
#                     risk_target=0.3,
#                     capital=capital,
#                     rule_names=rule_names,
#                     # rule_names=["breakout10","ewmac8_32","carry_fx",],
#                     short_cost=0.02,
#                     margin_cost=0.07,
#                     interest_on_balance=0.001,
#                     deviation_in_exposure_to_trade=1,
#                     )
# port.add_instrument("NVDA",
#                     risk_target=0.3,
#                     capital=capital,
#                     rule_names=rule_names,
#                     # rule_names=["breakout10","ewmac8_32","carry_fx",],
#                     short_cost=0.02,
#                     margin_cost=0.07,
#                     interest_on_balance=0.001,
#                     deviation_in_exposure_to_trade=1,
#                     )
# port.add_instrument("TSLA",
#                     risk_target=0.3,
#                     capital=capital,
#                     rule_names=rule_names,
#                     # rule_names=["breakout10","ewmac8_32","carry_fx",],
#                     short_cost=0.02,
#                     margin_cost=0.07,
#                     interest_on_balance=0.001,
#                     deviation_in_exposure_to_trade=1,
#                     )

# port.get_simulated_stats()
# port.optimize(
#   # fit_method="one_period"
#   )


# ief = port.instruments["IEF"]
# ief.risk_target = 0.12
# ief.re_trade(rules, optimization_method="one_period")
