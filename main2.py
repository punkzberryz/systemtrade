from lib.repository.repository import Repository, Instrument
from lib.util.object import get_methods
from lib.service.rules.trading_rules import TradingRule
from lib.service.rules.rules import Rules
from lib.service.rules.ewmac import ewmac_forecast
from lib.service.rules.breakout import breakout_forecast
from matplotlib import pyplot as plt
from lib.service.account.pandl_calculations.pandl_for_instrument_forecast import  get_position_from_forecast
from lib.service.vol import robust_daily_vol_given_price
import pandas as pd
import numpy as np
import datetime
from lib.service.optimization.optimization import markosolver, opt_and_plot, optimise_over_periods
from scipy.optimize import minimize
from lib.service.account.pandl_calculations.pandl_calculation import pandl_calculation
from lib.service.account.curves.account_curve import AccountCurve
from lib.util.frequency import Frequency
from lib.service.optimization.optimization import generate_fitting_dates
from lib.util.constants import arg_not_supplied
from lib.service.estimators.correlations import get_correlations
from lib.service.forecast_combine.diversification_multiplier import get_diversification_multiplier
from lib.service.forecast_combine.forecast_combine import combine_forecasts_for_instrument
from lib.service.portfolio.portfolio import Portfolio
from lib.service.account.pandl_calculations.simple_buy_and_hold import simple_buy_and_hold_pandl
# from lib.service.portfolio.vol_target import VolTargetParameters, PositionSizingConfig

repo = Repository()
# repo.add_instruments_by_codes(["AAPL", "MSFT", "GOOGL", "^GSPC"], fetch_data="yfinance", start_date="2008-01-01")
repo.add_instruments_by_codes(["AAPL", "MSFT", "GOOGL", "^GSPC", "^TYX", "EURUSD=X"], start_date="2008-01-01")
repo.get_instrument_list()
rule = TradingRule("ewma64_256", rule_func= lambda instrument : ewmac_forecast(instrument=instrument, Lfast=64, Lslow=256) , rule_arg={"Lfast": 64, "Lslow": 256})
aapl: Instrument = repo.instruments["AAPL"]

# add list of instruments into portfolio
# creating trading rules that will be used to forecast (we havn't assigned forecasts to instruments yet)
rule_list = [{ "name": "ewma64_256", "rule_func": lambda instrument : ewmac_forecast(instrument=instrument, Lfast=64, Lslow=256), "rule_arg": {"Lfast": 64, "Lslow": 256}},             
             { "name": "ewma16_64", "rule_func": lambda instrument : ewmac_forecast(instrument=instrument, Lfast=16, Lslow=64), "rule_arg": {"Lfast": 16, "Lslow": 64}},
             { "name": "ewma8_32", "rule_func": lambda instrument : ewmac_forecast(instrument=instrument, Lfast=8, Lslow=32), "rule_arg": {"Lfast": 8, "Lslow": 32}},
             { "name": "breakout20", "rule_func": lambda instrument : breakout_forecast(instrument, lookback=20), "rule_arg": {"lookback": 20}},
             { "name": "breakout10", "rule_func": lambda instrument : breakout_forecast(instrument, lookback=10), "rule_arg": {"lookback": 10}},
             ]
rules = Rules(rule_list=rule_list, instrument_list=repo.get_instrument_list())

# Initialize portfolio
port = Portfolio(
    repository=repo,
    rules=rules,
    instrument_code_list=["AAPL", "^TYX", "EURUSD=X"],
)




port.add_trading_rules_to_instrument("AAPL", ["ewma64_256", "ewma16_64", "ewma8_32", "breakout20", "breakout10"])
port.add_trading_rules_to_instrument("^TYX", ["ewma16_64", "ewma8_32", "breakout10"])
port.add_trading_rules_to_instrument("EURUSD=X", ["ewma64_256", "breakout10", "breakout20"])

position_aapl = port.get_instrument_position_from_forecast("AAPL")

# fit_method = "bootstrap"
# fit_method = "one_period"
# forecast_aapl = port.get_forecast_combined_for_instrument("AAPL", fit_method=fit_method)
# forecast_tyx = port.get_forecast_combined_for_instrument("^TYX", fit_method=fit_method)
# forecast_eurusd = port.get_forecast_combined_for_instrument("EURUSD=X", fit_method=fit_method)





# pos_aapl = get_position_from_forecast(
#     forecast=forecast_aapl,
#     price=repo.get_instrument("AAPL").data["PRICE"],
#     capital=1e3,
#     daily_returns_volatility=repo.get_instrument("AAPL").vol,)
# pandl_aapl = pandl_calculation(instrument_price=repo.get_instrument("AAPL").data["PRICE"],
#                                positions=pos_aapl,
#                                daily_returns_volatility=repo.get_instrument("AAPL").vol,
#                                capital=1e3,
#                                SR_cost=0.01,)
# account_aapl = AccountCurve(pandl_aapl)
# aapl = repo.get_instrument("AAPL")
# repo.get_instrument("EURUSD=X").data["returns"].hist()
# repo.get_instrument("^TYX").skew()
# print(aapl.sharpe())
# curve_aapl = port.get_curve_from_forecast("AAPL", fit_method=fit_method)
# curve_tyx = port.get_curve_from_forecast("^TYX", fit_method=fit_method)
# curve_eurusd = port.get_curve_from_forecast("EURUSD=X", fit_method=fit_method)


