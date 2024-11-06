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
repo = Repository()
# repo.add_instruments_by_codes(["AAPL", "MSFT", "GOOGL", "^GSPC"], fetch_data="yfinance", start_date="2008-01-01")
repo.add_instruments_by_codes(["AAPL", "MSFT", "GOOGL", "^GSPC", "^TYX", "EURUSD=X"], start_date="2008-01-01")
repo.get_instrument_list()
rule = TradingRule("ewma64_256", rule_func= lambda instrument : ewmac_forecast(instrument=instrument, Lfast=64, Lslow=256) , rule_arg={"Lfast": 64, "Lslow": 256})
aapl: Instrument = repo.instruments["AAPL"] 

instrument_codes = repo.instrument_codes
instr_list = repo.get_instrument_list()
rule.call_forecast(instr_list)

rule_list = [{ "name": "ewma64_256", "rule_func": lambda instrument : ewmac_forecast(instrument=instrument, Lfast=64, Lslow=256), "rule_arg": {"Lfast": 64, "Lslow": 256}},             
             { "name": "ewma16_64", "rule_func": lambda instrument : ewmac_forecast(instrument=instrument, Lfast=16, Lslow=64), "rule_arg": {"Lfast": 16, "Lslow": 64}},
             { "name": "ewma8_32", "rule_func": lambda instrument : ewmac_forecast(instrument=instrument, Lfast=8, Lslow=32), "rule_arg": {"Lfast": 8, "Lslow": 32}},
             { "name": "breakout20", "rule_func": lambda instrument : breakout_forecast(instrument, lookback=20), "rule_arg": {"lookback": 20}},]

rules = Rules(rule_list=rule_list, instrument_list=instr_list)
# let's plot some forecast

aapl_forecast_ewma64_256 = rules.get_forecast("AAPL", "ewma64_256")
aapl_forecast_ewma16_64 = rules.get_forecast("AAPL", "ewma16_64")
aapl_forecast_ewma8_32 = rules.get_forecast("AAPL", "ewma8_32")
aapl_forecast_breakout20 = rules.get_forecast("AAPL", "breakout20")
aapl_price = repo.get_instrument("AAPL").data["PRICE"]
aapl_vol = repo.get_instrument("AAPL").vol
SR_cost = 0.01

# next we want to calculate the P&L for each forecast, then we can later optimize the weights
aapl_forecast_ewma64_256_pos = get_position_from_forecast(forecast=aapl_forecast_ewma64_256, price=aapl_price, daily_returns_volatility=aapl_vol, capital=1e3)
aapl_forecast_ewma64_256_pandl = pandl_calculation(instrument_price=aapl_price, positions=aapl_forecast_ewma64_256_pos, capital=1e3, SR_cost=SR_cost, frequency=Frequency.BDay)
aapl_account_ewma64_256 = AccountCurve(aapl_forecast_ewma64_256_pandl, frequency=Frequency.BDay)
aapl_forecast_ewma16_64_pos = get_position_from_forecast(forecast=aapl_forecast_ewma16_64, price=aapl_price, daily_returns_volatility=aapl_vol, capital=1e3)
aapl_forecast_ewma16_64_pandl = pandl_calculation(instrument_price=aapl_price, positions=aapl_forecast_ewma16_64_pos, capital=1e3, SR_cost=SR_cost, frequency=Frequency.BDay)
aapl_account_ewma16_64 = AccountCurve(aapl_forecast_ewma16_64_pandl, frequency=Frequency.BDay)
aapl_forecast_ewma8_32_pos = get_position_from_forecast(forecast=aapl_forecast_ewma8_32, price=aapl_price, daily_returns_volatility=aapl_vol, capital=1e3)
aapl_forecast_ewma8_32_pandl = pandl_calculation(instrument_price=aapl_price, positions=aapl_forecast_ewma8_32_pos, capital=1e3, SR_cost=SR_cost, frequency=Frequency.BDay)
aapl_account_ewma8_32 = AccountCurve(aapl_forecast_ewma8_32_pandl, frequency=Frequency.BDay)
aapl_forecast_breakout20_pos = get_position_from_forecast(forecast=aapl_forecast_breakout20, price=aapl_price, daily_returns_volatility=aapl_vol, capital=1e3)
aapl_forecast_breakout20_pandl = pandl_calculation(instrument_price=aapl_price, positions=aapl_forecast_breakout20_pos, capital=1e3, SR_cost=SR_cost, frequency=Frequency.BDay)
aapl_account_breakout20 = AccountCurve(aapl_forecast_breakout20_pandl, frequency=Frequency.BDay)

#let's combine the forecasts

returns = pd.concat([aapl_forecast_ewma64_256_pandl, aapl_forecast_ewma16_64_pandl, aapl_forecast_ewma8_32_pandl, aapl_forecast_breakout20_pandl], axis=1)
returns.columns = ["ewma64_256", "ewma16_64", "ewma8_32", "breakout20"]
forecasts = pd.concat([aapl_forecast_ewma64_256, aapl_forecast_ewma16_64, aapl_forecast_ewma8_32, aapl_forecast_breakout20], axis=1)
forecasts.columns = ["ewma64_256", "ewma16_64", "ewma8_32", "breakout20"]

combined_forecast = combine_forecasts_for_instrument(forecast_returns=returns, forecasts=forecasts, fit_method="one_period", max_forecast_cap=20.0)
combined_pos = get_position_from_forecast(forecast=combined_forecast, price=aapl_price, daily_returns_volatility=aapl_vol, capital=1e3)
combined_pandl = pandl_calculation(instrument_price=aapl_price, positions=combined_pos, capital=1e3, SR_cost=SR_cost, frequency=Frequency.BDay)
combined_account = AccountCurve(combined_pandl, frequency=Frequency.BDay)



'''
# correlation estimator for subperiod
if floor_at_zero:
    corr_matrix_values[corr_matrix_values < 0] = 0
if clip is not arg_not_supplied:    
    ### clip upper values
    corr_matrix_values[corr_matrix_values > clip] = clip
    ### clip lower values
    corr_matrix_values[corr_matrix_values < -clip] = -clip
# calculate forecast diversification multiplier from 1 / (sqrt( W x H x W^T))
#convert weight index to datetime
weights.index = pd.to_datetime(weights.index)
weights.resample("B").sum()
weights_0 = weights.iloc[0]
h = corr_matrix_values

a = weights_0.dot(h)
b = a.dot(weights_0.transpose())
mult = 1 / np.sqrt(b)



### now let's combine forecasts
### first we multiply weights to its forecast
# first we need to re-index weights to match forecast
weights_reindex = weights.reindex(forecasts.index, method="ffill")
# then we multiply
weighted_forecasts = forecasts * weights_reindex
# then we combine
combined_forecast = weighted_forecasts.sum(axis=1)
# and multiply by diversification multiplier
combined_forecast_scaled = combined_forecast * mult
# don't forget to capped to +/- 20
combined_forecast_scaled[combined_forecast_scaled > 20] = 20
combined_forecast_scaled[combined_forecast_scaled < -20] = -20

## now let's try to calculate p&l for combined forecast
combined_pos = get_position_from_forecast(forecast=combined_forecast_scaled, price=aapl_price, daily_returns_volatility=aapl_vol, capital=1e3)
combined_pandl = pandl_calculation(instrument_price=aapl_price, positions=combined_pos, capital=1e3, SR_cost=SR_cost, frequency=Frequency.BDay)
combined_account = AccountCurve(combined_pandl, frequency=Frequency.BDay)
combined_account.curve().plot()
plt.show()
'''
'''






aapl_account_ewma16_64 = pandl_for_instrument_forecast(forecast=aapl_forecast_ewma16_64, price=aapl_price, daily_returns_volatility=aapl_vol, capital=1e3, SR_cost=SR_cost)
aapl_account_ewma8_32 = pandl_for_instrument_forecast(forecast=aapl_forecast_ewma8_32, price=aapl_price, daily_returns_volatility=aapl_vol, capital=1e3, SR_cost=SR_cost)
aapl_account_breakout20 = pandl_for_instrument_forecast(forecast=aapl_forecast_breakout20, price=aapl_price, daily_returns_volatility=aapl_vol, capital=1e3, SR_cost=SR_cost)

aapl_account = pd.concat([aapl_account_ewma64_256, aapl_account_ewma16_64, aapl_account_ewma8_32, aapl_account_breakout20], axis=1)
aapl_curve = pd.concat([aapl_account_ewma64_256.curve(), aapl_account_ewma16_64.curve(), aapl_account_ewma8_32.curve(), aapl_account_breakout20.curve()], axis=1)

aapl_account.columns = ["ewma64_256", "ewma16_64", "ewma8_32", "breakout20"]
aapl_curve.columns = ["ewma64_256", "ewma16_64", "ewma8_32", "breakout20"]
# output = markosolver(aapl_account)




#equalisevols
default_vol=0.2
returns = aapl_account
factors = (default_vol/16.0)/returns.std(axis=0)
index=returns.index
dullvalue=np.array([factors]*len(index))
dullname = returns.columns
# facmat=pd.DataFrame(dullvalue, index, columns=["A"])
facmat = pd.DataFrame(dullvalue, index, columns=dullname)
norm_returns=returns*facmat.values
norm_returns.columns=returns.columns
use_returns = norm_returns
sigma=use_returns.cov().values
mus=np.array([use_returns[asset_name].mean() for asset_name in use_returns.columns], ndmin=2).transpose()
## Starting weights
number_assets=use_returns.shape[1]
start_weights=[1.0/number_assets]*number_assets
bounds=[(0.0,1.0)]*number_assets
def addem(weights):
    ## Used for constraints
    return 1.0 - sum(weights)
cdict=[{'type':'eq', 'fun':addem}]

def variance(weights, sigma):
    ## returns the variance (NOT standard deviation) given weights and sigma
    return (np.matrix(weights)*sigma*np.matrix(weights).transpose())[0,0]

def neg_SR(weights, sigma, mus):
    ## Returns minus the Sharpe Ratio (as we're minimising)

    """    
    estreturn=250.0*((np.matrix(x)*mus)[0,0])
    variance=(variance(x,sigma)**.5)*16.0
    """
    estreturn=(np.matrix(weights)*mus)[0,0]
    std_dev=(variance(weights,sigma)**.5)

    
    return -estreturn/std_dev

ans=minimize(neg_SR, start_weights, (sigma, mus), method='SLSQP', bounds=bounds, constraints=cdict, tol=0.00001)
print(ans['x'])
# aapl_account_ewma64_256.curve().plot()
# aapl_account_ewma16_64.curve().plot()
# aapl_account_ewma8_32.curve().plot()
# aapl_account_breakout20.curve().plot()
# plt.show()

### optimization


'''