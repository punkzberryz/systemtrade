from lib.repository.repository import Repository, Instrument
from lib.service.rules.rules import Rules
from lib.service.rules.trading_rules import TradingRule
from lib.service.rules.ewmac import ewmac_calc_vol, ewmac
from lib.service.rules.breakout import breakout
from lib.util.pandas.strategy_functions import replace_all_zeros_with_nan
from lib.service.forecast_scale_cap import forecast_scalar, group_forecasts, get_capped_forecast
from lib.service.optimization.optimization import generate_fitting_dates, bootstrap_portfolio, optimise_over_periods, markosolver, opt_and_plot
import matplotlib.pyplot as plt
from lib.service.forecast_combine import *

# if __name__ == "__main__":
    # repo = Repository("AAPL")
    # repo.fetch_yf_data()
    
aapl = Instrument("AAPL")
# start_date = "2020-01-01"
start_date = "2007-01-01"
# aapl.fetch_yf_data(start_date=start_date)
# aapl.export_data("aapl_data.csv")
aapl.import_data("./data/aapl_data.csv")
meta = Instrument("META")
meta.import_data("./data/meta_data.csv")
ba = Instrument("BA")
ba.import_data("./data/ba_data.csv")
eurd = Instrument("EURUSD=X")
eurd.import_data("./data/euro_dollar_data.csv")
us30 = Instrument("^TYX")
us30.import_data("./data/us30_data.csv")
sp500 = Instrument("^GSPC")
sp500.import_data("./data/sp500_data.csv")


repo = Repository()
for instr in [aapl, meta, ba, us30, eurd, sp500]:    
    repo.add_instrument(instr)


#warm-up

# forecast us30
vol = us30.daily_returns_volatility()
#get raw price
price = us30.data["PRICE"]
# forecast with trading rule
raw_forecast = ewmac(price, Lfast=32, Lslow=128, vol=vol)
# replace all zero values with NaN
raw_forecast = replace_all_zeros_with_nan(raw_forecast)

cs_forecasts = group_forecasts([raw_forecast])
# cs_forecasts = group_forecasts([raw_forecast, aapl_forecast, meta_forecast])
# scaled the forecast
scalar = forecast_scalar(cs_forecasts)
scaled_forecast = raw_forecast * scalar
# capped the forecast to be between -20 and 20
capped_scaled_forecast = get_capped_forecast(scaled_forecast)


# step 1: list instruments we want to forecast
# we use 3 instruments: EU-DOLLAR-X, META, US30

# step 2: select trading rule(s) for each instrument
# for EU-DOLLAR-X, we use 1) EWMA-CROSSOVER at 32 and 128 days and 2) Breakout at lookback=10 days
eurd_vol = eurd.daily_returns_volatility()
eurd_raw_forecast_ewma_32_128 = ewmac(eurd.data["PRICE"], Lfast=32, Lslow=128, vol=eurd_vol)
eurd_raw_forecast_ewma_32_128 = replace_all_zeros_with_nan(eurd_raw_forecast_ewma_32_128)
eurd_raw_forecast_breakout = breakout(eurd.data["PRICE"], lookback=10)
eurd_raw_forecast_breakout = replace_all_zeros_with_nan(eurd_raw_forecast_breakout)

# for META, we use 1) EWMA-CROSSOVER at 16 and 64 days
meta_vol = meta.daily_returns_volatility()
meta_raw_forecast_ewma_16_64 = ewmac(meta.data["PRICE"], Lfast=16, Lslow=64, vol=meta_vol)
meta_raw_forecast_ewma_16_64 = replace_all_zeros_with_nan(meta_raw_forecast_ewma_16_64)

# for US30, we use 1) EWMA-CROSSOVER at 64 and 256 days and 2) Breakout at lookback=20 days
us30_vol = us30.daily_returns_volatility()
us30_raw_forecast_ewma_64_256 = ewmac(us30.data["PRICE"], Lfast=64, Lslow=256, vol=us30_vol)
us30_raw_forecast_ewma_64_256 = replace_all_zeros_with_nan(us30_raw_forecast_ewma_64_256)
us30_raw_forecast_breakout = breakout(us30.data["PRICE"], lookback=20)
us30_raw_forecast_breakout = replace_all_zeros_with_nan(us30_raw_forecast_breakout)

# step 3: apply scaling and capping to each trading rule
# let's start with ewma_32_128
# ideally we should perform ewma_32_128 to all instruments, let's do that for ewma_32_128 but not for the rest (too much work!!)
# to get accurate scaling, data needs to be at least 20 years, else we can use forecast scalar from textbook
forecasts_ewma_64_256 = []
forecasts_ewma_32_128 = []
forecasts_ewma_16_64 = []
forecasts_breakout_20 = []
forecasts_breakout_10 = []

for instr in [aapl, meta, ba, us30, eurd, sp500]:
    vol = instr.daily_returns_volatility()
    ewma_64_256 = ewmac(instr.data["PRICE"], Lfast=64, Lslow=256, vol=vol)
    ewma_32_128 = ewmac(instr.data["PRICE"], Lfast=32, Lslow=128, vol=vol)
    ewma_16_64 = ewmac(instr.data["PRICE"], Lfast=16, Lslow=64, vol=vol)
    breakout_20 = breakout(instr.data["PRICE"], lookback=20)
    breakout_10 = breakout(instr.data["PRICE"], lookback=10)
    forecasts_ewma_64_256.append(ewma_64_256)
    forecasts_ewma_32_128.append(ewma_32_128)
    forecasts_ewma_16_64.append(ewma_16_64)
    forecasts_breakout_20.append(breakout_20)
    forecasts_breakout_10.append(breakout_10)

cs_forecasts_ewma_64_256 = group_forecasts(forecasts_ewma_64_256)    
cs_forecasts_ewma_32_128 = group_forecasts(forecasts_ewma_32_128)
cs_forecasts_ewma_16_64 = group_forecasts(forecasts_ewma_16_64)
cs_forecasts_breakout_20 = group_forecasts(forecasts_breakout_20)
cs_forecasts_breakout_10 = group_forecasts(forecasts_breakout_10)

scalar_ewma_64_256 = forecast_scalar(cs_forecasts_ewma_64_256)
scalar_ewma_32_128 = forecast_scalar(cs_forecasts_ewma_32_128)
scalar_ewma_16_64 = forecast_scalar(cs_forecasts_ewma_16_64)
scalar_breakout_20 = forecast_scalar(cs_forecasts_breakout_20)
scalar_breakout_10 = forecast_scalar(cs_forecasts_breakout_10)

# now less apply scalar to each forecast
eurd_scaled_forecast_ewma_32_128 = eurd_raw_forecast_ewma_32_128 * scalar_ewma_32_128
eurd_scaled_forecast_breakout = eurd_raw_forecast_breakout * scalar_breakout_10

meta_scaled_forecast_ewma_16_64 = meta_raw_forecast_ewma_16_64 * scalar_ewma_16_64

us30_scaled_forecast_ewma_64_256 = us30_raw_forecast_ewma_64_256 * scalar_ewma_64_256
us30_scaled_forecast_breakout = us30_raw_forecast_breakout * scalar_breakout_20

# apply capped forecast to each forecast
eurd_capped_scaled_forecast_ewma_32_128 = get_capped_forecast(eurd_scaled_forecast_ewma_32_128)
eurd_capped_scaled_forecast_breakout = get_capped_forecast(eurd_scaled_forecast_breakout)
meta_capped_scaled_forecast_ewma_16_64 = get_capped_forecast(meta_scaled_forecast_ewma_16_64)
us30_capped_scaled_forecast_ewma_64_256 = get_capped_forecast(us30_scaled_forecast_ewma_64_256)
us30_capped_scaled_forecast_breakout = get_capped_forecast(us30_scaled_forecast_breakout)

# step 4: combine forecasts
#we do simple weighted average for now

eurd_forecasts2 = eurd_capped_scaled_forecast_ewma_32_128 * 0.5 + eurd_capped_scaled_forecast_breakout * 0.5
# meta_forecasts = meta_capped_scaled_forecast_ewma_16_64
# us30_forecasts = us30_capped_scaled_forecast_ewma_64_256 * 0.5 + us30_capped_scaled_forecast_breakout * 0.5


